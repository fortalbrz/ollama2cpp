"""Tests for the optimizer's benchmark sweep, validation and persistence (all subprocess/HTTP seams faked)."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from ollama2cpp import config, llamacpp, optimizer, proc, server as server_runtime
from ollama2cpp.config import Config, Profile, profile_key
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.llamacpp import LlamaCppError, ServeSettings
from ollama2cpp.ollama import MEDIA_MODEL, Layer, OllamaModel
from ollama2cpp.optimizer import (
    MIN_CTX,
    BenchResult,
    SeedPlan,
    candidate_steps,
    optimize_model,
    parse_bench_json,
    plan_to_bench_args,
    run_bench,
    seed_to_args,
    sweep,
    unload_ollama,
    validate,
    vram_preflight,
)
from tests.conftest import free_port
from tests.gguf_builder import write_gguf

SAMPLE = (Path(__file__).parent / "data" / "llama_bench.json").read_text(encoding="utf-8")
BENCH = Path("llama-bench")
MODEL = Path("model.gguf")
BLOCKS = 32


def _host(cores: int = 8, logical: int = 16) -> HostInfo:
    return HostInfo(devices=(), ram_total_mib=32768, ram_free_mib=16384, physical_cores=cores, logical_cores=logical)


def _seed(**kwargs: Any) -> SeedPlan:
    base = SeedPlan(ngl=BLOCKS + 1, ctx=8192, flash_attn=True, threads=8, batch=2048, ubatch=512, cache_type="f16")
    return replace(base, **kwargs)


def _flag(argv: Sequence[str], name: str) -> str:
    return argv[list(argv).index(name) + 1]


def _bench_json(pp: float | None, tg: float | None) -> str:
    rows: list[dict[str, Any]] = []
    if pp is not None:
        rows.append({"n_prompt": 512, "n_gen": 0, "avg_ts": pp})
    if tg is not None:
        rows.append({"n_prompt": 0, "n_gen": 128, "avg_ts": tg})
    return json.dumps(rows)


class FakeBench:
    """Replacement for ``proc.run`` that answers like ``llama-bench`` using ``speed(argv) -> (pp, tg) | str``."""

    def __init__(self, speed: Callable[[list[str]], tuple[float, float] | str], clock: Any = None, cost: float = 0.0):
        self.speed = speed
        self.calls: list[list[str]] = []
        self.clock = clock
        self.cost = cost

    def __call__(self, argv: Sequence[str], *, timeout: float | None = None) -> proc.ProcResult:
        argv = list(argv)
        self.calls.append(argv)
        if self.clock is not None:
            self.clock.now += self.cost
        answer = self.speed(argv)
        if isinstance(answer, str):
            return proc.ProcResult(1, "", answer)
        return proc.ProcResult(0, _bench_json(*answer), "ggml_vulkan: noise\n")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeBench) -> FakeBench:
    monkeypatch.setattr(proc, "run", fake)
    return fake


def _tg_by_ubatch(table: dict[int, float]) -> Callable[[list[str]], tuple[float, float]]:
    def speed(argv: list[str]) -> tuple[float, float]:
        return 100.0, table.get(int(_flag(argv, "-ub")), 10.0)

    return speed


# --- parse_bench_json -------------------------------------------------------------------------------------------


def test_parse_bench_json_real_sample():
    pp, tg = parse_bench_json(SAMPLE)
    assert pp is not None
    assert tg is not None
    assert pp > 0
    assert tg > 0


def test_parse_bench_json_real_sample_values():
    assert parse_bench_json(SAMPLE) == pytest.approx((608.367910, 10.274604))


@pytest.mark.parametrize("text", ["", "   ", "not json", "{", "[]", "{}", "[1, 2]", "null", '"str"'])
def test_parse_bench_json_total_on_garbage(text: str):
    assert parse_bench_json(text) == (None, None)


def test_parse_bench_json_missing_rows():
    assert parse_bench_json(_bench_json(50.0, None)) == (50.0, None)
    assert parse_bench_json(_bench_json(None, 7.5)) == (None, 7.5)


def test_parse_bench_json_ignores_bad_rows():
    rows = [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": "fast"},
        {"n_prompt": 512, "n_gen": 0, "avg_ts": True},
        {"n_prompt": 512, "n_gen": 0, "avg_ts": float("nan")},
        {"n_prompt": 0, "n_gen": 0, "avg_ts": 5.0},
        {"n_prompt": 512, "n_gen": 0},
        "junk",
        {"n_prompt": "x", "n_gen": 0, "avg_ts": 1.0},
        {"n_prompt": 0, "n_gen": 16, "avg_ts": 12.0},
    ]
    assert parse_bench_json(json.dumps(rows)) == (None, 12.0)


def test_parse_bench_json_tolerates_leading_noise_and_jsonl():
    noisy = "load_backend: loaded\n" + _bench_json(10.0, 4.0)
    assert parse_bench_json(noisy) == (10.0, 4.0)
    lines = "\n".join(json.dumps(r) for r in json.loads(_bench_json(10.0, 4.0)))
    assert parse_bench_json(lines) == (10.0, 4.0)
    single = json.dumps({"n_prompt": 0, "n_gen": 8, "avg_ts": 3.0})
    assert parse_bench_json(single) == (None, 3.0)


# --- plan_to_bench_args / run_bench -----------------------------------------------------------------------------


def test_plan_to_bench_args_matches_seed_and_adds_device():
    plan = _seed()
    assert plan_to_bench_args(plan) == seed_to_args(plan)
    args = plan_to_bench_args(plan, device="Vulkan0")
    assert args[-2:] == ["-dev", "Vulkan0"]
    assert args[:-2] == seed_to_args(plan)


def test_run_bench_builds_argv_and_parses(monkeypatch: pytest.MonkeyPatch):
    fake = _install(monkeypatch, FakeBench(lambda _argv: (321.0, 45.5)))
    result = run_bench(BENCH, MODEL, ["-ngl", "99", "-fa", "on"], timeout=60.0, repetitions=2)
    assert result == BenchResult(("-ngl", "99", "-fa", "on"), 45.5, 321.0)
    argv = fake.calls[0]
    assert argv[0] == str(BENCH)
    assert _flag(argv, "-m") == str(MODEL)
    assert _flag(argv, "-o") == "json"
    assert _flag(argv, "-r") == "2"
    assert argv[-4:] == ["-ngl", "99", "-fa", "on"]


def test_run_bench_tolerates_stderr_noise(monkeypatch: pytest.MonkeyPatch):
    def run(argv: Sequence[str], *, timeout: float | None = None) -> proc.ProcResult:
        return proc.ProcResult(0, _bench_json(9.0, 3.0), "warning: something\nggml_vulkan: Found 2 devices\n")

    monkeypatch.setattr(proc, "run", run)
    result = run_bench(BENCH, MODEL, ["-t", "4"], timeout=5.0)
    assert (result.pp_tps, result.tg_tps, result.error) == (9.0, 3.0, "")


@pytest.mark.parametrize(
    ("rc", "stdout", "stderr", "expected"),
    [
        (1, "", "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\nout of memory", "out of memory"),
        (1, "", "failed to allocate Vulkan0 buffer of size 123", "out of memory"),
        (0, "", "ggml_backend_vk: Failed to allocate memory", "out of memory"),
        (proc.RC_TIMEOUT, "", "", "timed out"),
        (proc.RC_NOT_FOUND, "", "no such file", "not found"),
        (3, "", "error: failed to load model\n  details", "exit code 3"),
        (0, "", "", "no generation"),
        (0, "garbage", "", "no generation"),
        (0, _bench_json(5.0, None), "", "no generation"),
    ],
)
def test_run_bench_failures_set_error_and_never_raise(
    monkeypatch: pytest.MonkeyPatch, rc: int, stdout: str, stderr: str, expected: str
):
    monkeypatch.setattr(proc, "run", lambda *_a, **_k: proc.ProcResult(rc, stdout, stderr))
    result = run_bench(BENCH, MODEL, ["-t", "4"], timeout=5.0)
    assert result.tg_tps is None
    assert expected in result.error
    assert "\n" not in result.error
    assert len(result.error) <= optimizer.MAX_ERROR_CHARS


def test_run_bench_oom_returns_error_not_raise(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(proc, "run", lambda *_a, **_k: proc.ProcResult(1, "", "ErrorOutOfDeviceMemory\x1b[0m\n" * 50))
    result = run_bench(BENCH, MODEL, [], timeout=5.0)
    assert result.error == "out of memory"


def test_run_bench_error_text_is_sanitised_and_truncated(monkeypatch: pytest.MonkeyPatch):
    stderr = "boom\x00\x1b[31m " + "x" * 1000
    monkeypatch.setattr(proc, "run", lambda *_a, **_k: proc.ProcResult(2, "", stderr))
    result = run_bench(BENCH, MODEL, [], timeout=5.0)
    assert "\x00" not in result.error
    assert "\x1b" not in result.error
    assert len(result.error) <= optimizer.MAX_ERROR_CHARS


# --- candidate_steps --------------------------------------------------------------------------------------------


def test_candidate_steps_dimension_order_and_values():
    steps = candidate_steps(_seed(ngl=20), _host(), allow_kv_quant=True, block_count=BLOCKS)
    assert [name for name, _ in steps] == ["flash_attn", "ubatch", "threads", "ngl", "cache_type"]
    by_name = dict(steps)
    assert [p.flash_attn for p in by_name["flash_attn"]] == [False, True]
    assert [p.ubatch for p in by_name["ubatch"]] == [256, 512, 1024, 2048]
    assert [p.threads for p in by_name["threads"]] == [6, 8, 10]
    assert sorted(p.ngl for p in by_name["ngl"]) == [16, 18, 22, 24]
    assert [p.cache_type for p in by_name["cache_type"]] == ["q8_0"]


def test_kv_quant_only_when_opted_in():
    steps = candidate_steps(_seed(ngl=20), _host(), allow_kv_quant=False, block_count=BLOCKS)
    assert "cache_type" not in [name for name, _ in steps]
    assert all(p.cache_type != "q8_0" for _, plans in steps for p in plans)


def test_candidate_steps_ngl_only_when_partial():
    full = candidate_steps(_seed(ngl=BLOCKS + 1), _host(), allow_kv_quant=False, block_count=BLOCKS)
    assert "ngl" not in [name for name, _ in full]
    unknown = candidate_steps(_seed(ngl=5), _host(), allow_kv_quant=False)
    assert "ngl" not in [name for name, _ in unknown]


def test_candidate_steps_ngl_clamped():
    steps = dict(candidate_steps(_seed(ngl=1), _host(), allow_kv_quant=False, block_count=BLOCKS))
    assert sorted(p.ngl for p in steps["ngl"]) == [0, 3, 5]
    near_full = dict(candidate_steps(_seed(ngl=BLOCKS), _host(), allow_kv_quant=False, block_count=BLOCKS))
    assert max(p.ngl for p in near_full["ngl"]) == BLOCKS + 1
    assert all(0 <= p.ngl <= BLOCKS + 1 for p in near_full["ngl"])


def test_candidate_steps_threads_deduped_and_positive():
    one = dict(candidate_steps(_seed(threads=1), _host(cores=1, logical=1), allow_kv_quant=False))
    assert [p.threads for p in one["threads"]] == [1]
    two = dict(candidate_steps(_seed(threads=2), _host(cores=2, logical=3), allow_kv_quant=False))
    assert [p.threads for p in two["threads"]] == [1, 2, 3]
    capped = dict(candidate_steps(_seed(threads=8), _host(cores=8, logical=8), allow_kv_quant=False))
    assert [p.threads for p in capped["threads"]] == [6, 8]


# --- sweep ------------------------------------------------------------------------------------------------------


def _sweep(**kwargs: Any) -> list[BenchResult]:
    params: dict[str, Any] = {"time_limit": 1e9, "block_count": BLOCKS}
    params.update(kwargs)
    return sweep(BENCH, MODEL, params.pop("seed", _seed()), _host(), **params)


def test_sweep_picks_fastest_tg(monkeypatch: pytest.MonkeyPatch):
    _install(monkeypatch, FakeBench(_tg_by_ubatch({256: 20.0, 512: 30.0, 1024: 55.0, 2048: 40.0})))
    results = _sweep()
    winners = max((r for r in results if r.tg_tps is not None), key=lambda r: r.tg_tps or 0.0)
    assert winners.tg_tps == pytest.approx(55.0)
    assert _flag(winners.args, "-ub") == "1024"


def test_sweep_carries_best_into_later_dimensions(monkeypatch: pytest.MonkeyPatch):
    fake = _install(monkeypatch, FakeBench(_tg_by_ubatch({1024: 55.0})))
    _sweep()
    thread_calls = [c for c in fake.calls if _flag(c, "-t") != "8"]
    assert thread_calls
    assert all(_flag(c, "-ub") == "1024" for c in thread_calls)


def test_sweep_never_remeasures_identical_candidates(monkeypatch: pytest.MonkeyPatch):
    fake = _install(monkeypatch, FakeBench(_tg_by_ubatch({})))
    results = _sweep()
    keys = [tuple(c[c.index("-ngl") :]) for c in fake.calls]
    assert len(keys) == len(set(keys))
    assert len(results) == len(fake.calls)
    assert len({r.args for r in results}) == len(results)


def test_sweep_tie_breaks_on_prompt_speed(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        return {"256": 90.0, "512": 100.0, "1024": 120.0, "2048": 80.0}[_flag(argv, "-ub")], 30.0

    fake = _install(monkeypatch, FakeBench(speed))
    _sweep()
    later = [c for c in fake.calls if _flag(c, "-t") != "8"]
    assert later
    assert all(_flag(c, "-ub") == "1024" for c in later)


def _res(tg: float | None, pp: float | None = 100.0) -> BenchResult:
    return BenchResult(("x",), tg, pp)


def test_beats_ignores_noise_level_generation_differences():
    assert not optimizer._beats(_res(30.5, 100.0), _res(30.0, 100.0))  # +1.7 %: noise
    assert not optimizer._beats(_res(29.5, 100.0), _res(30.0, 100.0))
    assert not optimizer._beats(_res(30.1, 101.0), _res(30.0, 100.0))  # pp gain is noise too


def test_beats_takes_a_real_generation_improvement():
    assert optimizer._beats(_res(30.7, 50.0), _res(30.0, 100.0))  # +2.3 %, even with slower prompts
    assert optimizer._beats(_res(45.0), _res(30.0))


def test_beats_within_tolerance_needs_a_real_prompt_speed_gain():
    assert optimizer._beats(_res(30.2, 103.0), _res(30.0, 100.0))  # tg tie, pp +3 %
    assert optimizer._beats(_res(29.8, 103.0), _res(30.0, 100.0))  # tg a hair lower but within tolerance
    assert not optimizer._beats(_res(30.0, 101.0), _res(30.0, 100.0))
    assert not optimizer._beats(_res(30.0, 100.0), _res(30.0, 100.0))


def test_beats_rejects_a_clearly_slower_candidate_whatever_its_prompt_speed():
    assert not optimizer._beats(_res(25.0, 900.0), _res(30.0, 100.0))


def test_beats_failed_and_first_results():
    assert not optimizer._beats(_res(None), _res(30.0))
    assert optimizer._beats(_res(1.0), None)
    assert optimizer._beats(_res(1.0), _res(None))
    assert optimizer._beats(_res(30.0, None), _res(30.0, None)) is False  # no pp on either side: incumbent stays


def test_sweep_keeps_the_incumbent_when_generation_speed_only_wiggles(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        tg = {"256": 30.2, "512": 30.0, "1024": 30.4, "2048": 29.9}[_flag(argv, "-ub")]
        return 100.0, tg

    fake = _install(monkeypatch, FakeBench(speed))
    _sweep()
    later = [c for c in fake.calls if _flag(c, "-t") != "8"]
    assert later
    assert all(_flag(c, "-ub") == "512" for c in later)  # the seed's ubatch stays: no step gained 2 %


def test_sweep_skips_failed_candidates(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float] | str:
        ub = _flag(argv, "-ub")
        if ub == "2048":
            return "ggml_vulkan: ErrorOutOfDeviceMemory"
        return 100.0, {"256": 20.0, "512": 30.0, "1024": 25.0}[ub]

    fake = _install(monkeypatch, FakeBench(speed))
    results = _sweep()
    failed = [r for r in results if r.error]
    assert failed
    assert all(r.tg_tps is None for r in failed)
    ok = [r for r in results if not r.error]
    assert max(r.tg_tps or 0.0 for r in ok) == pytest.approx(30.0)
    later = [c for c in fake.calls if _flag(c, "-t") != "8"]
    assert later
    assert all(_flag(c, "-ub") == "512" for c in later)


def test_sweep_survives_failed_seed(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float] | str:
        if _flag(argv, "-ub") == "512":
            return "out of memory"
        return 100.0, 50.0

    _install(monkeypatch, FakeBench(speed))
    results = _sweep()
    assert results[0].error
    assert any(r.tg_tps == pytest.approx(50.0) for r in results)


def test_sweep_respects_time_limit_zero_measures_only_seed(monkeypatch: pytest.MonkeyPatch):
    clock = Clock()
    fake = _install(monkeypatch, FakeBench(_tg_by_ubatch({}), clock=clock, cost=10.0))
    results = _sweep(time_limit=0, clock=clock)
    assert len(results) == 1
    assert len(fake.calls) == 1
    assert results[0].args == tuple(seed_to_args(_seed()))


def test_sweep_stops_when_budget_is_used_up(monkeypatch: pytest.MonkeyPatch):
    clock = Clock()
    fake = _install(monkeypatch, FakeBench(_tg_by_ubatch({}), clock=clock, cost=10.0))
    results = _sweep(time_limit=25.0, clock=clock)
    assert len(fake.calls) == 3
    assert len(results) == 3


def test_keyboard_interrupt_keeps_best(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def speed(argv: list[str]) -> tuple[float, float]:
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return 100.0, 10.0 * len(calls)

    _install(monkeypatch, FakeBench(speed))
    results = _sweep()
    assert len(results) == 3
    assert all(r.tg_tps is not None for r in results)


def test_sweep_object_records_that_it_was_interrupted(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        if len(fake.calls) == 3:
            raise KeyboardInterrupt
        return 100.0, 10.0 * len(fake.calls)

    fake = _install(monkeypatch, FakeBench(speed))
    run = optimizer._Sweep(
        BENCH,
        MODEL,
        _seed(),
        _host(),
        time_limit=1e9,
        block_count=BLOCKS,
        allow_kv_quant=False,
        device=None,
        on_progress=None,
        clock=lambda: 0.0,
    )
    assert not run.interrupted
    run.run()
    assert run.interrupted
    assert len(run.results) == 2


def test_sweep_ngl_dimension_when_partial(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        return 100.0, 100.0 - abs(int(_flag(argv, "-ngl")) - 22)

    fake = _install(monkeypatch, FakeBench(speed))
    results = _sweep(seed=_seed(ngl=20))
    best = max((r for r in results if r.tg_tps is not None), key=lambda r: r.tg_tps or 0.0)
    assert _flag(best.args, "-ngl") == "22"
    assert {_flag(c, "-ngl") for c in fake.calls} == {"16", "18", "20", "22", "24"}


def test_sweep_kv_quant_requires_flash_attention(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        return 100.0, 50.0 if _flag(argv, "-fa") == "off" else 10.0

    fake = _install(monkeypatch, FakeBench(speed))
    _sweep(allow_kv_quant=True, seed=_seed(flash_attn=False))
    assert all(_flag(c, "-ctk") == "f16" for c in fake.calls)


def test_sweep_kv_quant_measured_when_opted_in(monkeypatch: pytest.MonkeyPatch):
    def speed(argv: list[str]) -> tuple[float, float]:
        return 100.0, 80.0 if _flag(argv, "-ctk") == "q8_0" else 50.0

    fake = _install(monkeypatch, FakeBench(speed))
    results = _sweep(allow_kv_quant=True)
    quant = [c for c in fake.calls if _flag(c, "-ctk") == "q8_0"]
    assert len(quant) == 1
    assert _flag(quant[0], "-ctv") == "q8_0"
    assert _flag(quant[0], "-fa") == "on"
    assert results[-1].tg_tps == pytest.approx(80.0)


def test_sweep_passes_device_and_reports_progress(monkeypatch: pytest.MonkeyPatch):
    fake = _install(monkeypatch, FakeBench(_tg_by_ubatch({})))
    seen: list[BenchResult] = []
    results = _sweep(device="Vulkan0", on_progress=seen.append)
    assert seen == results
    assert all(_flag(c, "-dev") == "Vulkan0" for c in fake.calls)


def test_sweep_all_failures_returns_error_results(monkeypatch: pytest.MonkeyPatch):
    _install(monkeypatch, FakeBench(lambda _argv: "boom"))
    results = _sweep()
    assert results
    assert all(r.error for r in results)


# --- vram_preflight / unload_ollama ---------------------------------------------------------------------------


def test_vram_preflight_shortfall():
    device = Device("Vulkan0", "GPU", 8000, 600)
    assert vram_preflight(7000, device) == 6400


def test_vram_preflight_ok_and_missing_device():
    assert vram_preflight(100, Device("Vulkan0", "GPU", 8000, 600)) == 0
    assert vram_preflight(7000, None) == 0


def test_vram_preflight_model_larger_than_the_card_can_only_use_the_whole_card():
    assert vram_preflight(20000, Device("Vulkan0", "GPU", 8000, 8000)) == 0
    assert vram_preflight(20000, Device("Vulkan0", "GPU", 8000, 7000)) == 1000  # not 13000


def test_vram_preflight_need_below_free_is_no_shortfall():
    assert vram_preflight(5000, Device("Vulkan0", "GPU", 8000, 5000)) == 0
    assert vram_preflight(5000, Device("Vulkan0", "GPU", 8000, 7000)) == 0


def test_vram_preflight_small_and_large_holds():
    # 7544 needed on the real host: the desktop holds ~770 MiB (small), a game holds ~4900 MiB (large)
    assert vram_preflight(7544, Device("Vulkan0", "GPU", 7891, 7123)) == 421
    assert vram_preflight(7544, Device("Vulkan0", "GPU", 7891, 3000)) == 4544


def test_vram_preflight_unknown_free_memory_is_no_shortfall():
    assert vram_preflight(7000, Device("Vulkan0", "GPU", 8000, 0)) == 0  # free 0 means "not reported"


class _OllamaHandler(BaseHTTPRequestHandler):
    loaded: list[str] = []
    posts: list[dict[str, Any]] = []
    fail_posts = False

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/ps":
            self._send(200, {"models": [{"name": n, "model": n} for n in self.loaded]})
        else:
            self._send(404, {})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/api/generate" and not self.fail_posts:
            self.posts.append(json.loads(raw))
            self._send(200, {"done": True})
        else:
            self._send(500, {"error": "no"})

    def log_message(self, format: str, *args: Any) -> None:  # ruff: ignore[builtin-argument-shadowing]
        pass


@pytest.fixture
def ollama_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_OllamaHandler]]:
    handler = type("Handler", (_OllamaHandler,), {"loaded": [], "posts": [], "fail_posts": False})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{httpd.server_address[1]}")
    yield handler
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_unload_ollama_unloads_each_loaded_model(ollama_api: type[_OllamaHandler]):
    ollama_api.loaded = ["a:latest", "b:7b"]
    assert unload_ollama(timeout=5.0) is True
    assert ollama_api.posts == [{"model": "a:latest", "keep_alive": 0}, {"model": "b:7b", "keep_alive": 0}]


def test_unload_ollama_nothing_loaded(ollama_api: type[_OllamaHandler]):
    assert unload_ollama(timeout=5.0) is True
    assert ollama_api.posts == []


def test_unload_ollama_reports_failed_unload(ollama_api: type[_OllamaHandler]):
    ollama_api.loaded = ["a:latest"]
    ollama_api.fail_posts = True
    assert unload_ollama(timeout=5.0) is False


def test_unload_ollama_unreachable_returns_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(optimizer, "_http_json", lambda *_a, **_k: None)
    assert unload_ollama() is False


def test_unload_ollama_unreachable_real_socket(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{free_port()}")
    assert unload_ollama(timeout=2.0) is False


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (None, "http://127.0.0.1:11434"),
        ("", "http://127.0.0.1:11434"),
        ("localhost", "http://localhost:11434"),
        ("10.0.0.5:9999", "http://10.0.0.5:9999"),
        ("0.0.0.0:11500", "http://127.0.0.1:11500"),
        ("http://box:1234", "http://box:1234"),
        ("https://box", "https://box:11434"),
        (":8000", "http://127.0.0.1:8000"),
        ("[::1]:5000", "http://[::1]:5000"),
        ("host:notaport", "http://127.0.0.1:11434"),
    ],
)
def test_ollama_base_url(monkeypatch: pytest.MonkeyPatch, env: str | None, expected: str):
    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    assert optimizer._ollama_base_url() == expected


def test_http_json_roundtrip_and_failures(ollama_api: type[_OllamaHandler], monkeypatch: pytest.MonkeyPatch):
    base = optimizer._ollama_base_url()
    assert optimizer._http_json("GET", f"{base}/api/ps", None, 5.0) == (200, {"models": []})
    assert optimizer._http_json("GET", f"{base}/nope", None, 5.0) == (404, {})
    assert optimizer._http_json("POST", f"{base}/api/generate", {"x": 1}, 5.0) == (200, {"done": True})
    assert optimizer._http_json("GET", "http://127.0.0.1:1/x", None, 1.0) is None
    assert optimizer._http_json("GET", "not a url", None, 1.0) is None


# --- validate ---------------------------------------------------------------------------------------------------


def _settings(tmp_path: Path) -> ServeSettings:
    return ServeSettings(
        model_path=tmp_path / "m.gguf", alias="m", host="0.0.0.0", port=8080, ctx=4096, profile_args=["-ngl", "9"]
    )


class ServerSeam:
    """Fakes ``server.start`` / ``server.stop`` and records the calls."""

    def __init__(self, start_error: BaseException | None = None) -> None:
        self.start_error = start_error
        self.starts: list[dict[str, Any]] = []
        self.stops: list[Path | None] = []

    def start(self, argv: Sequence[str], **kwargs: Any) -> int:
        self.starts.append({"argv": list(argv), **kwargs})
        if self.start_error is not None:
            raise self.start_error
        return 0

    def stop(self, base: Path | None = None, *, timeout: float = 10.0) -> bool:
        self.stops.append(base)
        return True


def _seam(monkeypatch: pytest.MonkeyPatch, seam: ServerSeam) -> ServerSeam:
    monkeypatch.setattr(server_runtime, "start", seam.start)
    monkeypatch.setattr(server_runtime, "stop", seam.stop)
    return seam


def test_validate_success_uses_free_port_and_always_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seam = _seam(monkeypatch, ServerSeam())
    requests: list[tuple[str, str, Any]] = []

    def fake_http(method: str, url: str, payload: Any, timeout: float) -> tuple[int, Any]:
        requests.append((method, url, payload))
        return 200, {"content": " world"}

    monkeypatch.setattr(optimizer, "_http_json", fake_http)
    assert validate(Path("llama-server"), _settings(tmp_path), base=tmp_path) is True
    start = seam.starts[0]
    assert start["port"] not in {8080, 0}
    assert start["host"] == "127.0.0.1"
    assert start["detach"] is True
    assert start["base"] == tmp_path / "validate"
    assert "--port" in start["argv"]
    assert _flag(start["argv"], "--port") == str(start["port"])
    assert _flag(start["argv"], "-c") == "4096"
    assert seam.stops == [tmp_path / "validate"]
    method, url, payload = requests[0]
    assert method == "POST"
    assert url == f"http://127.0.0.1:{start['port']}/completion"
    assert payload == {"prompt": "Hello", "n_predict": 8}


@pytest.mark.parametrize(
    "error", [LlamaCppError("not ready"), OSError("spawn failed"), TimeoutError("slow")], ids=["llama", "os", "timeout"]
)
def test_validate_start_failure_returns_false_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    seam = _seam(monkeypatch, ServerSeam(start_error=error))
    assert validate(Path("llama-server"), _settings(tmp_path), base=tmp_path) is False
    assert len(seam.stops) == 1


def test_validate_interrupt_stops_the_probe_and_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seam = _seam(monkeypatch, ServerSeam(start_error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        validate(Path("llama-server"), _settings(tmp_path), base=tmp_path)
    assert seam.stops == [tmp_path / "validate"]  # the probe server never outlives Ctrl-C


@pytest.mark.parametrize(
    "answer", [None, (500, {}), (200, {"nope": 1}), (200, ["content"]), (200, {"content": 5})], ids=str
)
def test_validate_bad_completion_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Any):
    seam = _seam(monkeypatch, ServerSeam())
    monkeypatch.setattr(optimizer, "_http_json", lambda *_a, **_k: answer)
    assert validate(Path("llama-server"), _settings(tmp_path), base=tmp_path) is False
    assert len(seam.stops) == 1


def test_validate_default_base_resolves_home_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA2CPP_HOME", str(tmp_path / "h"))
    seam = _seam(monkeypatch, ServerSeam())
    monkeypatch.setattr(optimizer, "_http_json", lambda *_a, **_k: (200, {"content": ""}))
    assert validate(Path("llama-server"), _settings(tmp_path)) is True
    assert seam.starts[0]["base"] == tmp_path / "h" / "validate"


def test_free_port_is_bindable_and_not_default():
    port = optimizer._free_port()
    assert 0 < port < 65536
    assert port != 8080


# --- optimize_model ---------------------------------------------------------------------------------------------


DIGEST = "sha256:abc123"


class Env:
    """Everything ``optimize_model`` needs, with the process/HTTP seams faked."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.base = tmp_path / "store"
        self.blob = write_gguf(tmp_path / "blob.gguf", n_layer=8)
        self.model = OllamaModel(
            name="tiny:latest",
            manifest_path=tmp_path / "manifest",
            layers=(Layer(MEDIA_MODEL, DIGEST, 1000),),
            blob_path=self.blob,
            digest=DIGEST,
        )
        self.cfg = Config()
        self.host = _host()
        self.device: Device | None = Device("Vulkan0", "Fake GPU", 8192, 8192)
        self.validated: list[ServeSettings] = []
        self.verdicts: Callable[[ServeSettings], bool] = lambda _s: True
        self.bench = FakeBench(_tg_by_ubatch({1024: 55.0}))
        self.found = {"llama-server": Path("llama-server"), "llama-bench": Path("llama-bench")}
        monkeypatch.setattr(proc, "run", self.bench)
        monkeypatch.setattr(llamacpp, "find_binary", lambda name, _configured=None: self.found.get(name))
        monkeypatch.setattr(llamacpp, "version", lambda _binary: "b12721")
        monkeypatch.setattr(optimizer, "validate", self._validate)

    def _validate(self, server: Path, settings: ServeSettings, *, base: Path | None = None, timeout: float = 180.0):
        self.validated.append(settings)
        return self.verdicts(settings)

    def run(self, **kwargs: Any) -> Profile:
        params: dict[str, Any] = {
            "ctx": 8192,
            "quick": False,
            "time_limit": 1e9,
            "allow_kv_quant": False,
            "base": self.base,
        }
        params.update(kwargs)
        return optimize_model(self.model, self.cfg, self.host, self.device, **params)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _expected_seed(env: Env, ctx: int = 8192) -> SeedPlan:
    return optimizer.heuristic_seed(optimizer.read_gguf(env.blob), env.host, env.device, ctx)


def test_optimize_quick_marks_unmeasured(env: Env):
    profile = env.run(quick=True)
    seed = _expected_seed(env)
    assert profile.measured is False
    assert profile.tg_tps is None
    assert profile.pp_tps is None
    assert profile.args == seed_to_args(seed)
    assert profile.ctx == seed.ctx
    assert profile.bench == []
    assert env.bench.calls == []
    assert env.validated == []
    saved = config.load(env.base)
    assert saved is not None
    assert saved.profiles[profile_key(DIGEST, seed.ctx)] == profile
    assert profile.fingerprint == {
        "gpu": "Fake GPU",
        "vram_mib": "8192",
        "ram_mib": "32768",
        "cores": "8",
        "llama_build": "b12721",
    }
    assert profile.timestamp.endswith("+00:00")


def test_optimize_full_run_saves_measured_winner(env: Env):
    seen: list[BenchResult] = []
    profile = env.run(on_progress=seen.append)
    assert profile.measured is True
    assert profile.tg_tps == pytest.approx(55.0)
    assert profile.pp_tps == pytest.approx(100.0)
    assert _flag(profile.args, "-ub") == "1024"
    assert "-c" not in profile.args
    assert profile.ctx == 8192
    assert seen
    assert {"args", "tg_tps", "pp_tps", "error"} == set(profile.bench[0])
    assert all(isinstance(row["args"], list) for row in profile.bench)
    assert len(env.validated) == 1
    settings = env.validated[0]
    assert settings.ctx == 8192
    assert settings.device == "Vulkan0"
    assert settings.alias == "tiny:latest"
    assert settings.model_path == env.blob
    assert settings.profile_args == profile.args
    assert _flag(env.bench.calls[0], "-dev") == "Vulkan0"
    saved = config.load(env.base)
    assert saved is not None
    assert saved.profiles[profile_key(DIGEST, 8192)].tg_tps == pytest.approx(55.0)


def test_optimize_preserves_other_profiles(env: Env):
    other = Profile(args=["-ngl", "1"], ctx=4096, measured=True, tg_tps=9.0)
    env.cfg.profiles["other:4096"] = other
    env.run(quick=True)
    saved = config.load(env.base)
    assert saved is not None
    assert saved.profiles["other:4096"] == other
    assert len(saved.profiles) == 2


def test_validation_failure_steps_down_ctx_then_succeeds(env: Env):
    env.verdicts = lambda s: s.ctx <= 4096
    profile = env.run()
    assert [s.ctx for s in env.validated] == [8192, 4096]
    assert profile.ctx == 4096
    assert profile.measured is True
    assert profile.tg_tps == pytest.approx(55.0)
    saved = config.load(env.base)
    assert saved is not None
    assert set(saved.profiles) == {profile_key(DIGEST, 4096), profile_key(DIGEST, 8192)}
    assert saved.profiles[profile_key(DIGEST, 8192)] == saved.profiles[profile_key(DIGEST, 4096)]
    assert saved.profiles[profile_key(DIGEST, 8192)].ctx == 4096


def test_validation_step_down_lowers_ngl_when_ctx_at_minimum(env: Env):
    env.verdicts = lambda s: _flag(s.profile_args, "-ngl") != "9"
    profile = env.run(ctx=MIN_CTX, time_limit=0)
    assert [_flag(s.profile_args, "-ngl") for s in env.validated] == ["9", "5"]
    assert profile.ctx == MIN_CTX
    assert _flag(profile.args, "-ngl") == "5"
    assert profile.measured is False
    assert profile.tg_tps is None


def test_validation_failure_steps_down_then_falls_back_to_seed(env: Env):
    env.verdicts = lambda _s: False
    profile = env.run()
    seed = _expected_seed(env)
    assert len(env.validated) == 2
    assert profile.measured is False
    assert profile.tg_tps is None
    assert profile.pp_tps is None
    assert profile.args == seed_to_args(seed)
    assert profile.ctx == seed.ctx
    assert profile.bench
    saved = config.load(env.base)
    assert saved is not None
    assert profile_key(DIGEST, seed.ctx) in saved.profiles


def test_validation_failure_at_minimum_ctx_steps_ngl_then_saves_seed(env: Env):
    env.verdicts = lambda _s: False
    profile = env.run(ctx=MIN_CTX, time_limit=0)
    assert profile.measured is False
    # ctx is already at the minimum, so the single step-down lowers ngl (9 -> 5); then the seed is saved
    assert len(env.validated) == 2
    assert profile.args == seed_to_args(_expected_seed(env, MIN_CTX))


def test_step_down_returns_none_when_exhausted():
    assert optimizer._step_down(_seed(ctx=MIN_CTX, ngl=0)) is None
    assert optimizer._step_down(_seed(ctx=8192)) == _seed(ctx=4096)
    assert optimizer._step_down(_seed(ctx=3000)) == _seed(ctx=MIN_CTX)
    assert optimizer._step_down(_seed(ctx=MIN_CTX, ngl=3)) == _seed(ctx=MIN_CTX, ngl=0)


def test_optimize_every_candidate_fails_saves_unmeasured_seed(env: Env):
    env.bench.speed = lambda _argv: "out of memory"
    profile = env.run()
    assert profile.measured is False
    assert profile.tg_tps is None
    assert profile.args == seed_to_args(_expected_seed(env))
    assert all(row["error"] for row in profile.bench)
    assert len(env.validated) == 1  # the seed itself is still validated


def test_optimize_without_llama_bench_validates_seed_unmeasured(env: Env):
    del env.found["llama-bench"]
    profile = env.run()
    assert profile.measured is False
    assert profile.bench == []
    assert len(env.validated) == 1
    assert profile.args == seed_to_args(_expected_seed(env))


def test_optimize_without_llama_server_raises(env: Env):
    del env.found["llama-server"]
    with pytest.raises(LlamaCppError, match="llama-server"):
        env.run()


def test_optimize_cpu_only_has_no_device(env: Env, monkeypatch: pytest.MonkeyPatch):
    env.device = None
    profile = env.run()
    assert env.validated[0].device is None
    assert all("-dev" not in c for c in env.bench.calls)
    assert profile.fingerprint["gpu"] == "cpu"
    assert _flag(profile.args, "-ngl") == "0"


def test_cpu_only_never_sweeps_ngl_and_keeps_flash_attn_off_for_seed(env: Env):
    env.device = None
    env.run(time_limit=1e9)
    assert env.bench.calls
    assert {_flag(c, "-ngl") for c in env.bench.calls} == {"0"}
    assert _flag(env.bench.calls[0], "-fa") == "off"


def test_quick_mode_stores_shrunk_seed_under_requested_and_actual_ctx(env: Env):
    env.device = Device("Vulkan0", "Small GPU", 1200, 1200)
    profile = env.run(quick=True, ctx=32768)
    assert profile.ctx < 32768
    saved = config.load(env.base)
    assert saved is not None
    assert set(saved.profiles) == {profile_key(DIGEST, profile.ctx), profile_key(DIGEST, 32768)}
    assert {p.ctx for p in saved.profiles.values()} == {profile.ctx}


def test_rerun_overwrites_both_keys_and_leaves_unrelated_key(env: Env):
    env.device = Device("Vulkan0", "Small GPU", 1200, 1200)
    other = Profile(args=["-ngl", "1"], ctx=4096, measured=True, tg_tps=9.0)
    env.cfg.profiles["other:4096"] = other
    first = env.run(quick=True, ctx=32768)
    second = env.run(quick=True, ctx=32768)
    assert second is not first
    saved = config.load(env.base)
    assert saved is not None
    assert set(saved.profiles) == {profile_key(DIGEST, first.ctx), profile_key(DIGEST, 32768), "other:4096"}
    assert saved.profiles["other:4096"] == other
    assert saved.profiles[profile_key(DIGEST, 32768)].timestamp == second.timestamp
    assert saved.profiles[profile_key(DIGEST, first.ctx)].timestamp == second.timestamp


def test_optimize_save_failure_never_raises(env: Env, monkeypatch: pytest.MonkeyPatch):
    def boom(cfg: Config, base: Path | None = None) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(config, "save", boom)
    profile = env.run(quick=True)
    assert profile.measured is False
    assert profile_key(DIGEST, profile.ctx) in env.cfg.profiles


def test_optimize_default_base_resolves_home_at_call_time(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA2CPP_HOME", str(tmp_path / "home"))
    env.run(quick=True, base=None)
    assert (tmp_path / "home" / "config.json").is_file()


def test_optimize_seed_budgets_the_free_vram_not_the_total(env: Env):
    # the model fits the 8192 MiB card, but only 520 MiB are free now: sweep with partial offload instead of OOM-ing
    env.device = Device("Vulkan0", "Fake GPU", 8192, 520)
    profile = env.run()
    info = optimizer.read_gguf(env.blob)
    full = info.block_count + 1
    assert _flag(env.bench.calls[0], "-ngl") != str(full)  # the seed is a partial offload
    assert {int(_flag(c, "-ngl")) for c in env.bench.calls} > {int(_flag(env.bench.calls[0], "-ngl"))}  # ngl was swept
    assert info.block_count == 8
    assert profile.fingerprint["vram_mib"] == "8192"  # the fingerprint still describes the card, not what is free


def test_optimize_seed_with_everything_free_is_a_full_offload_without_ngl_sweep(env: Env):
    env.run()
    assert {_flag(c, "-ngl") for c in env.bench.calls} == {"9"}


def test_optimize_model_larger_than_the_card_is_swept_with_partial_offload(env: Env):
    env.device = Device("Vulkan0", "Fake GPU", 512, 512)  # not even the runtime overhead fits
    env.run()
    assert len({_flag(c, "-ngl") for c in env.bench.calls}) > 1


def test_optimize_free_vram_unknown_falls_back_to_the_total(env: Env):
    env.device = Device("Vulkan0", "Fake GPU", 8192, 0)
    env.run()
    assert {_flag(c, "-ngl") for c in env.bench.calls} == {"9"}


def test_optimize_seed_basis_is_shared_with_the_cli_report():
    info = optimizer.GgufInfo(
        architecture="llama",
        block_count=8,
        context_length=4096,
        embedding_length=64,
        head_count=4,
        head_count_kv=4,
        key_length=None,
        file_type=None,
        is_embedding=False,
        file_size=100,
    )
    device = Device("Vulkan0", "GPU", 8192, 520)
    assert optimizer.optimize_seed(info, _host(), device, 8192) == optimizer.heuristic_seed(
        info, _host(), replace(device, total_mib=520), 8192
    )
    assert optimizer.optimize_seed(info, _host(), None, 8192) == optimizer.heuristic_seed(info, _host(), None, 8192)


def test_interrupt_during_the_sweep_skips_validation_and_saves_the_measured_winner(env: Env):
    calls: list[int] = []

    def speed(argv: list[str]) -> tuple[float, float]:
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return 100.0, 20.0 * len(calls)

    env.bench.speed = speed
    profile = env.run()
    assert env.validated == []  # no real server was started
    assert profile.measured is True
    assert profile.validated is False
    assert profile.tg_tps == pytest.approx(40.0)
    saved = config.load(env.base)
    assert saved is not None
    stored = saved.profiles[profile_key(DIGEST, 8192)]
    assert stored.validated is False
    assert stored.measured is True


def test_interrupt_before_any_result_saves_the_unmeasured_unvalidated_seed(env: Env):
    def speed(argv: list[str]) -> tuple[float, float]:
        raise KeyboardInterrupt

    env.bench.speed = speed
    profile = env.run()
    assert env.validated == []
    assert profile.measured is False
    assert profile.validated is False
    assert profile.args == seed_to_args(_expected_seed(env))


def test_interrupt_during_validation_keeps_the_measured_winner(env: Env):
    def verdict(_settings: ServeSettings) -> bool:
        raise KeyboardInterrupt

    env.verdicts = verdict
    profile = env.run()
    assert profile.measured is True
    assert profile.validated is False
    assert profile.tg_tps == pytest.approx(55.0)
    assert _flag(profile.args, "-ub") == "1024"
    saved = config.load(env.base)
    assert saved is not None
    assert saved.profiles[profile_key(DIGEST, 8192)].validated is False


def test_normal_run_is_validated(env: Env):
    assert env.run().validated is True
    assert env.run(quick=True).validated is True  # quick launches nothing and says so through `measured`, not this


def test_decode_body_and_loaded_models_tolerate_garbage():
    assert optimizer._decode_body(b"not json") is None
    assert optimizer._decode_body(b"") is None
    assert optimizer._loaded_models(None) == []
    assert optimizer._loaded_models({"models": "x"}) == []
    assert optimizer._loaded_models({"models": [1, {"name": ""}, {"model": "m"}, {"name": "n"}]}) == ["m", "n"]


def test_http_json_http_error_with_unreadable_body(monkeypatch: pytest.MonkeyPatch):

    class Boom(io.BytesIO):
        def read(self, *_args: Any) -> bytes:
            raise OSError("reset")

    error = urllib.error.HTTPError("http://x/", 502, "bad", {}, Boom())  # type: ignore[arg-type]

    class Opener:
        def open(self, *_a: Any, **_k: Any) -> Any:
            raise error

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: Opener())
    assert optimizer._http_json("GET", "http://x/", None, 1.0) == (502, None)


def test_optimize_model_without_model_layer_raises(env: Env):
    env.model = replace(env.model, blob_path=None, digest=None)
    with pytest.raises(ValueError, match="no model layer"):
        env.run()
