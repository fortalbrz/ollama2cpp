"""CLI tests for list, info, status, stop, doctor, update, config and optimize (external seams are monkeypatched)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ollama2cpp import (
    commands,
    config,
    constants,
    gguf,
    hardware,
    llamacpp,
    ollama,
    optimize_cmd,
    optimizer,
    proc,
    selector,
    server,
    ui,
)
from ollama2cpp.__main__ import app
from ollama2cpp.config import Profile
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.optimizer import BenchResult
from tests.gguf_builder import write_gguf

runner = CliRunner()
GB = 1024**3
GPU = Device("CUDA0", "Test GPU", 8192, 6000)
FULL_GPU = Device("CUDA0", "Test GPU", 8192, 8000)
BUSY_GPU = Device("CUDA0", "Test GPU", 8192, 100)
BUILD = "b1234"
BIG = "big:latest"
TINY = "tiny:latest"


def _host(*devices: Device) -> HostInfo:
    return HostInfo(devices=devices, ram_total_mib=32768, ram_free_mib=16000, physical_cores=8, logical_cores=16)


def _chat(name: str, size: int, tag: str = "latest", **options: Any) -> tuple[str, str, int, dict[str, Any]]:
    """Spec for a model whose manifest declares ``size`` bytes and whose blob is a tiny valid GGUF."""
    return (name, tag, size, {"blob_writer": lambda path: write_gguf(path, n_layer=2), **options})


@dataclass
class Rig:
    """Recorded calls and switchable state for the monkeypatched seams."""

    root: Path
    binary: Path
    bench: Path
    devices: list[Device] = field(default_factory=lambda: [GPU])
    binary_present: bool = True
    bench_present: bool = True
    running: tuple[str, str | None] = ("none", None)
    ready: bool = True
    stops: list[int] = field(default_factory=list)
    updates: list[list[str]] = field(default_factory=list)

    def invoke(self, *args: str) -> Any:
        """Run the CLI."""
        return runner.invoke(app, list(args))

    def save_config(self, **overrides: Any) -> config.Config:
        """Write a ready-to-use config (as if a first run happened)."""
        values: dict[str, Any] = {
            "ollama_path": str(self.root),
            "default_model": BIG,
            "llama_server": str(self.binary),
            "llama_bench": str(self.bench),
        }
        cfg = config.Config(**{**values, **overrides})
        config.save(cfg)
        return cfg

    def model(self, name: str) -> ollama.OllamaModel:
        """Return an installed model by name."""
        return ollama.resolve_model(self.root, name)


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Rich from wrapping tables and panels at the 80 columns of a non-terminal."""
    monkeypatch.setattr(ui.console, "width", 220)


@pytest.fixture
def rig(fake_ollama: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """Build a fake Ollama store plus seams for llama.cpp, hardware and the server."""
    root = fake_ollama(
        [
            _chat("big", 6 * GB, projector=True),
            _chat("tiny", 1 * GB),
            _chat("huge", 20 * GB),
            ("nomic-embed", "latest", 1 * GB, {"blob_writer": lambda p: write_gguf(p, arch="bert", pooling=1)}),
            ("whisper-small", "latest", 1 * GB, {"blob_writer": lambda p: write_gguf(p, n_layer=2)}),
            ("garbled", "latest", 2 * GB, {}),  # the default 16-byte stub is not a readable GGUF
            ("dangling", "latest", 2 * GB, {}),
        ],
        missing_blob={"dangling:latest"},
    )
    binary = tmp_path / "bin" / "llama-server.exe"
    bench = tmp_path / "bin" / "llama-bench.exe"
    state = Rig(root=root, binary=binary, bench=bench)
    monkeypatch.setenv(constants.ENV_OLLAMA_MODELS, str(root))

    def find_binary(name: str, configured: str | None = None) -> Path | None:
        del configured
        if name == "llama-server":
            return state.binary if state.binary_present else None
        return state.bench if state.bench_present else None

    def describe(host: str, port: int, base: Path | None = None) -> tuple[str, str | None]:
        del host, port, base
        return state.running

    def stop(base: Path | None = None, **_kwargs: Any) -> bool:
        del base
        state.stops.append(1)
        was_ours = state.running[0] == "ours"
        if was_ours:
            state.running = ("none", None)
        return was_ours

    def run(argv: Any, **_kwargs: Any) -> proc.ProcResult:
        state.updates.append(list(argv))
        return proc.ProcResult(0, "Successfully installed\n", "")

    monkeypatch.setattr(llamacpp, "find_binary", find_binary)
    monkeypatch.setattr(llamacpp, "version", lambda _b: BUILD)
    monkeypatch.setattr(hardware, "host_info", lambda _b: _host(*state.devices))
    monkeypatch.setattr(hardware, "detect_devices", lambda _b: list(state.devices))
    monkeypatch.setattr(server, "describe_running", describe)
    monkeypatch.setattr(server, "is_ready", lambda _h, _p: state.ready)
    monkeypatch.setattr(server, "stop", stop)
    monkeypatch.setattr(server, "port_in_use", lambda _h, _p: False)
    monkeypatch.setattr(proc, "run", run)
    monkeypatch.setattr(optimize_cmd, "UNLOAD_WAIT_S", 0.05)
    monkeypatch.setattr(optimize_cmd, "UNLOAD_POLL_S", 0.01)
    return state


def _rows(output: str) -> dict[str, str]:
    """Map the first cell of every table row (``name  | ...``) to the whole line, for loose assertions."""
    return {line.strip("│ ").split("│")[0].strip(): line for line in output.splitlines() if "│" in line}


# --- list ----------------------------------------------------------------------------------------------------------


def test_list_table_shows_models_and_marks_embedding(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("list")
    assert result.exit_code == 0, result.output
    rows = _rows(result.output)
    assert "chat" in rows[BIG]
    assert "6.0" in rows[BIG]
    assert "embedding" in rows["nomic-embed:latest"]
    assert "huge:latest" in rows
    assert "Traceback" not in result.output


def test_list_table_marks_fit_and_optimized(rig: Rig) -> None:
    cfg = rig.save_config()
    digest = rig.model(BIG).digest
    cfg.profiles[config.profile_key(digest or "", 8192)] = Profile(args=[], ctx=8192, tg_tps=30.0, measured=True)
    cfg.profiles[config.profile_key(rig.model(TINY).digest or "", 8192)] = Profile(args=[], ctx=8192)
    config.save(cfg)
    rows = _rows(rig.invoke("list").output)
    big, tiny, huge = (rows[n].replace("│", " ").split() for n in (BIG, TINY, "huge:latest"))
    assert big[-2:] == ["yes", "yes"]  # fits the 8 GiB card, optimized
    assert tiny[-2:] == ["yes", "seed"]  # fits, only the unmeasured seed is saved
    assert huge[-2:] == ["no", "no"]  # 20 GiB model does not fit, never optimized


def test_list_json_is_valid_and_has_fields(rig: Rig) -> None:
    cfg = rig.save_config()
    cfg.profiles[config.profile_key(rig.model(BIG).digest or "", 8192)] = Profile(args=[], ctx=8192, measured=True)
    config.save(cfg)
    result = rig.invoke("list", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    by_name = {row["name"]: row for row in data}
    assert set(by_name[BIG]) == {
        "name",
        "size_gib",
        "quant",
        "arch",
        "ctx",
        "kind",
        "need_mib",
        "fits_vram",
        "optimized",
        "reason",
    }
    assert by_name[BIG]["kind"] == "chat"
    assert by_name[BIG]["size_gib"] == pytest.approx(6.0)
    assert by_name[BIG]["arch"] == "llama"
    assert by_name[BIG]["ctx"] == 4096
    assert by_name[BIG]["fits_vram"] is True
    assert by_name[BIG]["optimized"] == "yes"
    assert by_name["huge:latest"]["fits_vram"] is False
    assert by_name["nomic-embed:latest"]["kind"] == "embedding"
    assert by_name["nomic-embed:latest"]["fits_vram"] is None
    assert result.output.lstrip().startswith("[")
    assert "\x1b" not in result.output  # machine output carries no Rich styling


def test_list_json_with_no_models_is_empty_array(rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty-ollama"
    (empty / "manifests").mkdir(parents=True)
    rig.save_config(ollama_path=str(empty))
    result = rig.invoke("list", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    assert "No Ollama models" in rig.invoke("list").output


def test_list_shows_unsupported_rows(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("list")
    assert result.exit_code == 0, result.output  # one bad model never aborts the listing
    rows = _rows(result.output)
    for name in ("garbled:latest", "dangling:latest", "whisper-small:latest"):
        assert "unsupported" in rows[name]
    assert "missing blob" in result.output
    assert "audio model" in result.output
    data = {row["name"]: row for row in json.loads(rig.invoke("list", "--json").output)}
    assert data["dangling:latest"]["reason"] == "missing blob"
    assert data["dangling:latest"]["fits_vram"] is None
    assert data["garbled:latest"]["reason"]


def test_list_without_gpu_cannot_tell_if_it_fits(rig: Rig) -> None:
    rig.devices = []
    rig.save_config()
    big = _rows(rig.invoke("list").output)[BIG].replace("│", " ").split()
    assert big[-2:] == ["?", "no"]
    assert json.loads(rig.invoke("list", "--json").output)[0]["fits_vram"] is None


def test_list_without_ollama_directory_exits_1(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config(ollama_path=None)
    monkeypatch.setattr(ollama, "find_ollama_path", lambda *_a, **_k: None)
    result = rig.invoke("list")
    assert result.exit_code == 1
    assert "Ollama models directory" in result.output
    assert "Traceback" not in result.output


def test_list_runs_first_run_setup_when_unconfigured(rig: Rig) -> None:
    result = rig.invoke("list")
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.default_model == BIG


# --- info ----------------------------------------------------------------------------------------------------------


def test_info_unknown_model_exit_1(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("info", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert BIG in result.output  # the available models are named
    assert "Traceback" not in result.output


def test_info_shows_vram_need(rig: Rig) -> None:
    cfg = rig.save_config()
    model = rig.model(BIG)
    cfg.profiles[config.profile_key(model.digest or "", 8192)] = Profile(
        args=["-ngl", "3"], ctx=4096, tg_tps=42.5, pp_tps=900.0, measured=True
    )
    config.save(cfg)
    info = gguf.read_gguf(model.blob_path or Path())
    need_default = selector.estimate_need_mib(model, info, 8192)
    need_profile = selector.estimate_need_mib(model, info, 4096)
    result = rig.invoke("info", "big")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert f"VRAM at ctx 8192: about {need_default} MiB (fits 8192 MiB of VRAM)" in text
    assert f"VRAM at ctx 4096: about {need_profile} MiB" in text
    assert "Architecture : llama" in text
    assert "Layers : 2" in text
    assert "Trained ctx : 4096" in text
    assert "tg 42.5 t/s" in text
    assert "-ngl 3" in text
    for layer in ("model", "projector", "template"):
        assert layer in result.output


def test_info_unsupported_model_shows_reason_and_layers(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("info", "dangling")
    assert result.exit_code == 0, result.output
    assert "missing blob" in result.output
    assert "MISSING" in result.output


def test_info_without_profiles_hints_at_optimize(rig: Rig) -> None:
    rig.save_config()
    assert "none saved" in rig.invoke("info", "tiny").output


# --- status --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("running", "expected"),
    [
        (("none", None), "not running"),
        (("ours", BIG), "running (ours)"),
        (("foreign", "someone"), "another process"),
    ],
)
def test_status_none_ours_foreign(rig: Rig, running: tuple[str, str | None], expected: str) -> None:
    rig.save_config()
    rig.running = running
    result = rig.invoke("status")
    assert result.exit_code == 0, result.output
    assert expected in result.output
    if running[0] == "ours":
        assert BIG in result.output
        assert "ready" in result.output
        rig.ready = False
        assert "still loading" in rig.invoke("status").output


def test_status_warns_low_free_vram(rig: Rig) -> None:
    rig.save_config()
    rig.devices = [BUSY_GPU]
    result = rig.invoke("status")
    assert result.exit_code == 0, result.output
    assert "something else is holding VRAM" in result.output
    rig.devices = [GPU]
    assert "holding VRAM" not in rig.invoke("status").output


def test_status_shows_config_and_default_profile(rig: Rig) -> None:
    cfg = rig.save_config()
    cfg.profiles[config.profile_key(rig.model(BIG).digest or "", 8192)] = Profile(
        args=[], ctx=8192, tg_tps=12.0, measured=True
    )
    config.save(cfg)
    text = " ".join(rig.invoke("status").output.split())
    assert str(config.config_file()) in text
    assert "tg 12.0 t/s" in text
    assert "Test GPU: 6000 MiB free of 8192 MiB" in text


def test_status_works_without_config_and_without_gpu(rig: Rig) -> None:
    rig.devices = []
    result = rig.invoke("status")
    assert result.exit_code == 0, result.output
    assert "not found" in result.output
    assert "CPU only" in result.output
    assert not config.config_file().exists()  # no first-run setup


def test_status_reports_a_corrupt_config_without_touching_it(rig: Rig) -> None:
    path = config.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    result = rig.invoke("status")
    assert result.exit_code == 0, result.output
    assert "unreadable or invalid" in result.output
    assert path.read_text(encoding="utf-8") == "{not json"


# --- stop ----------------------------------------------------------------------------------------------------------


def test_stop_without_server_exit_0_message(rig: Rig) -> None:
    result = rig.invoke("stop")
    assert result.exit_code == 0, result.output
    assert "No ollama2cpp server is running" in result.output


def test_stop_stops_our_server(rig: Rig) -> None:
    rig.running = ("ours", BIG)
    result = rig.invoke("stop")
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output
    assert rig.stops == [1]


def test_stop_leaves_a_foreign_listener_alone(rig: Rig) -> None:
    rig.running = ("foreign", "other")
    result = rig.invoke("stop")
    assert result.exit_code == 0, result.output
    assert "did not start" in result.output
    assert "left alone" in result.output


# --- doctor --------------------------------------------------------------------------------------------------------


def test_doctor_reports_missing_llama_server_with_hint(rig: Rig) -> None:
    rig.binary_present = False
    rig.save_config()
    result = rig.invoke("doctor")
    assert result.exit_code == 1, result.output
    rows = _rows(result.output)
    assert "fail" in rows["llama-server"]
    assert "winget install --id ggml.llamacpp" in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_doctor_all_ok_exit_0(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("doctor")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert BUILD in text
    assert "backend: CUDA" in text
    assert "(7 models)" in text
    assert "fail" not in result.output


def test_doctor_warns_without_bench_gpu_or_config(rig: Rig) -> None:
    rig.bench_present = False
    rig.devices = []
    result = rig.invoke("doctor")
    assert result.exit_code == 0, result.output  # only a `fail` makes the exit code non-zero
    rows = _rows(result.output)
    assert "warn" in rows["llama-bench"]
    assert "warn" in rows["GPU"]
    assert "warn" in rows["Config"]
    assert not config.config_file().exists()  # no first-run setup


def test_doctor_warns_on_corrupt_config_and_keeps_the_file(rig: Rig) -> None:
    path = config.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"port": "not a number"}', encoding="utf-8")
    result = rig.invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "warn" in _rows(result.output)["Config"]
    assert path.is_file()


def test_doctor_warns_when_vram_is_held(rig: Rig) -> None:
    rig.devices = [BUSY_GPU]
    rig.save_config()
    assert "warn" in _rows(rig.invoke("doctor").output)["Free VRAM"]


def test_doctor_fails_without_ollama_directory(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config(ollama_path=None)
    monkeypatch.setattr(ollama, "find_ollama_path", lambda *_a, **_k: None)
    result = rig.invoke("doctor")
    assert result.exit_code == 1
    assert "fail" in _rows(result.output)["Ollama models"]


def test_doctor_never_raises_when_a_check_crashes(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.save_config()

    def boom(_binary: Path) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(llamacpp, "version", boom)
    result = rig.invoke("doctor")
    assert result.exit_code == 1
    assert "the check crashed: boom" in " ".join(result.output.split())
    assert "Traceback" not in result.output


# --- config --------------------------------------------------------------------------------------------------------


def test_config_shows_path_and_json(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("config")
    assert result.exit_code == 0, result.output
    assert str(config.config_file()) in " ".join(result.output.split())
    body = result.output[result.output.index("{") :]
    assert json.loads(body)["default_model"] == BIG


def test_config_set_and_reset(rig: Rig, tmp_path: Path) -> None:
    rig.save_config()
    assert rig.invoke("config", "--set", "port=9000").exit_code == 0
    cfg = config.load()
    assert cfg is not None
    assert cfg.port == 9000
    other = tmp_path / "other-ollama"
    other.mkdir()
    result = rig.invoke(
        "config",
        "--set",
        "host=0.0.0.0",
        "--set",
        "device=CUDA0",
        "--set",
        f"ollama_path={other}",
        "--set",
        "default_model=Tiny",
    )
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert (cfg.host, cfg.port, cfg.device, cfg.default_model) == ("0.0.0.0", 9000, "CUDA0", TINY)
    assert cfg.ollama_path == str(other.resolve())
    assert cfg.llama_server == str(rig.binary)  # untouched keys survive
    assert rig.invoke("config", "--set", "device=none", "--set", "default_model=").exit_code == 0
    cfg = config.load()
    assert cfg is not None
    assert cfg.device is None
    assert cfg.default_model is None
    # --reset refuses to delete without --yes when stdin is not a terminal, then deletes with it
    refused = rig.invoke("config", "--reset")
    assert refused.exit_code == 1
    assert "--yes" in refused.output
    assert config.config_file().is_file()
    assert rig.invoke("config", "--reset", "--yes").exit_code == 0
    assert not config.config_file().exists()
    assert "no configuration to reset" in rig.invoke("config", "--reset").output


def test_config_set_sets_llama_server(rig: Rig) -> None:
    rig.save_config()
    rig.binary.parent.mkdir(parents=True)
    rig.binary.write_text("x", encoding="utf-8")
    assert rig.invoke("config", "--set", f"llama_server={rig.binary}").exit_code == 0
    cfg = config.load()
    assert cfg is not None
    assert cfg.llama_server == str(rig.binary.resolve())


@pytest.mark.parametrize(
    "assignment",
    [
        "bogus=1",
        "port=0",
        "port=70000",
        "port=abc",
        "port",
        "host=",
        "ollama_path=/definitely/not/here",
        "llama_server=/definitely/not/here.exe",
    ],
)
def test_config_set_rejects_bad_input_and_writes_nothing(rig: Rig, assignment: str) -> None:
    rig.save_config()
    before = config.config_file().read_text(encoding="utf-8")
    result = rig.invoke("config", "--set", "default_model=tiny", "--set", assignment)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert config.config_file().read_text(encoding="utf-8") == before  # not even the valid assignment


def test_config_set_and_reset_together_is_an_error(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("config", "--set", "port=9000", "--reset", "--yes")
    assert result.exit_code == 1
    assert config.config_file().is_file()


@pytest.mark.parametrize("answer", [True, False])
def test_config_reset_asks_on_a_terminal(rig: Rig, monkeypatch: pytest.MonkeyPatch, answer: bool) -> None:
    rig.save_config()
    questions: list[tuple[str, bool]] = []

    def confirm(question: str, *, assume_yes: bool = False, default: bool = True) -> bool:
        del assume_yes
        questions.append((question, default))
        return answer

    monkeypatch.setattr(commands, "_interactive", lambda: True)
    monkeypatch.setattr(ui, "confirm", confirm)
    assert rig.invoke("config", "--reset").exit_code == 0
    assert config.config_file().exists() is (not answer)
    assert questions[0][1] is False  # destructive: the default answer is no


# --- update --------------------------------------------------------------------------------------------------------


def test_update_calls_winget_upgrade(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("update")
    assert result.exit_code == 0, result.output
    assert rig.updates == [["winget", "upgrade", "--id", "ggml.llamacpp", "-e"]]
    assert "Successfully installed" in result.output
    assert BUILD in result.output


def test_update_records_relocated_binaries(rig: Rig) -> None:
    rig.save_config(llama_server="old/llama-server.exe", llama_bench=None)
    assert rig.invoke("update").exit_code == 0
    cfg = config.load()
    assert cfg is not None
    assert (cfg.llama_server, cfg.llama_bench) == (str(rig.binary), str(rig.bench))


def test_update_failure_exits_1(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "update", lambda: proc.ProcResult(1, "", "installer failed\n"))
    result = rig.invoke("update")
    assert result.exit_code == 1
    assert "installer failed" in result.output
    assert "winget upgrade failed" in result.output
    assert "Traceback" not in result.output


def test_update_without_winget_prints_instructions(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "update", lambda: proc.ProcResult(proc.RC_NOT_FOUND, "", "no winget"))
    result = rig.invoke("update")
    assert result.exit_code == 1
    assert "winget is not available" in result.output
    assert "Install llama.cpp manually" in " ".join(result.output.split())


def test_update_already_current_is_not_an_error(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "update", lambda: proc.ProcResult(2316632107, "No available upgrade found.\n", ""))
    result = rig.invoke("update")
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output


# --- optimize ------------------------------------------------------------------------------------------------------


@dataclass
class OptimizeRig:
    """Fake optimizer entry points that record how they were used."""

    unloads: list[int] = field(default_factory=list)
    optimized: list[str] = field(default_factory=list)
    benches: list[tuple[str, ...]] = field(default_factory=list)
    fail_for: set[str] = field(default_factory=set)
    free_after_unload: Device | None = None
    bench_script: Callable[[int], BenchResult] | None = None


@pytest.fixture
def opt(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> OptimizeRig:
    """Replace the optimizer's slow parts (bench, validation, Ollama unload) with recorders."""
    state = OptimizeRig()
    real = optimizer.optimize_model

    def optimize_model(model: ollama.OllamaModel, *args: Any, **kwargs: Any) -> Profile:
        state.optimized.append(model.name)
        if model.name in state.fail_for:
            msg = f"cannot optimize {model.name}"
            raise RuntimeError(msg)
        return real(model, *args, **kwargs)

    def run_bench(bench: Path, model: Path, args: Any, **kwargs: Any) -> BenchResult:
        del bench, model, kwargs
        state.benches.append(tuple(args))
        if state.bench_script is not None:
            return state.bench_script(len(state.benches))
        return BenchResult(tuple(args), 10.0 + len(state.benches), 100.0)

    def unload(timeout: float = 30.0) -> bool:
        del timeout
        state.unloads.append(1)
        if state.free_after_unload is not None:
            rig.devices = [state.free_after_unload]
        return True

    monkeypatch.setattr(optimizer, "optimize_model", optimize_model)
    monkeypatch.setattr(optimizer, "run_bench", run_bench)
    monkeypatch.setattr(optimizer, "validate", lambda *_a, **_k: True)
    monkeypatch.setattr(optimizer, "unload_ollama", unload)
    return state


def _profile_for(name: str, rig: Rig, ctx: int = 8192) -> Profile | None:
    cfg = config.load()
    assert cfg is not None
    return cfg.profiles.get(config.profile_key(rig.model(name).digest or "", ctx))


def test_optimize_refuses_when_vram_occupied(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [BUSY_GPU]
    rig.save_config()
    result = rig.invoke("optimize", "big")
    assert result.exit_code == 1
    assert "short by" in result.output
    assert "MiB" in result.output
    assert "--unload-ollama" in result.output
    assert opt.benches == []  # the benchmark never ran
    assert opt.optimized == []
    assert opt.unloads == []  # nothing is stopped without the explicit flag
    assert _profile_for(BIG, rig) is None
    assert "Traceback" not in result.output


def test_optimize_unload_flag_calls_unload(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [BUSY_GPU]
    opt.free_after_unload = FULL_GPU
    rig.save_config()
    result = rig.invoke("optimize", "big", "--unload-ollama")
    assert result.exit_code == 0, result.output
    assert opt.unloads == [1]
    assert opt.optimized == [BIG]
    assert opt.benches  # the benchmark ran once VRAM was free
    profile = _profile_for(BIG, rig)
    assert profile is not None
    assert profile.measured


def test_optimize_unload_that_frees_too_little_refuses(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [BUSY_GPU]
    rig.save_config()
    result = rig.invoke("optimize", "big", "--unload-ollama")
    assert result.exit_code == 1
    assert opt.unloads == [1]
    assert "still short by" in result.output
    assert opt.benches == []


def test_optimize_unload_unreachable_ollama_warns(rig: Rig, opt: OptimizeRig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.devices = [BUSY_GPU]
    rig.save_config()
    monkeypatch.setattr(optimizer, "unload_ollama", lambda *_a, **_k: False)
    result = rig.invoke("optimize", "big", "--unload-ollama")
    assert result.exit_code == 1
    assert "did not confirm" in result.output


def test_optimize_quick_saves_unmeasured_profile(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [BUSY_GPU]  # quick mode launches nothing, so busy VRAM is no obstacle
    rig.bench_present = False
    rig.save_config()
    result = rig.invoke("optimize", "big", "--quick")
    assert result.exit_code == 0, result.output
    profile = _profile_for(BIG, rig)
    assert profile is not None
    assert profile.measured is False
    assert profile.args
    assert opt.benches == []
    assert "unmeasured" in result.output
    assert "quick mode" in result.output
    assert "Traceback" not in result.output


def test_optimize_measured_run_shows_before_and_after(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [FULL_GPU]
    rig.save_config()
    result = rig.invoke("optimize", "tiny")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "seed (measured)" in text
    assert "optimized" in text
    assert "% vs the heuristic seed" in text
    profile = _profile_for(TINY, rig)
    assert profile is not None
    assert profile.measured
    assert profile.tg_tps == pytest.approx(10.0 + len(opt.benches))


def test_optimize_benchmark_failure_saves_the_seed_and_says_so(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [FULL_GPU]
    rig.save_config()
    opt.bench_script = lambda _n: BenchResult((), None, None, "crashed")
    result = rig.invoke("optimize", "tiny")
    assert result.exit_code == 0, result.output
    assert "benchmark failed - heuristic settings saved" in " ".join(result.output.split())
    profile = _profile_for(TINY, rig)
    assert profile is not None
    assert profile.measured is False


def test_optimize_all_survives_one_failure(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config()
    opt.fail_for = {TINY}
    result = rig.invoke("optimize", "--all", "--quick")
    assert result.exit_code == 0, result.output
    assert opt.optimized == [BIG, "huge:latest", TINY]  # the failure did not stop the rest
    assert _profile_for(BIG, rig) is not None
    assert _profile_for("huge:latest", rig) is not None
    assert _profile_for(TINY, rig) is None
    text = " ".join(result.output.split())
    assert "cannot optimize tiny:latest" in text
    assert "Optimization summary" in text
    assert "fail" in _rows(result.output)[TINY]
    assert "Traceback" not in result.output
    assert "nomic-embed" not in result.output  # only chat models are optimized


def test_optimize_all_failing_everywhere_exits_1(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config()
    opt.fail_for = {BIG, TINY, "huge:latest"}
    result = rig.invoke("optimize", "--all", "--quick")
    assert result.exit_code == 1
    assert opt.optimized == [BIG, "huge:latest", TINY]


def test_optimize_all_refuses_the_models_that_do_not_fit(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [FULL_GPU]
    rig.save_config()
    result = rig.invoke("optimize", "--all")
    assert result.exit_code == 0, result.output
    assert "huge:latest" in result.output
    assert "short by" in result.output
    assert BIG in opt.optimized
    assert "huge:latest" not in opt.optimized


def test_optimize_ctrl_c_saves_best_so_far(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [FULL_GPU]
    rig.save_config()

    def script(n: int) -> BenchResult:
        if n == 3:
            raise KeyboardInterrupt
        return BenchResult(("-n", str(n)), 20.0 * n, 100.0)

    opt.bench_script = script
    result = rig.invoke("optimize", "tiny")
    assert result.exit_code == 0, result.output
    assert len(opt.benches) == 3  # the sweep stopped at the interrupt
    profile = _profile_for(TINY, rig)
    assert profile is not None
    assert profile.measured
    assert profile.tg_tps == pytest.approx(40.0)  # the best of the two candidates measured before Ctrl-C
    assert "Traceback" not in result.output


def test_optimize_interrupt_outside_the_sweep_exits_130(
    rig: Rig, opt: OptimizeRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.save_config()

    def interrupted(*_args: Any, **_kwargs: Any) -> Profile:
        raise KeyboardInterrupt

    monkeypatch.setattr(optimizer, "optimize_model", interrupted)
    result = rig.invoke("optimize", "big", "--quick")
    assert result.exit_code == 130
    assert "Interrupted" in result.output
    assert "Traceback" not in result.output


def test_optimize_refuses_non_chat_models(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config()
    for name in ("nomic-embed", "garbled"):
        result = rig.invoke("optimize", name, "--quick")
        assert result.exit_code == 1
        assert "only chat models" in result.output
    assert opt.optimized == []


def test_optimize_needs_llama_bench_unless_quick(rig: Rig, opt: OptimizeRig) -> None:
    rig.bench_present = False
    rig.save_config()
    result = rig.invoke("optimize", "tiny")
    assert result.exit_code == 1
    assert "llama-bench was not found" in result.output
    assert "ollama2cpp update" in result.output
    assert rig.invoke("optimize", "tiny", "--quick").exit_code == 0


def test_optimize_needs_llama_server(rig: Rig, opt: OptimizeRig) -> None:
    rig.binary_present = False
    rig.save_config()
    result = rig.invoke("optimize", "tiny", "--quick")
    assert result.exit_code == 1
    assert "llama-server was not found" in result.output


def test_optimize_refuses_while_our_server_runs_unless_told_to_stop_it(rig: Rig, opt: OptimizeRig) -> None:
    rig.devices = [FULL_GPU]
    rig.running = ("ours", BIG)
    rig.save_config()
    refused = rig.invoke("optimize", "tiny")
    assert refused.exit_code == 1
    assert "--stop-server" in refused.output
    assert rig.stops == []
    assert opt.optimized == []
    result = rig.invoke("optimize", "tiny", "--stop-server")
    assert result.exit_code == 0, result.output
    assert rig.stops == [1]
    assert opt.optimized == [TINY]


def test_optimize_uses_the_saved_default_model_and_ctx(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config(default_model=TINY)
    result = rig.invoke("optimize", "--quick", "--ctx", "4096")
    assert result.exit_code == 0, result.output
    assert opt.optimized == [TINY]
    assert _profile_for(TINY, rig, 4096) is not None


def test_optimize_unknown_model_exits_1(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config()
    result = rig.invoke("optimize", "nope", "--quick")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_optimize_all_and_model_together_is_an_error(rig: Rig, opt: OptimizeRig) -> None:
    rig.save_config()
    result = rig.invoke("optimize", "tiny", "--all", "--quick")
    assert result.exit_code == 1
    assert opt.optimized == []


def test_optimize_all_without_chat_models_is_an_error(rig: Rig, opt: OptimizeRig, tmp_path: Path) -> None:
    empty = tmp_path / "empty-ollama"
    (empty / "manifests").mkdir(parents=True)
    rig.save_config(ollama_path=str(empty))
    result = rig.invoke("optimize", "--all", "--quick")
    assert result.exit_code == 1
    assert "No chat models" in result.output


# --- review fixes: machine output, refusal logging, OLLAMA_MODEL --------------------------------------------------


def test_list_json_first_run_keeps_stdout_pure(rig: Rig) -> None:
    result = rig.invoke("list", "--json")
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)) == 7  # stdout is only the JSON array
    assert "First run" in result.stderr  # the setup messages are not swallowed
    assert "First run" not in result.stdout
    assert config.config_file().is_file()


def test_list_json_with_unsupported_models_keeps_stdout_pure(rig: Rig) -> None:
    rig.save_config()
    result = rig.invoke("list", "--json")
    assert result.exit_code == 0, result.output
    assert {r["kind"] for r in json.loads(result.stdout)} == {"chat", "embedding", "unsupported"}


def test_use_stderr_routes_messages_and_restores_the_console(capsys: pytest.CaptureFixture[str]) -> None:
    with ui.use_stderr():
        ui.show_notice("info", "to stderr")
    ui.show_notice("info", "to stdout")
    out, err = capsys.readouterr()
    assert "to stderr" in err
    assert "to stderr" not in out
    assert "to stdout" in out
    assert ui.console.stderr is False


def _fail_inside_stderr_block() -> None:
    with ui.use_stderr():
        msg = "boom"
        raise RuntimeError(msg)


def test_use_stderr_restores_the_console_after_an_error() -> None:
    with pytest.raises(RuntimeError):
        _fail_inside_stderr_block()
    assert ui.console.stderr is False


def test_optimize_refusal_logs_no_traceback(rig: Rig, opt: OptimizeRig, caplog: pytest.LogCaptureFixture) -> None:
    rig.devices = [BUSY_GPU]
    rig.save_config()
    with caplog.at_level(logging.DEBUG, logger="ollama2cpp"):
        result = rig.invoke("optimize", "--all")
    assert result.exit_code == 1
    ours = [r for r in caplog.records if r.name.startswith("ollama2cpp")]
    loud = [r for r in ours if r.levelno >= logging.WARNING]
    assert any("Optimizing big:latest failed" in r.getMessage() for r in loud)  # one line per failure...
    assert all(r.exc_info is None for r in loud)  # ...and never a stack trace at WARNING or above
    assert any(r.exc_info for r in ours if r.levelno == logging.DEBUG)  # the details stay available with --verbose


def test_optimize_all_ignores_a_model_from_the_environment(
    rig: Rig, opt: OptimizeRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    rig.save_config()
    result = rig.invoke("optimize", "--all", "--quick")
    assert result.exit_code == 0, result.output
    assert sorted(opt.optimized) == sorted([BIG, TINY, "huge:latest"])


def test_optimize_uses_the_model_from_the_environment(
    rig: Rig, opt: OptimizeRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    rig.save_config()
    assert rig.invoke("optimize", "--quick").exit_code == 0
    assert opt.optimized == [TINY]


def test_optimize_all_with_an_explicit_model_flag_still_conflicts(
    rig: Rig, opt: OptimizeRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(constants.ENV_OLLAMA_MODEL, "tiny")
    rig.save_config()
    result = rig.invoke("--model", "big", "optimize", "--all", "--quick")
    assert result.exit_code == 1
    assert "Give either a MODEL or --all, not both." in result.output
    assert opt.optimized == []
