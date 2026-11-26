"""Tune llama-server launch settings for maximum generation speed.

Two halves: a pure, deterministic heuristic seed (no I/O), then a ``llama-bench`` coordinate-descent sweep around it,
a real ``llama-server`` validation of the winner, and persistence of the resulting profile.

Flag spellings verified against ``llama-server --help`` and ``llama-bench --help`` (build 12721, Windows): ``-ngl``,
``-fa on|off|auto`` (takes a value, not a bare switch), ``-t``, ``-b``, ``-ub``, ``-ctk``/``-ctv``, and ``-dev <id>``
(bench only; the server uses ``--device``). ``llama-bench`` sweeps comma-separated value lists, but this module always
passes a single value per flag so every run measures exactly one candidate.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ollama2cpp import config, constants, hardware, llamacpp, proc, server as server_runtime
from ollama2cpp.config import Config, Profile, profile_key
from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.gguf import GgufInfo, kv_cache_bytes, read_gguf
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.llamacpp import LlamaCppError, ServeSettings
from ollama2cpp.ollama import OllamaModel
from ollama2cpp.selector import HEADROOM_FACTOR, OVERHEAD_MIB

log = logging.getLogger(__name__)

MIN_CTX = 2048  # the seed never shrinks the context below this
SEED_BATCH = 2048
SEED_UBATCH = 512
SEED_CACHE_TYPE = "f16"
_MIB = 1024 * 1024

BENCH_PROMPT_TOKENS = 512
BENCH_GEN_TOKENS = 128
BENCH_TIMEOUT = 300.0  # per candidate (the seed always gets all of it)
MIN_BENCH_TIMEOUT = 60.0
MAX_ERROR_CHARS = 200
UBATCH_VALUES = (256, 512, 1024, 2048)
NGL_DELTAS = (2, -2, 4, -4)
KV_QUANT_TYPE = "q8_0"
TIE_TOLERANCE = 0.02  # speeds within this fraction are benchmark noise: only a clear gain replaces the incumbent
VALIDATE_DIRNAME = "validate"  # scratch storage so validation never touches the user's own server record
VALIDATE_PROMPT = "Hello"
VALIDATE_TOKENS = 8
COMPLETION_TIMEOUT = 120.0
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_PS_TIMEOUT = 5.0
HTTP_OK = 200
_OOM_MARKERS = ("out of memory", "failed to allocate", "outofdevicememory", "outofhostmemory")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class SeedPlan:
    """Starting launch settings for the sweep.

    Attributes:
        ngl: GPU layers (``block_count + 1`` is a full offload including the output layer; 0 is CPU-only).
        ctx: Context length in tokens.
        flash_attn: Whether to enable flash attention.
        threads: CPU threads for generation.
        batch: Logical batch size.
        ubatch: Physical batch size.
        cache_type: KV cache element type, used for both K and V.
    """

    ngl: int
    ctx: int
    flash_attn: bool
    threads: int
    batch: int
    ubatch: int
    cache_type: str


def _need_mib(weights: float, kv: float) -> int:
    """Estimate MiB for ``weights`` + ``kv`` bytes, using the selector's headroom and overhead model."""
    return math.ceil((weights + kv) * HEADROOM_FACTOR / _MIB + OVERHEAD_MIB)


def _fit_ctx(info: GgufInfo, budget_mib: int, ctx: int) -> tuple[int, bool]:
    """Halve ``ctx`` (floor ``MIN_CTX``) until the whole model fits ``budget_mib``.

    Args:
        info: Parsed GGUF facts.
        budget_mib: Memory budget in MiB.
        ctx: Requested context length.

    Returns:
        tuple[int, bool]: The chosen context and whether the whole model fits at it.
    """
    while _need_mib(info.file_size, kv_cache_bytes(info, ctx)) > budget_mib:
        if ctx <= MIN_CTX:
            return ctx, False
        ctx = max(MIN_CTX, ctx // 2)
    return ctx, True


def _largest_layer_count(info: GgufInfo, budget_mib: int, ctx: int) -> int:
    """Return the largest layer count (``0..block_count``) whose weights and KV shares fit ``budget_mib``."""
    kv = kv_cache_bytes(info, ctx)
    blocks = max(info.block_count, 1)
    total_layers = info.block_count + 1
    best = 0
    for layers in range(info.block_count + 1):
        weights = info.file_size * layers / total_layers
        if _need_mib(weights, kv * layers / blocks) > budget_mib:
            break
        best = layers
    return best


def heuristic_seed(info: GgufInfo, host: HostInfo, device: Device | None, ctx: int = DEFAULT_CTX) -> SeedPlan:
    """Pick a starting plan from the model and hardware facts alone (pure, no I/O).

    With a device the budget is its *total* VRAM. The context is halved (floor ``MIN_CTX``) before any layer is left on
    the CPU; only when the whole model still does not fit at the floor is the layer count reduced. Without a device
    the plan is CPU-only and the context is reduced only to fit total RAM.

    Args:
        info: Parsed GGUF facts (``info.file_size`` is the weight size).
        host: Host snapshot (physical core count and RAM).
        device: Chosen GPU, or None for a CPU-only run.
        ctx: Requested context length in tokens.

    Returns:
        SeedPlan: The starting settings.
    """
    threads = max(1, host.physical_cores)
    if device is None:
        fitted, _ = _fit_ctx(info, host.ram_total_mib, ctx)
        return SeedPlan(0, fitted, False, threads, SEED_BATCH, SEED_UBATCH, SEED_CACHE_TYPE)
    fitted, fits = _fit_ctx(info, device.total_mib, ctx)
    ngl = info.block_count + 1 if fits else _largest_layer_count(info, device.total_mib, fitted)
    log.debug("Seed for %s: ngl=%d ctx=%d (requested ctx=%d)", device.id, ngl, fitted, ctx)
    return SeedPlan(ngl, fitted, True, threads, SEED_BATCH, SEED_UBATCH, SEED_CACHE_TYPE)


def optimize_seed(info: GgufInfo, host: HostInfo, device: Device | None, ctx: int = DEFAULT_CTX) -> SeedPlan:
    """Pick the starting plan of an ``optimize`` run: like :func:`heuristic_seed`, but budgeting the FREE memory.

    ``serve`` without a profile seeds from the card's *total* VRAM (it assumes nothing else needs the GPU), while a
    benchmark runs now with whatever is free: a model that does not fully fit in the free memory starts as a partial
    offload (so the ``-ngl`` dimension is swept) instead of running out of memory on every candidate.

    Args:
        info: Parsed GGUF facts.
        host: Host snapshot.
        device: Chosen GPU, or None for a CPU-only run.
        ctx: Requested context length in tokens.

    Returns:
        SeedPlan: The starting settings for the sweep.
    """
    if device is not None:
        device = replace(device, total_mib=min(device.total_mib, available_mib(device)))
    return heuristic_seed(info, host, device, ctx)


def seed_to_args(plan: SeedPlan) -> list[str]:
    """Render ``plan`` as ``llama-server`` arguments (the context is passed separately by the caller).

    Args:
        plan: The launch settings.

    Returns:
        list[str]: Argument vector fragment using the spellings verified against ``llama-server --help``.
    """
    return [
        "-ngl",
        str(plan.ngl),
        "-fa",
        "on" if plan.flash_attn else "off",
        "-t",
        str(plan.threads),
        "-b",
        str(plan.batch),
        "-ub",
        str(plan.ubatch),
        "-ctk",
        plan.cache_type,
        "-ctv",
        plan.cache_type,
    ]


# --- benchmark -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    """One measured (or failed) benchmark candidate.

    Attributes:
        args: The ``llama-bench`` arguments that were measured.
        tg_tps: Generation speed in tokens/s, or None when the run failed.
        pp_tps: Prompt-processing speed in tokens/s, or None when unavailable.
        error: Short sanitised failure reason; empty on success.
    """

    args: tuple[str, ...]
    tg_tps: float | None
    pp_tps: float | None
    error: str = ""


def plan_to_bench_args(plan: SeedPlan, device: str | None = None) -> list[str]:
    """Render ``plan`` as ``llama-bench`` arguments (same spellings as ``llama-server``; ``-dev`` selects a GPU).

    Args:
        plan: The launch settings.
        device: Device id (``Vulkan0``...) to pin the run to, or None for llama.cpp's default.

    Returns:
        list[str]: Argument vector fragment.
    """
    args = seed_to_args(plan)
    return [*args, "-dev", device] if device else args


def _decode_rows(text: str) -> list[Any]:
    """Decode ``llama-bench -o json`` output (an array, a single object, or JSON lines) without raising.

    Args:
        text: Raw stdout, possibly with noise around the JSON.

    Returns:
        list[Any]: The decoded rows (not yet validated).
    """
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    for attempt in (text, text[start : end + 1] if 0 <= start < end else ""):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        return data if isinstance(data, list) else [data]
    rows: list[Any] = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _number(value: object) -> float | None:
    """Return ``value`` as a finite float, or None for non-numbers, booleans and NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def parse_bench_json(text: str) -> tuple[float | None, float | None]:
    """Extract prompt-processing and generation speed from ``llama-bench -o json`` output.

    Prompt rows have ``n_prompt > 0`` and ``n_gen == 0``; generation rows have ``n_gen > 0`` and ``n_prompt == 0``;
    the speed is ``avg_ts``. Total function: anything unusable yields ``None`` for the missing figure.

    Args:
        text: Raw stdout of ``llama-bench``.

    Returns:
        tuple[float | None, float | None]: ``(pp_tps, tg_tps)``.
    """
    pp: float | None = None
    tg: float | None = None
    for row in _decode_rows(text):
        if not isinstance(row, dict):
            continue
        speed, n_prompt, n_gen = _number(row.get("avg_ts")), _number(row.get("n_prompt")), _number(row.get("n_gen"))
        if speed is None or speed <= 0 or n_prompt is None or n_gen is None:
            continue
        if n_prompt > 0 and n_gen == 0 and pp is None:
            pp = speed
        elif n_gen > 0 and n_prompt == 0 and tg is None:
            tg = speed
    return pp, tg


def _sanitize(text: str) -> str:
    """Make ``text`` safe for one-line display: no escape codes or control characters, bounded length."""
    text = _ANSI_RE.sub("", text)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(text.split())[:MAX_ERROR_CHARS]


def _bench_error(result: proc.ProcResult, timeout: float) -> str:
    """Explain why a ``llama-bench`` run produced no generation figure.

    Args:
        result: The finished (or failed) process.
        timeout: The timeout that was applied, in seconds.

    Returns:
        str: A short, sanitised reason.
    """
    if result.returncode == proc.RC_NOT_FOUND:
        return "llama-bench not found"
    if result.returncode == proc.RC_TIMEOUT:
        return f"timed out after {timeout:g}s"
    output = f"{result.stderr}\n{result.stdout}"
    if any(marker in output.lower() for marker in _OOM_MARKERS):
        return "out of memory"
    if result.returncode != 0:
        lines = [line for line in output.splitlines() if line.strip()]
        return _sanitize(f"exit code {result.returncode}: {lines[-1] if lines else ''}")
    return "no generation result in llama-bench output"


def run_bench(bench: Path, model: Path, args: Sequence[str], *, timeout: float, repetitions: int = 3) -> BenchResult:
    """Run ``llama-bench`` once for one candidate; never raises.

    Args:
        bench: The ``llama-bench`` executable.
        model: GGUF file to measure.
        args: Candidate arguments (see ``plan_to_bench_args``).
        timeout: Seconds before the run is killed.
        repetitions: Repetitions per test (``-r``).

    Returns:
        BenchResult: The measurement, or ``error`` set (out of memory, timeout, bad exit, unparseable output).
    """
    argv = [str(bench), "-m", str(model), "-o", "json", "-r", str(repetitions)]
    argv += ["-p", str(BENCH_PROMPT_TOKENS), "-n", str(BENCH_GEN_TOKENS), *args]
    result = proc.run(argv, timeout=timeout)
    pp, tg = parse_bench_json(result.stdout) if result.returncode == 0 else (None, None)
    if tg is not None:
        return BenchResult(tuple(args), tg, pp)
    return BenchResult(tuple(args), None, pp, _bench_error(result, timeout))


# --- sweep ---------------------------------------------------------------------------------------------------------


def _thread_values(host: HostInfo) -> list[int]:
    """Thread counts to try: ``cores - 2``, ``cores``, ``min(logical, cores + 2)``; deduplicated, at least 1."""
    cores = max(1, host.physical_cores)
    wanted = (max(1, cores - 2), cores, max(1, min(host.logical_cores, cores + 2)))
    return list(dict.fromkeys(wanted))


def _ngl_values(seed_ngl: int, block_count: int) -> list[int]:
    """Layer counts around a partial offload: ``seed +-2`` and ``+-4``, clamped to ``0..block_count + 1``."""
    values = [min(block_count + 1, max(0, seed_ngl + delta)) for delta in NGL_DELTAS]
    return list(dict.fromkeys(v for v in values if v != seed_ngl))


def candidate_steps(
    seed: SeedPlan, host: HostInfo, *, allow_kv_quant: bool, block_count: int | None = None
) -> list[tuple[str, list[SeedPlan]]]:
    """List the coordinate-descent dimensions in sweep order.

    Each step is ``(field_name, candidates)`` where every candidate is ``seed`` with only that field changed; the sweep
    re-applies the field to the best plan found so far.

    Args:
        seed: The starting plan.
        host: Host snapshot (core counts bound the thread candidates).
        allow_kv_quant: Include the ``q8_0`` KV-cache candidate (opt-in because it affects quality).
        block_count: Layers in the model; the ``ngl`` step needs it and is skipped when None.

    Returns:
        list[tuple[str, list[SeedPlan]]]: ``flash_attn``, ``ubatch``, ``threads``, ``ngl`` (only when the seed is a
            partial offload), ``cache_type`` (only when ``allow_kv_quant``), in that order.
    """
    steps = [
        ("flash_attn", [replace(seed, flash_attn=v) for v in (False, True)]),
        ("ubatch", [replace(seed, ubatch=v) for v in UBATCH_VALUES]),
        ("threads", [replace(seed, threads=v) for v in _thread_values(host)]),
    ]
    if block_count is not None and seed.ngl < block_count + 1:
        steps.append(("ngl", [replace(seed, ngl=v) for v in _ngl_values(seed.ngl, block_count)]))
    if allow_kv_quant:
        steps.append(("cache_type", [replace(seed, cache_type=KV_QUANT_TYPE)]))
    return steps


def _beats(new: BenchResult, best: BenchResult | None) -> bool:
    """Tell whether ``new`` replaces ``best``, treating differences within ``TIE_TOLERANCE`` as noise.

    ``new`` wins with a generation speed more than ``TIE_TOLERANCE`` above ``best``. When the two are within that
    tolerance on generation speed (either way), it wins only with a prompt speed more than ``TIE_TOLERANCE`` above
    ``best``'s. Otherwise the incumbent stays, so batch/thread steps are not decided by measurement noise.
    """
    if new.tg_tps is None:
        return False
    if best is None or best.tg_tps is None:
        return True
    if new.tg_tps > best.tg_tps * (1 + TIE_TOLERANCE):
        return True
    if new.tg_tps >= best.tg_tps * (1 - TIE_TOLERANCE):
        return (new.pp_tps or 0.0) > (best.pp_tps or 0.0) * (1 + TIE_TOLERANCE)
    return False


class _Sweep:
    """State of one coordinate-descent run (results, cache, best plan)."""

    def __init__(
        self,
        bench: Path,
        model: Path,
        seed: SeedPlan,
        host: HostInfo,
        *,
        time_limit: float,
        block_count: int | None,
        allow_kv_quant: bool,
        device: str | None,
        on_progress: Callable[[BenchResult], None] | None,
        clock: Callable[[], float],
    ) -> None:
        """Store the run's inputs; nothing is measured until ``run``."""
        self._bench, self._model, self._seed, self._host = bench, model, seed, host
        self._time_limit, self._block_count, self._allow_kv_quant = time_limit, block_count, allow_kv_quant
        self._device, self._on_progress, self._clock = device, on_progress, clock
        self._started = 0.0
        self._cache: dict[tuple[str, ...], BenchResult] = {}
        self.results: list[BenchResult] = []
        self.best: BenchResult | None = None
        self.best_plan = seed
        self.interrupted = False

    def measured(self, plan: SeedPlan) -> BenchResult | None:
        """Return the result already measured for ``plan``, if any."""
        return self._cache.get(tuple(plan_to_bench_args(plan, self._device)))

    def _remaining(self) -> float:
        return self._time_limit - (self._clock() - self._started)

    def _measure(self, plan: SeedPlan) -> None:
        """Measure ``plan`` unless an identical candidate was already measured; track the best."""
        args = tuple(plan_to_bench_args(plan, self._device))
        if args in self._cache:
            return
        timeout = min(BENCH_TIMEOUT, max(MIN_BENCH_TIMEOUT, self._remaining())) if self.results else BENCH_TIMEOUT
        result = run_bench(self._bench, self._model, args, timeout=timeout)
        self._cache[args] = result
        self.results.append(result)
        if result.error:
            log.info("Candidate %s failed: %s", " ".join(args), result.error)
        if _beats(result, self.best):
            self.best, self.best_plan = result, plan
        if self._on_progress is not None:
            self._on_progress(result)

    def _step(self, field: str, candidates: Sequence[SeedPlan]) -> None:
        """Try every value of one dimension on top of the best plan so far."""
        base = self.best_plan
        for candidate in candidates:
            if self._remaining() <= 0:
                return
            plan = replace(base, **{field: getattr(candidate, field)})
            if field == "cache_type" and not plan.flash_attn:
                continue  # a quantised V cache needs flash attention
            self._measure(plan)

    def run(self) -> list[BenchResult]:
        """Measure the seed, then each dimension in turn until the budget is spent or the user interrupts.

        Ctrl-C ends the sweep early and sets ``interrupted``; what was measured so far is kept.

        Returns:
            list[BenchResult]: Every measured candidate in order (failures included).
        """
        self._started = self._clock()
        steps = candidate_steps(
            self._seed, self._host, allow_kv_quant=self._allow_kv_quant, block_count=self._block_count
        )
        try:
            self._measure(self._seed)
            for field, candidates in steps:
                self._step(field, candidates)
        except KeyboardInterrupt:
            self.interrupted = True
            log.warning("Sweep interrupted; keeping the best result so far")
        return self.results


def sweep(
    bench: Path,
    model: Path,
    seed: SeedPlan,
    host: HostInfo,
    *,
    time_limit: float,
    allow_kv_quant: bool = False,
    block_count: int | None = None,
    device: str | None = None,
    on_progress: Callable[[BenchResult], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[BenchResult]:
    """Coordinate-descent over launch settings, one dimension at a time, keeping the best after each.

    The seed is always measured; ``time_limit`` seconds bound the rest (0 measures only the seed). Failed candidates
    are recorded and skipped, identical candidates are never re-measured, and Ctrl-C returns what was measured.

    Args:
        bench: The ``llama-bench`` executable.
        model: GGUF file to measure.
        seed: Starting plan.
        host: Host snapshot.
        time_limit: Budget in seconds, measured with ``clock``.
        allow_kv_quant: Also try a ``q8_0`` KV cache.
        block_count: Model layer count (enables the ``ngl`` dimension for partial offloads).
        device: Device id to pin ``llama-bench`` to (``-dev``).
        on_progress: Called with every new result.
        clock: Monotonic time source (injectable for tests).

    Returns:
        list[BenchResult]: All measured candidates in order.
    """
    return _Sweep(
        bench,
        model,
        seed,
        host,
        time_limit=time_limit,
        block_count=block_count,
        allow_kv_quant=allow_kv_quant,
        device=device,
        on_progress=on_progress,
        clock=clock,
    ).run()


# --- HTTP ----------------------------------------------------------------------------------------------------------


def _decode_body(raw: bytes) -> Any:
    """Parse a JSON body, returning None when it is empty or not JSON."""
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _http_json(method: str, url: str, payload: Any, timeout: float) -> tuple[int, Any] | None:
    """Send one JSON request to a local service, bypassing proxies.

    Args:
        method: ``GET`` or ``POST``.
        url: Absolute ``http(s)`` URL.
        payload: JSON-serialisable body, or None for no body.
        timeout: Socket timeout in seconds.

    Returns:
        tuple[int, Any] | None: ``(status, parsed JSON or None)``, or None when nothing answered.
    """
    if not url.startswith(("http://", "https://")):
        return None
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    # the scheme was restricted to http(s) above
    request = urllib.request.Request(url, data=body, headers=headers, method=method)  # ruff: ignore[suspicious-url-open-usage]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as resp:
            return resp.status, _decode_body(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, _decode_body(exc.read())
        except OSError:
            return exc.code, None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def _free_port() -> int:
    """Return a free ephemeral loopback port (never the default serve port)."""
    with socket.socket() as sock:
        sock.bind((constants.DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


# --- validation ----------------------------------------------------------------------------------------------------


def _completes(host: str, port: int, timeout: float) -> bool:
    """Ask the server for a few tokens; True when it answers 200 with a ``content`` string."""
    payload = {"prompt": VALIDATE_PROMPT, "n_predict": VALIDATE_TOKENS}
    answer = _http_json("POST", f"http://{host}:{port}/completion", payload, min(timeout, COMPLETION_TIMEOUT))
    return (
        answer is not None
        and answer[0] == HTTP_OK
        and isinstance(answer[1], dict)
        and isinstance(answer[1].get("content"), str)
    )


def validate(server: Path, settings: ServeSettings, *, base: Path | None = None, timeout: float = 180.0) -> bool:
    """Launch the real ``llama-server`` with ``settings`` and check that it generates; never raises.

    The probe listens on a free ephemeral loopback port and keeps its pid file and log in ``<base>/validate``, so it
    cannot collide with, or ever stop, the user's own server. The probe is always stopped, also on Ctrl-C (a
    ``KeyboardInterrupt`` is not swallowed: the caller decides what to keep).

    Args:
        server: The ``llama-server`` executable.
        settings: Launch settings to verify (host and port are replaced by the probe's).
        base: Storage directory (None resolves ``constants.home()`` at call time).
        timeout: Seconds to wait for readiness (the completion has its own, shorter limit).

    Returns:
        bool: True when the server became ready and completed a short generation.
    """
    scratch = (base if base is not None else constants.home()) / VALIDATE_DIRNAME
    try:
        port = _free_port()
        probe = replace(settings, host=constants.DEFAULT_HOST, port=port)
        argv = llamacpp.build_serve_argv(server, probe)
        server_runtime.start(
            argv, base=scratch, host=probe.host, port=port, model=probe.alias, detach=True, ready_timeout=timeout
        )
        return _completes(probe.host, port, timeout)
    except (LlamaCppError, OSError) as exc:
        log.warning("Validation launch failed: %s", exc)
        return False
    finally:
        server_runtime.stop(scratch)


# --- Ollama / VRAM -------------------------------------------------------------------------------------------------


def available_mib(device: Device) -> int:
    """Return the memory a run can use on ``device`` now: its free memory (the total when that is unknown)."""
    return device.free_mib if device.free_mib > 0 else device.total_mib


def vram_preflight(model_need_mib: int, device: Device | None) -> int:
    """Return how many MiB the model is short of on ``device`` (0 when it fits or there is no device).

    A model larger than the card can only ever use the whole card, so its need is capped at ``device.total_mib``;
    the shortfall is then just the memory other processes hold. An unknown free figure (0) is no shortfall.

    Args:
        model_need_mib: Memory the model needs, in MiB.
        device: The chosen GPU, or None for CPU-only.

    Returns:
        int: Shortfall in MiB computed from ``device.free_mib``.
    """
    if device is None or device.free_mib <= 0:
        return 0
    return max(0, min(model_need_mib, device.total_mib) - device.free_mib)


def _ollama_base_url() -> str:
    """Resolve Ollama's API base URL from ``OLLAMA_HOST`` (``host[:port]`` or a URL); default ``127.0.0.1:11434``."""
    default = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}"
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return default
    scheme, _, rest = raw.rpartition("://")
    parts = urlsplit(f"//{rest.split('/', 1)[0]}")
    try:
        port = parts.port or OLLAMA_DEFAULT_PORT
    except ValueError:
        return default
    host = parts.hostname or constants.DEFAULT_HOST
    host = constants.DEFAULT_HOST if host in {"0.0.0.0", "::"} else host  # a bind-all address is not connectable
    return f"{scheme or 'http'}://{f'[{host}]' if ':' in host else host}:{port}"


def _loaded_models(data: Any) -> list[str]:
    """Extract model names from an ``/api/ps`` document, tolerating unexpected shapes."""
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    names = (m.get("name") or m.get("model") for m in models if isinstance(m, dict))
    return [n for n in names if isinstance(n, str) and n]


def unload_ollama(timeout: float = 30.0) -> bool:
    """Ask a local Ollama to unload every loaded model (``keep_alive: 0``) to free VRAM.

    Args:
        timeout: Seconds allowed for each unload request.

    Returns:
        bool: True when Ollama answered and every loaded model was unloaded (also when none were loaded); False when
            Ollama is unreachable or an unload failed.
    """
    base = _ollama_base_url()
    answer = _http_json("GET", f"{base}/api/ps", None, min(timeout, OLLAMA_PS_TIMEOUT))
    if answer is None or answer[0] != HTTP_OK:
        log.info("Ollama is not reachable at %s", base)
        return False
    unloaded = True
    for name in _loaded_models(answer[1]):
        reply = _http_json("POST", f"{base}/api/generate", {"model": name, "keep_alive": 0}, timeout)
        if reply is None or reply[0] != HTTP_OK:
            log.warning("Could not unload Ollama model %s", name)
            unloaded = False
    return unloaded


# --- optimize ------------------------------------------------------------------------------------------------------


def _step_down(plan: SeedPlan) -> SeedPlan | None:
    """Return a more conservative plan: ctx halved (floor ``MIN_CTX``), then ``ngl`` lowered by 4; None if exhausted."""
    if plan.ctx > MIN_CTX:
        return replace(plan, ctx=max(MIN_CTX, plan.ctx // 2))
    if plan.ngl > 0:
        return replace(plan, ngl=max(0, plan.ngl - 4))
    return None


def _validated(
    server_bin: Path, model: OllamaModel, cfg: Config, device: Device | None, plan: SeedPlan, base: Path | None
) -> SeedPlan | None:
    """Validate ``plan`` with the real server; on failure step down once and re-validate.

    Args:
        server_bin: The ``llama-server`` executable.
        model: The model being tuned.
        cfg: Configuration (host and port defaults).
        device: Chosen GPU, or None.
        plan: The plan to verify.
        base: Storage directory.

    Returns:
        SeedPlan | None: The plan that generated successfully, or None when both attempts failed.
    """

    def works(candidate: SeedPlan) -> bool:
        settings = ServeSettings(
            model_path=model.blob_path or Path(),
            alias=model.name,
            host=cfg.host,
            port=cfg.port,
            ctx=candidate.ctx,
            device=device.id if device else None,
            profile_args=seed_to_args(candidate),
        )
        return validate(server_bin, settings, base=base)

    if works(plan):
        return plan
    lower = _step_down(plan)
    return lower if lower is not None and works(lower) else None


def _bench_table(results: Sequence[BenchResult]) -> list[dict[str, Any]]:
    """Convert results to the JSON-friendly rows stored in ``Profile.bench``."""
    return [{"args": list(r.args), "tg_tps": r.tg_tps, "pp_tps": r.pp_tps, "error": r.error} for r in results]


def _store(cfg: Config, base: Path | None, keys: Sequence[str], profile: Profile) -> Profile:
    """Record ``profile`` under every key in ``keys`` and persist the config; a failed save is logged, never raised."""
    for key in keys:
        cfg.profiles[key] = profile
    try:
        config.save(cfg, base)
    except Exception:
        log.warning("Could not save the profile for %s", ", ".join(keys), exc_info=True)
    return profile


def optimize_model(
    model: OllamaModel,
    cfg: Config,
    host: HostInfo,
    device: Device | None,
    *,
    ctx: int,
    quick: bool,
    time_limit: float,
    allow_kv_quant: bool,
    base: Path | None = None,
    on_progress: Callable[[BenchResult], None] | None = None,
) -> Profile:
    """Find, validate and save the fastest launch settings for ``model``.

    Seed from the GGUF and hardware; unless ``quick``, sweep with ``llama-bench``, then launch the winner as a real
    server. A failed validation steps down once (ctx, then ``ngl``); if that fails too the unmeasured seed is saved.
    Ctrl-C (during the sweep or the validation) skips or abandons the validation and saves the best result so far with
    ``validated=False``. The seed budgets the free VRAM (see :func:`optimize_seed`). ``quick`` saves the unmeasured
    seed without launching anything. The profile is stored in ``cfg`` under ``profile_key(digest, final_ctx)`` (and
    also under the requested ctx when that differs, so a lookup by the requested ctx finds it); the config is
    written (a write failure is logged, never raised).

    Args:
        model: The installed Ollama model.
        cfg: Configuration to update (other profiles are preserved).
        host: Host snapshot.
        device: Chosen GPU, or None for CPU-only.
        ctx: Requested context length.
        quick: Skip benchmarking and validation; save the heuristic seed.
        time_limit: Sweep budget in seconds.
        allow_kv_quant: Also try a ``q8_0`` KV cache.
        base: Storage directory (None resolves ``constants.home()`` at call time).
        on_progress: Called with every benchmark result.

    Returns:
        Profile: The saved profile.

    Raises:
        LlamaCppError: ``llama-server`` cannot be found.
        ValueError: The model has no GGUF layer.
    """
    server_bin = llamacpp.find_binary("llama-server", cfg.llama_server)
    if server_bin is None:
        raise LlamaCppError("llama-server was not found; install llama.cpp first")
    if model.blob_path is None or model.digest is None:
        raise ValueError(f"{model.name} has no model layer")
    info = read_gguf(model.blob_path)
    seed = optimize_seed(info, host, device, ctx)  # budgets the free VRAM; the fingerprint keeps describing the card
    fingerprint = hardware.fingerprint(host, device, llamacpp.version(server_bin))
    profile = Profile(args=seed_to_args(seed), ctx=seed.ctx, fingerprint=fingerprint, timestamp=_now())
    if not quick:
        bench_bin = llamacpp.find_binary("llama-bench", cfg.llama_bench)
        run = None
        if bench_bin is None:
            log.warning("llama-bench not found; validating the heuristic seed without benchmarking")
        else:
            run = _Sweep(
                bench_bin,
                model.blob_path,
                seed,
                host,
                time_limit=time_limit,
                block_count=info.block_count if device else None,  # no ngl sweep without a GPU
                allow_kv_quant=allow_kv_quant,
                device=device.id if device else None,
                on_progress=on_progress,
                clock=time.monotonic,
            )
            run.run()
            profile.bench = _bench_table(run.results)
        winner = run.best_plan if run is not None and run.best is not None else seed
        interrupted = run is not None and run.interrupted
        final: SeedPlan | None = winner  # an interrupted run launches nothing: the measured winner is kept as is
        if not interrupted:
            try:
                final = _validated(server_bin, model, cfg, device, winner, base)
            except KeyboardInterrupt:
                interrupted = True
                log.warning("Validation interrupted; keeping the measured winner without validating it")
                final = winner
        if final is not None:
            result = run.measured(final) if run is not None else None
            profile.args, profile.ctx = seed_to_args(final), final.ctx
            if result is not None and result.tg_tps is not None:
                profile.tg_tps, profile.pp_tps, profile.measured = result.tg_tps, result.pp_tps, True
        profile.validated = not interrupted
    # The same profile is also reachable by the requested ctx, so `serve --ctx <requested>` finds it even when the
    # actual ctx (``profile.ctx``) was reduced to fit or by validation step-down.
    keys = list(dict.fromkeys([profile_key(model.digest, profile.ctx), profile_key(model.digest, ctx)]))
    return _store(cfg, base, keys, profile)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")
