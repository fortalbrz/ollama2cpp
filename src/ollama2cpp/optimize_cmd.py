"""The ``optimize`` command: pre-flight checks, one isolated optimizer run per model and the before/after report.

No CLI parsing lives here (see ``__main__``); :func:`run_optimize` returns the process exit code.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from ollama2cpp import constants, hardware, llamacpp, ollama, optimizer, selector, server, ui, workflow
from ollama2cpp.commands import find_server, load_config, models_root, pick_device, server_endpoint
from ollama2cpp.config import Config, Profile
from ollama2cpp.constants import DEFAULT_CTX, EXIT_INTERRUPTED
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.optimizer import BenchResult, SeedPlan
from ollama2cpp.selector import Candidate
from ollama2cpp.workflow import WorkflowError

__all__ = ["OptimizeOptions", "run_optimize"]

logger = logging.getLogger(__name__)

DEFAULT_TIME_LIMIT = 300.0
UNLOAD_WAIT_S = 15.0  # how long to wait for Ollama's VRAM to come back after asking it to unload
UNLOAD_POLL_S = 1.0
HOLD_TOLERANCE_MIB = 1024  # other processes may hold this much VRAM (a desktop session) without stopping a benchmark


@dataclass(frozen=True)
class OptimizeOptions:
    """Everything the ``optimize`` command accepts.

    Attributes:
        model: Model name (default: the saved default model, else an automatic choice).
        model_from_env: ``model`` came from ``$OLLAMA_MODEL``; when it is not installed the saved default (or an
            automatic choice) is used with a warning.
        quick: Save the heuristic seed without benchmarking.
        all_models: Optimize every chat model.
        time_limit: Sweep budget per model, in seconds.
        ctx: Context size to optimize for (default 8192).
        allow_kv_quant: Also try a ``q8_0`` KV cache.
        unload_ollama: Ask Ollama to unload its models when they hold the VRAM the model needs.
        stop_server: Stop our own running server first (it holds VRAM).
        assume_yes: Answer yes to questions (first-run installation of llama.cpp).
    """

    model: str | None = None
    model_from_env: bool = False
    quick: bool = False
    all_models: bool = False
    time_limit: float = DEFAULT_TIME_LIMIT
    ctx: int | None = None
    allow_kv_quant: bool = False
    unload_ollama: bool = False
    stop_server: bool = False
    assume_yes: bool = False


@dataclass(frozen=True)
class _Outcome:
    """The result of optimizing one model: a profile and its seed, or an error."""

    name: str
    profile: Profile | None = None
    seed: SeedPlan | None = None
    error: str = ""


# --- pre-flight ----------------------------------------------------------------------------------------------------


def _prepare(cfg: Config, opts: OptimizeOptions) -> Path:
    """Check the tools; record discovered paths in ``cfg`` (saved with the profile). Stops and refuses nothing."""
    binary = find_server(cfg)
    if binary is None:
        raise WorkflowError("llama-server was not found; run `ollama2cpp serve` or `ollama2cpp update` first.")
    cfg.llama_server = str(binary)
    cfg.ollama_path = str(models_root(cfg))
    if opts.quick:
        return binary
    bench = llamacpp.find_binary("llama-bench", cfg.llama_bench)
    if bench is None:
        raise WorkflowError(
            "llama-bench was not found. Run `ollama2cpp update` (or `ollama2cpp doctor`), "
            "or use --quick to save the heuristic settings without benchmarking."
        )
    cfg.llama_bench = str(bench)
    return binary


def _release_server(cfg: Config, opts: OptimizeOptions) -> None:
    """Refuse (or, with ``--stop-server``, stop) our own running server: it holds the VRAM a benchmark needs.

    Runs after the targets are resolved, so a typo never stops the user's server. The server is looked up at the
    address in its pid record (it may have been started with another ``--port``).
    """
    if opts.quick:
        return
    host, port = server_endpoint(cfg)
    state, alias = server.describe_running(host, port, constants.home())
    if state != "ours":
        return
    if not opts.stop_server:
        raise WorkflowError(
            f"ollama2cpp is serving {alias or 'a model'} and that holds VRAM. "
            "Run `ollama2cpp stop` first, or pass --stop-server."
        )
    ui.show_notice("info", "Stopping the running ollama2cpp server.")
    server.stop(constants.home())


def _targets(cfg: Config, opts: OptimizeOptions, ctx: int) -> list[Candidate]:
    """Resolve the models to optimize (chat models only)."""
    if opts.all_models:
        if opts.model:
            raise WorkflowError("Give either a MODEL or --all, not both.")
        cands = [c for c in selector.classify_all(ollama.list_models(models_root(cfg)), ctx) if c.kind == "chat"]
        if not cands:
            raise WorkflowError("No chat models are installed; pull one first, e.g. `ollama pull llama3.2`.")
        return cands
    model = workflow.select_model(cfg, workflow.ServeOptions(model=opts.model, model_from_env=opts.model_from_env))
    cand = selector.classify(model, ctx)
    if cand.kind != "chat":
        why = f": {cand.reason}" if cand.reason else ""
        raise WorkflowError(f"{model.name} is not a chat model ({cand.kind}{why}); only chat models can be optimized.")
    return [cand]


def _wait_for_vram(cand: Candidate, cfg: Config, binary: Path, device: Device | None, wait: float) -> Device | None:
    """Poll the devices until the model fits in free VRAM or ``wait`` seconds pass; return the latest device."""
    deadline = time.monotonic() + wait
    while True:
        fresh = hardware.best_device(hardware.detect_devices(binary), cfg.device) or device
        if optimizer.vram_preflight(cand.need_mib, fresh) <= 0 or time.monotonic() >= deadline:
            return fresh
        time.sleep(UNLOAD_POLL_S)


def _clear_vram(
    cand: Candidate, cfg: Config, binary: Path, device: Device | None, opts: OptimizeOptions
) -> Device | None:
    """Decide whether the benchmark can run with the VRAM that is free now; unload Ollama first when asked.

    The seed budgets the free memory (see ``optimizer.optimize_seed``), so a model that does not fully fit is
    benchmarked with a partial offload. The run is refused only when other processes hold more than
    ``HOLD_TOLERANCE_MIB`` and the model needs memory that is not free.

    Raises:
        WorkflowError: When the VRAM held by other processes leaves the model short.
    """
    if device is None:
        return device
    name = cand.model.name
    short = optimizer.vram_preflight(cand.need_mib, device)
    if short > 0 and opts.unload_ollama:
        problem = f"{name} needs about {cand.need_mib} MiB of VRAM but only {device.free_mib} MiB are free"
        ui.show_notice("info", f"{problem} (short by {short} MiB); asking Ollama to unload its models.")
        unloaded = optimizer.unload_ollama()
        if not unloaded:
            ui.show_notice("warn", "Ollama did not confirm the unload (is it running?).")
        device = _wait_for_vram(cand, cfg, binary, device, UNLOAD_WAIT_S if unloaded else 0.0) or device
        short = optimizer.vram_preflight(cand.need_mib, device)
        after = " after unloading Ollama"
    else:
        after = ""
    if cand.need_mib > device.total_mib:
        ui.show_notice(
            "info",
            f"{name} does not fit this GPU entirely ({cand.need_mib} MiB needed, {device.total_mib} MiB total); "
            "layers will be offloaded partially.",
        )
    if short <= 0:
        return device
    held = device.total_mib - device.free_mib
    if held <= HOLD_TOLERANCE_MIB:
        ui.show_notice(
            "info", f"Free VRAM is {device.free_mib} MiB of {device.total_mib} MiB; optimizing for what is free."
        )
        return device
    if opts.unload_ollama:
        raise WorkflowError(
            f"{name} is still short by {short} MiB of VRAM{after}: {held} MiB of VRAM is in use by other processes; "
            "close the other GPU users."
        )
    raise WorkflowError(
        f"{name} needs about {cand.need_mib} MiB of VRAM but only {device.free_mib} MiB are free "
        f"(short by {short} MiB): {held} MiB of VRAM is in use by other processes. "
        "Free it (for example `ollama stop MODEL` for Ollama), or pass --unload-ollama."
    )


# --- running -------------------------------------------------------------------------------------------------------


def _speed(result: BenchResult) -> str:
    return f"{result.tg_tps:.1f} t/s" if result.tg_tps is not None else "failed"


def _optimize(cand: Candidate, cfg: Config, host: HostInfo, device: Device | None, opts: OptimizeOptions) -> Profile:
    """Run the optimizer for one model, with a progress display while it benchmarks."""
    name = cand.model.name

    def run(on_progress: Callable[[BenchResult], None] | None) -> Profile:
        return optimizer.optimize_model(
            cand.model,
            cfg,
            host,
            device,
            ctx=opts.ctx or DEFAULT_CTX,
            quick=opts.quick,
            time_limit=opts.time_limit,
            allow_kv_quant=opts.allow_kv_quant,
            on_progress=on_progress,
        )

    if opts.quick:
        return run(None)
    with ui.steps_progress() as bar:
        task = bar.add_task(name, total=None)

        def step(result: BenchResult) -> None:
            bar.update(task, advance=1, description=f"{name} (last: {_speed(result)})")

        return run(step)


def _optimize_one(
    cand: Candidate, cfg: Config, binary: Path, host: HostInfo, device: Device | None, opts: OptimizeOptions
) -> _Outcome:
    """Optimize one model; any failure becomes an error outcome so the other models still run."""
    name = cand.model.name
    try:
        if not opts.quick:
            device = _clear_vram(cand, cfg, binary, device, opts)
        profile = _optimize(cand, cfg, host, device, opts)
        seed = optimizer.optimize_seed(cand.gguf, host, device, opts.ctx or DEFAULT_CTX) if cand.gguf else None
    except Exception as exc:
        logger.warning("Optimizing %s failed: %s", name, exc)
        logger.debug("Optimizing %s failed", name, exc_info=True)
        ui.show_notice("error", f"{name}: {exc}")
        return _Outcome(name, error=str(exc))
    return _Outcome(name, profile, seed)


# --- report --------------------------------------------------------------------------------------------------------


def _rate(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _seed_measurement(profile: Profile) -> tuple[float | None, float | None]:
    """Return the tg and pp speeds of the seed, which the sweep measures first (``(None, None)`` when it failed)."""
    first = profile.bench[0] if profile.bench else {}
    if first.get("error") or first.get("tg_tps") is None:
        return None, None
    return first.get("tg_tps"), first.get("pp_tps")


def _show_outcome(outcome: _Outcome, opts: OptimizeOptions) -> None:
    """Print the before/after table of one successful run."""
    profile, seed = outcome.profile, outcome.seed
    if profile is None:
        return
    seed_args = " ".join(optimizer.seed_to_args(seed)) if seed else "-"
    seed_ctx = str(seed.ctx) if seed else "-"
    rows: list[list[str | Text]]
    if profile.measured:
        seed_tg, seed_pp = _seed_measurement(profile)
        label = "seed (measured)" if seed_tg is not None else "seed (unmeasured)"
        rows = [
            [label, _rate(seed_tg), _rate(seed_pp), seed_ctx, seed_args],
            ["optimized", _rate(profile.tg_tps), _rate(profile.pp_tps), str(profile.ctx), " ".join(profile.args)],
        ]
        if seed_tg and profile.tg_tps:
            gain = (profile.tg_tps / seed_tg - 1) * 100
            ui.show_notice("ok", f"{outcome.name}: generation speed {gain:+.0f}% vs the heuristic seed.")
    else:
        rows = [["seed (unmeasured)", "-", "-", str(profile.ctx), " ".join(profile.args)]]
        if opts.quick:
            ui.show_notice("info", f"{outcome.name}: quick mode saved the heuristic settings (not measured).")
        else:
            ui.show_notice("warn", f"{outcome.name}: benchmark failed - heuristic settings saved (unmeasured).")
    ui.show_table(f"{outcome.name}", ["Settings", "tg t/s", "pp t/s", "ctx", "args"], rows)
    if not profile.validated:
        ui.show_notice(
            "warn",
            f"{outcome.name}: interrupted before validation - the best result so far was saved, not validated "
            "against a real server. Re-run `ollama2cpp optimize` to finish it.",
        )


def _show_summary(outcomes: list[_Outcome]) -> None:
    rows = []
    for o in outcomes:
        if o.profile is None:
            rows.append([o.name, ui.status_text("fail"), "-", o.error])
        elif o.profile.measured:
            rows.append([o.name, ui.status_text("ok"), _rate(o.profile.tg_tps), f"ctx {o.profile.ctx}"])
        else:
            rows.append([o.name, ui.status_text("warn"), "-", "heuristic settings only (unmeasured)"])
    ui.show_table("Optimization summary", ["Model", "Result", "tg t/s", "Detail"], rows)


def run_optimize(opts: OptimizeOptions) -> int:
    """Optimize one model (or every chat model) and save the profiles.

    Ctrl-C keeps the results so far: the interrupted model keeps its best measurement (saved unvalidated) and the
    remaining models of ``--all`` are not touched.

    Args:
        opts: The command options.

    Returns:
        int: 0 when at least one model got a profile, 130 when interrupted before any was saved, else 1.
    """
    cfg = load_config(assume_yes=opts.assume_yes)
    binary = _prepare(cfg, opts)
    ctx = opts.ctx or DEFAULT_CTX
    targets = _targets(cfg, opts, ctx)  # validated before anything is stopped
    _release_server(cfg, opts)
    host, device = workflow.probe(binary, cfg.device)
    outcomes: list[_Outcome] = []
    interrupted = False
    for index, cand in enumerate(targets):
        if index and not opts.quick:
            device = pick_device(cfg, binary) or device  # free VRAM changes between models
        try:
            outcome = _optimize_one(cand, cfg, binary, host, device, opts)
        except KeyboardInterrupt:
            interrupted = True
            break
        _show_outcome(outcome, opts)
        outcomes.append(outcome)
        if outcome.profile is not None and not outcome.profile.validated:
            interrupted = True
            break
    if interrupted:
        ui.show_notice("warn", "Interrupted - kept the results so far.")
    if len(outcomes) > 1:
        _show_summary(outcomes)
    if any(o.profile is not None for o in outcomes):
        return 0
    return EXIT_INTERRUPTED if interrupted else 1
