"""Serve orchestration: first-run setup, configuration upkeep, launch planning and starting the server.

No CLI parsing lives here (see ``__main__``); every user-facing failure is a :class:`WorkflowError` (or one of the
domain errors ``LlamaCppError`` / ``ModelNotFoundError``) that the CLI turns into one error line and exit code 1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ollama2cpp import config, constants, gguf, hardware, llamacpp, ollama, optimizer, proc, selector, server, ui
from ollama2cpp.config import Config, Profile
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.llamacpp import ServeSettings
from ollama2cpp.ollama import OllamaModel

__all__ = [
    "ServeOptions",
    "ServePlan",
    "WorkflowError",
    "ensure_config",
    "first_run",
    "plan_serve",
    "run_serve",
    "select_model",
]

logger = logging.getLogger(__name__)

PORT_FREE_TIMEOUT = 15.0
PORT_POLL_INTERVAL = 0.2
_MIB = 1024 * 1024


class WorkflowError(RuntimeError):
    """A user-facing failure; the message is shown as-is."""


@dataclass(frozen=True)
class ServeOptions:
    """Everything the ``serve`` command accepts.

    Attributes:
        model: Model name for this run (and the stored default on a first run).
        ollama_path: Explicit Ollama models directory.
        host: Listen address override.
        port: Listen port override.
        ctx: Requested context size override.
        detach: Run in the background and return once ready.
        restart: Replace our own running server when it serves another model.
        dry_run: Print the command instead of starting anything.
        vision: Pass the model's multimodal projector (``--mmproj``) when it has one.
        assume_yes: Answer yes to confirmation questions.
        extra_args: Extra ``llama-server`` arguments, appended last.
    """

    model: str | None = None
    ollama_path: Path | None = None
    host: str | None = None
    port: int | None = None
    ctx: int | None = None
    detach: bool = False
    restart: bool = False
    dry_run: bool = False
    vision: bool = False
    assume_yes: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServePlan:
    """What to run: the final command plus the facts that explain it.

    Attributes:
        argv: The full ``llama-server`` command line.
        settings: The settings the command was built from.
        model: The model being served.
        host: Listen address.
        port: Listen port.
        profile_note: One line saying which profile or seed was used.
        warnings: Warnings to show before starting.
    """

    argv: tuple[str, ...]
    settings: ServeSettings
    model: OllamaModel
    host: str
    port: int
    profile_note: str
    warnings: tuple[str, ...] = ()


# --- hardware and model selection ----------------------------------------------------------------------------------


def _probe(llama_server: Path, override: str | None) -> tuple[HostInfo, Device | None]:
    """Probe the host and choose the device to run on.

    Args:
        llama_server: The ``llama-server`` binary (used to list devices).
        override: A device id configured by the user.

    Returns:
        tuple[HostInfo, Device | None]: The host snapshot and the chosen GPU (None for CPU-only).
    """
    host = hardware.host_info(llama_server)
    return host, hardware.best_device(host.devices, override)


def _require_server(cfg: Config) -> Path:
    """Return the configured ``llama-server`` path.

    Args:
        cfg: The configuration (already completed by :func:`ensure_config`).

    Returns:
        Path: The binary.

    Raises:
        WorkflowError: When no binary is configured.
    """
    if not cfg.llama_server:
        raise WorkflowError("llama-server is not configured; run `ollama2cpp serve` again to set it up")
    return Path(cfg.llama_server)


def _require_root(cfg: Config) -> Path:
    """Return the configured Ollama models directory.

    Args:
        cfg: The configuration (already completed by :func:`ensure_config`).

    Returns:
        Path: The models directory.

    Raises:
        WorkflowError: When none is configured.
    """
    if not cfg.ollama_path:
        raise WorkflowError("The Ollama models path is not configured; run `ollama2cpp serve` again to set it up")
    return Path(cfg.ollama_path)


def _require_blob(model: OllamaModel) -> tuple[Path, str]:
    """Return the blob path and digest of a model whose blob is on disk.

    Args:
        model: The installed model.

    Returns:
        tuple[Path, str]: The GGUF blob and its digest.

    Raises:
        WorkflowError: For a dangling manifest (no model layer or blob missing).
    """
    if model.blob_path is None or model.digest is None or not model.blob_exists:
        raise WorkflowError(f"Model blob missing for {model.name}; re-pull it with `ollama pull {model.name}`")
    return model.blob_path, model.digest


def _pick_best(root: Path, host: HostInfo, device: Device | None) -> tuple[selector.Candidate, bool]:
    """Choose the best installed chat model for the hardware.

    Args:
        root: The Ollama models directory.
        host: Host snapshot (RAM is the budget on CPU-only hosts).
        device: Chosen GPU (its total VRAM is the budget), or None.

    Returns:
        tuple[selector.Candidate, bool]: The candidate and whether it fits.

    Raises:
        WorkflowError: When there is no chat model installed.
    """
    cands = selector.classify_all(ollama.list_models(root))
    best = selector.choose_best(cands, device.total_mib if device else 0, host.ram_total_mib)
    if best is None:
        raise WorkflowError(f"No chat model found in {root}. Pull one first, e.g. `ollama pull llama3.2`.")
    return best


def _hardware_line(host: HostInfo, device: Device | None) -> str:
    """Describe the hardware for the summary panel."""
    if device is None:
        return f"CPU only ({host.ram_total_mib} MiB RAM)"
    return f"{device.name} ({device.total_mib} MiB VRAM), {host.ram_total_mib} MiB RAM"


def _auto_select(cfg: Config, root: Path) -> tuple[OllamaModel, str]:
    """Choose a model automatically and say why.

    Args:
        cfg: The configuration (its llama-server and device are used to probe the hardware).
        root: The Ollama models directory.

    Returns:
        tuple[OllamaModel, str]: The model and the reason line.
    """
    host, device = _probe(_require_server(cfg), cfg.device)
    cand, fits = _pick_best(root, host, device)
    hw = _hardware_line(host, device)
    if fits:
        return cand.model, f"largest chat model that fits ({cand.need_mib} MiB needed) on {hw}"
    ui.show_notice("warn", f"No installed chat model fits {hw}; using the smallest one ({cand.model.name}).")
    return cand.model, f"smallest chat model (nothing fits: {cand.need_mib} MiB needed) on {hw}"


# --- configuration and first run -----------------------------------------------------------------------------------


def _ensure_binaries(cfg: Config, opts: ServeOptions) -> bool:
    """Make sure ``llama-server`` exists (installing it when needed) and record the binary paths.

    Args:
        cfg: The configuration, updated in place.
        opts: The serve options (``assume_yes`` is passed to the installer).

    Returns:
        bool: True when the configuration changed.
    """
    changed = False
    binary = llamacpp.find_binary("llama-server", cfg.llama_server)
    if binary is None:
        ui.show_notice("info", "llama-server was not found; installing llama.cpp.")
        binary = llamacpp.install(assume_yes=opts.assume_yes)
    if cfg.llama_server != str(binary):
        cfg.llama_server = str(binary)
        changed = True
    bench = llamacpp.find_binary("llama-bench", cfg.llama_bench)
    if bench is not None and cfg.llama_bench != str(bench):
        cfg.llama_bench = str(bench)
        changed = True
    return changed


def _ensure_root(cfg: Config, opts: ServeOptions) -> bool:
    """Make sure the configured Ollama models directory exists, discovering it when needed.

    Order: ``--ollama-path``, the configured path, ``$OLLAMA_MODELS`` and the common locations.

    Args:
        cfg: The configuration, updated in place.
        opts: The serve options (``ollama_path`` is the explicit candidate).

    Returns:
        bool: True when the configuration changed.

    Raises:
        WorkflowError: When no Ollama models directory can be found.
    """
    configured = Path(cfg.ollama_path) if cfg.ollama_path else None
    if opts.ollama_path is None and configured is not None and (configured / "manifests").is_dir():
        return False
    found = ollama.find_ollama_path(opts.ollama_path)
    if opts.ollama_path is not None and found != opts.ollama_path:
        ui.show_notice(
            "warn", f"--ollama-path {opts.ollama_path} is not an Ollama models directory; searching instead."
        )
    if found is None:
        raise WorkflowError(
            "Could not find the Ollama models directory. Set OLLAMA_MODELS or pass --ollama-path "
            "(it is the folder that contains `manifests` and `blobs`)."
        )
    if opts.ollama_path is None and configured is not None:
        ui.show_notice("warn", f"Ollama path {configured} no longer exists; using {found}.")
    changed = cfg.ollama_path != str(found)
    cfg.ollama_path = str(found)
    return changed


def first_run(opts: ServeOptions) -> Config:
    """Set ollama2cpp up from scratch and save the configuration.

    Installs llama.cpp when missing, finds the Ollama models directory, chooses the default model (``--model`` or
    the best one for the hardware), saves everything and prints a summary.

    Args:
        opts: The serve options.

    Returns:
        Config: The saved configuration.
    """
    ui.show_notice("info", "First run: setting up ollama2cpp.")
    cfg = Config()
    _ensure_binaries(cfg, opts)
    _ensure_root(cfg, opts)
    root = _require_root(cfg)
    if opts.model:
        model, reason = ollama.resolve_model(root, opts.model), "requested with --model"
        _require_blob(model)
    else:
        model, reason = _auto_select(cfg, root)
    cfg.default_model = model.name
    config.save(cfg)
    ui.panel(
        "ollama2cpp first run",
        "\n".join(
            [
                f"llama-server : {cfg.llama_server}",
                f"Ollama models: {cfg.ollama_path}",
                f"Default model: {model.name}",
                f"Why          : {reason}",
                f"Config file  : {config.config_file()}",
            ]
        ),
    )
    return cfg


def ensure_config(opts: ServeOptions) -> Config:
    """Load the configuration, running the first-run setup when there is none, and repair what went stale.

    A missing or corrupt config file means a first run. Otherwise a vanished ``llama-server`` is reinstalled and a
    vanished Ollama directory is rediscovered; changes are saved.

    Args:
        opts: The serve options.

    Returns:
        Config: A configuration with ``llama_server`` and ``ollama_path`` set.
    """
    cfg = config.load()
    if cfg is None:
        return first_run(opts)
    changed = _ensure_binaries(cfg, opts)
    changed = _ensure_root(cfg, opts) or changed
    if changed:
        config.save(cfg)
    return cfg


def select_model(cfg: Config, opts: ServeOptions) -> OllamaModel:
    """Resolve the model to serve: ``--model``, else the saved default, else a fresh automatic choice.

    A saved default that is no longer installed is replaced (with a notice) and the config is updated. A
    ``--model`` given on a regular run is not saved.

    Args:
        cfg: The configuration.
        opts: The serve options.

    Returns:
        OllamaModel: The model.
    """
    root = _require_root(cfg)
    if opts.model:
        return ollama.resolve_model(root, opts.model)
    if cfg.default_model:
        try:
            return ollama.resolve_model(root, cfg.default_model)
        except ollama.ModelNotFoundError:
            ui.show_notice("warn", f"Default model {cfg.default_model} is no longer installed; choosing another.")
    model, reason = _auto_select(cfg, root)
    ui.show_notice("info", f"Selected {model.name}: {reason}")
    cfg.default_model = model.name
    config.save(cfg)
    return model


# --- planning ------------------------------------------------------------------------------------------------------


def _projector(model: OllamaModel, blob: Path) -> Path | None:
    """Locate the model's multimodal projector blob.

    Args:
        model: The installed model.
        blob: The model blob (the projector lives in the same directory).

    Returns:
        Path | None: The projector blob, or None when the model has none or it is not on disk.
    """
    layer = next((x for x in model.layers if x.media_type == ollama.MEDIA_PROJECTOR), None)
    if layer is None:
        return None
    path = blob.parent / layer.digest.replace(":", "-")
    return path if path.is_file() else None


def _choose_profile(
    cfg: Config, model: OllamaModel, digest: str, blob: Path, requested: int, host: HostInfo, device: Device | None
) -> tuple[list[str], int, str, list[str]]:
    """Pick the saved profile for ``(model, ctx)`` or fall back to the heuristic seed.

    Args:
        cfg: The configuration (profiles and the ``llama-server`` path).
        model: The model being served.
        digest: The model blob digest.
        blob: The model blob.
        requested: The requested context size.
        host: Host snapshot.
        device: Chosen GPU, or None.

    Returns:
        tuple[list[str], int, str, list[str]]: Profile arguments, the context to use, a one-line description and
            warnings.

    Raises:
        WorkflowError: When the GGUF header cannot be read.
    """
    profile: Profile | None = cfg.profiles.get(config.profile_key(digest, requested))
    if profile is not None:
        warnings: list[str] = []
        current = hardware.fingerprint(host, device, llamacpp.version(_require_server(cfg)))
        if profile.fingerprint != current:
            warnings.append("The saved profile is stale (hardware or llama.cpp changed); re-run `ollama2cpp optimize`.")
        if profile.measured and profile.tg_tps is not None:
            note = f"Using optimized profile (tg {profile.tg_tps:.1f} t/s)"
        else:
            note = "Using unmeasured profile (heuristic settings; run `ollama2cpp optimize` to measure)"
        return list(profile.args), profile.ctx, note, warnings
    try:
        info = gguf.read_gguf(blob)
    except gguf.GgufError as exc:
        raise WorkflowError(f"Cannot read the GGUF header of {model.name}: {exc}") from exc
    seed = optimizer.heuristic_seed(info, host, device, requested)
    warnings = []
    if seed.ctx != requested:
        warnings.append(f"Context reduced from {requested} to {seed.ctx} tokens to fit the available memory.")
    note = "No optimized profile for this model/context; using a heuristic seed (run `ollama2cpp optimize` to tune it)"
    return optimizer.seed_to_args(seed), seed.ctx, note, warnings


def plan_serve(cfg: Config, opts: ServeOptions, model: OllamaModel) -> ServePlan:
    """Work out the exact ``llama-server`` command for ``model``.

    Args:
        cfg: The configuration.
        opts: The serve options.
        model: The model to serve.

    Returns:
        ServePlan: The command and the facts behind it.
    """
    blob, digest = _require_blob(model)
    binary = _require_server(cfg)
    host_addr, port = opts.host or cfg.host, opts.port or cfg.port
    host, device = _probe(binary, cfg.device)
    args, ctx, note, warnings = _choose_profile(
        cfg, model, digest, blob, opts.ctx or constants.DEFAULT_CTX, host, device
    )
    pinned = device.id if device is not None and (cfg.device or len(host.devices) > 1) else None
    if device is not None and 0 < device.free_mib < model.size // _MIB:
        warnings.append(
            f"Only {device.free_mib} MiB of VRAM is free but the model is {model.size // _MIB} MiB; "
            "another process (e.g. Ollama) may be holding it."
        )
    mmproj = None
    if opts.vision:
        mmproj = _projector(model, blob)
        if mmproj is None:
            warnings.append(f"{model.name} has no usable multimodal projector; continuing without it.")
    settings = ServeSettings(
        model_path=blob,
        alias=model.name,
        host=host_addr,
        port=port,
        ctx=ctx,
        device=pinned,
        mmproj=mmproj,
        profile_args=args,
        extra_args=list(opts.extra_args),
    )
    argv = tuple(llamacpp.build_serve_argv(binary, settings))
    return ServePlan(argv, settings, model, host_addr, port, note, tuple(warnings))


# --- running -------------------------------------------------------------------------------------------------------


def _wait_port_free(host: str, port: int) -> None:
    """Wait until nothing listens on ``host:port``.

    Args:
        host: Address.
        port: Port.

    Raises:
        WorkflowError: When the port is still in use after ``PORT_FREE_TIMEOUT`` seconds.
    """
    deadline = time.monotonic() + PORT_FREE_TIMEOUT
    while server.port_in_use(host, port):
        if time.monotonic() >= deadline:
            raise WorkflowError(f"Port {port} is still in use after stopping the server.")
        time.sleep(PORT_POLL_INTERVAL)


def _handle_running(model: OllamaModel, opts: ServeOptions, host: str, port: int) -> int | None:
    """Deal with a server that already listens on the target port.

    Args:
        model: The model about to be served.
        opts: The serve options.
        host: Target address.
        port: Target port.

    Returns:
        int | None: An exit code when the run must end here, None to continue and start the server.

    Raises:
        WorkflowError: When a foreign process holds the port.
    """
    state, alias = server.describe_running(host, port, constants.home())
    if state == "none":
        return None
    if state == "foreign":
        raise WorkflowError(
            f"Port {port} on {host} is held by another process (not started by ollama2cpp), "
            "which is never stopped. Pick another port with --port."
        )
    url = f"http://{host}:{port}"
    if alias is not None and ollama.normalize_name(alias) == model.name:
        ui.show_notice("ok", f"llama-server is already running {model.name} at {url}.")
        return 0
    if not opts.restart:
        ui.show_notice(
            "warn",
            f"ollama2cpp is already serving {alias or 'another model'} at {url}; "
            f"use --restart to replace it with {model.name}.",
        )
        return 1
    ui.show_notice("info", f"Stopping the running server ({alias}) to start {model.name}.")
    server.stop(constants.home())
    _wait_port_free(host, port)
    return None


def run_serve(opts: ServeOptions) -> int:
    """Run the whole ``serve`` flow: setup, already-running check, planning and starting.

    Args:
        opts: The serve options.

    Returns:
        int: The process exit code.
    """
    cfg = ensure_config(opts)
    model = select_model(cfg, opts)
    host, port = opts.host or cfg.host, opts.port or cfg.port
    if not opts.dry_run:
        early = _handle_running(model, opts, host, port)
        if early is not None:
            return early
    plan = plan_serve(cfg, opts, model)
    for warning in plan.warnings:
        ui.show_notice("warn", warning)
    ui.show_notice("info", plan.profile_note)
    command = proc.format_command(plan.argv)
    if opts.dry_run:
        ui.show_command(command)
        return 0
    url = f"http://{plan.host}:{plan.port}"
    ui.show_notice("info", f"Starting {model.name} at {url} (Ctrl-C to stop).")
    logger.info("Command: %s", command)
    code = server.start(
        plan.argv, base=constants.home(), host=plan.host, port=plan.port, model=model.name, detach=opts.detach
    )
    if opts.detach:
        ui.show_notice("ok", f"{model.name} is ready at {url}")
    return code
