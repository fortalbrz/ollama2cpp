"""The inspection and maintenance commands: list, info, status, stop, doctor, update and config.

No CLI parsing lives here (see ``__main__``): every ``run_*`` function prints through ``ui`` and returns the process
exit code; user-facing failures are :class:`~ollama2cpp.workflow.WorkflowError` (shown as one error line by the CLI).
The optimizer command lives in :mod:`ollama2cpp.optimize_cmd`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import typer

from ollama2cpp import config, constants, gguf, hardware, llamacpp, ollama, proc, selector, server, ui, workflow
from ollama2cpp.config import Config, Profile
from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.hardware import Device
from ollama2cpp.ollama import OllamaModel
from ollama2cpp.selector import Candidate
from ollama2cpp.workflow import WorkflowError

__all__ = [
    "SETTABLE_KEYS",
    "describe_profile",
    "find_server",
    "load_config",
    "models_root",
    "pick_device",
    "profiles_for",
    "run_config",
    "run_doctor",
    "run_info",
    "run_list",
    "run_status",
    "run_stop",
    "run_update",
]

logger = logging.getLogger(__name__)

_GIB = 1024**3
LOW_FREE_FRACTION = 0.5  # free VRAM below this share of the total means another process is holding it
UPDATE_TAIL_LINES = 12
SETTABLE_KEYS = ("host", "port", "device", "ollama_path", "default_model", "llama_server")
_OLLAMA_MEDIA = "application/vnd.ollama.image."
_UNSET_WORDS = frozenset({"", "none"})
_MAX_PORT = 65535
_UP_TO_DATE_MARKERS = ("no available upgrade", "no newer package")

ConfigState = Literal["ok", "missing", "corrupt"]


# --- shared helpers ------------------------------------------------------------------------------------------------


def load_config(*, assume_yes: bool = False) -> Config:
    """Load the saved configuration, running the first-run setup when there is none.

    Args:
        assume_yes: Answer yes to the first-run installation question.

    Returns:
        Config: The saved configuration (never an unsaved per-run copy).
    """
    cfg = config.load()
    return cfg if cfg is not None else workflow.ensure_config(workflow.ServeOptions(assume_yes=assume_yes))


def peek_config() -> tuple[Config | None, ConfigState]:
    """Read the saved configuration without side effects (no first run, no ``.bak`` rename of a corrupt file).

    Returns:
        tuple[Config | None, ConfigState]: The configuration and ``"ok"``, or ``None`` with ``"missing"`` (no file)
            or ``"corrupt"`` (unreadable or invalid).
    """
    path = config.config_file()
    if not path.is_file():
        return None, "missing"
    with tempfile.TemporaryDirectory(prefix="o2c-peek-") as tmp:
        try:
            shutil.copyfile(path, Path(tmp) / constants.CONFIG_FILENAME)
        except OSError:
            return None, "corrupt"
        cfg = config.load(Path(tmp))
    return (cfg, "ok") if cfg is not None else (None, "corrupt")


def models_root(cfg: Config | None) -> Path:
    """Return the Ollama models directory: the configured one while it holds ``manifests/``, else discovery.

    Args:
        cfg: The configuration, or None.

    Returns:
        Path: The models directory.

    Raises:
        WorkflowError: When no Ollama models directory can be found.
    """
    if cfg is not None and cfg.ollama_path and (Path(cfg.ollama_path) / "manifests").is_dir():
        return Path(cfg.ollama_path)
    found = ollama.find_ollama_path()
    if found is None:
        raise WorkflowError(
            "Could not find the Ollama models directory. Set OLLAMA_MODELS or run "
            "`ollama2cpp config --set ollama_path=<folder with manifests and blobs>`."
        )
    return found


def find_server(cfg: Config | None) -> Path | None:
    """Locate ``llama-server`` (configured path first)."""
    return llamacpp.find_binary("llama-server", cfg.llama_server if cfg else None)


def pick_device(cfg: Config | None, binary: Path | None) -> Device | None:
    """Detect the GPUs and choose the one to run on (None for CPU-only)."""
    return hardware.best_device(hardware.detect_devices(binary), cfg.device if cfg else None)


def profiles_for(cfg: Config, digest: str | None) -> list[tuple[str, Profile]]:
    """Return every saved profile of the model with blob ``digest`` as ``(requested ctx, profile)``, by ctx."""
    if digest is None:
        return []
    prefix = f"{digest}:"
    found = [(key.removeprefix(prefix), p) for key, p in cfg.profiles.items() if key.startswith(prefix)]
    return sorted(found, key=lambda item: (len(item[0]), item[0]))


def describe_profile(profile: Profile) -> str:
    """Summarise a profile in one line."""
    if profile.measured and profile.tg_tps is not None:
        speed = f"tg {profile.tg_tps:.1f} t/s" + (f", pp {profile.pp_tps:.1f} t/s" if profile.pp_tps else "")
        return f"ctx {profile.ctx}, {speed} (measured)"
    return f"ctx {profile.ctx}, heuristic settings (unmeasured)"


def _layer_size(size: int) -> str:
    for unit, label in ((_GIB, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if size >= unit:
            return f"{size / unit:.1f} {label}"
    return f"{size} B"


def _gib(size: int) -> str:
    return f"{size / _GIB:.2f} GiB" if size else "-"


def _free_text(device: Device) -> str:
    free = f"{device.free_mib}" if device.free_mib else "?"
    return f"{device.name}: {free} MiB free of {device.total_mib} MiB"


def _is_low(device: Device) -> bool:
    return 0 < device.free_mib < device.total_mib * LOW_FREE_FRACTION


def _low_vram_warning() -> str:
    return "something else is holding VRAM (e.g. Ollama) - `optimize` may refuse; see --unload-ollama"


def _interactive() -> bool:
    return sys.stdin.isatty()


# --- list ----------------------------------------------------------------------------------------------------------


def _optimized(cfg: Config, model: OllamaModel) -> str:
    found = [p for _, p in profiles_for(cfg, model.digest)]
    if not found:
        return "no"
    return "yes" if any(p.measured for p in found) else "seed"


def _list_row(cand: Candidate, cfg: Config, device: Device | None) -> dict[str, Any]:
    """Build the JSON-friendly record of one model (``None`` marks a field that does not apply)."""
    info, chat = cand.gguf, cand.kind == "chat"
    return {
        "name": cand.model.name,
        "size_gib": round(cand.model.size / _GIB, 2),
        "quant": gguf.file_type_name(info.file_type) if info else None,
        "arch": info.architecture if info else None,
        "ctx": info.context_length if info else None,
        "kind": cand.kind,
        "need_mib": cand.need_mib if info else None,
        "fits_vram": cand.need_mib <= device.total_mib if chat and device else None,
        "optimized": _optimized(cfg, cand.model) if chat else None,
        "reason": cand.reason,
    }


def _table_cells(row: dict[str, Any]) -> list[str]:
    chat = row["kind"] == "chat"
    fits = "?" if chat and row["fits_vram"] is None else {True: "yes", False: "no"}.get(row["fits_vram"], "-")
    shown = {k: "-" if row[k] is None else str(row[k]) for k in ("quant", "arch", "ctx")}
    return [
        row["name"],
        f"{row['size_gib']:.1f}" if row["size_gib"] else "-",
        shown["quant"],
        shown["arch"],
        shown["ctx"],
        row["kind"],
        fits,
        row["optimized"] or "-",
        row["reason"],
    ]


def run_list(*, as_json: bool = False) -> int:
    """List the installed Ollama models with their classification (never aborts on one bad model).

    Args:
        as_json: Print a JSON array instead of a table.

    Returns:
        int: The exit code.
    """
    with ui.use_stderr() if as_json else contextlib.nullcontext():  # stdout carries only the JSON
        cfg = load_config()
        root = models_root(cfg)
        device = pick_device(cfg, find_server(cfg))
        rows = [_list_row(c, cfg, device) for c in selector.classify_all(ollama.list_models(root), DEFAULT_CTX)]
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return 0
    if not rows:
        ui.show_notice("warn", f"No Ollama models found in {root}.")
        return 0
    columns = ["Name", "Size GiB", "Quant", "Arch", "Ctx", "Kind", "Fits VRAM", "Optimized", "Note"]
    ui.show_table(f"Ollama models in {root}", columns, [_table_cells(r) for r in rows])
    return 0


# --- info ----------------------------------------------------------------------------------------------------------


def _vram_lines(cand: Candidate, profiles: Sequence[tuple[str, Profile]], device: Device | None) -> list[str]:
    """Describe the memory need at the default context and at each saved profile context."""
    if cand.gguf is None:
        return []
    ctxs = [DEFAULT_CTX, *sorted({p.ctx for _, p in profiles} - {DEFAULT_CTX})]
    lines = []
    for ctx in ctxs:
        need = selector.estimate_need_mib(cand.model, cand.gguf, ctx)
        verdict = ""
        if device is not None:
            verdict = f" ({'fits' if need <= device.total_mib else 'does not fit'} {device.total_mib} MiB of VRAM)"
        lines.append(f"VRAM at ctx {ctx}: about {need} MiB{verdict}")
    return lines


def _info_lines(cand: Candidate, device: Device | None, profiles: Sequence[tuple[str, Profile]]) -> list[str]:
    model, info = cand.model, cand.gguf
    lines = [
        f"Model        : {model.name}",
        f"Manifest     : {model.manifest_path}",
        f"Size         : {_gib(model.size)}",
    ]
    if info is None:
        lines.append(f"GGUF         : unreadable ({cand.reason})")
    else:
        lines += [
            f"Kind         : {cand.kind}" + (f" ({cand.reason})" if cand.reason else ""),
            f"Architecture : {info.architecture}",
            f"Layers       : {info.block_count}",
            f"Heads        : {info.head_count} (KV heads {info.head_count_kv})",
            f"Trained ctx  : {info.context_length}",
            f"File type    : {gguf.file_type_name(info.file_type)}",
        ]
    lines += _vram_lines(cand, profiles, device)
    if not profiles:
        lines.append("Profiles     : none saved (run `ollama2cpp optimize`)")
    for requested, profile in profiles:
        lines.append(f"Profile @ {requested}: {describe_profile(profile)}")
        lines.append(f"    args: {' '.join(profile.args)}")
    return lines


def run_info(name: str) -> int:
    """Show the GGUF details, layers, memory need and saved profiles of one model.

    Args:
        name: The model name as typed.

    Returns:
        int: The exit code.
    """
    cfg = load_config()
    root = models_root(cfg)
    model = ollama.resolve_model(root, name)
    cand = selector.classify(model, DEFAULT_CTX)
    device = pick_device(cfg, find_server(cfg))
    ui.panel(model.name, "\n".join(_info_lines(cand, device, profiles_for(cfg, model.digest))))
    blobs = root / "blobs"
    layers = [
        [
            layer.media_type.removeprefix(_OLLAMA_MEDIA),
            _layer_size(layer.size),
            layer.digest[:19],
            "yes" if (blobs / layer.digest.replace(":", "-")).is_file() else "MISSING",
        ]
        for layer in model.layers
    ]
    ui.show_table("Layers", ["Type", "Size", "Digest", "On disk"], layers)
    return 0


# --- status --------------------------------------------------------------------------------------------------------


def _server_line(host: str, port: int) -> str:
    state, alias = server.describe_running(host, port, constants.home())
    if state == "none":
        return f"not running ({host}:{port})"
    if state == "foreign":
        return f"port {port} on {host} is held by another process (not ollama2cpp){f': {alias}' if alias else ''}"
    ready = "ready" if server.is_ready(host, port) else "still loading"
    return f"running (ours): {alias or 'unknown model'} at http://{host}:{port} ({ready})"


def _config_line(state: ConfigState) -> str:
    detail = {"ok": "ok", "missing": "not found - run `ollama2cpp serve` to set up", "corrupt": "unreadable or invalid"}
    return f"{config.config_file()} ({detail[state]})"


def _default_profile_line(cfg: Config) -> str:
    if not cfg.default_model:
        return "no default model saved"
    try:
        model = ollama.resolve_model(models_root(cfg), cfg.default_model)
    except (WorkflowError, ollama.ModelNotFoundError):
        return f"{cfg.default_model} (not installed)"
    profiles = profiles_for(cfg, model.digest)
    summary = "; ".join(f"@{req}: {describe_profile(p)}" for req, p in profiles) or "no profile saved"
    return f"{model.name} - {summary}"


def run_status() -> int:
    """Show the server state, configuration, GPU memory and the default model's saved profile (needs no config).

    Returns:
        int: The exit code.
    """
    cfg, state = peek_config()
    host = cfg.host if cfg else constants.DEFAULT_HOST
    port = cfg.port if cfg else constants.DEFAULT_PORT
    devices = hardware.detect_devices(find_server(cfg))
    lines = [f"Server       : {_server_line(host, port)}", f"Config       : {_config_line(state)}"]
    lines += [f"GPU          : {_free_text(d)}" for d in devices] or ["GPU          : none detected (CPU only)"]
    if cfg is not None:
        lines.append(f"Default model: {_default_profile_line(cfg)}")
    ui.panel("ollama2cpp status", "\n".join(lines))
    if any(_is_low(d) for d in devices):
        ui.show_notice("warn", f"Less than half of the VRAM is free: {_low_vram_warning()}.")
    return 0


# --- stop ----------------------------------------------------------------------------------------------------------


def run_stop() -> int:
    """Stop the server started by ollama2cpp; a foreign process on the port is never touched.

    Returns:
        int: The exit code (0 also when there was nothing to stop).
    """
    if server.stop(constants.home()):
        ui.show_notice("ok", "Stopped the ollama2cpp server.")
        return 0
    cfg, _ = peek_config()
    host = cfg.host if cfg else constants.DEFAULT_HOST
    port = cfg.port if cfg else constants.DEFAULT_PORT
    if server.describe_running(host, port, constants.home())[0] == "foreign":
        ui.show_notice(
            "warn", f"Port {port} on {host} is held by a process ollama2cpp did not start; it was left alone."
        )
    else:
        ui.show_notice("info", "No ollama2cpp server is running.")
    return 0


# --- doctor --------------------------------------------------------------------------------------------------------

Check = tuple[Literal["ok", "warn", "fail"], str, str]  # level, detail, fix hint


def _check_server(cfg: Config | None) -> Check:
    binary = find_server(cfg)
    if binary is None:
        hint = "run `ollama2cpp serve` (installs llama.cpp) or `winget install --id ggml.llamacpp -e`"
        return "fail", "llama-server was not found", hint
    backends = sorted({d.id.rstrip("0123456789") for d in hardware.detect_devices(binary)})
    return "ok", f"{binary} ({llamacpp.version(binary)}, backend: {', '.join(backends) or 'CPU'})", ""


def _check_bench(cfg: Config | None) -> Check:
    binary = llamacpp.find_binary("llama-bench", cfg.llama_bench if cfg else None)
    if binary is None:
        return "warn", "llama-bench was not found (only `optimize --quick` works)", "run `ollama2cpp update`"
    return "ok", str(binary), ""


def _check_ollama(cfg: Config | None) -> Check:
    try:
        root = models_root(cfg)
    except WorkflowError:
        hint = "set OLLAMA_MODELS or `ollama2cpp config --set ollama_path=...`"
        return "fail", "Ollama models directory not found", hint
    count = len(ollama.list_models(root))
    if count == 0:
        return "warn", f"{root} (no models)", "pull one, e.g. `ollama pull llama3.2`"
    return "ok", f"{root} ({count} models)", ""


def _check_gpu(cfg: Config | None) -> Check:
    devices = hardware.detect_devices(find_server(cfg))
    if not devices:
        hint = "install GPU drivers or a CUDA/Vulkan llama.cpp build"
        return "warn", "no GPU detected - inference runs on the CPU", hint
    return "ok", "; ".join(f"{d.name} ({d.total_mib} MiB)" for d in devices), ""


def _check_vram(cfg: Config | None) -> Check:
    devices = hardware.detect_devices(find_server(cfg))
    if not devices:
        return "ok", "no GPU, nothing to check", ""
    low = [d for d in devices if _is_low(d)]
    if low:
        return "warn", "; ".join(_free_text(d) for d in low), "close other GPU users or use `optimize --unload-ollama`"
    return "ok", "; ".join(_free_text(d) for d in devices), ""


def _check_config(state: ConfigState) -> Check:
    if state == "corrupt":
        hint = "`ollama2cpp config --reset` or re-run `serve`"
        return "warn", f"{config.config_file()} is unreadable or invalid", hint
    if state == "missing":
        return "warn", "not configured yet", "run `ollama2cpp serve` to set up"
    return "ok", str(config.config_file()), ""


def _safely(name: str, check: Callable[[], Check]) -> tuple[str, Check]:
    """Run one check; a crash becomes a ``fail`` row instead of aborting the doctor."""
    try:
        return name, check()
    except Exception as exc:
        logger.debug("Doctor check %s crashed", name, exc_info=True)
        return name, ("fail", f"the check crashed: {exc}", "re-run with --verbose and report it")


def run_doctor() -> int:
    """Run the health checks (needs no config, never raises) and print them with fix hints.

    Returns:
        int: 1 when any check failed, else 0.
    """
    cfg, state = peek_config()
    checks: list[tuple[str, Callable[[], Check]]] = [
        ("llama-server", lambda: _check_server(cfg)),
        ("llama-bench", lambda: _check_bench(cfg)),
        ("Ollama models", lambda: _check_ollama(cfg)),
        ("GPU", lambda: _check_gpu(cfg)),
        ("Free VRAM", lambda: _check_vram(cfg)),
        ("Config", lambda: _check_config(state)),
    ]
    results = [_safely(name, check) for name, check in checks]
    ui.show_table(
        "ollama2cpp doctor",
        ["Check", "Status", "Detail", "Fix"],
        [[name, ui.status_text(level), detail, fix] for name, (level, detail, fix) in results],
    )
    return 1 if any(level == "fail" for _, (level, _, _) in results) else 0


# --- update --------------------------------------------------------------------------------------------------------


def _tail(text: str, count: int = UPDATE_TAIL_LINES) -> str:
    return "\n".join([line for line in text.splitlines() if line.strip()][-count:])


def _refresh_paths() -> str:
    """Record the (possibly relocated) binaries in the saved configuration; return the new ``llama-server`` version."""
    saved = config.load()
    binary = llamacpp.find_binary("llama-server", saved.llama_server if saved else None)
    if binary is None:
        return "unknown"
    if saved is not None:
        bench = llamacpp.find_binary("llama-bench", saved.llama_bench)
        found = (str(binary), str(bench) if bench else saved.llama_bench)
        if found != (saved.llama_server, saved.llama_bench):
            saved.llama_server, saved.llama_bench = found
            config.save(saved)
    return llamacpp.version(binary)


def run_update() -> int:
    """Upgrade llama.cpp with winget and show the new version.

    Returns:
        int: The exit code.

    Raises:
        WorkflowError: When winget is missing or the upgrade failed.
    """
    ui.show_notice("info", "Upgrading llama.cpp with winget.")
    result = llamacpp.update()
    output = f"{result.stdout}\n{result.stderr}"
    tail = _tail(output)
    if tail:
        ui.console.print(tail, markup=False, highlight=False)
    if result.returncode == proc.RC_NOT_FOUND:
        raise WorkflowError(f"winget is not available. {llamacpp.manual_instructions()}")
    if result.returncode != 0 and not any(m in output.lower() for m in _UP_TO_DATE_MARKERS):
        raise WorkflowError(f"winget upgrade failed (exit code {result.returncode}). {llamacpp.manual_instructions()}")
    ui.show_notice("ok", f"llama.cpp is up to date (llama-server {_refresh_paths()}).")
    return 0


# --- config --------------------------------------------------------------------------------------------------------


def _existing(raw: str, *, directory: bool, key: str) -> str:
    path = Path(raw).expanduser()
    if not (path.is_dir() if directory else path.is_file()):
        raise WorkflowError(f"{key}: {raw!r} is not an existing {'directory' if directory else 'file'}")
    return str(path.resolve())


def _coerce(key: str, raw: str) -> str | int | None:
    """Validate ``raw`` for ``key`` and return the value to store."""
    if key == "port":
        if not raw.isdigit() or not 1 <= int(raw) <= _MAX_PORT:
            raise WorkflowError(f"port: {raw!r} is not a port number (1-{_MAX_PORT})")
        return int(raw)
    if key in {"device", "default_model"} and raw.strip().lower() in _UNSET_WORDS:
        return None
    if key == "host":
        if not raw.strip():
            raise WorkflowError("host: the address cannot be empty")
        return raw.strip()
    if key == "default_model":
        return ollama.normalize_name(raw)
    if key == "device":
        return raw.strip()
    return _existing(raw, directory=key == "ollama_path", key=key)


def _parse_assignments(pairs: Sequence[str]) -> dict[str, str | int | None]:
    values: dict[str, str | int | None] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep:
            raise WorkflowError(f"Expected key=value, got {pair!r}")
        if key not in SETTABLE_KEYS:
            raise WorkflowError(f"Unknown setting {key!r}. Settable keys: {', '.join(SETTABLE_KEYS)}")
        values[key] = _coerce(key, raw)
    return values


def _reset(*, assume_yes: bool) -> int:
    path = config.config_file()
    if not path.is_file():
        ui.show_notice("info", f"There is no configuration to reset ({path}).")
        return 0
    if not assume_yes:
        if not _interactive():
            raise WorkflowError("Refusing to delete the configuration without --yes (stdin is not a terminal).")
        if not ui.confirm(f"Delete {path}?", default=False):
            ui.show_notice("info", "Configuration kept.")
            return 0
    path.unlink()
    ui.show_notice("ok", f"Deleted {path}; the next `ollama2cpp serve` runs the first-run setup again.")
    return 0


def run_config(*, set_values: Sequence[str] = (), reset: bool = False, assume_yes: bool = False) -> int:
    """Show the configuration, change settable keys, or delete it.

    Args:
        set_values: ``key=value`` assignments (validated all together; nothing is written when one is invalid).
        reset: Delete the configuration file.
        assume_yes: Skip the confirmation of ``reset``.

    Returns:
        int: The exit code.

    Raises:
        WorkflowError: For an invalid assignment or a refused reset.
    """
    if reset and set_values:
        raise WorkflowError("Use either --set or --reset, not both.")
    if reset:
        return _reset(assume_yes=assume_yes)
    values = _parse_assignments(set_values)  # validated before first-run or any write
    cfg = load_config()
    if values:
        for key, value in values.items():
            setattr(cfg, key, value)
        config.save(cfg)
        for key, value in values.items():
            ui.show_notice("ok", f"{key} = {value}")
    ui.show_notice("info", f"Config file: {config.config_file()}")
    typer.echo(config.config_file().read_text(encoding="utf-8").rstrip())
    return 0
