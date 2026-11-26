"""Persistent configuration store (JSON, atomic writes, never crashes on bad files)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ollama2cpp.constants import CONFIG_FILENAME, DEFAULT_HOST, DEFAULT_PORT, home

log = logging.getLogger(__name__)

BACKUP_SUFFIX = ".bak"


class _InvalidConfigError(ValueError):
    """Raised internally when the parsed JSON does not match the config schema."""


@dataclass
class Profile:
    """Saved llama-server settings for one model at one context size."""

    args: list[str]
    ctx: int
    tg_tps: float | None = None
    pp_tps: float | None = None
    measured: bool = False
    fingerprint: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    bench: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Config:
    """The whole persisted configuration.

    ``extra`` holds unknown top-level keys so they survive a load/save round trip.
    """

    ollama_path: str | None = None
    default_model: str | None = None
    llama_server: str | None = None
    llama_bench: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str | None = None
    profiles: dict[str, Profile] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


_KNOWN_KEYS = frozenset(Config.__dataclass_fields__) - {"extra"}


def config_file(base: Path | None = None) -> Path:
    """Return the config file path.

    Args:
        base: Storage directory; ``None`` resolves ``constants.home()`` at call time.

    Returns:
        Path: ``<base>/config.json``.
    """
    return (home() if base is None else base) / CONFIG_FILENAME


def profile_key(digest: str, ctx: int) -> str:
    """Build the key under which a model profile is stored.

    Args:
        digest: Model blob digest.
        ctx: Context size.

    Returns:
        str: ``"<digest>:<ctx>"``.
    """
    return f"{digest}:{ctx}"


def _str_or_none(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise _InvalidConfigError(key)
    return value


def _number_or_none(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _InvalidConfigError(key)
    return float(value)


def _int(data: dict[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidConfigError(key)
    return value


def _profile_from_dict(data: object) -> Profile:
    if not isinstance(data, dict):
        raise _InvalidConfigError("profile")
    args = data.get("args")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise _InvalidConfigError("args")
    fingerprint = data.get("fingerprint", {})
    if not isinstance(fingerprint, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in fingerprint.items()
    ):
        raise _InvalidConfigError("fingerprint")
    bench = data.get("bench", [])
    if not isinstance(bench, list) or not all(isinstance(b, dict) for b in bench):
        raise _InvalidConfigError("bench")
    measured = data.get("measured", False)
    if not isinstance(measured, bool):
        raise _InvalidConfigError("measured")
    timestamp = data.get("timestamp", "")
    if not isinstance(timestamp, str):
        raise _InvalidConfigError("timestamp")
    return Profile(
        args=list(args),
        ctx=_int(data, "ctx"),
        tg_tps=_number_or_none(data, "tg_tps"),
        pp_tps=_number_or_none(data, "pp_tps"),
        measured=measured,
        fingerprint=dict(fingerprint),
        timestamp=timestamp,
        bench=list(bench),
    )


def _config_from_dict(data: object) -> Config:
    if not isinstance(data, dict):
        raise _InvalidConfigError("top level")
    host = data.get("host", DEFAULT_HOST)
    if not isinstance(host, str):
        raise _InvalidConfigError("host")
    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise _InvalidConfigError("profiles")
    return Config(
        ollama_path=_str_or_none(data, "ollama_path"),
        default_model=_str_or_none(data, "default_model"),
        llama_server=_str_or_none(data, "llama_server"),
        llama_bench=_str_or_none(data, "llama_bench"),
        host=host,
        port=_int(data, "port", DEFAULT_PORT),
        device=_str_or_none(data, "device"),
        profiles={str(k): _profile_from_dict(v) for k, v in raw_profiles.items()},
        extra={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
    )


def _to_dict(cfg: Config) -> dict[str, Any]:
    data: dict[str, Any] = dict(cfg.extra)
    for name in _KNOWN_KEYS:
        value = getattr(cfg, name)
        if name == "profiles":
            value = {k: vars(p).copy() for k, p in value.items()}
        data[name] = value
    return data


def _back_up(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    try:
        Path(path).replace(backup)
    except OSError:
        log.warning("Could not back up corrupt config %s", path, exc_info=True)
    else:
        log.warning("Corrupt config moved to %s; starting fresh", backup)


def load(base: Path | None = None) -> Config | None:
    """Load the config.

    A corrupt or schema-invalid file is moved to ``config.json.bak`` and treated as absent.

    Args:
        base: Storage directory; ``None`` resolves ``constants.home()`` at call time.

    Returns:
        Config | None: The config, or ``None`` on first run (or after a corrupt-file backup).
    """
    path = config_file(base)
    if not path.exists():
        return None
    try:
        cfg = _config_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):  # ValueError covers JSON, unicode and schema errors
        log.warning("Config %s is unreadable or invalid", path, exc_info=True)
        if path.is_file():
            _back_up(path)
        return None
    return cfg


def save(cfg: Config, base: Path | None = None) -> None:
    """Atomically write the config (temp file in the same directory, then ``os.replace``).

    Args:
        cfg: The configuration to persist.
        base: Storage directory; ``None`` resolves ``constants.home()`` at call time.
    """
    path = config_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_to_dict(cfg), fh, indent=2)
            fh.write("\n")
        Path(tmp_name).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_name).unlink()
        raise
