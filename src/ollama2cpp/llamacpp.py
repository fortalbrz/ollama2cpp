"""llama.cpp binary management and ``llama-server`` argv construction."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ollama2cpp import proc, ui
from ollama2cpp.constants import WINGET_ID

__all__ = [
    "LlamaCppError",
    "ServeSettings",
    "build_serve_argv",
    "find_binary",
    "install",
    "update",
    "version",
]

logger = logging.getLogger(__name__)

RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"
VERSION_TIMEOUT = 15.0
INSTALL_TIMEOUT = 1800.0
_BUILD_RE = re.compile(r"\bversion:.*?\(build\s+(\d+)")


class LlamaCppError(RuntimeError):
    """Raised when llama.cpp cannot be installed or located."""


@dataclass
class ServeSettings:
    """Everything needed to build a ``llama-server`` command line.

    Attributes:
        model_path: GGUF file to serve.
        alias: Model name exposed by the API.
        host: Address to listen on.
        port: Port to listen on.
        ctx: Context size in tokens.
        device: Device to offload to (``--device``), or None for llama.cpp's default.
        mmproj: Multimodal projector file, or None.
        profile_args: Extra arguments from the hardware profile.
        extra_args: User-supplied arguments; passed last so they override the profile.
    """

    model_path: Path
    alias: str
    host: str
    port: int
    ctx: int
    device: str | None = None
    mmproj: Path | None = None
    profile_args: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)


def _is_windows() -> bool:
    """Report whether the host is Windows (a seam for tests).

    Returns:
        bool: True on Windows.
    """
    return sys.platform == "win32"


def _which(name: str) -> Path | None:
    """Look ``name`` up on PATH, also trying the ``.exe`` suffix on Windows.

    Args:
        name: Executable name without suffix.

    Returns:
        Path | None: The resolved executable, or None.
    """
    candidates = [name, f"{name}.exe"] if _is_windows() else [name]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def _winget_glob(name: str) -> Path | None:
    """Search the WinGet package directory for ``name``.

    Args:
        name: Executable name without suffix.

    Returns:
        Path | None: The first match (sorted), or None.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
    for candidate in (f"{name}.exe", name):
        matches = sorted(packages.glob(f"{WINGET_ID}_*/{candidate}"))
        if matches:
            return matches[0]
    return None


def find_binary(name: str, configured: str | None = None) -> Path | None:
    """Locate a llama.cpp executable.

    Order: the configured path (if it exists), then PATH, then the WinGet package directory.

    Args:
        name: Executable name without suffix, e.g. ``llama-server``.
        configured: Explicit path from the user's configuration.

    Returns:
        Path | None: The executable, or None when it cannot be found.
    """
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
        logger.warning("Configured binary %s does not exist; searching elsewhere", configured)
    return _which(name) or _winget_glob(name)


def _manual_instructions() -> str:
    """Build the manual installation hint for this platform.

    Returns:
        str: Human-readable instructions.
    """
    if _is_windows():
        first = f"winget install --id {WINGET_ID} -e"
    else:
        first = "brew install llama.cpp"
    return f"Install llama.cpp manually with `{first}` or download a build from {RELEASES_URL}"


def install(*, assume_yes: bool = False) -> Path:
    """Install llama.cpp with winget and return the ``llama-server`` path.

    Args:
        assume_yes: Skip the confirmation question.

    Returns:
        Path: The installed ``llama-server`` executable.

    Raises:
        LlamaCppError: Not on Windows, the user declined, winget is missing or failed,
            or the binary still cannot be found afterwards.
    """
    if not _is_windows():
        msg = f"Automatic installation is only supported on Windows. {_manual_instructions()}"
        raise LlamaCppError(msg)
    if not ui.confirm(f"Install llama.cpp with winget ({WINGET_ID})?", assume_yes=assume_yes):
        msg = f"Installation declined. {_manual_instructions()}"
        raise LlamaCppError(msg)
    result = proc.run(
        [
            "winget",
            "install",
            "--id",
            WINGET_ID,
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
        timeout=INSTALL_TIMEOUT,
    )
    if result.returncode != 0:
        reason = "winget is not available" if result.returncode == proc.RC_NOT_FOUND else "winget failed"
        msg = f"{reason} (exit code {result.returncode}). {_manual_instructions()}"
        raise LlamaCppError(msg)
    found = find_binary("llama-server")
    if found is None:
        msg = f"winget finished but llama-server was not found. {_manual_instructions()}"
        raise LlamaCppError(msg)
    return found


def update() -> proc.ProcResult:
    """Upgrade llama.cpp with winget.

    Returns:
        proc.ProcResult: The winget outcome (never raises).
    """
    return proc.run(["winget", "upgrade", "--id", WINGET_ID, "-e"], timeout=INSTALL_TIMEOUT)


def version(binary: Path) -> str:
    """Return the compact build string of a llama.cpp binary.

    ``llama-server --version`` prints ``version: 1.0.0-dev (build 12721, commit 79c69f4b0)`` on stderr
    (after any backend-loading lines); the result is ``b12721``.

    Args:
        binary: Path to a llama.cpp executable.

    Returns:
        str: ``b<build>``, or ``"unknown"`` when it cannot be determined.
    """
    result = proc.run([str(binary), "--version"], timeout=VERSION_TIMEOUT)
    if result.returncode != 0:
        return "unknown"
    match = _BUILD_RE.search(f"{result.stdout}\n{result.stderr}")
    return f"b{match.group(1)}" if match else "unknown"


def build_serve_argv(binary: Path, s: ServeSettings) -> list[str]:
    """Build the ``llama-server`` command line.

    Later arguments override earlier ones in llama-server, so ``extra_args`` come last. Paths are
    single argv elements (never split or quoted).

    Args:
        binary: The ``llama-server`` executable.
        s: Serve settings.

    Returns:
        list[str]: The argv list.
    """
    argv = [
        str(binary),
        "-m",
        str(s.model_path),
        "--alias",
        s.alias,
        "--host",
        s.host,
        "--port",
        str(s.port),
        "-c",
        str(s.ctx),
        "--jinja",
    ]
    if s.device:
        argv += ["--device", s.device]
    if s.mmproj:
        argv += ["--mmproj", str(s.mmproj)]
    argv += s.profile_args
    argv += s.extra_args
    return argv
