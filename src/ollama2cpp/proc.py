"""Subprocess seam: the only module that launches external processes."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Handle", "ProcResult", "run", "spawn"]

Handle = subprocess.Popen[str]  # running-process handle, re-exported so other modules need not import subprocess

RC_NOT_FOUND = 127
RC_TIMEOUT = 124


@dataclass(frozen=True)
class ProcResult:
    """Outcome of a finished process.

    Attributes:
        returncode: Exit code (127 = executable missing, 124 = timed out).
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    returncode: int
    stdout: str
    stderr: str


def _text(value: str | bytes | None) -> str:
    """Normalise captured output (``TimeoutExpired`` may carry bytes or None).

    Args:
        value: Raw captured output.

    Returns:
        str: Decoded text, empty when nothing was captured.
    """
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def run(argv: Sequence[str], *, timeout: float | None = None) -> ProcResult:
    """Run a command to completion without ever raising.

    Args:
        argv: Command and arguments (never passed through a shell).
        timeout: Seconds before the process is killed.

    Returns:
        ProcResult: Captured output; ``returncode`` is 127 if the executable is missing, 124 on timeout.
    """
    try:
        # argv list, shell=False, caller-controlled executable
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return ProcResult(RC_NOT_FOUND, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return ProcResult(RC_TIMEOUT, _text(exc.stdout), _text(exc.stderr))
    except OSError as exc:
        return ProcResult(RC_NOT_FOUND, "", str(exc))
    return ProcResult(done.returncode, done.stdout, done.stderr)


def spawn(argv: Sequence[str], *, log_path: Path | None = None, detach: bool = False) -> subprocess.Popen[str]:
    """Start a long-running process without waiting for it.

    Args:
        argv: Command and arguments (never passed through a shell).
        log_path: When given, stdout and stderr are appended to this file (parent dirs are created).
        detach: Start in a new session / process group so it outlives the parent.

    Returns:
        subprocess.Popen[str]: The running process handle.
    """
    stdout = stderr = subprocess.DEVNULL
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        stdout, stderr = log_file, subprocess.STDOUT
    flags = 0
    new_session = False
    if detach:
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            new_session = True
    try:
        # argv list, shell=False, caller-controlled executable
        return subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            creationflags=flags,
            start_new_session=new_session,
        )
    finally:
        if log_file is not None:
            log_file.close()
