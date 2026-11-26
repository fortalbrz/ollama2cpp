"""``llama-server`` runtime: start, stop, health checks and the pid record.

Safety rule: only a process that this module started and recorded (pid **and** creation time) is ever
terminated. A bare pid is not proof, because operating systems reuse pids.
"""

from __future__ import annotations

import codecs
import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import psutil

from ollama2cpp import constants, proc, ui
from ollama2cpp.llamacpp import LlamaCppError

__all__ = [
    "LOG_FILENAME",
    "ServerRecord",
    "clear_record",
    "describe_running",
    "health",
    "is_ready",
    "port_in_use",
    "props",
    "read_record",
    "start",
    "stop",
    "write_record",
]

logger = logging.getLogger(__name__)

LOG_FILENAME = "server.log"
LOG_TAIL_LINES = 20
CREATE_TIME_TOLERANCE = 1.0
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 0.3  # loopback answers instantly; Windows needs ~1 s to report a refused connection
KILL_GRACE = 3.0
HTTP_OK = 200
MAX_PORT = 65536
_ANY_V4 = {"", "0.0.0.0"}
_ANY_V6 = {"::", "[::]"}


@dataclass
class ServerRecord:
    """The server this tool started, persisted in the pid file.

    Attributes:
        pid: Process id of ``llama-server``.
        host: Address it listens on.
        port: Port it listens on.
        model: Model alias being served.
        started: ISO-8601 start timestamp.
        create_time: ``psutil`` creation time of the process; ``0.0`` asks ``write_record`` to capture it.
    """

    pid: int
    host: str
    port: int
    model: str
    started: str
    create_time: float = 0.0


# --- HTTP / socket probes ------------------------------------------------------------------------------------------


def _connect_host(host: str) -> str:
    """Map a bind-all address to one that can be connected to.

    Args:
        host: Configured listen address.

    Returns:
        str: Loopback for wildcard addresses and ``localhost``, otherwise ``host`` lower-cased without brackets.
    """
    host = host.strip().lower()
    if host in _ANY_V4 or host == "localhost":
        return "127.0.0.1"
    if host in _ANY_V6:
        return "::1"
    return host.strip("[]")


def _get(host: str, port: int, path: str, timeout: float) -> tuple[int, bytes] | None:
    """GET a path from the local server, bypassing any proxy.

    Args:
        host: Server address.
        port: Server port.
        path: Request path starting with ``/``.
        timeout: Socket timeout in seconds.

    Returns:
        tuple[int, bytes] | None: ``(status, body)``, or None when nothing answered (HTTP errors give their status).
    """
    target = _connect_host(host)
    netloc = f"[{target}]" if ":" in target else target
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{netloc}:{port}{path}", timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def health(host: str, port: int, timeout: float = 1.0) -> bool:
    """Tell whether a ``llama-server`` is ready (``/health`` returns 200; 503 means still loading).

    Args:
        host: Server address.
        port: Server port.
        timeout: Socket timeout in seconds.

    Returns:
        bool: True only for an HTTP 200 answer.
    """
    answer = _get(host, port, "/health", timeout)
    return answer is not None and answer[0] == HTTP_OK


def props(host: str, port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    """Fetch the server's ``/props`` document.

    Args:
        host: Server address.
        port: Server port.
        timeout: Socket timeout in seconds.

    Returns:
        dict[str, Any] | None: Parsed JSON object, or None when unavailable or not an object.
    """
    answer = _get(host, port, "/props", timeout)
    if answer is None or answer[0] != HTTP_OK:
        return None
    try:
        data = json.loads(answer[1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def port_in_use(host: str, port: int) -> bool:
    """Tell whether something accepts TCP connections on ``host:port``.

    Args:
        host: Address to probe.
        port: Port to probe.

    Returns:
        bool: True when a connection succeeds.
    """
    try:
        with socket.create_connection((_connect_host(host), port), timeout=PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _model_alias(data: dict[str, Any] | None) -> str | None:
    """Extract a model name from a ``/props`` document, tolerating several layouts.

    Args:
        data: Parsed ``/props`` JSON, or None.

    Returns:
        str | None: Alias, model-path basename, or generation-settings model; None if absent.
    """
    if not data:
        return None
    alias = data.get("model_alias")
    if isinstance(alias, str) and alias:
        return alias
    path = data.get("model_path")
    if isinstance(path, str) and path:
        return PureWindowsPath(path).name
    settings = data.get("default_generation_settings")
    if isinstance(settings, dict):
        model = settings.get("model")
        if isinstance(model, str) and model:
            return model
    return None


# --- pid record ----------------------------------------------------------------------------------------------------


def _base(base: Path | None) -> Path:
    """Resolve the storage directory at call time.

    Args:
        base: Explicit directory, or None for ``constants.home()``.

    Returns:
        Path: Directory holding the pid file and log.
    """
    return base if base is not None else constants.home()


def _create_time(pid: int) -> float:
    """Return a process creation time, or ``0.0`` when it cannot be read.

    Args:
        pid: Process id.

    Returns:
        float: Epoch creation time, ``0.0`` if the process is gone or inaccessible.
    """
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return 0.0


def _is_ours(rec: ServerRecord) -> bool:
    """Check that the recorded pid is alive and is the very process we started.

    Args:
        rec: Record to validate.

    Returns:
        bool: True when the pid exists, is not a zombie and its creation time matches the record.
    """
    try:
        process = psutil.Process(rec.pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        return abs(process.create_time() - rec.create_time) <= CREATE_TIME_TOLERANCE
    except psutil.Error:
        return False


def write_record(base: Path | None, rec: ServerRecord) -> None:
    """Persist ``rec`` atomically; captures ``create_time`` for ``rec.pid`` when it is ``0.0``.

    Args:
        base: Storage directory (None for the default).
        rec: Record to write.
    """
    if not rec.create_time:
        rec.create_time = _create_time(rec.pid)
    directory = _base(base)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / constants.PID_FILENAME
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(rec)), encoding="utf-8")
    tmp.replace(target)


def clear_record(base: Path | None = None) -> None:
    """Remove the pid file if present.

    Args:
        base: Storage directory (None for the default).
    """
    try:
        (_base(base) / constants.PID_FILENAME).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove pid file: %s", exc)


def _parse_record(data: object) -> ServerRecord | None:
    """Build a record from decoded JSON, validating every field's type and range.

    Args:
        data: Decoded JSON document.

    Returns:
        ServerRecord | None: The record, or None when ``data`` is not a well-formed record.
    """
    if not isinstance(data, dict):
        return None
    pid, host, port = data.get("pid"), data.get("host"), data.get("port")
    model, started, ctime = data.get("model"), data.get("started"), data.get("create_time")
    if not (isinstance(pid, int) and not isinstance(pid, bool) and pid > 0):
        return None
    if not (isinstance(port, int) and not isinstance(port, bool) and 0 < port < MAX_PORT):
        return None
    if not (isinstance(ctime, int | float) and not isinstance(ctime, bool)):
        return None
    if not (isinstance(host, str) and isinstance(model, str) and isinstance(started, str)):
        return None
    return ServerRecord(pid, host, port, model, started, float(ctime))


def read_record(base: Path | None = None) -> ServerRecord | None:
    """Load the pid record, discarding it when stale, corrupt or pointing at a different process.

    Args:
        base: Storage directory (None for the default).

    Returns:
        ServerRecord | None: The record of a live process we started, else None (and the file is removed).
    """
    path = _base(base) / constants.PID_FILENAME
    if not path.is_file():
        return None
    try:
        rec = _parse_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        rec = None
    if rec is None:
        logger.warning("discarding unreadable or invalid pid record %s", path)
        clear_record(base)
        return None
    if not _is_ours(rec):
        clear_record(base)
        return None
    return rec


# --- inspection ----------------------------------------------------------------------------------------------------


def is_ready(host: str, port: int) -> bool:
    """Tell whether the server on ``host:port`` is up and finished loading.

    Args:
        host: Server address.
        port: Server port.

    Returns:
        bool: True when ``/health`` returns 200 (readiness only; identity comes from the record).
    """
    return health(host, port)


def _same_host(left: str, right: str) -> bool:
    """Compare two listen addresses, treating wildcard and ``localhost`` spellings as equivalent.

    Args:
        left: First address.
        right: Second address.

    Returns:
        bool: True when both resolve to the same connect address.
    """
    return _connect_host(left) == _connect_host(right)


def describe_running(host: str, port: int, base: Path | None = None) -> tuple[str, str | None]:
    """Classify whatever is on ``host:port``.

    Identity comes from the pid record (valid pid and creation time, same host and port); health only
    describes state, so a server of ours that is still loading is still ``"ours"`` (see ``is_ready``).

    Args:
        host: Address to inspect.
        port: Port to inspect.
        base: Storage directory (None for the default).

    Returns:
        tuple[str, str | None]: ``("ours", alias)`` when a valid record matches host and port, where ``alias``
            comes from ``/props`` and falls back to the recorded model; ``("foreign", alias)`` for a listener
            without such a record; ``("none", None)`` when nothing is there.
    """
    rec = read_record(base)
    listening = port_in_use(host, port)
    alias = _model_alias(props(host, port)) if listening else None
    if rec is not None and rec.port == port and _same_host(rec.host, host):
        return "ours", alias or rec.model
    return ("foreign", alias) if listening else ("none", None)


def _refuse_if_occupied(host: str, port: int, base: Path | None) -> None:
    """Raise unless it is safe to start a server on ``host:port``.

    Args:
        host: Address to listen on.
        port: Port to listen on.
        base: Storage directory (None for the default).

    Raises:
        LlamaCppError: When our own server is already recorded, or a foreign process holds the port.
    """
    rec = read_record(base)
    if rec is not None:
        raise LlamaCppError(
            f"an ollama2cpp server is already running (pid {rec.pid}, {rec.host}:{rec.port}); stop it first"
        )
    if port_in_use(host, port):
        raise LlamaCppError(f"port {port} on {host} is held by another process (not started by ollama2cpp)")


# --- lifecycle -----------------------------------------------------------------------------------------------------


def _terminate_tree(pid: int, timeout: float) -> None:
    """Terminate a process and its children, escalating to kill after ``timeout``.

    Args:
        pid: Root process id (must already be verified as ours).
        timeout: Seconds to wait before killing survivors.
    """
    try:
        root = psutil.Process(pid)
        targets = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.Error:
            logger.debug("terminate failed for pid %s", target.pid)
    _gone, alive = psutil.wait_procs(targets, timeout=timeout)
    for target in alive:
        try:
            target.kill()
        except psutil.Error:
            logger.debug("kill failed for pid %s", target.pid)
    psutil.wait_procs(alive, timeout=KILL_GRACE)


def stop(base: Path | None = None, *, timeout: float = 10.0) -> bool:
    """Stop the server recorded in our pid file (and its children); never touches other processes.

    Args:
        base: Storage directory (None for the default).
        timeout: Seconds to wait after terminate before killing.

    Returns:
        bool: True when a recorded, verified process was stopped; False when there was none.
    """
    rec = read_record(base)
    if rec is None:
        return False
    _terminate_tree(rec.pid, timeout)
    clear_record(base)
    return True


def _log_tail(log: Path, offset: int) -> str:
    """Return the last lines the child wrote since ``offset``.

    Args:
        log: Log file path.
        offset: Byte offset where this run's output begins.

    Returns:
        str: Up to ``LOG_TAIL_LINES`` lines.
    """
    try:
        with log.open("rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(deque(text.splitlines(), maxlen=LOG_TAIL_LINES))


def _failure(message: str, log: Path, offset: int) -> LlamaCppError:
    """Build an error that carries the log tail.

    Args:
        message: Headline.
        log: Log file path.
        offset: Byte offset where this run's output begins.

    Returns:
        LlamaCppError: Error with the last log lines appended.
    """
    return LlamaCppError(f"{message}\n--- last {LOG_TAIL_LINES} log lines ---\n{_log_tail(log, offset)}")


class _LogPump:
    """Copies new bytes of a log file to the console, as the child writes them."""

    def __init__(self, path: Path, offset: int) -> None:
        """Remember where to start reading.

        Args:
            path: Log file to follow.
            offset: Byte offset to start from.
        """
        self._path = path
        self._offset = offset
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def pump(self) -> None:
        """Print whatever was appended since the last call."""
        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return
        self._offset += len(chunk)
        text = self._decoder.decode(chunk)
        if text:
            ui.console.print(text, markup=False, highlight=False, end="")


def _record_for(pid: int, host: str, port: int, model: str) -> ServerRecord:
    """Build the record for a freshly spawned process.

    Args:
        pid: Child process id.
        host: Listen address.
        port: Listen port.
        model: Model alias.

    Returns:
        ServerRecord: Record with the process creation time captured.
    """
    started = datetime.now(UTC).isoformat(timespec="seconds")
    return ServerRecord(pid, host, port, model, started, _create_time(pid))


def _run_foreground(child: proc.Handle, log: Path, offset: int) -> int:
    """Stream the child's output until it exits or the user interrupts.

    Args:
        child: Running server process.
        log: Log file the child writes to.
        offset: Byte offset where this run's output begins.

    Returns:
        int: The child's exit code.
    """
    pump = _LogPump(log, offset)
    try:
        while child.poll() is None:
            pump.pump()
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        ui.show_notice("info", "Stopping server...")
        _terminate_tree(child.pid, KILL_GRACE)
    code = child.wait()
    pump.pump()
    return code


def _wait_ready(child: proc.Handle, host: str, port: int, ready_timeout: float, log: Path, offset: int) -> None:
    """Block until ``/health`` returns 200, the child dies, or the deadline passes.

    A child that exits first, or is still not ready at the deadline, raises
    ``LlamaCppError`` with the log tail.

    Args:
        child: Running server process.
        host: Listen address.
        port: Listen port.
        ready_timeout: Seconds to wait.
        log: Log file the child writes to.
        offset: Byte offset where this run's output begins.
    """
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        code = child.poll()
        if code is not None:
            raise _failure(f"llama-server exited with code {code} before becoming ready", log, offset)
        if health(host, port, timeout=PROBE_TIMEOUT):
            return
        time.sleep(POLL_INTERVAL)
    raise _failure(f"llama-server was not ready after {ready_timeout:g}s", log, offset)


def _supervise(
    child: proc.Handle, rec: ServerRecord, base: Path | None, log: Path, offset: int, ready_timeout: float | None
) -> int:
    """Record the child, then wait for readiness (detach) or stream until it exits (foreground).

    Args:
        child: Running server process.
        rec: Record to persist for it.
        base: Storage directory (None for the default).
        log: Log file the child writes to.
        offset: Byte offset where this run's output begins.
        ready_timeout: Seconds to wait for readiness, or None for foreground mode.

    Returns:
        int: 0 once ready (detach) or the child's exit code (foreground).
    """
    write_record(base, rec)
    if ready_timeout is None:
        return _run_foreground(child, log, offset)
    _wait_ready(child, rec.host, rec.port, ready_timeout, log, offset)
    return 0


def start(
    argv: Sequence[str],
    *,
    base: Path | None = None,
    host: str,
    port: int,
    model: str,
    detach: bool,
    ready_timeout: float = 120.0,
) -> int:
    """Start ``llama-server`` and record it.

    Foreground mode streams the server output to the console until it exits or Ctrl-C; detach mode returns
    once ``/health`` answers 200. Output always goes to ``<base>/server.log``. Raises ``LlamaCppError`` when the
    port is occupied or the server fails to become ready. On any failure the child is terminated and the record
    cleared, so no unrecorded server is left running.

    Args:
        argv: Full ``llama-server`` command line.
        base: Storage directory (None for the default).
        host: Address the server listens on.
        port: Port the server listens on.
        model: Model alias, stored in the record.
        detach: Run in the background and return after it is ready.
        ready_timeout: Seconds to wait for readiness in detach mode.

    Returns:
        int: Exit code of the server (foreground) or 0 once ready (detach).
    """
    _refuse_if_occupied(host, port, base)
    log = _base(base) / LOG_FILENAME
    offset = log.stat().st_size if log.is_file() else 0
    child = proc.spawn(argv, log_path=log, detach=detach)
    try:
        rec = _record_for(child.pid, host, port, model)
        code = _supervise(child, rec, base, log, offset, ready_timeout if detach else None)
    except BaseException:
        if child.poll() is None:
            _terminate_tree(child.pid, KILL_GRACE)
        clear_record(base)
        raise
    if not detach:
        clear_record(base)
    return code
