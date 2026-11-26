"""Tests for the server runtime (a tiny HTTP server stands in for llama-server)."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest

from ollama2cpp import constants, proc, server
from ollama2cpp.llamacpp import LlamaCppError
from tests.conftest import FAKE_SERVER, FakeServer, free_port

Track = Callable[[proc.Handle], proc.Handle]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
LOCAL = "127.0.0.1"


def _rec(pid: int, port: int = 1, **kw: Any) -> server.ServerRecord:
    return server.ServerRecord(pid=pid, host=LOCAL, port=port, model="m", started="now", **kw)


def _wait(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _fake_argv(port: int, load: float = 0.0) -> list[str]:
    return [sys.executable, str(FAKE_SERVER), str(port), "", str(load)]


def _dead(pid: int) -> bool:
    return not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.fixture
def base(tmp_path: Path) -> Iterator[Path]:
    """Storage dir; whatever server was recorded there is stopped at teardown."""
    yield tmp_path
    server.stop(tmp_path, timeout=3)


def test_health_true_for_fake_server(fake_server: FakeServer):
    assert server.health(fake_server.host, fake_server.port)


def test_health_false_when_nothing_listening():
    assert not server.health(LOCAL, free_port())


def test_health_false_on_503(reaper: Track):
    port = free_port()
    reaper(proc.spawn(_fake_argv(port, 30)))
    assert _wait(lambda: server.port_in_use(LOCAL, port))
    assert not server.health(LOCAL, port)


def test_health_wildcard_host_connects_to_loopback(fake_server: FakeServer):
    assert server.health("0.0.0.0", fake_server.port)
    assert server.port_in_use("", fake_server.port)


def test_props(fake_server: FakeServer):
    assert server.props(fake_server.host, fake_server.port) == {"model_alias": "fake-model"}
    assert server.props(LOCAL, free_port()) is None


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, None),
        ({}, None),
        ({"model_alias": "a"}, "a"),
        ({"model_path": "C:\\models\\x.gguf"}, "x.gguf"),
        ({"model_path": "/m/y.gguf"}, "y.gguf"),
        ({"default_generation_settings": {"model": "z"}}, "z"),
        ({"default_generation_settings": 3}, None),
    ],
)
def test_model_alias_variants(data: dict[str, Any] | None, expected: str | None):
    assert server._model_alias(data) == expected


def test_port_in_use(fake_server: FakeServer):
    assert server.port_in_use(fake_server.host, fake_server.port)
    assert not server.port_in_use(LOCAL, free_port())


def test_record_roundtrip(tmp_path: Path, reaper: Track):
    p = reaper(proc.spawn(SLEEP))
    server.write_record(tmp_path, _rec(p.pid, 8081))
    rec = server.read_record(tmp_path)
    assert rec is not None
    assert (rec.pid, rec.port, rec.model) == (p.pid, 8081, "m")
    assert rec.create_time == pytest.approx(psutil.Process(p.pid).create_time())
    assert (tmp_path / constants.PID_FILENAME).is_file()
    server.clear_record(tmp_path)
    assert server.read_record(tmp_path) is None


def test_record_uses_home_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaper: Track):
    monkeypatch.setenv(constants.ENV_HOME, str(tmp_path / "h"))
    p = reaper(proc.spawn(SLEEP))
    server.write_record(None, _rec(p.pid))
    assert (tmp_path / "h" / constants.PID_FILENAME).is_file()
    assert server.read_record() is not None
    server.clear_record()
    assert server.read_record() is None


def test_stale_record_cleared(tmp_path: Path):
    p = proc.spawn([sys.executable, "-c", "pass"])
    p.wait(timeout=10)
    server.write_record(tmp_path, _rec(p.pid, create_time=1.0))
    assert server.read_record(tmp_path) is None
    assert not (tmp_path / constants.PID_FILENAME).exists()


def test_record_with_reused_pid_is_stale(tmp_path: Path, reaper: Track):
    p = reaper(proc.spawn(SLEEP))
    server.write_record(tmp_path, _rec(p.pid, create_time=psutil.Process(p.pid).create_time() - 3600))
    assert server.read_record(tmp_path) is None
    assert not (tmp_path / constants.PID_FILENAME).exists()


def test_corrupt_record_cleared(tmp_path: Path):
    (tmp_path / constants.PID_FILENAME).write_text("{not json", encoding="utf-8")
    assert server.read_record(tmp_path) is None
    assert not (tmp_path / constants.PID_FILENAME).exists()


def test_clear_and_read_without_file(tmp_path: Path):
    server.clear_record(tmp_path)
    assert server.read_record(tmp_path) is None


def test_start_refuses_foreign_port(fake_server: FakeServer, base: Path):
    with pytest.raises(LlamaCppError, match=str(fake_server.port)):
        server.start(SLEEP, base=base, host=fake_server.host, port=fake_server.port, model="m", detach=True)
    assert fake_server.process.poll() is None  # foreign process untouched
    assert server.read_record(base) is None


def test_start_refuses_when_ours_already_running(tmp_path: Path, reaper: Track):
    p = reaper(proc.spawn(SLEEP))
    server.write_record(tmp_path, _rec(p.pid, 9))
    with pytest.raises(LlamaCppError, match="already running"):
        server.start(SLEEP, base=tmp_path, host=LOCAL, port=free_port(), model="m", detach=True)
    assert p.poll() is None


def test_stop_only_recorded_pid(tmp_path: Path, reaper: Track):
    bystander = reaper(proc.spawn(SLEEP))
    assert server.stop(tmp_path) is False
    assert bystander.poll() is None


def test_stop_terminates_recorded(tmp_path: Path, reaper: Track):
    p = reaper(proc.spawn(SLEEP))
    server.write_record(tmp_path, _rec(p.pid))
    assert server.stop(tmp_path, timeout=5) is True
    assert p.poll() is not None
    assert not (tmp_path / constants.PID_FILENAME).exists()


def test_stop_ignores_reused_pid(tmp_path: Path, reaper: Track):
    victim = reaper(proc.spawn(SLEEP))
    server.write_record(tmp_path, _rec(victim.pid, create_time=psutil.Process(victim.pid).create_time() - 3600))
    assert server.stop(tmp_path, timeout=1) is False
    assert victim.poll() is None


def test_stop_terminates_children_too(tmp_path: Path, reaper: Track):
    code = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    out = tmp_path / "out.log"
    parent = reaper(proc.spawn([sys.executable, "-c", code], log_path=out))
    assert _wait(lambda: out.is_file() and out.read_text(encoding="utf-8").strip().isdigit())
    child_pid = int(out.read_text(encoding="utf-8").strip())
    server.write_record(tmp_path, _rec(parent.pid))
    assert server.stop(tmp_path, timeout=5)
    assert _wait(lambda: _dead(child_pid))


def test_stop_escalates_to_kill(tmp_path: Path, reaper: Track):
    stubborn = (
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('r', flush=True)\ntime.sleep(60)\n"
    )
    out = tmp_path / "o.log"
    p = reaper(proc.spawn([sys.executable, "-c", stubborn], log_path=out))
    assert _wait(lambda: out.is_file() and out.read_text(encoding="utf-8").strip() == "r")
    server.write_record(tmp_path, _rec(p.pid))
    assert server.stop(tmp_path, timeout=0.3)
    assert p.poll() is not None


def test_terminate_tree_missing_pid():
    p = proc.spawn([sys.executable, "-c", "pass"])
    p.wait(timeout=10)
    server._terminate_tree(p.pid, 0.1)  # must not raise


def test_describe_running_none(tmp_path: Path):
    assert server.describe_running(LOCAL, free_port(), tmp_path) == ("none", None)


def test_describe_running_foreign_vs_ours(fake_server: FakeServer, tmp_path: Path):
    host, port = fake_server.host, fake_server.port
    assert server.describe_running(host, port, tmp_path) == ("foreign", "fake-model")
    server.write_record(tmp_path, _rec(fake_server.process.pid, port))
    assert server.describe_running(host, port, tmp_path) == ("ours", "fake-model")
    # a record for a different port does not make this one ours
    server.write_record(tmp_path, _rec(fake_server.process.pid, port + 1))
    assert server.describe_running(host, port, tmp_path)[0] == "foreign"


def test_describe_running_wrong_create_time_is_foreign(fake_server: FakeServer, tmp_path: Path):
    ct = psutil.Process(fake_server.process.pid).create_time() - 3600
    server.write_record(tmp_path, _rec(fake_server.process.pid, fake_server.port, create_time=ct))
    assert server.describe_running(fake_server.host, fake_server.port, tmp_path)[0] == "foreign"


def test_start_failure_includes_log_tail(tmp_path: Path):
    argv = [sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"]
    with pytest.raises(LlamaCppError, match="boom"):
        server.start(argv, base=tmp_path, host=LOCAL, port=free_port(), model="m", detach=True, ready_timeout=10)
    assert server.read_record(tmp_path) is None


def test_start_failure_tail_is_last_20_lines_of_this_run(tmp_path: Path):
    (tmp_path / server.LOG_FILENAME).write_text("OLD-RUN\n", encoding="utf-8")
    argv = [sys.executable, "-c", "import sys\nfor i in range(30): print('line', i)\nsys.exit(2)"]
    with pytest.raises(LlamaCppError) as err:
        server.start(argv, base=tmp_path, host=LOCAL, port=free_port(), model="m", detach=True, ready_timeout=10)
    text = str(err.value)
    assert "line 29" in text
    assert "line 9" not in text
    assert "OLD-RUN" not in text


def test_start_detach_waits_for_health(base: Path):
    port = free_port()
    began = time.monotonic()
    code = server.start(_fake_argv(port, 0.6), base=base, host=LOCAL, port=port, model="alias", detach=True)
    assert code == 0
    assert time.monotonic() - began >= 0.4  # kept polling through the 503s
    rec = server.read_record(base)
    assert rec is not None
    assert (rec.port, rec.model) == (port, "alias")
    assert server.describe_running(LOCAL, port, base) == ("ours", "fake-model")
    assert server.stop(base, timeout=5)
    assert _wait(lambda: not server.port_in_use(LOCAL, port))


def test_start_detach_timeout(base: Path):
    port = free_port()
    with pytest.raises(LlamaCppError, match="not ready"):
        server.start(_fake_argv(port, 60), base=base, host=LOCAL, port=port, model="m", detach=True, ready_timeout=0.6)
    assert server.read_record(base) is None
    assert _wait(lambda: not server.port_in_use(LOCAL, port))


def test_start_foreground_streams_and_returns_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    argv = [sys.executable, "-c", "print('hello from child'); raise SystemExit(3)"]
    code = server.start(argv, base=tmp_path, host=LOCAL, port=free_port(), model="m", detach=False)
    assert code == 3
    assert "hello from child" in capsys.readouterr().out
    assert not (tmp_path / constants.PID_FILENAME).exists()


def test_start_foreground_stopped_from_elsewhere(base: Path):
    port = free_port()
    result: list[int] = []

    def run() -> None:
        result.append(server.start(_fake_argv(port), base=base, host=LOCAL, port=port, model="m", detach=False))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert _wait(lambda: server.health(LOCAL, port))
        assert server.describe_running(LOCAL, port, base) == ("ours", "fake-model")
        assert server.stop(base, timeout=5)
    finally:
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert len(result) == 1
    assert not (base / constants.PID_FILENAME).exists()


def test_start_foreground_ctrl_c_terminates_child(base: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}
    real_pump = server._LogPump.pump

    def interrupt(self: server._LogPump) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        real_pump(self)

    pids: list[int] = []
    real_spawn = proc.spawn

    def spy(*args: Any, **kw: Any) -> proc.Handle:
        handle = real_spawn(*args, **kw)
        pids.append(handle.pid)
        return handle

    monkeypatch.setattr(server._LogPump, "pump", interrupt)
    monkeypatch.setattr(proc, "spawn", spy)
    server.start(SLEEP, base=base, host=LOCAL, port=free_port(), model="m", detach=False)
    assert _dead(pids[0])
    assert not (base / constants.PID_FILENAME).exists()


def test_log_helpers_tolerate_missing_file(tmp_path: Path):
    assert not server._log_tail(tmp_path / "nope.log", 0)
    server._LogPump(tmp_path / "nope.log", 0).pump()


# --- review fixes -------------------------------------------------------------------------------------------------


def test_describe_running_ours_while_loading(base: Path, reaper: Track):
    port = free_port()
    child = reaper(proc.spawn(_fake_argv(port, 30)))
    assert _wait(lambda: server.port_in_use(LOCAL, port))
    server.write_record(base, _rec(child.pid, port))
    assert not server.is_ready(LOCAL, port)
    assert server.describe_running(LOCAL, port, base) == ("ours", "m")  # /props is 503: recorded model


def test_describe_running_ours_prefers_props_alias_when_ready(base: Path, fake_server: FakeServer):
    server.write_record(base, _rec(fake_server.process.pid, fake_server.port))
    assert server.is_ready(fake_server.host, fake_server.port)
    assert server.describe_running(fake_server.host, fake_server.port, base) == ("ours", "fake-model")


def test_describe_running_ours_falls_back_to_recorded_model_on_props_failure(base: Path, reaper: Track):
    p = reaper(proc.spawn(SLEEP))
    server.write_record(base, _rec(p.pid, free_port()))  # nothing bound yet
    rec = server.read_record(base)
    assert rec is not None
    assert server.describe_running(LOCAL, rec.port, base) == ("ours", "m")


@pytest.mark.parametrize("spelling", ["", "0.0.0.0", "localhost", "LOCALHOST", "127.0.0.1"])
def test_describe_running_host_aliases(base: Path, reaper: Track, spelling: str):
    p = reaper(proc.spawn(SLEEP))
    port = free_port()
    server.write_record(base, _rec(p.pid, port))
    assert server.describe_running(spelling, port, base)[0] == "ours"


def test_describe_running_foreign_without_record(fake_server: FakeServer, base: Path):
    assert server.describe_running(fake_server.host, fake_server.port, base) == ("foreign", "fake-model")


@pytest.mark.parametrize(
    "payload",
    [
        {"pid": "abc", "host": LOCAL, "port": 1, "model": "m", "started": "n", "create_time": 1.0},
        {"pid": -5, "host": LOCAL, "port": 1, "model": "m", "started": "n", "create_time": 1.0},
        {"pid": True, "host": LOCAL, "port": 1, "model": "m", "started": "n", "create_time": 1.0},
        {"pid": 5, "host": LOCAL, "port": "80", "model": "m", "started": "n", "create_time": 1.0},
        {"pid": 5, "host": LOCAL, "port": 70000, "model": "m", "started": "n", "create_time": 1.0},
        {"pid": 5, "host": 7, "port": 1, "model": "m", "started": "n", "create_time": 1.0},
        {"pid": 5, "host": LOCAL, "port": 1, "model": "m", "started": "n", "create_time": "x"},
        {"pid": 5, "host": LOCAL},
        [1, 2, 3],
        "text",
        None,
    ],
)
def test_invalid_record_is_cleared(tmp_path: Path, payload: object):
    (tmp_path / constants.PID_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert server.read_record(tmp_path) is None
    assert not (tmp_path / constants.PID_FILENAME).exists()
    assert server.stop(tmp_path) is False


def _garbage_server(reply: bytes) -> tuple[socket.socket, int, threading.Thread]:
    sock = socket.socket()
    sock.bind((LOCAL, 0))
    sock.listen(5)
    sock.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(0.2)
                try:
                    conn.recv(1024)
                    if reply:
                        conn.sendall(reply)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    thread.stop = stop  # type: ignore[attr-defined]
    return sock, sock.getsockname()[1], thread


@pytest.mark.parametrize("reply", [b"garbage not http\r\n\r\n", b""])
def test_non_http_listener_is_foreign_not_a_crash(reply: bytes, tmp_path: Path):
    sock, port, thread = _garbage_server(reply)
    try:
        assert not server.health(LOCAL, port)
        assert server.props(LOCAL, port) is None
        assert server.describe_running(LOCAL, port, tmp_path) == ("foreign", None)
    finally:
        thread.stop.set()  # type: ignore[attr-defined]
        thread.join(timeout=3)
        sock.close()


def test_start_terminates_child_when_record_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pids: list[int] = []
    real_spawn = proc.spawn

    def spy(*args: Any, **kw: Any) -> proc.Handle:
        handle = real_spawn(*args, **kw)
        pids.append(handle.pid)
        return handle

    def boom(*_a: Any, **_k: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(proc, "spawn", spy)
    monkeypatch.setattr(server, "write_record", boom)
    with pytest.raises(PermissionError):
        server.start(SLEEP, base=tmp_path, host=LOCAL, port=free_port(), model="m", detach=True)
    assert _wait(lambda: _dead(pids[0]))


def test_start_foreground_pump_error_terminates_child_and_clears_record(base: Path, monkeypatch: pytest.MonkeyPatch):
    pids: list[int] = []
    real_spawn = proc.spawn

    def spy(*args: Any, **kw: Any) -> proc.Handle:
        handle = real_spawn(*args, **kw)
        pids.append(handle.pid)
        return handle

    def explode(self: server._LogPump) -> None:
        raise UnicodeEncodeError("ascii", "x", 0, 1, "console")

    monkeypatch.setattr(proc, "spawn", spy)
    monkeypatch.setattr(server._LogPump, "pump", explode)
    with pytest.raises(UnicodeEncodeError):
        server.start(SLEEP, base=base, host=LOCAL, port=free_port(), model="m", detach=False)
    assert _wait(lambda: _dead(pids[0]))
    assert not (base / constants.PID_FILENAME).exists()


def test_start_detach_keyboard_interrupt_terminates_child(base: Path, monkeypatch: pytest.MonkeyPatch):
    pids: list[int] = []
    real_spawn = proc.spawn

    def spy(*args: Any, **kw: Any) -> proc.Handle:
        handle = real_spawn(*args, **kw)
        pids.append(handle.pid)
        return handle

    def interrupted(*_a: Any, **_k: Any) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(proc, "spawn", spy)
    monkeypatch.setattr(server, "health", interrupted)
    with pytest.raises(KeyboardInterrupt):
        server.start(SLEEP, base=base, host=LOCAL, port=free_port(), model="m", detach=True, ready_timeout=10)
    assert _wait(lambda: _dead(pids[0]))
    assert not (base / constants.PID_FILENAME).exists()
