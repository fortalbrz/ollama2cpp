"""Tests for the subprocess seam."""

from __future__ import annotations

import sys

from ollama2cpp import proc


def test_run_missing_exe():
    assert proc.run(["definitely-not-a-binary-xyz"]).returncode == 127


def test_run_captures_output():
    assert proc.run([sys.executable, "-c", "print('hi')"]).stdout.strip() == "hi"


def test_run_captures_stderr_and_code():
    r = proc.run([sys.executable, "-c", "import sys; sys.stderr.write('e'); sys.exit(3)"])
    assert r.returncode == 3
    assert r.stderr == "e"


def test_run_timeout():
    assert proc.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2).returncode == 124


def test_spawn_logs_to_file(tmp_path):
    log = tmp_path / "sub" / "out.log"
    p = proc.spawn([sys.executable, "-c", "print('logged')"], log_path=log)
    p.wait(timeout=10)
    assert "logged" in log.read_text(encoding="utf-8")


def test_spawn_without_log():
    p = proc.spawn([sys.executable, "-c", "pass"])
    assert p.wait(timeout=10) == 0


def test_spawn_detach():
    p = proc.spawn([sys.executable, "-c", "pass"], detach=True)
    assert p.wait(timeout=10) == 0


def test_spawn_detach_posix_branch(monkeypatch):
    monkeypatch.setattr(proc.sys, "platform", "linux")
    p = proc.spawn([sys.executable, "-c", "pass"], detach=True)
    assert p.wait(timeout=10) == 0


def test_run_oserror_is_sentinel(monkeypatch):
    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(proc.subprocess, "run", boom)
    r = proc.run(["x"])
    assert r.returncode == 127
    assert "denied" in r.stderr


def test_text_normalises():
    assert not proc._text(None)
    assert proc._text(b"ab") == "ab"
    assert proc._text("ab") == "ab"


def test_format_command_round_trips_spaces(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    text = proc.format_command([r"C:\Program Files\llama-server.exe", "-m", "a b.gguf"])
    assert text == r'"C:\Program Files\llama-server.exe" -m "a b.gguf"'


def test_format_command_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert proc.format_command(["llama-server", "-m", "a b.gguf"]) == "llama-server -m 'a b.gguf'"
