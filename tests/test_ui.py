"""Tests for the ui seam."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.progress import Progress

from ollama2cpp import ui


@pytest.fixture
def captured(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", Console(file=buf, force_terminal=False, width=100))
    return buf


@pytest.mark.parametrize("level", ["info", "ok", "warn", "error"])
def test_show_notice_levels(captured, level):
    ui.show_notice(level, "hello world")
    assert "hello world" in captured.getvalue()


def test_confirm_assume_yes():
    assert ui.confirm("ok?", assume_yes=True) is True


def test_confirm_non_tty_is_yes(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert ui.confirm("ok?") is True


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_confirm_tty_asks(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr(ui.Confirm, "ask", lambda *_a, **_k: False)
    assert ui.confirm("ok?") is False


def test_progress_returns_progress():
    assert isinstance(ui.progress(), Progress)
