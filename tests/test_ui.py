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


def test_panel_shows_title_and_body(captured):
    ui.panel("First run", "Models: [C:/x]")
    out = captured.getvalue()
    assert "First run" in out
    assert "Models: [C:/x]" in out


def test_show_command_does_not_wrap(captured):
    command = "llama-server -m " + "x" * 200
    ui.show_command(command)
    assert command in captured.getvalue()


def test_confirm_passes_the_default_answer(monkeypatch):
    seen = {}
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr(ui.Confirm, "ask", lambda *_a, **kwargs: seen.update(kwargs) or True)
    assert ui.confirm("delete?", default=False) is True
    assert seen["default"] is False


def test_steps_progress_returns_progress():
    assert isinstance(ui.steps_progress(), Progress)


def test_show_table_prints_cells_as_plain_text(captured):
    ui.show_table("Things", ["Name", "State"], [["[bold]x", ui.status_text("ok")], ["y", ui.status_text("fail")]])
    out = captured.getvalue()
    assert "Things" in out
    assert "[bold]x" in out  # markup in a cell is not interpreted
    assert "ok" in out
    assert "fail" in out
