"""Console output seam: every user-facing message goes through Rich here."""

from __future__ import annotations

import sys
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
from rich.prompt import Confirm
from rich.text import Text

__all__ = ["Level", "confirm", "console", "panel", "progress", "show_command", "show_notice"]

Level = Literal["info", "ok", "warn", "error"]

console = Console()

_STYLES: dict[str, tuple[str, str]] = {
    "info": ("blue", "i"),
    "ok": ("green", "ok"),
    "warn": ("yellow", "!"),
    "error": ("red", "x"),
}


def show_notice(level: Level, message: str) -> None:
    """Print a one-line notice styled by severity.

    Args:
        level: One of ``info``, ``ok``, ``warn`` or ``error``.
        message: Text to show (Rich markup is not interpreted).
    """
    style, tag = _STYLES[level]
    console.print(f"[{style}]{tag}[/{style}] ", end="")
    console.print(message, markup=False, highlight=False)


def confirm(question: str, *, assume_yes: bool = False) -> bool:
    """Ask a yes/no question.

    Args:
        question: The question to show.
        assume_yes: Answer yes without asking.

    Returns:
        bool: True when ``assume_yes`` is set or stdin is not a TTY, else the user's answer.
    """
    if assume_yes or not sys.stdin.isatty():
        return True
    return Confirm.ask(question, console=console, default=True)


def panel(title: str, body: str) -> None:
    """Print a bordered panel (the body is plain text, Rich markup is not interpreted).

    Args:
        title: Panel title.
        body: Panel text, may span several lines.
    """
    console.print(Panel(Text(body), title=title, title_align="left", expand=False))


def show_command(command: str) -> None:
    """Print a command line verbatim on one logical line (never wrapped or restyled).

    Args:
        command: The command line.
    """
    console.print(command, markup=False, highlight=False, soft_wrap=True)


def progress() -> Progress:
    """Build a download-style progress bar bound to the shared console.

    Returns:
        Progress: A Rich progress instance, to be used as a context manager.
    """
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console,
    )
