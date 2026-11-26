"""Console output seam: every user-facing message goes through Rich here."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator, Sequence
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

__all__ = [
    "Level",
    "confirm",
    "console",
    "panel",
    "progress",
    "show_command",
    "show_notice",
    "show_table",
    "status_text",
    "steps_progress",
    "use_stderr",
]

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


def confirm(question: str, *, assume_yes: bool = False, default: bool = True) -> bool:
    """Ask a yes/no question.

    Args:
        question: The question to show.
        assume_yes: Answer yes without asking.
        default: The answer when the user just presses Enter (use False for destructive actions).

    Returns:
        bool: True when ``assume_yes`` is set or stdin is not a TTY, else the user's answer.
    """
    if assume_yes or not sys.stdin.isatty():
        return True
    return Confirm.ask(question, console=console, default=default)


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


def steps_progress() -> Progress:
    """Build a progress display for work of unknown length (a spinner, a step counter and the elapsed time).

    Returns:
        Progress: A Rich progress instance, to be used as a context manager.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        TextColumn("{task.completed} candidates"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def status_text(level: Literal["ok", "warn", "fail"]) -> Text:
    """Render a check status word in its severity colour.

    Args:
        level: One of ``ok``, ``warn`` or ``fail``.

    Returns:
        Text: The word, coloured green, yellow or red.
    """
    style = {"ok": "green", "warn": "yellow", "fail": "red"}[level]
    return Text(level, style=style)


def show_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str | Text]]) -> None:
    """Print a table whose cells are plain text (Rich markup in cells is not interpreted).

    Args:
        title: Table title.
        columns: Column headers.
        rows: One sequence of cells per row; a ``Text`` cell keeps its own style.
    """
    table = Table(title=title, title_justify="left", header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(cell if isinstance(cell, Text) else Text(cell) for cell in row))
    console.print(table)


@contextlib.contextmanager
def use_stderr() -> Generator[None]:
    """Send every message, panel, prompt and progress display to stderr for the duration of the block.

    Used by commands whose stdout is machine output (``list --json``), so notices stay visible without polluting it.

    Yields:
        None: Control to the caller; the console goes back to stdout afterwards.
    """
    previous = console.stderr
    console.stderr = True
    try:
        yield
    finally:
        console.stderr = previous
