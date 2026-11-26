#!/usr/bin/env python3
"""ollama2cpp — command line interface.

Console app to start llammacpp from ollama available models.

"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import warnings
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import typer

from ollama2cpp import __version__, ui
from ollama2cpp.constants import (
    DEFAULT_OLLAMA_MODEL,
    ENV_OLLAMA_MODEL,
    DEFAULT_PATH,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
__all__ = ["app", "main"]


# ---------------------------------------------------------------------------
# Global state — populated by the @app.callback before each command
# ---------------------------------------------------------------------------


@dataclass
class _State:
    verbose: bool = False
    model: str = DEFAULT_OLLAMA_MODEL
    sha256: str = ""
    path: Path = field(default_factory=lambda: DEFAULT_PATH)


_state = _State()

app = typer.Typer(
    name="ollama2cpp",
    help=t("cli.app.help"),
    epilog=t(
        "cli.epilog",
        model=ENV_OLLAMA_MODEL        
    ),
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
items_app = typer.Typer(help=t("cli.items.help"), no_args_is_help=True)
app.add_typer(items_app, name="items", rich_help_panel=t("cli.panel.connections"))


@app.callback()
def _configure(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help=t("cli.option.verbose"))] = False,
    model: Annotated[
        str, typer.Option("--model", "-m", help=t("cli.option.model"), envvar=ENV_OLLAMA_MODEL)
    ] = DEFAULT_OLLAMA_MODEL,
    version: Annotated[bool, typer.Option("--version", help=t("cli.option.version"))] = False,
) -> None:
    """Record the global options and start the chat when no command is given."""
    if version:
        ui.show_notice(Notice("info", f"llmopenfinance {__version__}"))
        raise typer.Exit(code=0)
    _state.verbose = verbose
    _state.model = model
    _state.path = DEFAULT_PATH
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if ctx.invoked_subcommand is None:
        raise typer.Exit(code=_handle_chat())


# ---------------------------------------------------------------------------
# Factories — monkeypatched in tests
# ---------------------------------------------------------------------------



def _llamma_server(model: str, *, think: bool = False) -> Any:
    """Instantiate the configured LLM chat backend.

    Args:
        model: Model ollama id (empty = default).
        think: Enable thinking for Ollama models.

    Returns:
        Any: ``AbstractChat`` instance.
    """
    from llmopenfinance.chat import get_llm

    kwargs: dict[str, Any] = {"think": think} if model else {}
    return start_llamma_server(model, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _help_text() -> str:
    """Return the in-chat help in the process language.

    Evaluated on every call so it follows the process language; never cache it in a module constant.

    Returns:
        str: The command list on the first line, then example questions.
    """
    return t("chat.help")


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------



@app.command(help=t("cli.status.help"), rich_help_panel=t("cli.panel.data"))
def status() -> None:
    """Show registered connections, consents and local data counts."""
    raise typer.Exit(code=_handle_status())




@app.command("ver", help=t("cli.ver.help"))
def version_cmd() -> None:
    """Print the version."""
    ui.show_notice(Notice("info", f"llmopenfinance {__version__}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
