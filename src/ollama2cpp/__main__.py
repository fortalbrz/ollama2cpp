#!/usr/bin/env python3
"""ollama2cpp command line interface.

Console app to start llama-server (llama.cpp) from Ollama available models.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Annotated

import typer

from ollama2cpp import __version__, ui
from ollama2cpp.constants import ENV_OLLAMA_MODEL

logger = logging.getLogger(__name__)
__all__ = ["app", "main"]


@dataclass
class _State:
    """Global options populated by the app callback before each command."""

    verbose: bool = False
    model: str | None = None


_state = _State()

app = typer.Typer(
    name="ollama2cpp",
    help="Start llama-server (llama.cpp) on the GGUF blobs of your Ollama models.",
    epilog=f"Set [bold]{ENV_OLLAMA_MODEL}[/bold] to choose the default model.",
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _configure(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Ollama model name (default: auto-select).", envvar=ENV_OLLAMA_MODEL),
    ] = None,
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    """Record the global options; show help when no command is given."""
    if version:
        ui.show_notice("info", f"ollama2cpp {__version__}")
        raise typer.Exit(code=0)
    _state.verbose = verbose
    _state.model = model
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if ctx.invoked_subcommand is None:
        ui.console.print(ctx.get_help())
        raise typer.Exit(code=0)


@app.command("ver")
def version_cmd() -> None:
    """Print the version."""
    ui.show_notice("info", f"ollama2cpp {__version__}")


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
