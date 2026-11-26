#!/usr/bin/env python3
"""ollama2cpp command line interface.

Console app to start llama-server (llama.cpp) from Ollama available models.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from ollama2cpp import __version__, commands, optimize_cmd, ui, workflow
from ollama2cpp.constants import ENV_OLLAMA_MODEL
from ollama2cpp.llamacpp import LlamaCppError
from ollama2cpp.ollama import ModelNotFoundError

logger = logging.getLogger(__name__)
__all__ = ["app", "main"]


@dataclass
class _State:
    """Global options populated by the app callback before each command."""

    verbose: bool = False
    model: str | None = None
    model_explicit: str | None = None  # the model from a `--model` flag, not from the environment


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
    source = ctx.get_parameter_source("model")
    _state.model_explicit = None if source is not None and source.name == "ENVIRONMENT" else model
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if ctx.invoked_subcommand is None:
        _finish(lambda: workflow.run_serve(workflow.ServeOptions(model=model)))


EXIT_INTERRUPTED = 130


def _finish(action: Callable[[], int]) -> None:
    """Run ``action`` and exit with its code; failures become one error line (traceback only with ``--verbose``).

    Args:
        action: The command body, returning the exit code.

    Raises:
        Exit: Always, carrying the exit code.
    """
    try:
        code = action()
    except (workflow.WorkflowError, LlamaCppError, ModelNotFoundError) as exc:
        ui.show_notice("error", str(exc))
        code = 1
    except KeyboardInterrupt:
        ui.show_notice("warn", "Interrupted.")
        code = EXIT_INTERRUPTED
    except Exception as exc:
        logger.debug("Unexpected failure", exc_info=True)
        ui.show_notice("error", f"Unexpected error: {exc}")
        if _state.verbose:
            ui.console.print_exception(show_locals=False)
        code = 1
    raise typer.Exit(code=code)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def serve(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Ollama model name (default: saved/auto).")] = None,
    ollama_path: Annotated[
        Path | None, typer.Option("--ollama-path", help="Ollama models directory (contains manifests/ and blobs/).")
    ] = None,
    host: Annotated[
        str | None, typer.Option("--host", help="Address to listen on (default: config or 127.0.0.1).")
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", min=1, max=65535, help="Port to listen on (default 8080).")
    ] = None,
    context: Annotated[int | None, typer.Option("--ctx", min=1, help="Context size in tokens (default 8192).")] = None,
    detach: Annotated[bool, typer.Option("--detach", help="Run in the background; return once ready.")] = False,
    restart: Annotated[bool, typer.Option("--restart", help="Replace our own running server.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the command and exit.")] = False,
    vision: Annotated[
        bool, typer.Option("--vision", help="Load the multimodal projector when the model has one.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Answer yes to questions (e.g. installing llama.cpp).")
    ] = False,
) -> None:
    """Start llama-server on an Ollama model. Extra llama-server arguments go after `--`."""
    opts = workflow.ServeOptions(
        model=model or _state.model,
        ollama_path=ollama_path,
        host=host,
        port=port,
        ctx=context,
        detach=detach,
        restart=restart,
        dry_run=dry_run,
        vision=vision,
        assume_yes=yes,
        extra_args=tuple(ctx.args),
    )
    _finish(lambda: workflow.run_serve(opts))


@app.command("list")
def list_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Print a JSON array instead of a table.")] = False,
) -> None:
    """List the installed Ollama models (kind, size, quantisation, fit, optimization state)."""
    _finish(lambda: commands.run_list(as_json=as_json))


@app.command()
def info(model: Annotated[str, typer.Argument(help="Ollama model name.")]) -> None:
    """Show the GGUF details, layers, VRAM need and saved profiles of a model."""
    _finish(lambda: commands.run_info(model))


@app.command()
def status() -> None:
    """Show the server state, configuration, GPU memory and the default model's profile."""
    _finish(commands.run_status)


@app.command()
def stop() -> None:
    """Stop the server started by ollama2cpp (never touches other processes)."""
    _finish(commands.run_stop)


@app.command()
def doctor() -> None:
    """Check llama.cpp, Ollama, the GPU and the configuration, with fix hints."""
    _finish(commands.run_doctor)


@app.command()
def update() -> None:
    """Upgrade llama.cpp with winget."""
    _finish(commands.run_update)


@app.command("config")
def config_cmd(
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set", metavar="KEY=VALUE", help=f"Set a key ({', '.join(commands.SETTABLE_KEYS)}); repeatable."
        ),
    ] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Delete the configuration file.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before --reset.")] = False,
) -> None:
    """Show the configuration and its path; change settings with --set, or delete it with --reset."""
    _finish(lambda: commands.run_config(set_values=set_values or [], reset=reset, assume_yes=yes))


@app.command()
def optimize(
    model: Annotated[str | None, typer.Argument(help="Ollama model name (default: saved/auto).")] = None,
    quick: Annotated[bool, typer.Option("--quick", help="Save the heuristic settings without benchmarking.")] = False,
    all_models: Annotated[bool, typer.Option("--all", help="Optimize every chat model.")] = False,
    time_limit: Annotated[
        float, typer.Option("--time-limit", min=0, help="Benchmark budget per model, in seconds.")
    ] = optimize_cmd.DEFAULT_TIME_LIMIT,
    context: Annotated[
        int | None, typer.Option("--ctx", min=1, help="Context size to optimize for (default 8192).")
    ] = None,
    allow_kv_quant: Annotated[
        bool, typer.Option("--allow-kv-quant", help="Also try a q8_0 KV cache (may affect quality).")
    ] = False,
    unload_ollama: Annotated[
        bool, typer.Option("--unload-ollama", help="Ask Ollama to unload its models if they hold the VRAM.")
    ] = False,
    stop_server: Annotated[
        bool, typer.Option("--stop-server", help="Stop our running server first (it holds VRAM).")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Answer yes to questions (e.g. installing llama.cpp).")
    ] = False,
) -> None:
    """Find the fastest llama-server settings for a model on this machine and save them."""
    opts = optimize_cmd.OptimizeOptions(
        # `--all` ignores a model that only came from $OLLAMA_MODEL; an explicit MODEL or `--model` still conflicts
        model=model or (_state.model_explicit if all_models else _state.model),
        quick=quick,
        all_models=all_models,
        time_limit=time_limit,
        ctx=context,
        allow_kv_quant=allow_kv_quant,
        unload_ollama=unload_ollama,
        stop_server=stop_server,
        assume_yes=yes,
    )
    _finish(lambda: optimize_cmd.run_optimize(opts))


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
