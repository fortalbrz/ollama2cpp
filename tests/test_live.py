"""Live smoke tests against the real Ollama directory and the real ``llama-server``.

Deselected by default (``-m "not live"``); run them with ``uv run pytest -m live --no-cov``.

They never start a server and never run a benchmark. Discovery is the default one (``~/.ollama/models`` and the
usual ``llama-server`` locations) because the autouse fixture in ``conftest.py`` clears ``OLLAMA_MODELS`` and
``OLLAMA_MODEL`` and points ``OLLAMA2CPP_HOME`` at a temporary directory, so nothing here touches the real
configuration. The tests skip when the host has no Ollama directory or no ``llama-server``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ollama2cpp import config, llamacpp, ollama
from ollama2cpp.__main__ import app

pytestmark = pytest.mark.live

runner = CliRunner()
_MODEL_FLAG = re.compile(r'\s-m\s+(?:"(?P<quoted>[^"]+)"|(?P<plain>\S+))')


@pytest.fixture(autouse=True)
def _real_host() -> None:
    """Skip the test unless the host has an Ollama directory and a ``llama-server``."""
    if ollama.find_ollama_path() is None:
        pytest.skip("no Ollama models directory found with default discovery")
    if llamacpp.find_binary("llama-server") is None:
        pytest.skip("llama-server not found")


def test_list_shows_models() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    models = ollama.list_models(ollama.find_ollama_path() or Path())
    assert models, "the Ollama directory has no models"
    assert "chat" in result.output or "embedding" in result.output


def test_list_json_is_valid_with_a_chat_model() -> None:
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert any(row["kind"] == "chat" for row in rows)


def test_serve_dry_run_points_at_an_existing_blob() -> None:
    result = runner.invoke(app, ["serve", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    match = _MODEL_FLAG.search(result.stdout)
    assert match is not None, result.stdout
    blob = Path(match["quoted"] or match["plain"])
    assert blob.is_file()
    assert blob.name.startswith("sha256-")
    assert "." not in blob.name  # Ollama blobs have no .gguf extension
    assert config.load() is not None  # the first run saved its configuration in the isolated home


def test_doctor_exits_zero() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output


def test_optimize_quick_saves_a_profile() -> None:
    result = runner.invoke(app, ["optimize", "--quick", "--yes"])
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg is not None
    assert cfg.profiles
    profile = next(iter(cfg.profiles.values()))
    assert profile.args
    assert not profile.measured
