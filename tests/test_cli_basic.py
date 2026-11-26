"""Basic CLI tests."""

from __future__ import annotations

from typer.testing import CliRunner

from ollama2cpp import constants
from ollama2cpp.__main__ import app

runner = CliRunner()


def test_version_flag():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert "ollama2cpp" in r.output


def test_ver_command():
    r = runner.invoke(app, ["ver"])
    assert r.exit_code == 0
    assert "ollama2cpp" in r.output


def test_no_args_exits_cleanly():
    r = runner.invoke(app, [])
    assert r.exit_code == 0


def test_verbose_flag_accepted():
    r = runner.invoke(app, ["-v", "ver"])
    assert r.exit_code == 0


def test_home_default(monkeypatch):
    monkeypatch.delenv(constants.ENV_HOME, raising=False)
    assert constants.home() == constants.DEFAULT_PATH


def test_home_empty_env_uses_default(monkeypatch):
    monkeypatch.setenv(constants.ENV_HOME, "")
    assert constants.home() == constants.DEFAULT_PATH


def test_home_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(constants.ENV_HOME, str(tmp_path))
    assert constants.home() == tmp_path


def test_defaults():
    assert constants.DEFAULT_HOST == "127.0.0.1"
    assert constants.DEFAULT_PORT == 8080
    assert constants.DEFAULT_CTX == 8192
    assert not hasattr(constants, "DEFAULT_OLLAMA_MODEL")


def test_main_runs_app(monkeypatch):
    from ollama2cpp import __main__ as entry

    called = []
    monkeypatch.setattr(entry, "app", lambda: called.append(True))
    entry.main()
    assert called == [True]
