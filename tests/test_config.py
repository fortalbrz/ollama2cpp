"""Tests for the config store."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ollama2cpp import config, constants
from ollama2cpp.config import Config, Profile, load, profile_key, save


def _profile() -> Profile:
    return Profile(
        args=["-ngl", "99", "-c", "8192"],
        ctx=8192,
        tg_tps=42.5,
        pp_tps=900.0,
        measured=True,
        fingerprint={"gpu": "RTX", "vram": "8192"},
        timestamp="2026-11-25T10:00:00",
        bench=[{"ngl": 99, "tg": 42.5}],
    )


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load(tmp_path) is None


def test_config_file_path(tmp_path: Path) -> None:
    assert config.config_file(tmp_path) == tmp_path / constants.CONFIG_FILENAME


def test_defaults_are_not_shared() -> None:
    a, b = Config(), Config()
    a.profiles["x"] = _profile()
    a.extra["k"] = 1
    assert b.profiles == {}
    assert b.extra == {}
    assert a.host == constants.DEFAULT_HOST
    assert a.port == constants.DEFAULT_PORT


def test_roundtrip(tmp_path: Path) -> None:
    cfg = Config(ollama_path="X", device="CUDA0", profiles={"d:8192": _profile()})
    save(cfg, tmp_path)
    assert load(tmp_path) == cfg


def test_profile_validated_defaults_true_and_roundtrips(tmp_path: Path) -> None:
    assert _profile().validated is True
    unvalidated = Profile(args=["-ngl", "1"], ctx=2048, measured=True, tg_tps=3.0, validated=False)
    save(Config(profiles={"d:1": unvalidated, "d:2": _profile()}), tmp_path)
    loaded = load(tmp_path)
    assert loaded is not None
    assert loaded.profiles["d:1"].validated is False
    assert loaded.profiles["d:2"].validated is True


def test_profile_without_validated_key_loads_as_validated(tmp_path: Path) -> None:
    data = {"profiles": {"k": {"args": [], "ctx": 1, "measured": True}}}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = load(tmp_path)
    assert loaded is not None
    assert loaded.profiles["k"].validated is True


def test_roundtrip_profile_with_none_speeds(tmp_path: Path) -> None:
    prof = Profile(args=[], ctx=4096, tg_tps=None, pp_tps=None, measured=False, fingerprint={}, timestamp="", bench=[])
    cfg = Config(profiles={"d:4096": prof})
    save(cfg, tmp_path)
    assert load(tmp_path) == cfg


def test_save_creates_directory(tmp_path: Path) -> None:
    base = tmp_path / "nested" / "dir"
    save(Config(), base)
    assert (base / "config.json").is_file()


def test_unknown_keys_preserved(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"future": 1, "port": 9}), encoding="utf-8")
    cfg = load(tmp_path)
    assert cfg is not None
    assert cfg.port == 9
    assert cfg.extra == {"future": 1}
    save(cfg, tmp_path)
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["future"] == 1
    assert data["port"] == 9


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        "\xff\xfe",
        json.dumps({"port": "8080"}),
        json.dumps({"port": True}),
        json.dumps({"host": 5}),
        json.dumps({"ollama_path": 5}),
        json.dumps({"profiles": []}),
        json.dumps({"profiles": {"k": 1}}),
        json.dumps({"profiles": {"k": {"args": "x", "ctx": 1}}}),
        json.dumps({"profiles": {"k": {"args": [1], "ctx": 1}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": "1"}}}),
        json.dumps({"profiles": {"k": {"args": []}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "tg_tps": "fast"}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "measured": "yes"}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "fingerprint": {"a": 1}}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "timestamp": 3}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "bench": [1]}}}),
        json.dumps({"profiles": {"k": {"args": [], "ctx": 1, "validated": "no"}}}),
    ],
)
def test_corrupt_file_backed_up(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(content.encode("latin-1"))
    assert load(tmp_path) is None
    assert (tmp_path / "config.json.bak").exists()
    assert not path.exists()


def test_corrupt_backup_overwrites_old_backup(tmp_path: Path) -> None:
    (tmp_path / "config.json.bak").write_text("old", encoding="utf-8")
    (tmp_path / "config.json").write_text("{bad", encoding="utf-8")
    assert load(tmp_path) is None
    assert (tmp_path / "config.json.bak").read_text(encoding="utf-8") == "{bad"


def test_corrupt_backup_failure_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text("{bad", encoding="utf-8")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr(os, "replace", boom)
    assert load(tmp_path) is None


def test_unreadable_config_raises_and_is_left_alone(tmp_path: Path) -> None:
    # A directory where the file should be: the read fails with OSError, which is not corruption.
    (tmp_path / "config.json").mkdir()
    with pytest.raises(OSError, match=r"config\.json"):
        load(tmp_path)
    assert (tmp_path / "config.json").is_dir()
    assert not (tmp_path / "config.json.bak").exists()


def test_os_error_on_read_of_valid_file_does_not_back_it_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(Config(ollama_path="keep", profiles={"d:1": _profile()}), tmp_path)
    path = tmp_path / "config.json"
    before = path.read_text(encoding="utf-8")

    def denied(*_a: object, **_k: object) -> str:
        raise PermissionError("denied")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", denied)
        with pytest.raises(PermissionError):
            load(tmp_path)
    assert path.read_text(encoding="utf-8") == before  # the valid file (and its profiles) were not moved
    assert not (tmp_path / "config.json.bak").exists()
    loaded = load(tmp_path)
    assert loaded is not None
    assert "d:1" in loaded.profiles


def test_empty_file_is_corrupt_and_backed_up(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("", encoding="utf-8")
    assert load(tmp_path) is None
    assert (tmp_path / "config.json.bak").exists()


def test_int_speed_accepted_as_float(tmp_path: Path) -> None:
    data = {"profiles": {"k": {"args": [], "ctx": 1, "tg_tps": 5}}}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    cfg = load(tmp_path)
    assert cfg is not None
    assert cfg.profiles["k"].tg_tps == pytest.approx(5.0)
    assert cfg.profiles["k"].measured is False


def test_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save(Config(ollama_path="orig"), tmp_path)
    path = tmp_path / "config.json"
    before = path.read_text(encoding="utf-8")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        save(Config(ollama_path="new"), tmp_path)
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]


def test_profile_key() -> None:
    assert profile_key("abc", 8192) == "abc:8192"


def test_env_home_redirects_load_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(constants.ENV_HOME, str(tmp_path / "h"))
    assert load() is None
    save(Config(ollama_path="E"))
    assert (tmp_path / "h" / "config.json").is_file()
    assert config.config_file() == tmp_path / "h" / "config.json"
    loaded = load()
    assert loaded is not None
    assert loaded.ollama_path == "E"


def test_corrupt_config_warning_has_no_traceback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "config.json").write_text("{bad", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="ollama2cpp.config"):
        assert load(tmp_path) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert all(r.exc_info is None for r in warnings)
    assert any(r.levelno == logging.DEBUG and r.exc_info for r in caplog.records)
