"""Tests for the Ollama store reader."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from ollama2cpp import ollama
from ollama2cpp.constants import ENV_OLLAMA_MODELS
from ollama2cpp.ollama import Layer, ModelNotFoundError, OllamaModel, find_ollama_path, list_models, resolve_model

Builder = Callable[..., Path]


def _model_hex(root: Path, manifest: str) -> str:
    data = json.loads((root / "manifests" / manifest).read_text(encoding="utf-8"))
    digest = next(x["digest"] for x in data["layers"] if x["mediaType"].endswith("image.model"))
    return digest.split(":", 1)[1]


def _store(path: Path) -> Path:
    (path / "manifests").mkdir(parents=True)
    return path


def test_find_prefers_explicit_then_env(tmp_path: Path) -> None:
    explicit = _store(tmp_path / "explicit")
    from_env = _store(tmp_path / "env")
    env = {ENV_OLLAMA_MODELS: str(from_env)}
    assert find_ollama_path(explicit, env) == explicit
    assert find_ollama_path(None, env) == from_env


def test_find_skips_dir_without_manifests(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    from_env = _store(tmp_path / "env")
    assert find_ollama_path(empty, {ENV_OLLAMA_MODELS: str(from_env)}) == from_env


def test_find_falls_back_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    store = _store(tmp_path / ".ollama" / "models")
    assert find_ollama_path(None, {}) == store


def test_find_localappdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    store = _store(tmp_path / "local" / "Ollama" / "models")
    assert find_ollama_path(None, {"LOCALAPPDATA": str(tmp_path / "local")}) == store


def test_find_system_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    store = _store(tmp_path / "system")
    monkeypatch.setattr(ollama, "SYSTEM_MODELS_PATH", store)
    assert find_ollama_path(None, {}) == store


def test_find_none_when_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(ollama, "SYSTEM_MODELS_PATH", tmp_path / "nosystem")
    assert find_ollama_path(None, {}) is None


def test_list_models_names(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("nomic-embed-text", "latest", 300, {}), ("gemma4", "12b", 5000, {})])
    assert [m.name for m in list_models(root)] == ["gemma4:12b", "nomic-embed-text:latest"]


def test_list_models_no_manifests_dir(tmp_path: Path) -> None:
    assert list_models(tmp_path) == []


def test_blob_path_maps_digest(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("gemma4", "12b", 5000, {})])
    (m,) = list_models(root)
    hex_ = _model_hex(root, "registry.ollama.ai/library/gemma4/12b")
    assert m.digest == "sha256:" + hex_
    assert m.blob_path is not None
    assert m.blob_path.name == "sha256-" + hex_
    assert m.blob_path.parent == root / "blobs"
    assert m.blob_exists
    assert m.size == 5000


def test_dangling_manifest(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("gemma4", "12b", 5000, {})], missing_blob={"gemma4:12b"})
    (m,) = list_models(root)
    assert m.blob_path is not None
    assert m.blob_exists is False


def test_no_model_layer(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("tmpl", "latest", 0, {"no_model": True})])
    (m,) = list_models(root)
    assert m.digest is None
    assert m.blob_path is None
    assert m.blob_exists is False
    assert m.size == 0


def test_projector_and_draft_flags(fake_ollama: Builder) -> None:
    root = fake_ollama(
        models=[
            ("plain", "latest", 1, {}),
            ("vision", "latest", 1, {"projector": True}),
            ("spec", "latest", 1, {"draft": True}),
        ]
    )
    by_name = {m.name: m for m in list_models(root)}
    assert (by_name["plain:latest"].has_projector, by_name["plain:latest"].has_draft) == (False, False)
    assert (by_name["vision:latest"].has_projector, by_name["vision:latest"].has_draft) == (True, False)
    assert (by_name["spec:latest"].has_projector, by_name["spec:latest"].has_draft) == (False, True)


def test_corrupt_manifest_skipped(fake_ollama: Builder, caplog: pytest.LogCaptureFixture) -> None:
    root = fake_ollama(models=[("good", "latest", 1, {})], corrupt=[("bad", "latest")])
    with caplog.at_level(logging.WARNING):
        names = [m.name for m in list_models(root)]
    assert names == ["good:latest"]
    assert "bad" in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"layers": "nope"}',
        '{"layers": [1]}',
        '{"layers": [{"mediaType": "x", "digest": "sha256:../../evil", "size": 1}]}',
        '{"layers": [{"mediaType": "x", "digest": "sha256:ab", "size": "big"}]}',
        '{"layers": [{"mediaType": "x", "digest": 5, "size": 1}]}',
    ],
)
def test_malformed_manifest_shapes_skipped(fake_ollama: Builder, content: str) -> None:
    root = fake_ollama(models=[("good", "latest", 1, {})])
    bad = root / "manifests" / "registry.ollama.ai" / "library" / "bad" / "latest"
    bad.parent.mkdir(parents=True)
    bad.write_text(content, encoding="utf-8")
    assert [m.name for m in list_models(root)] == ["good:latest"]


def test_manifest_without_layers_key(fake_ollama: Builder) -> None:
    root = fake_ollama()
    path = root / "manifests" / "registry.ollama.ai" / "library" / "empty" / "latest"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    (m,) = list_models(root)
    assert m.layers == ()
    assert m.blob_path is None


def test_shallow_manifest_path_skipped(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("good", "latest", 1, {})])
    stray = root / "manifests" / "stray"
    stray.write_text("{}", encoding="utf-8")
    assert [m.name for m in list_models(root)] == ["good:latest"]


def test_resolve_normalizes(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("gemma4", "latest", 1, {}), ("gemma4", "12b", 1, {})])
    assert resolve_model(root, "Gemma4").name == "gemma4:latest"
    assert resolve_model(root, "GEMMA4:12B").name == "gemma4:12b"
    assert resolve_model(root, "registry.ollama.ai/library/gemma4:12b").name == "gemma4:12b"
    assert resolve_model(root, "library/gemma4").name == "gemma4:latest"


def test_resolve_unknown_lists_available(fake_ollama: Builder) -> None:
    root = fake_ollama(models=[("gemma4", "12b", 1, {})])
    with pytest.raises(ModelNotFoundError, match="gemma4:12b"):
        resolve_model(root, "nope")


def test_resolve_unknown_in_empty_store(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="no models"):
        resolve_model(tmp_path, "nope")


def test_nonlibrary_namespace_and_registry_names(fake_ollama: Builder) -> None:
    root = fake_ollama(
        models=[
            ("model", "tag", 1, {"namespace": "user"}),
            ("repo", "q4", 1, {"registry": "hf.co", "namespace": "org"}),
        ]
    )
    names = [m.name for m in list_models(root)]
    assert names == ["hf.co/org/repo:q4", "user/model:tag"]
    assert resolve_model(root, "user/model:tag").name == "user/model:tag"
    assert resolve_model(root, "HF.co/org/repo:q4").name == "hf.co/org/repo:q4"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Gemma4", "gemma4:latest"),
        ("gemma4:12B", "gemma4:12b"),
        (" gemma4 ", "gemma4:latest"),
        ("registry.ollama.ai/library/gemma4", "gemma4:latest"),
        ("registry.ollama.ai/user/x:t", "user/x:t"),
        ("hf.co/org/repo:q4", "hf.co/org/repo:q4"),
        ("localhost:5000/ns/x", "localhost:5000/ns/x:latest"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert ollama.normalize_name(raw) == expected


def test_dataclasses_frozen() -> None:
    layer = Layer("application/vnd.ollama.image.model", "sha256:ab", 3)
    m = OllamaModel("a:b", Path("m"), (layer,), Path("b"), "sha256:ab")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(m, "name", "x")  # ruff: ignore[set-attr-with-constant]
