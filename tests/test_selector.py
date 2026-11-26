"""Tests for the model selector."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ollama2cpp import selector
from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.gguf import kv_cache_bytes, read_gguf
from ollama2cpp.ollama import OllamaModel, list_models
from ollama2cpp.selector import Candidate, choose_best, classify, classify_all
from tests.gguf_builder import write_gguf

Builder = Callable[..., Path]
Spec = tuple[str, str, int, dict[str, Any]]

GB = 1024**3
MIB = 1024**2
BUDGET_8GB = 8 * 1024


def _chat(name: str, size: int, **kwargs: Any) -> Spec:
    """Spec for a model whose manifest declares ``size`` bytes and whose blob is a tiny valid GGUF (small KV cache)."""
    opts: dict[str, Any] = {"n_layer": 2, **kwargs}
    return (name, "latest", size, {"blob_writer": lambda path: write_gguf(path, **opts)})


def _models(builder: Builder, specs: list[Spec], **kwargs: Any) -> list[OllamaModel]:
    return list_models(builder(specs, **kwargs))


def _cands(builder: Builder, sizes: dict[str, int]) -> list[Candidate]:
    return classify_all(_models(builder, [_chat(n, s) for n, s in sizes.items()]))


def test_chat_model_classified(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("chat", 4 * GB)])
    result = classify(model)
    assert result.kind == "chat"
    assert result.gguf is not None
    assert result.gguf.architecture == "llama"
    assert not result.reason


def test_need_includes_kv_and_overhead(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("chat", 4 * GB)])
    cand = classify(model)
    assert model.blob_path is not None
    info = read_gguf(model.blob_path)
    expected = math.ceil(
        (model.size + kv_cache_bytes(info, DEFAULT_CTX)) * selector.HEADROOM_FACTOR / MIB + selector.OVERHEAD_MIB
    )
    assert cand.need_mib == expected
    assert cand.need_mib > model.size // MIB
    assert cand.need_mib > 4 * 1024 + selector.OVERHEAD_MIB


def test_need_uses_manifest_size_not_blob_size(fake_ollama: Builder) -> None:
    small, big = _models(fake_ollama, [_chat("a", GB), _chat("b", 9 * GB)])
    assert classify(big).need_mib > classify(small).need_mib + 8 * 1024


def test_need_grows_with_ctx(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("chat", GB)])
    assert classify(model, 32768).need_mib > classify(model, 2048).need_mib


def test_embedding_by_pooling_excluded(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("vec", GB, pooling=1)])
    assert classify(model).kind == "embedding"


def test_embedding_by_name(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("Nomic-EMBED-text", GB)])
    assert classify(model).kind == "embedding"


def test_unsupported_when_blob_missing(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("gone", GB)], missing_blob=["gone:latest"])
    cand = classify(model)
    assert cand.kind == "unsupported"
    assert cand.reason == "missing blob"
    assert cand.gguf is None


def test_unsupported_when_no_model_layer(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [("nolayer", "latest", GB, {"no_model": True})])
    cand = classify(model)
    assert cand.kind == "unsupported"
    assert cand.reason == "no model layer"


def test_unreadable_gguf_is_unsupported_not_raise(fake_ollama: Builder, caplog: pytest.LogCaptureFixture) -> None:
    (model,) = _models(fake_ollama, [("junk", "latest", GB, {})])
    with caplog.at_level(logging.WARNING):
        cand = classify(model)
    assert cand.kind == "unsupported"
    assert cand.reason
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_non_llm_architecture_unsupported(fake_ollama: Builder) -> None:
    (model,) = _models(fake_ollama, [_chat("proj", GB, arch="clip")])
    cand = classify(model)
    assert cand.kind == "unsupported"
    assert "clip" in cand.reason
    assert choose_best([cand], 100_000) is None


def test_unexpected_exception_is_unsupported(fake_ollama: Builder, monkeypatch: pytest.MonkeyPatch) -> None:
    (model,) = _models(fake_ollama, [_chat("boom", GB)])

    def explode(*_a: object, **_k: object) -> int:
        msg = "kaboom"
        raise RuntimeError(msg)

    monkeypatch.setattr(selector, "kv_cache_bytes", explode)
    cand = classify(model)
    assert cand.kind == "unsupported"
    assert "kaboom" in cand.reason


def test_classify_all_never_raises(fake_ollama: Builder) -> None:
    models = _models(
        fake_ollama,
        [_chat("a", GB), ("junk", "latest", GB, {}), _chat("z", GB, pooling=1), _chat("gone", GB)],
        missing_blob=["gone:latest"],
    )
    kinds = {c.model.name: c.kind for c in classify_all(models)}
    assert kinds == {
        "a:latest": "chat",
        "gone:latest": "unsupported",
        "junk:latest": "unsupported",
        "z:latest": "embedding",
    }


def test_choose_largest_that_fits(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"s": 2 * GB, "m": 6 * GB, "l": 20 * GB})
    best = choose_best(cands, BUDGET_8GB)
    assert best is not None
    assert best[0].model.name == "m:latest"
    assert best[1] is True


def test_none_fit_picks_smallest_with_warning_flag(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"s": 2 * GB, "m": 6 * GB, "l": 20 * GB})
    best = choose_best(cands, 1024)
    assert best is not None
    assert best[0].model.name == "s:latest"
    assert best[1] is False


def test_cpu_only_uses_ram_budget(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"s": 2 * GB, "m": 6 * GB, "l": 20 * GB})
    best = choose_best(cands, 0, 32 * 1024)
    assert best is not None
    assert best[0].model.name == "l:latest"
    assert best[1] is True
    small_ram = choose_best(cands, 0, BUDGET_8GB)
    assert small_ram is not None
    assert small_ram[0].model.name == "m:latest"


def test_vram_takes_precedence_over_ram(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"s": 2 * GB, "m": 6 * GB, "l": 20 * GB})
    best = choose_best(cands, BUDGET_8GB, 64 * 1024)
    assert best is not None
    assert best[0].model.name == "m:latest"


def test_both_zero_picks_smallest_not_fitting(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"s": 2 * GB, "m": 6 * GB})
    best = choose_best(cands, 0, 0)
    assert best is not None
    assert best[0].model.name == "s:latest"
    assert best[1] is False


def test_no_chat_models_returns_none(fake_ollama: Builder) -> None:
    models = _models(fake_ollama, [_chat("vec", GB, pooling=1), ("junk", "latest", GB, {})])
    assert choose_best(classify_all(models), 100_000) is None
    assert choose_best([], 100_000) is None


def test_equal_sizes_tie_break_by_name(fake_ollama: Builder) -> None:
    cands = _cands(fake_ollama, {"b": 2 * GB, "a": 2 * GB, "c": 2 * GB})
    fits = choose_best(cands, 100_000)
    nofit = choose_best(cands, 1)
    assert fits is not None
    assert nofit is not None
    assert fits[0].model.name == "a:latest"
    assert nofit[0].model.name == "a:latest"
    assert choose_best(list(reversed(cands)), 100_000) == fits


def test_embedding_never_chosen_even_if_largest(fake_ollama: Builder) -> None:
    models = _models(fake_ollama, [_chat("vec", 5 * GB, pooling=1), _chat("chat", 2 * GB)])
    best = choose_best(classify_all(models), 100_000)
    assert best is not None
    assert best[0].model.name == "chat:latest"
