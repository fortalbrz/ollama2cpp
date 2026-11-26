"""Classify installed Ollama models and pick the best one for the available memory.

Memory estimate (``need_mib``): ``ceil((weights + kv_cache) * HEADROOM_FACTOR / MiB + OVERHEAD_MIB)``.

``weights`` is the manifest's model-layer size (:attr:`OllamaModel.size`), a single deliberate choice: it is the real
model size even when the blob on disk is not that large (tests use tiny blobs); the GGUF header only supplies the
layer and head counts for the KV-cache estimate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.gguf import GgufError, GgufInfo, kv_cache_bytes, read_gguf
from ollama2cpp.ollama import OllamaModel

log = logging.getLogger(__name__)

HEADROOM_FACTOR = 1.10  # compute buffers and fragmentation on top of weights + KV cache
OVERHEAD_MIB = 512  # fixed runtime overhead (CUDA/Vulkan context, scratch buffers)
# GGUF ``general.architecture`` values that are not text LLMs and so are never chat candidates:
#   clip    - vision/projector encoders (CLIP, multimodal projector GGUFs)
#   mmproj  - llama.cpp's name for multimodal projector files
#   whisper - speech-to-text models
# The real store's audio model (lfm3-audio) reports the text backbone arch ``lfm3``, so it is caught by name instead.
NON_LLM_ARCHITECTURES = frozenset({"clip", "mmproj", "whisper"})
# Case-insensitive model-name substrings marking speech/audio models that llama-server cannot serve as text chat.
NON_TEXT_NAME_PATTERNS = ("audio", "whisper", "tts", "speech")
_MIB = 1024 * 1024

Kind = Literal["chat", "embedding", "unsupported"]


@dataclass(frozen=True)
class Candidate:
    """An installed model with its classification and estimated memory need."""

    model: OllamaModel
    gguf: GgufInfo | None
    kind: Kind
    need_mib: int
    reason: str = ""


def _unsupported(model: OllamaModel, reason: str, info: GgufInfo | None = None) -> Candidate:
    """Build an ``unsupported`` candidate (``need_mib`` is ``0``: nothing was estimated)."""
    return Candidate(model, info, "unsupported", 0, reason)


def estimate_need_mib(model: OllamaModel, info: GgufInfo, ctx: int) -> int:
    """Estimate the memory needed to run ``model`` at ``ctx`` tokens.

    Known limitation: the KV-cache term ignores sliding-window attention, so it over-estimates such models (the
    selector is conservative; ``choose_best`` falls back to the smallest chat model with ``fits=False``).

    Args:
        model: The installed model (its manifest size is the weight size).
        info: Parsed GGUF facts.
        ctx: Context length in tokens.

    Returns:
        int: Estimated MiB, rounded up.
    """
    total = model.size + kv_cache_bytes(info, ctx)
    return math.ceil(total * HEADROOM_FACTOR / _MIB + OVERHEAD_MIB)


def _classify(model: OllamaModel, ctx: int) -> Candidate:
    """Classify one model; may raise (see :func:`classify`)."""
    if model.blob_path is None:
        return _unsupported(model, "no model layer")
    if not model.blob_exists:
        return _unsupported(model, "missing blob")
    info = read_gguf(model.blob_path)
    need = estimate_need_mib(model, info, ctx)
    name = model.name.lower()
    if info.is_embedding or "embed" in name:  # embedding wins over the non-text rules below
        return Candidate(model, info, "embedding", need)
    if info.architecture in NON_LLM_ARCHITECTURES:
        return _unsupported(model, f"non-LLM architecture {info.architecture!r} (not a text chat model)", info)
    if any(pattern in name for pattern in NON_TEXT_NAME_PATTERNS):
        return _unsupported(model, "audio model (not a text chat model)", info)
    return Candidate(model, info, "chat", need)


def classify(model: OllamaModel, ctx: int = DEFAULT_CTX) -> Candidate:
    """Classify a model as chat, embedding or unsupported and estimate its memory need.

    Never raises: any failure yields an ``unsupported`` candidate whose ``reason`` carries the message.

    Args:
        model: The installed model.
        ctx: Context length in tokens used for the KV-cache estimate.

    Returns:
        Candidate: The classification.
    """
    try:
        return _classify(model, ctx)
    except (GgufError, OSError) as exc:
        log.warning("Cannot read model %s: %s", model.name, exc)
        return _unsupported(model, str(exc))
    except Exception as exc:
        log.warning("Unexpected error classifying model %s: %s", model.name, exc)
        return _unsupported(model, f"unexpected error: {exc}")


def classify_all(models: Sequence[OllamaModel], ctx: int = DEFAULT_CTX) -> list[Candidate]:
    """Classify every model; one bad model becomes an ``unsupported`` row instead of raising.

    Args:
        models: Installed models.
        ctx: Context length in tokens used for the KV-cache estimate.

    Returns:
        list[Candidate]: One candidate per model, in input order.
    """
    return [classify(model, ctx) for model in models]


def choose_best(cands: Sequence[Candidate], vram_mib: int, ram_mib: int = 0) -> tuple[Candidate, bool] | None:
    """Pick the largest chat model that fits, else the smallest chat model.

    The memory budget is ``vram_mib``, or ``ram_mib`` on a CPU-only host (``vram_mib == 0``). Ties are broken by
    model name so the choice is deterministic.

    Args:
        cands: Classified candidates; only ``kind == "chat"`` is eligible.
        vram_mib: Free GPU memory in MiB (``0`` when there is no GPU).
        ram_mib: Available system RAM in MiB, used as the budget when ``vram_mib`` is ``0``.

    Returns:
        tuple[Candidate, bool] | None: ``(candidate, fits)``, or ``None`` when there is no chat model.
    """
    chats = [c for c in cands if c.kind == "chat"]
    if not chats:
        return None
    budget = vram_mib if vram_mib > 0 else ram_mib
    fitting = [c for c in chats if c.need_mib <= budget]
    if fitting:
        return min(fitting, key=lambda c: (-c.need_mib, c.model.name)), True
    return min(chats, key=lambda c: (c.need_mib, c.model.name)), False
