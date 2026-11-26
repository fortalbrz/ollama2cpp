"""Tune llama-server launch settings for maximum generation speed.

First half: a pure, deterministic heuristic seed (no I/O) that the benchmark sweep starts from.

Flag spellings verified against ``llama-server --help`` (build 12721, Windows): ``-ngl``, ``-fa on|off|auto`` (takes
a value, not a bare switch), ``-t``, ``-b``, ``-ub``, ``-ctk``/``-ctv``. ``llama-bench`` of the same build accepts the
same short spellings (``-fa`` also takes ``on|off|auto`` there; ``-nkvo`` takes ``0|1``), but sweeps comma-separated
value lists, e.g. ``-ngl 0,33``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.gguf import GgufInfo, kv_cache_bytes
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.selector import HEADROOM_FACTOR, OVERHEAD_MIB

log = logging.getLogger(__name__)

MIN_CTX = 2048  # the seed never shrinks the context below this
SEED_BATCH = 2048
SEED_UBATCH = 512
SEED_CACHE_TYPE = "f16"
_MIB = 1024 * 1024


@dataclass(frozen=True)
class SeedPlan:
    """Starting launch settings for the sweep.

    Attributes:
        ngl: GPU layers (``block_count + 1`` is a full offload including the output layer; 0 is CPU-only).
        ctx: Context length in tokens.
        flash_attn: Whether to enable flash attention.
        threads: CPU threads for generation.
        batch: Logical batch size.
        ubatch: Physical batch size.
        cache_type: KV cache element type, used for both K and V.
    """

    ngl: int
    ctx: int
    flash_attn: bool
    threads: int
    batch: int
    ubatch: int
    cache_type: str


def _need_mib(weights: float, kv: float) -> int:
    """Estimate MiB for ``weights`` + ``kv`` bytes, using the selector's headroom and overhead model."""
    return math.ceil((weights + kv) * HEADROOM_FACTOR / _MIB + OVERHEAD_MIB)


def _fit_ctx(info: GgufInfo, budget_mib: int, ctx: int) -> tuple[int, bool]:
    """Halve ``ctx`` (floor ``MIN_CTX``) until the whole model fits ``budget_mib``.

    Args:
        info: Parsed GGUF facts.
        budget_mib: Memory budget in MiB.
        ctx: Requested context length.

    Returns:
        tuple[int, bool]: The chosen context and whether the whole model fits at it.
    """
    while _need_mib(info.file_size, kv_cache_bytes(info, ctx)) > budget_mib:
        if ctx <= MIN_CTX:
            return ctx, False
        ctx = max(MIN_CTX, ctx // 2)
    return ctx, True


def _largest_layer_count(info: GgufInfo, budget_mib: int, ctx: int) -> int:
    """Return the largest layer count (``0..block_count``) whose weights and KV shares fit ``budget_mib``."""
    kv = kv_cache_bytes(info, ctx)
    blocks = max(info.block_count, 1)
    total_layers = info.block_count + 1
    best = 0
    for layers in range(info.block_count + 1):
        weights = info.file_size * layers / total_layers
        if _need_mib(weights, kv * layers / blocks) > budget_mib:
            break
        best = layers
    return best


def heuristic_seed(info: GgufInfo, host: HostInfo, device: Device | None, ctx: int = DEFAULT_CTX) -> SeedPlan:
    """Pick a starting plan from the model and hardware facts alone (pure, no I/O).

    With a device the budget is its *total* VRAM. The context is halved (floor ``MIN_CTX``) before any layer is left on
    the CPU; only when the whole model still does not fit at the floor is the layer count reduced. Without a device
    the plan is CPU-only and the context is reduced only to fit total RAM.

    Args:
        info: Parsed GGUF facts (``info.file_size`` is the weight size).
        host: Host snapshot (physical core count and RAM).
        device: Chosen GPU, or None for a CPU-only run.
        ctx: Requested context length in tokens.

    Returns:
        SeedPlan: The starting settings.
    """
    threads = max(1, host.physical_cores)
    if device is None:
        fitted, _ = _fit_ctx(info, host.ram_total_mib, ctx)
        return SeedPlan(0, fitted, False, threads, SEED_BATCH, SEED_UBATCH, SEED_CACHE_TYPE)
    fitted, fits = _fit_ctx(info, device.total_mib, ctx)
    ngl = info.block_count + 1 if fits else _largest_layer_count(info, device.total_mib, fitted)
    log.debug("Seed for %s: ngl=%d ctx=%d (requested ctx=%d)", device.id, ngl, fitted, ctx)
    return SeedPlan(ngl, fitted, True, threads, SEED_BATCH, SEED_UBATCH, SEED_CACHE_TYPE)


def seed_to_args(plan: SeedPlan) -> list[str]:
    """Render ``plan`` as ``llama-server`` arguments (the context is passed separately by the caller).

    Args:
        plan: The launch settings.

    Returns:
        list[str]: Argument vector fragment using the spellings verified against ``llama-server --help``.
    """
    return [
        "-ngl",
        str(plan.ngl),
        "-fa",
        "on" if plan.flash_attn else "off",
        "-t",
        str(plan.threads),
        "-b",
        str(plan.batch),
        "-ub",
        str(plan.ubatch),
        "-ctk",
        plan.cache_type,
        "-ctv",
        plan.cache_type,
    ]
