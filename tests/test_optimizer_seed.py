"""Tests for the optimizer's heuristic seed."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from ollama2cpp import selector
from ollama2cpp.constants import DEFAULT_CTX
from ollama2cpp.gguf import GgufInfo, kv_cache_bytes
from ollama2cpp.hardware import Device, HostInfo
from ollama2cpp.optimizer import SeedPlan, heuristic_seed, seed_to_args

GB = 1024**3
MIB = 1024**2


def _info(file_size: int = 4 * GB, **kwargs: Any) -> GgufInfo:
    base = GgufInfo(
        architecture="llama",
        block_count=32,
        context_length=131072,
        embedding_length=4096,
        head_count=32,
        head_count_kv=8,
        key_length=None,
        file_type=None,
        is_embedding=False,
        file_size=file_size,
    )
    return replace(base, **kwargs)


def _host(ram_mib: int = 32 * 1024, cores: int = 8) -> HostInfo:
    return HostInfo(
        devices=(), ram_total_mib=ram_mib, ram_free_mib=ram_mib // 2, physical_cores=cores, logical_cores=cores * 2
    )


def _gpu(total_mib: int, free_mib: int | None = None) -> Device:
    return Device(id="CUDA0", name="GPU", total_mib=total_mib, free_mib=total_mib if free_mib is None else free_mib)


def _need(info: GgufInfo, ctx: int) -> int:
    total = info.file_size + kv_cache_bytes(info, ctx)
    return math.ceil(total * selector.HEADROOM_FACTOR / MIB + selector.OVERHEAD_MIB)


def test_full_offload_when_fits() -> None:
    info = _info()
    seed = heuristic_seed(info, _host(), _gpu(24 * 1024))
    assert seed.ngl == info.block_count + 1
    assert seed.ctx == DEFAULT_CTX
    assert seed.flash_attn is True
    assert (seed.batch, seed.ubatch, seed.cache_type) == (2048, 512, "f16")
    assert seed.threads == 8


def test_partial_offload_when_too_big() -> None:
    info = _info(file_size=20 * GB)
    seed = heuristic_seed(info, _host(), _gpu(8 * 1024))
    assert 0 < seed.ngl <= info.block_count
    assert seed.ctx == 2048  # ctx was reduced to the floor first
    layers = seed.ngl
    share = (
        info.file_size * layers / (info.block_count + 1) + kv_cache_bytes(info, seed.ctx) * layers / info.block_count
    )
    assert share * selector.HEADROOM_FACTOR / MIB + selector.OVERHEAD_MIB <= 8 * 1024
    one_more = (
        info.file_size * (layers + 1) / (info.block_count + 1)
        + kv_cache_bytes(info, seed.ctx) * (layers + 1) / info.block_count
    )
    assert one_more * selector.HEADROOM_FACTOR / MIB + selector.OVERHEAD_MIB > 8 * 1024


def test_ctx_reduced_before_dropping_layers() -> None:
    info = _info(file_size=2 * GB)
    vram = 4 * 1024
    assert _need(info, 32768) > vram >= _need(info, 2048)  # precondition: only the KV overflows
    seed = heuristic_seed(info, _host(), _gpu(vram), ctx=32768)
    assert seed.ctx < 32768
    assert seed.ctx >= 2048
    assert seed.ngl == info.block_count + 1
    assert _need(info, seed.ctx) <= vram
    assert seed.ctx == 2048 or _need(info, seed.ctx * 2) > vram  # halving stops at the first fit


def test_uses_total_not_free_vram() -> None:
    info = _info()
    seed = heuristic_seed(info, _host(), _gpu(24 * 1024, free_mib=100))
    assert seed.ngl == info.block_count + 1


def test_nothing_fits_gives_zero_layers() -> None:
    seed = heuristic_seed(_info(), _host(), _gpu(selector.OVERHEAD_MIB // 2))
    assert seed.ngl == 0
    assert seed.ctx == 2048


def test_no_device_is_cpu_only() -> None:
    host = _host(cores=6)
    seed = heuristic_seed(_info(), host, None)
    assert seed.ngl == 0
    assert seed.flash_attn is False
    assert seed.threads == host.physical_cores
    assert seed.ctx == DEFAULT_CTX


def test_no_device_ctx_reduced_to_fit_ram() -> None:
    info = _info(file_size=2 * GB)
    ram = 4 * 1024
    assert _need(info, 32768) > ram >= _need(info, 2048)
    seed = heuristic_seed(info, _host(ram_mib=ram), None, ctx=32768)
    assert 2048 <= seed.ctx < 32768
    assert _need(info, seed.ctx) <= ram
    assert seed.ngl == 0


def test_no_device_ram_too_small_keeps_floor() -> None:
    seed = heuristic_seed(_info(file_size=20 * GB), _host(ram_mib=1024), None)
    assert seed.ctx == 2048


def test_ctx_below_floor_is_kept() -> None:
    seed = heuristic_seed(_info(file_size=20 * GB), _host(), _gpu(1024), ctx=1024)
    assert seed.ctx == 1024


def test_all_blocks_fit_but_not_output_layer() -> None:
    info = _info(file_size=33 * GB)
    kv = kv_cache_bytes(info, 2048)
    budget = math.ceil((info.file_size * 32 / 33 + kv) * selector.HEADROOM_FACTOR / MIB + selector.OVERHEAD_MIB)
    assert _need(info, 2048) > budget
    seed = heuristic_seed(info, _host(), _gpu(budget), ctx=2048)
    assert seed.ngl == info.block_count


@pytest.mark.parametrize("cores", [0, -3])
def test_threads_at_least_one(cores: int) -> None:
    assert heuristic_seed(_info(), _host(cores=cores), None).threads == 1


def test_odd_inputs_do_not_raise() -> None:
    odd = _info(file_size=0, block_count=0, head_count=1, head_count_kv=1, embedding_length=1)
    seed = heuristic_seed(odd, _host(), _gpu(1))
    assert seed.ngl >= 0
    full = heuristic_seed(odd, _host(), _gpu(24 * 1024))
    assert full.ngl == 1
    cpu = heuristic_seed(odd, _host(), None)
    assert cpu.ngl == 0


def test_deterministic() -> None:
    args = (_info(file_size=9 * GB), _host(), _gpu(6 * 1024))
    assert heuristic_seed(*args) == heuristic_seed(*args)


def test_seed_to_args_roundtrip_flags() -> None:
    plan = SeedPlan(ngl=33, ctx=8192, flash_attn=True, threads=8, batch=2048, ubatch=512, cache_type="f16")
    assert seed_to_args(plan) == [
        "-ngl",
        "33",
        "-fa",
        "on",
        "-t",
        "8",
        "-b",
        "2048",
        "-ub",
        "512",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
    ]


def test_seed_to_args_flash_attn_off() -> None:
    plan = SeedPlan(ngl=0, ctx=2048, flash_attn=False, threads=1, batch=2048, ubatch=512, cache_type="q8_0")
    args = seed_to_args(plan)
    assert args[args.index("-fa") + 1] == "off"
    assert args[args.index("-ctk") + 1] == args[args.index("-ctv") + 1] == "q8_0"
    assert "-c" not in args  # ctx is passed separately by the caller
