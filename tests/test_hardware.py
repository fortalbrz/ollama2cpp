"""Tests for the hardware probe."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

from ollama2cpp import hardware
from ollama2cpp.hardware import (
    Device,
    HostInfo,
    best_device,
    detect_devices,
    fingerprint,
    host_info,
    is_integrated,
    parse_list_devices,
    parse_nvidia_smi,
)
from ollama2cpp.proc import ProcResult

DATA = Path(__file__).parent / "data"
VULKAN_SAMPLE = (DATA / "list_devices_vulkan.txt").read_text(encoding="utf-8")
NVIDIA_SAMPLE = (DATA / "nvidia_smi.csv").read_text(encoding="utf-8")

CUDA_SAMPLE = """\
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
load_backend: loaded CUDA backend from /opt/llama/libggml-cuda.so
Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12288 MiB, 12000 MiB free)
"""

CPU_ONLY_SAMPLE = """\
load_backend: loaded CPU backend from /opt/llama/libggml-cpu.so
Available devices:
"""

CPU_ENTRY_SAMPLE = """\
Available devices:
  CPU: AMD Ryzen 9 (65536 MiB, 60000 MiB free)
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)
"""

IGPU = Device("Vulkan1", "AMD Radeon(TM) Graphics", 32568, 30939)
DGPU = Device("Vulkan0", "NVIDIA GeForce RTX 5060 Ti", 7891, 5198)
BIG_CPU_LIKE = Device("CPU", "cpu", 65536, 60000)

ARGV_LD = "--list-devices"


def _fake_run(outputs: dict[str, ProcResult], calls: list[list[str]] | None = None):
    """Build a ``proc.run`` stand-in keyed on a marker argument."""

    def run(argv: Sequence[str], *, timeout: float | None = None) -> ProcResult:
        del timeout
        if calls is not None:
            calls.append(list(argv))
        for marker, result in outputs.items():
            if marker in argv[0] or marker in argv:
                return result
        return ProcResult(127, "", "missing")

    return run


def test_parse_list_devices_real_sample() -> None:
    devs = parse_list_devices(VULKAN_SAMPLE)
    assert [d.id for d in devs] == ["Vulkan0", "Vulkan1"]
    assert devs[0].id.startswith("Vulkan")
    assert devs[0].name == "NVIDIA GeForce RTX 5060 Ti"
    assert devs[0].total_mib == 7891
    assert devs[0].free_mib == 5198
    assert devs[1].name == "AMD Radeon(TM) Graphics"
    assert devs[1].total_mib == 32568


def test_parse_list_devices_cuda_style_skips_noise() -> None:
    devs = parse_list_devices(CUDA_SAMPLE)
    assert [(d.id, d.total_mib, d.free_mib) for d in devs] == [("CUDA0", 24564, 23000), ("CUDA1", 12288, 12000)]


def test_parse_list_devices_ignores_cpu_entry() -> None:
    assert [d.id for d in parse_list_devices(CPU_ENTRY_SAMPLE)] == ["CUDA0"]


def test_parse_list_devices_cpu_only_and_garbage() -> None:
    assert parse_list_devices(CPU_ONLY_SAMPLE) == []
    assert parse_list_devices("") == []
    assert parse_list_devices("\x00\xff garbage ((( MiB\n  : ()\n") == []


def test_parse_list_devices_without_free_figure() -> None:
    devs = parse_list_devices("Available devices:\n  Metal0: Apple M2 (16384 MiB)\n")
    assert devs == [Device("Metal0", "Apple M2", 16384, 0)]


def test_parse_nvidia_smi() -> None:
    assert parse_nvidia_smi("NVIDIA GeForce RTX 5060 Ti, 8151, 659\n")[0].total_mib == 8151
    devs = parse_nvidia_smi(NVIDIA_SAMPLE)
    assert devs[0].id == "CUDA0"
    assert devs[0].name == "NVIDIA GeForce RTX 5060 Ti"
    assert devs[0].total_mib == 8151
    assert devs[0].free_mib > 0


def test_parse_nvidia_smi_multi_and_garbage() -> None:
    devs = parse_nvidia_smi("A, 100, 50\nB, 200, 150\n")
    assert [(d.id, d.name, d.total_mib, d.free_mib) for d in devs] == [
        ("CUDA0", "A", 100, 50),
        ("CUDA1", "B", 200, 150),
    ]
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate with the driver\n") == []
    assert parse_nvidia_smi("X, [N/A], [N/A]\n, 5, 5\nonly one\n") == []


def test_parse_nvidia_smi_name_with_comma() -> None:
    devs = parse_nvidia_smi("Weird, Name, 100, 40\n")
    assert devs == [Device("CUDA0", "Weird, Name", 100, 40)]


def test_best_device_prefers_discrete_over_larger_igpu() -> None:
    assert best_device([DGPU, IGPU]) == DGPU
    assert best_device([IGPU, DGPU]) == DGPU


def test_best_device_real_fixture_picks_rtx_not_igpu() -> None:
    chosen = best_device(parse_list_devices(VULKAN_SAMPLE))
    assert chosen is not None
    assert chosen.id == "Vulkan0"
    assert chosen.name == "NVIDIA GeForce RTX 5060 Ti"


def test_best_device_igpu_only_returns_igpu() -> None:
    assert best_device([IGPU]) == IGPU
    assert best_device([BIG_CPU_LIKE, IGPU]) == IGPU


def test_best_device_largest_among_discrete() -> None:
    small = Device("CUDA1", "NVIDIA GeForce RTX 3060", 12288, 12000)
    big = Device("CUDA0", "NVIDIA GeForce RTX 4090", 24564, 23000)
    assert best_device([small, IGPU, big]) == big


def test_best_device_override_honours_igpu() -> None:
    assert best_device([DGPU, IGPU], override="Vulkan1") == IGPU
    assert best_device([IGPU, DGPU], override="Vulkan0") == DGPU


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("AMD Radeon(TM) Graphics", True),
        ("AMD Radeon Graphics", True),
        ("AMD Radeon(TM) Vega 8 Graphics", True),
        ("AMD Radeon 780M Graphics", True),
        ("Intel(R) UHD Graphics 770", True),
        ("Intel(R) Iris(R) Xe Graphics", True),
        ("Intel(R) HD Graphics 630", True),
        ("Intel(R) Arc(TM) A770 Graphics", False),
        ("NVIDIA GeForce RTX 5060 Ti", False),
        ("AMD Radeon RX 7900 XTX", False),
        ("AMD Radeon(TM) RX 6600 Graphics", False),
        ("Apple M2 Pro", False),
    ],
)
def test_is_integrated_heuristic(name: str, expected: bool) -> None:
    assert is_integrated(Device("X0", name, 1000, 500)) is expected


def test_best_device_override_missing_falls_back_to_discrete() -> None:
    assert best_device([DGPU, IGPU], override="CUDA9") == DGPU


def test_best_device_never_picks_cpu() -> None:
    assert best_device([BIG_CPU_LIKE, DGPU]) == DGPU
    assert best_device([BIG_CPU_LIKE]) is None
    assert best_device([BIG_CPU_LIKE, DGPU], override="CPU") == DGPU


def test_best_device_empty_is_none() -> None:
    assert best_device([]) is None
    assert best_device([], override="Vulkan0") is None


def test_detect_uses_list_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        hardware.proc,
        "run",
        _fake_run({"llama-server": ProcResult(0, VULKAN_SAMPLE, "")}, calls),
    )
    devs = detect_devices(Path("llama-server"))
    assert [d.id for d in devs] == ["Vulkan0", "Vulkan1"]
    assert calls == [[str(Path("llama-server")), ARGV_LD]]


def test_detect_reads_stderr_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.proc, "run", _fake_run({"llama-server": ProcResult(0, "", CUDA_SAMPLE)}))
    assert [d.id for d in detect_devices(Path("llama-server"))] == ["CUDA0", "CUDA1"]


def test_detect_falls_back_to_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware.proc,
        "run",
        _fake_run({"llama-server": ProcResult(0, CPU_ONLY_SAMPLE, ""), "nvidia-smi": ProcResult(0, NVIDIA_SAMPLE, "")}),
    )
    devs = detect_devices(Path("llama-server"))
    assert [d.id for d in devs] == ["CUDA0"]
    assert devs[0].total_mib == 8151


def test_detect_without_llama_server_uses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(hardware.proc, "run", _fake_run({"nvidia-smi": ProcResult(0, NVIDIA_SAMPLE, "")}, calls))
    assert [d.id for d in detect_devices(None)] == ["CUDA0"]
    assert len(calls) == 1
    assert calls[0][0] == "nvidia-smi"


def test_detect_failed_list_devices_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware.proc,
        "run",
        _fake_run(
            {"llama-server": ProcResult(1, VULKAN_SAMPLE, "boom"), "nvidia-smi": ProcResult(0, NVIDIA_SAMPLE, "")}
        ),
    )
    assert [d.id for d in detect_devices(Path("llama-server"))] == ["CUDA0"]


def test_no_gpu_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.proc, "run", _fake_run({}))
    devs = detect_devices(Path("llama-server"))
    assert devs == []
    assert best_device(devs) is None


def test_nvidia_smi_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware.proc,
        "run",
        _fake_run({"llama-server": ProcResult(0, CPU_ONLY_SAMPLE, ""), "nvidia-smi": ProcResult(9, "", "no driver")}),
    )
    assert detect_devices(Path("llama-server")) == []


def test_host_info_uses_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_devices", lambda _server: [DGPU])
    info = host_info(None)
    assert info.devices == (DGPU,)
    assert info.ram_total_mib > 0
    assert 0 < info.ram_free_mib <= info.ram_total_mib
    assert info.logical_cores >= 1
    assert info.physical_cores >= 1


def test_host_info_physical_cores_falls_back_to_logical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_devices", lambda _server: [])
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda logical=True: 8 if logical else None)
    info = host_info(None)
    assert info.physical_cores == info.logical_cores == 8
    assert info.devices == ()


def test_host_info_cpu_count_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_devices", lambda _server: [])
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda **_kw: None)
    info = host_info(None)
    assert info.physical_cores == info.logical_cores == 1


def test_dataclasses_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DGPU.total_mib = 1  # type: ignore[misc]
    info = HostInfo((DGPU,), 1000, 500, 4, 8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.ram_total_mib = 1  # type: ignore[misc]


def test_fingerprint_stable_keys() -> None:
    info = HostInfo((DGPU,), 32000, 16000, 8, 16)
    fp = fingerprint(info, DGPU, "b1234")
    assert set(fp) == {"gpu", "vram_mib", "ram_mib", "cores", "llama_build"}
    assert fp == {
        "gpu": "NVIDIA GeForce RTX 5060 Ti",
        "vram_mib": "7891",
        "ram_mib": "32000",
        "cores": "8",
        "llama_build": "b1234",
    }
    assert all(isinstance(v, str) for v in fp.values())


def test_fingerprint_without_gpu_has_same_keys() -> None:
    info = HostInfo((), 32000, 16000, 8, 16)
    fp = fingerprint(info, None, "b1")
    assert set(fp) == {"gpu", "vram_mib", "ram_mib", "cores", "llama_build"}
    assert fp["gpu"] == "cpu"
    assert fp["vram_mib"] == "0"
