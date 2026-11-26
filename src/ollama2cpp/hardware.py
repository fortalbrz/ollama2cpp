"""Hardware probe: GPU devices (via ``llama-server --list-devices`` or ``nvidia-smi``) and host RAM / cores."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil

from ollama2cpp import proc

__all__ = [
    "Device",
    "HostInfo",
    "best_device",
    "detect_devices",
    "fingerprint",
    "host_info",
    "is_integrated",
    "parse_list_devices",
    "parse_nvidia_smi",
]

log = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 30.0
_BYTES_PER_MIB = 1024 * 1024
_CPU_ID_PREFIX = "CPU"

# Conservative integrated-GPU name heuristics (shared system memory, not VRAM). Apple/Metal is deliberately absent.
_INTEGRATED_NAME_PATTERNS = (
    # AMD: "Radeon(TM) Graphics", "Radeon Graphics", "Radeon(TM) Vega 8 Graphics", "Radeon 780M Graphics".
    re.compile(r"\bRadeon(?:\(TM\))?\s+(?:(?:Vega\s+\d+|\d{3,4}M)\s+)?Graphics\b", re.IGNORECASE),
    # Intel: "UHD Graphics", "Iris(R) Xe Graphics", "HD Graphics".
    re.compile(r"\bIntel\b.*\b(?:UHD|Iris|HD)\b.*\bGraphics\b", re.IGNORECASE),
    # Intel Core Ultra: "Intel(R) Graphics", "Intel(R) Arc(TM) Graphics", "Intel(R) Arc(TM) 140V GPU" (no A/B model).
    re.compile(r"\bIntel(?:\(R\))?\s+Graphics\b", re.IGNORECASE),
    re.compile(r"\bArc(?:\(TM\))?\s+(?:Graphics\b|\d{3}V\b)", re.IGNORECASE),
)
# Discrete Intel Arc names carry a model number: "Arc(TM) A770", "Arc(TM) B580", "Arc(TM) A370M", "Arc(TM) Pro A60".
_DISCRETE_NAME_PATTERN = re.compile(r"\bArc(?:\(TM\))?\s+(?:Pro\s+)?[A-Z]\d{2,3}M?\b", re.IGNORECASE)

# "  Vulkan0: NVIDIA GeForce RTX 5060 Ti (7891 MiB, 5198 MiB free)"; the free figure is optional.
_DEVICE_LINE = re.compile(
    r"^\s*(?P<id>[A-Za-z][A-Za-z0-9_]*):\s+(?P<name>.+?)\s+\((?P<total>\d+) MiB(?:, (?P<free>\d+) MiB free)?\)\s*$"
)


@dataclass(frozen=True)
class Device:
    """A compute device that ``llama-server --device`` can target.

    Attributes:
        id: Device id exactly as ``--device`` accepts it (e.g. ``Vulkan0``, ``CUDA0``).
        name: Human-readable device name.
        total_mib: Total device memory in MiB.
        free_mib: Free device memory in MiB at probe time (0 when unknown).
    """

    id: str
    name: str
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class HostInfo:
    """Snapshot of the host's compute resources.

    Attributes:
        devices: Detected GPU devices (empty on a CPU-only host).
        ram_total_mib: Total system RAM in MiB.
        ram_free_mib: Available system RAM in MiB.
        physical_cores: Physical CPU cores (falls back to the logical count when unknown).
        logical_cores: Logical CPU cores.
    """

    devices: tuple[Device, ...]
    ram_total_mib: int
    ram_free_mib: int
    physical_cores: int
    logical_cores: int


def _is_cpu(device: Device) -> bool:
    """Tell whether a device is the CPU pseudo-device.

    Args:
        device: Device to test.

    Returns:
        bool: True when the id denotes the CPU backend.
    """
    return device.id.upper().startswith(_CPU_ID_PREFIX)


def is_integrated(device: Device) -> bool:
    """Guess whether a device is an integrated GPU that borrows system RAM instead of having its own VRAM.

    The guess is name-based and conservative. It flags AMD ``Radeon(TM) Graphics`` / ``Radeon Graphics`` /
    ``Radeon(TM) Vega N Graphics`` / ``Radeon 780M Graphics`` style names (no discrete model number) and Intel
    ``UHD`` / ``Iris`` / ``HD`` Graphics, plus the Core Ultra iGPU names (``Intel(R) Graphics``,
    ``Arc(TM) Graphics``, ``Arc(TM) 140V GPU``). Intel Arc names with a model number (``A770``, ``B580``,
    ``Pro A60``), NVIDIA, discrete Radeon (``RX ...``) and Apple/Metal GPUs are never flagged.
    ``llama-server --list-devices`` prints no explicit integrated/UMA marker, so none is read.

    Args:
        device: Device to classify.

    Returns:
        bool: True when the name looks like an integrated GPU.
    """
    if _DISCRETE_NAME_PATTERN.search(device.name):
        return False
    return any(pattern.search(device.name) for pattern in _INTEGRATED_NAME_PATTERNS)


def parse_list_devices(text: str) -> list[Device]:
    """Parse the output of ``llama-server --list-devices``.

    Backend-loading noise and the ``Available devices:`` header are ignored, as is the CPU pseudo-device.
    Never raises: unparseable text yields ``[]``.

    Args:
        text: Combined stdout and stderr of the command.

    Returns:
        list[Device]: GPU devices in the order printed.
    """
    devices: list[Device] = []
    for line in text.splitlines():
        match = _DEVICE_LINE.match(line)
        if match is None:
            continue
        free = match["free"]
        device = Device(match["id"], match["name"], int(match["total"]), int(free) if free else 0)
        if not _is_cpu(device):
            devices.append(device)
    return devices


def parse_nvidia_smi(text: str) -> list[Device]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`` output.

    Devices get ids ``CUDA0``, ``CUDA1``, ... by row position (best-effort id, only used when ``--list-devices``
    finds no GPU). Never raises: unparseable rows are skipped.

    Args:
        text: Command output, one ``name, total, free`` row per GPU.

    Returns:
        list[Device]: One device per valid row.
    """
    devices: list[Device] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        devices.append(Device(f"CUDA{len(devices)}", parts[0], int(parts[1]), int(parts[2])))
    return devices


def _list_devices(llama_server: Path) -> list[Device]:
    """Ask ``llama-server`` which devices it sees.

    Args:
        llama_server: Path to the ``llama-server`` executable.

    Returns:
        list[Device]: Devices found, ``[]`` when the command fails.
    """
    result = proc.run([str(llama_server), "--list-devices"], timeout=PROBE_TIMEOUT_S)
    if result.returncode != 0:
        log.debug("llama-server --list-devices failed (rc=%s)", result.returncode)
        return []
    return parse_list_devices(result.stdout + "\n" + result.stderr)


def _nvidia_smi() -> list[Device]:
    """Query ``nvidia-smi`` for NVIDIA GPUs.

    Returns:
        list[Device]: Devices found, ``[]`` when the tool is missing or fails.
    """
    argv = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
    result = proc.run(argv, timeout=PROBE_TIMEOUT_S)
    if result.returncode != 0:
        log.debug("nvidia-smi failed (rc=%s)", result.returncode)
        return []
    return parse_nvidia_smi(result.stdout)


def detect_devices(llama_server: Path | None) -> list[Device]:
    """Detect GPU devices, preferring ``llama-server --list-devices`` over ``nvidia-smi``.

    Args:
        llama_server: Path to ``llama-server``, or None when it is not installed.

    Returns:
        list[Device]: Detected GPUs; ``[]`` on a CPU-only host or when no probe works.
    """
    devices = _list_devices(llama_server) if llama_server is not None else []
    return devices or _nvidia_smi()


def best_device(devices: Sequence[Device], override: str | None = None) -> Device | None:
    """Choose the device to run on.

    Args:
        devices: Candidate devices.
        override: A device id that wins when it matches a non-CPU candidate (even an integrated GPU).

    Returns:
        Device | None: The override match, else the discrete GPU with the most memory. An integrated GPU is chosen
            (largest first) only when it is the sole kind of GPU; None when there is no GPU.
    """
    gpus = [d for d in devices if not _is_cpu(d)]
    if override is not None:
        for device in gpus:
            if device.id == override:
                return device
    return max(gpus, key=lambda d: (not is_integrated(d), d.total_mib), default=None)


def host_info(llama_server: Path | None) -> HostInfo:
    """Probe the host's GPUs, RAM and CPU cores.

    Args:
        llama_server: Path to ``llama-server``, or None when it is not installed.

    Returns:
        HostInfo: Snapshot of the host.
    """
    memory = psutil.virtual_memory()
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    return HostInfo(
        devices=tuple(detect_devices(llama_server)),
        ram_total_mib=memory.total // _BYTES_PER_MIB,
        ram_free_mib=memory.available // _BYTES_PER_MIB,
        physical_cores=physical,
        logical_cores=logical,
    )


def fingerprint(host: HostInfo, device: Device | None, llama_build: str) -> dict[str, str]:
    """Summarise the hardware for cache keys (tuned-setting invalidation).

    Args:
        host: Host snapshot.
        device: Chosen GPU, or None for CPU-only.
        llama_build: ``llama-server`` build identifier.

    Returns:
        dict[str, str]: Exactly the keys ``gpu``, ``vram_mib``, ``ram_mib``, ``cores`` and ``llama_build``.
    """
    return {
        "gpu": device.name if device is not None else "cpu",
        "vram_mib": str(device.total_mib if device is not None else 0),
        "ram_mib": str(host.ram_total_mib),
        "cores": str(host.physical_cores),
        "llama_build": llama_build,
    }
