"""Minimal GGUF metadata reader (no dependencies).

Only the header's key/value metadata is parsed (little-endian GGUF v2/v3); tensor info is never read. Arrays are
skipped without materialising them, except short integer arrays (per-layer head counts), which are kept.

Metadata keys are looked up under the architecture prefix, e.g. ``llama.block_count`` or ``gemma4.block_count``.
Every malformed input raises :class:`GgufError`.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = frozenset({2, 3})
EMBEDDING_ARCHITECTURES = frozenset({"bert", "nomic-bert", "modern-bert", "jina-bert-v2"})

_T_STRING = 8
_T_ARRAY = 9
_MAX_DEPTH = 4
_MAX_KEPT_ARRAY = 4096  # per-layer arrays are far shorter than this
_CHUNK = 1 << 20
_U64 = struct.Struct("<Q")
_MIN_KV_BYTES = 12  # key length (8) + value type (4)

# value type id -> struct format of the fixed-size scalar
_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_SCALAR_SIZES = {t: struct.calcsize(fmt) for t, fmt in _SCALAR_FORMATS.items()}
_INT_TYPES = frozenset({0, 1, 2, 3, 4, 5, 10, 11})

_REQUIRED_SUFFIXES = (
    "block_count",
    "context_length",
    "embedding_length",
    "attention.head_count",
)


class GgufError(ValueError):
    """Raised when a file is not a readable GGUF file."""


@dataclass(frozen=True)
class GgufInfo:
    """Facts about a GGUF model needed to plan a llama-server launch.

    ``head_count`` and ``head_count_kv`` are single ints. Newer architectures store them as per-layer arrays; in that
    case the maximum is used, which over-estimates (never under-estimates) the KV cache.
    """

    architecture: str
    block_count: int
    context_length: int
    embedding_length: int
    head_count: int
    head_count_kv: int
    key_length: int | None
    file_type: int | None
    is_embedding: bool
    file_size: int


class _Reader:
    """Bounded little-endian reader over a binary file."""

    def __init__(self, fh: IO[bytes], size: int) -> None:
        self._fh = fh
        self.size = size

    def remaining(self) -> int:
        """Return the number of bytes left in the file."""
        return self.size - self._fh.tell()

    def read(self, n: int) -> bytes:
        """Read exactly ``n`` bytes.

        Args:
            n: Byte count.

        Returns:
            The bytes read.

        Raises:
            GgufError: If fewer than ``n`` bytes remain.
        """
        if n < 0 or n > self.remaining():
            raise GgufError("truncated GGUF file")
        return self._fh.read(n)

    def skip(self, n: int) -> None:
        """Skip ``n`` bytes.

        Args:
            n: Byte count.

        Raises:
            GgufError: If fewer than ``n`` bytes remain.
        """
        if n < 0 or n > self.remaining():
            raise GgufError("truncated GGUF file")
        self._fh.seek(n, 1)

    def unpack(self, fmt: str) -> Any:
        """Unpack one value of struct format ``fmt``.

        Args:
            fmt: A little-endian struct format with a single item.

        Returns:
            The unpacked value.
        """
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        """Read a length-prefixed UTF-8 string.

        Returns:
            The decoded string.

        Raises:
            GgufError: On absurd length, truncation or invalid UTF-8.
        """
        raw = self.read(self.unpack("<Q"))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufError("invalid UTF-8 in GGUF string") from exc

    def skip_strings(self, count: int) -> None:
        """Skip ``count`` consecutive length-prefixed strings using buffered reads (fast for token tables).

        Args:
            count: Number of strings to skip.

        Raises:
            GgufError: If a string runs past the end of the file.
        """
        # The caller has checked count * 8 <= remaining, so every refill holds at least one length prefix.
        fh = self._fh
        buf = b""
        off = 0
        done = 0
        while done < count:
            if len(buf) - off < _U64.size:
                fh.seek(off - len(buf), 1)  # drop the unread tail of the buffer
                buf = fh.read(_CHUNK)
                off = 0
            end = off + _U64.size + _U64.unpack_from(buf, off)[0]
            done += 1
            if end <= len(buf):
                off = end
                continue
            overshoot = end - len(buf)  # the string extends past the buffer
            if overshoot > self.remaining():
                raise GgufError("truncated GGUF file")
            fh.seek(overshoot, 1)
            buf = b""
            off = 0
        fh.seek(off - len(buf), 1)


def _skip_array(rd: _Reader, depth: int) -> None:
    """Skip an array whose element type and count are next in the stream."""
    _skip_elements(rd, rd.unpack("<I"), rd.unpack("<Q"), depth)


def _skip_elements(rd: _Reader, elem_type: int, count: int, depth: int) -> None:
    """Skip ``count`` array elements of ``elem_type``."""
    if depth > _MAX_DEPTH:
        raise GgufError("GGUF arrays nested too deeply")
    if elem_type in _SCALAR_SIZES:
        rd.skip(count * _SCALAR_SIZES[elem_type])
        return
    if elem_type not in {_T_STRING, _T_ARRAY}:
        raise GgufError(f"unknown GGUF value type {elem_type}")
    if count > rd.remaining() // 8:  # every string / nested array costs at least 8 bytes
        raise GgufError("GGUF array count exceeds file size")
    if elem_type == _T_STRING:
        rd.skip_strings(count)
        return
    for _ in range(count):
        _skip_array(rd, depth + 1)


def _read_value(rd: _Reader, vtype: int) -> Any:
    """Read one metadata value; arrays come back as short numeric lists, otherwise ``None`` (skipped)."""
    if vtype in _SCALAR_FORMATS:
        return rd.unpack(_SCALAR_FORMATS[vtype])
    if vtype == _T_STRING:
        return rd.string()
    if vtype != _T_ARRAY:
        raise GgufError(f"unknown GGUF value type {vtype}")
    elem_type = rd.unpack("<I")
    count = rd.unpack("<Q")
    if elem_type in _INT_TYPES and count <= _MAX_KEPT_ARRAY:
        if count * _SCALAR_SIZES[elem_type] > rd.remaining():
            raise GgufError("GGUF array count exceeds file size")
        fmt = _SCALAR_FORMATS[elem_type]
        return [rd.unpack(fmt) for _ in range(count)]
    _skip_elements(rd, elem_type, count, 1)
    return None


def _as_int(info: dict[str, Any], key: str) -> int | None:
    """Return ``info[key]`` as an int (max of a non-empty int array), or ``None`` when absent."""
    if key not in info:
        return None
    value = info[key]
    if isinstance(value, list) and value:
        value = max(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GgufError(f"GGUF key {key!r} is not an integer")
    return value


def _read_metadata(rd: _Reader) -> dict[str, Any]:
    """Read the header and return every metadata key/value pair (large arrays are skipped, not kept).

    The whole KV section is scanned: ``general.file_type`` follows the tokenizer tables in real files. String
    arrays are skipped with buffered reads, so a 260k-token vocabulary costs roughly 0.1 s.
    """
    if rd.read(4) != GGUF_MAGIC:
        raise GgufError("bad GGUF magic")
    version = rd.unpack("<I")
    if version not in SUPPORTED_VERSIONS:
        raise GgufError(f"unsupported GGUF version {version}")
    rd.unpack("<Q")  # tensor count: tensor info is not read
    kv_count = rd.unpack("<Q")
    if kv_count > rd.remaining() // _MIN_KV_BYTES:
        raise GgufError("GGUF key/value count exceeds file size")
    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        key = rd.string()
        kv[key] = _read_value(rd, rd.unpack("<I"))
    return kv


def _build_info(kv: dict[str, Any], file_size: int) -> GgufInfo:
    """Turn raw metadata into a :class:`GgufInfo`."""
    arch = kv.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GgufError("GGUF is missing general.architecture")
    ints: dict[str, int] = {}
    for suffix in _REQUIRED_SUFFIXES:
        value = _as_int(kv, f"{arch}.{suffix}")
        if value is None:
            raise GgufError(f"GGUF is missing {arch}.{suffix}")
        ints[suffix] = value
    if ints["attention.head_count"] < 1:
        raise GgufError(f"GGUF {arch}.attention.head_count must be positive")
    head_count = ints["attention.head_count"]
    head_count_kv = _as_int(kv, f"{arch}.attention.head_count_kv")
    return GgufInfo(
        architecture=arch,
        block_count=ints["block_count"],
        context_length=ints["context_length"],
        embedding_length=ints["embedding_length"],
        head_count=head_count,
        head_count_kv=head_count_kv if head_count_kv and head_count_kv > 0 else head_count,
        key_length=_as_int(kv, f"{arch}.attention.key_length"),
        file_type=_as_int(kv, "general.file_type"),
        is_embedding=f"{arch}.pooling_type" in kv or arch in EMBEDDING_ARCHITECTURES,
        file_size=file_size,
    )


def read_gguf(path: Path) -> GgufInfo:
    """Read the metadata header of a GGUF file.

    Args:
        path: Path to the GGUF file (opened read-only).

    Returns:
        The parsed model facts.

    Raises:
        GgufError: If the file is missing, malformed, truncated or lacks required keys.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            kv = _read_metadata(_Reader(fh, size))
    except OSError as exc:
        raise GgufError(f"cannot read {path}: {exc}") from exc
    return _build_info(kv, size)


def head_dim(info: GgufInfo) -> int:
    """Return the attention head dimension.

    Args:
        info: Parsed GGUF facts.

    Returns:
        ``key_length`` when present, else ``embedding_length // head_count``.
    """
    return info.key_length or info.embedding_length // info.head_count


def kv_cache_bytes(info: GgufInfo, ctx: int, bytes_per_elem: float = 2.0) -> int:
    """Estimate the KV cache size.

    Args:
        info: Parsed GGUF facts.
        ctx: Context length in tokens.
        bytes_per_elem: Bytes per cached element (2.0 for f16).

    Returns:
        ``block_count * 2 * ctx * head_count_kv * head_dim * bytes_per_elem`` as an int.
    """
    return int(info.block_count * 2 * ctx * info.head_count_kv * head_dim(info) * bytes_per_elem)
