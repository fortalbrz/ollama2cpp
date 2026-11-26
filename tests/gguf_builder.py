"""Build tiny synthetic GGUF files for tests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"

T_U32 = 4
T_F32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_U64 = 10

_U32_LIMIT = 2**32


def pack_string(value: str) -> bytes:
    """Encode a GGUF string (u64 length prefix + UTF-8 bytes).

    Args:
        value: Text to encode.

    Returns:
        The encoded bytes.
    """
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(value: Any) -> tuple[int, bytes]:
    """Encode ``value`` as a GGUF ``(type id, payload)`` pair."""
    if isinstance(value, bool):
        return T_BOOL, struct.pack("<B", int(value))
    if isinstance(value, int):
        if 0 <= value < _U32_LIMIT:
            return T_U32, struct.pack("<I", value)
        return T_U64, struct.pack("<Q", value)
    if isinstance(value, float):
        return T_F32, struct.pack("<f", value)
    if isinstance(value, str):
        return T_STRING, pack_string(value)
    if isinstance(value, (list, tuple)):
        items = [_pack_value(v) for v in value]
        elem_type = items[0][0] if items else T_U32
        body = b"".join(payload for _, payload in items)
        return T_ARRAY, struct.pack("<IQ", elem_type, len(items)) + body
    msg = f"unsupported test value: {value!r}"
    raise TypeError(msg)


def pack_kv(key: str, value: Any) -> bytes:
    """Encode one metadata key/value pair.

    Args:
        key: Metadata key.
        value: bool, int, float, str, or a list of those.

    Returns:
        The encoded bytes.
    """
    vtype, payload = _pack_value(value)
    return pack_string(key) + struct.pack("<I", vtype) + payload


def pack_header(
    kv_blobs: list[bytes], *, version: int = 3, tensor_count: int = 0, kv_count: int | None = None
) -> bytes:
    """Assemble a GGUF file image from already encoded key/value blobs.

    Args:
        kv_blobs: Encoded key/value pairs, in file order.
        version: GGUF version field.
        tensor_count: Declared tensor count (no tensor info is written).
        kv_count: Declared KV count; defaults to ``len(kv_blobs)``.

    Returns:
        The file image.
    """
    count = len(kv_blobs) if kv_count is None else kv_count
    return MAGIC + struct.pack("<IQQ", version, tensor_count, count) + b"".join(kv_blobs)


def write_gguf(
    path: Path,
    *,
    arch: str = "llama",
    n_layer: int = 32,
    n_ctx: int = 4096,
    n_embd: int = 4096,
    n_head: int | list[int] = 32,
    n_head_kv: int | list[int] | None = 8,
    pooling: int | None = None,
    extra_kv: dict[str, Any] | None = None,
    tokens: int = 0,
    tokens_first: bool = False,
) -> Path:
    """Write a valid minimal GGUF v3 file with metadata only.

    Args:
        path: Destination file.
        arch: ``general.architecture`` and key prefix.
        n_layer: ``{arch}.block_count``.
        n_ctx: ``{arch}.context_length``.
        n_embd: ``{arch}.embedding_length``.
        n_head: ``{arch}.attention.head_count`` (int or per-layer list).
        n_head_kv: ``{arch}.attention.head_count_kv`` (int, per-layer list, or ``None`` to omit).
        pooling: ``{arch}.pooling_type`` when not ``None``.
        extra_kv: Additional keys, written after the core keys.
        tokens: Size of a ``tokenizer.ggml.tokens`` string array (0 for none).
        tokens_first: Write the token array before the core keys (forces a real array skip).

    Returns:
        ``path``.
    """
    blobs: list[bytes] = []
    token_blob = pack_kv("tokenizer.ggml.tokens", [f"tok{i}" for i in range(tokens)]) if tokens else b""
    if token_blob and tokens_first:
        blobs.append(token_blob)
    blobs.append(pack_kv("general.architecture", arch))
    blobs.append(pack_kv(f"{arch}.block_count", n_layer))
    blobs.append(pack_kv(f"{arch}.context_length", n_ctx))
    blobs.append(pack_kv(f"{arch}.embedding_length", n_embd))
    blobs.append(pack_kv(f"{arch}.attention.head_count", n_head))
    if n_head_kv is not None:
        blobs.append(pack_kv(f"{arch}.attention.head_count_kv", n_head_kv))
    if pooling is not None:
        blobs.append(pack_kv(f"{arch}.pooling_type", pooling))
    blobs.extend(pack_kv(k, v) for k, v in (extra_kv or {}).items())
    if token_blob and not tokens_first:
        blobs.append(token_blob)
    path.write_bytes(pack_header(blobs))
    return path
