"""Tests for the GGUF header reader."""

from __future__ import annotations

import dataclasses
import struct
import time
from pathlib import Path

import pytest

from ollama2cpp import gguf
from ollama2cpp.gguf import GgufError, GgufInfo, head_dim, kv_cache_bytes, read_gguf
from tests.gguf_builder import T_ARRAY, T_STRING, pack_header, pack_kv, pack_string, write_gguf


def _info(**over: object) -> GgufInfo:
    base: dict[str, object] = {
        "architecture": "llama",
        "block_count": 32,
        "context_length": 8192,
        "embedding_length": 4096,
        "head_count": 32,
        "head_count_kv": 8,
        "key_length": None,
        "file_type": None,
        "is_embedding": False,
        "file_size": 1,
    }
    base.update(over)
    return GgufInfo(**base)  # type: ignore[arg-type]


def _raw(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "raw.gguf"
    path.write_bytes(data)
    return path


def _core(arch: str = "llama") -> list[bytes]:
    return [
        pack_kv("general.architecture", arch),
        pack_kv(f"{arch}.block_count", 4),
        pack_kv(f"{arch}.context_length", 2048),
        pack_kv(f"{arch}.embedding_length", 256),
        pack_kv(f"{arch}.attention.head_count", 8),
    ]


def test_reads_basic_fields(tmp_path: Path) -> None:
    p = write_gguf(tmp_path / "m.gguf")
    info = read_gguf(p)
    assert (info.architecture, info.block_count, info.head_count_kv) == ("llama", 32, 8)
    assert (info.context_length, info.embedding_length, info.head_count) == (4096, 4096, 32)
    assert info.file_size == p.stat().st_size
    assert info.is_embedding is False
    assert info.key_length is None
    assert info.file_type is None


def test_info_is_frozen(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.block_count = 1  # type: ignore[misc]


def test_embedding_detected_by_pooling(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", arch="foo", pooling=1))
    assert info.is_embedding is True


@pytest.mark.parametrize("arch", ["bert", "nomic-bert", "modern-bert", "jina-bert-v2"])
def test_embedding_detected_by_arch(tmp_path: Path, arch: str) -> None:
    assert read_gguf(write_gguf(tmp_path / "m.gguf", arch=arch)).is_embedding is True


def test_optional_keys(tmp_path: Path) -> None:
    extra = {"llama.attention.key_length": 128, "general.file_type": 15}
    info = read_gguf(write_gguf(tmp_path / "m.gguf", extra_kv=extra))
    assert (info.key_length, info.file_type) == (128, 15)


@pytest.mark.parametrize("tokens_first", [True, False])
def test_skips_large_string_array(tmp_path: Path, tokens_first: bool) -> None:
    p = write_gguf(tmp_path / "m.gguf", tokens=50_000, tokens_first=tokens_first)
    start = time.perf_counter()
    info = read_gguf(p)
    assert time.perf_counter() - start < 2.0
    assert (info.architecture, info.block_count) == ("llama", 32)


def test_keys_after_tokenizer_section_are_found(tmp_path: Path) -> None:
    blobs = [
        pack_kv("general.architecture", "llama"),
        pack_kv("tokenizer.ggml.tokens", ["a", "b"]),
        *_core()[1:],
        pack_kv("general.file_type", 7),
    ]
    info = read_gguf(_raw(tmp_path, pack_header(blobs)))
    assert (info.block_count, info.head_count, info.file_type) == (4, 8, 7)


def test_string_array_with_strings_larger_than_read_buffer(tmp_path: Path) -> None:
    big = "x" * (3 * 2**20)
    tokens = ["a"] * 10 + [big] + ["b"] * 10 + [big, big] + ["c"] * 100_000
    blobs = [pack_kv("tokenizer.ggml.tokens", tokens), *_core(), pack_kv("general.file_type", 2)]
    info = read_gguf(_raw(tmp_path, pack_header(blobs)))
    assert (info.block_count, info.file_type) == (4, 2)


def test_every_truncation_of_a_string_array_file_raises_gguf_error(tmp_path: Path) -> None:
    long_tokens = ["abc", "", "de" * 10, "f"] * 5
    blobs = [pack_kv("tokenizer.ggml.tokens", long_tokens), *_core(), pack_kv("general.file_type", 2)]
    data = pack_header(blobs)
    assert read_gguf(_raw(tmp_path, data)).file_type == 2
    for cut in range(len(data)):
        with pytest.raises(GgufError):
            read_gguf(_raw(tmp_path, data[:cut]))


def test_every_truncation_with_tiny_read_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "_CHUNK", 16)
    data = pack_header([pack_kv("tokenizer.ggml.tokens", ["abc", "de" * 10, "f"] * 5), *_core()])
    assert read_gguf(_raw(tmp_path, data)).block_count == 4
    for cut in range(len(data)):
        with pytest.raises(GgufError):
            read_gguf(_raw(tmp_path, data[:cut]))


def test_string_longer_than_cap_but_within_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "_MAX_STRING_BYTES", 16)
    with pytest.raises(GgufError, match="limit"):
        read_gguf(_raw(tmp_path, pack_header([pack_kv("general.name", "x" * 64), *_core()])))


def test_string_array_ending_the_file(tmp_path: Path) -> None:
    data = pack_header([pack_kv("tokenizer.ggml.tokens", ["abc", "de"])])
    with pytest.raises(GgufError, match="architecture"):
        read_gguf(_raw(tmp_path, data))


def test_bad_magic(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="magic"):
        read_gguf(_raw(tmp_path, b"NOPE" + b"\x00" * 64))


def test_unsupported_version(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="version"):
        read_gguf(_raw(tmp_path, pack_header(_core(), version=1)))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GgufError):
        read_gguf(tmp_path / "nope.gguf")


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, b""))


def test_truncated_file(tmp_path: Path) -> None:
    data = write_gguf(tmp_path / "m.gguf").read_bytes()
    for cut in (6, 20, 40, len(data) - 3):
        with pytest.raises(GgufError):
            read_gguf(_raw(tmp_path, data[:cut]))


def test_absurd_kv_count(tmp_path: Path) -> None:
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header(_core(), kv_count=2**60)))


def test_absurd_string_length(tmp_path: Path) -> None:
    bad = pack_string("general.name") + struct.pack("<I", T_STRING) + struct.pack("<Q", 2**62)
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_absurd_array_count(tmp_path: Path) -> None:
    bad = pack_string("x.arr") + struct.pack("<I", T_ARRAY) + struct.pack("<IQ", 4, 2**60)
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_absurd_string_array_count(tmp_path: Path) -> None:
    bad = pack_string("x.arr") + struct.pack("<I", T_ARRAY) + struct.pack("<IQ", T_STRING, 2**40)
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_unknown_value_type(tmp_path: Path) -> None:
    bad = pack_string("x.y") + struct.pack("<I", 99) + b"\x00" * 8
    with pytest.raises(GgufError, match="type"):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_unknown_array_element_type(tmp_path: Path) -> None:
    bad = pack_string("x.y") + struct.pack("<I", T_ARRAY) + struct.pack("<IQ", 99, 1) + b"\x00" * 8
    with pytest.raises(GgufError, match="type"):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_invalid_utf8_string(tmp_path: Path) -> None:
    bad = pack_string("general.name") + struct.pack("<I", T_STRING) + struct.pack("<Q", 2) + b"\xff\xfe"
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_deeply_nested_arrays(tmp_path: Path) -> None:
    deep = (
        pack_string("x.y") + struct.pack("<I", T_ARRAY) + b"".join(struct.pack("<IQ", T_ARRAY, 1) for _ in range(200))
    )
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([deep, *_core()])))


def test_nested_array_and_scalars_are_skipped(tmp_path: Path) -> None:
    extra = {"x.nested": [[1, 2], [3]], "x.f": 1.5, "x.b": True, "x.big": 2**40, "x.s": ["a", "b"]}
    info = read_gguf(write_gguf(tmp_path / "m.gguf", extra_kv=extra))
    assert info.block_count == 32


def test_missing_required_key(tmp_path: Path) -> None:
    blobs = [pack_kv("general.architecture", "llama"), pack_kv("llama.block_count", 4)]
    with pytest.raises(GgufError, match="context_length"):
        read_gguf(_raw(tmp_path, pack_header(blobs)))


def test_missing_architecture(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="architecture"):
        read_gguf(_raw(tmp_path, pack_header([pack_kv("llama.block_count", 4)])))


def test_non_integer_required_key(tmp_path: Path) -> None:
    blobs = [*_core()[:1], pack_kv("llama.block_count", "many"), *_core()[2:]]
    with pytest.raises(GgufError, match="block_count"):
        read_gguf(_raw(tmp_path, pack_header(blobs)))


def test_head_count_kv_defaults(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", n_head=16, n_head_kv=None))
    assert info.head_count_kv == 16


def test_per_layer_head_arrays_use_max(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", arch="gemma4", n_head=[8, 8, 16], n_head_kv=[0, 2, 4, 2]))
    assert (info.head_count, info.head_count_kv) == (16, 4)
    assert info.architecture == "gemma4"


def test_empty_head_array_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="head_count"):
        read_gguf(write_gguf(tmp_path / "m.gguf", n_head=[]))


def test_head_dim_and_kv_cache_bytes() -> None:
    info = _info(key_length=128)
    assert head_dim(info) == 128
    assert kv_cache_bytes(info, 8192) == 32 * 2 * 8192 * 8 * 128 * 2 == 1_073_741_824
    assert kv_cache_bytes(info, 8192, 1.0) == 1_073_741_824 // 2


def test_head_dim_falls_back_to_embedding() -> None:
    assert head_dim(_info(key_length=None)) == 128


def test_short_int_array_with_absurd_count(tmp_path: Path) -> None:
    bad = pack_string("x.arr") + struct.pack("<I", T_ARRAY) + struct.pack("<IQ", 4, 4000) + b"\0" * 8
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad, *_core()])))


def test_zero_head_count_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="head_count"):
        read_gguf(write_gguf(tmp_path / "m.gguf", n_head=0))


def test_string_array_with_absurd_element_length(tmp_path: Path) -> None:
    arr = struct.pack("<IQ", T_STRING, 2) + struct.pack("<Q", 2**40) + b"\x00" * 24
    bad = pack_string("x.arr") + struct.pack("<I", T_ARRAY) + arr
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad])))


def test_string_array_missing_elements(tmp_path: Path) -> None:
    arr = struct.pack("<IQ", T_STRING, 3) + struct.pack("<Q", 0) * 2 + b"\x00" * 4
    bad = pack_string("x.arr") + struct.pack("<I", T_ARRAY) + arr
    with pytest.raises(GgufError):
        read_gguf(_raw(tmp_path, pack_header([bad])))


_FILE_TYPE_CASES = [(None, "?"), (0, "F32"), (1, "F16"), (7, "Q8_0"), (15, "Q4_K_M"), (18, "Q6_K"), (30, "IQ4_XS")]
_FILE_TYPE_CASES += [(32, "BF16"), (38, "MXFP4"), (4, "ftype 4"), (99, "ftype 99")]


@pytest.mark.parametrize(("file_type", "name"), _FILE_TYPE_CASES)
def test_file_type_name(file_type: int | None, name: str) -> None:
    assert gguf.file_type_name(file_type) == name
