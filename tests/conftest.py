"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pytest

from ollama2cpp import proc

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"

ModelSpec = tuple[str, str, int, dict[str, Any]]


def _digest(*parts: str) -> str:
    return "sha256:" + hashlib.sha256(":".join(parts).encode()).hexdigest()


def _layer(media: str, digest: str, size: int) -> dict[str, Any]:
    return {"mediaType": f"application/vnd.ollama.image.{media}", "digest": digest, "size": size}


@pytest.fixture
def fake_ollama(tmp_path: Path) -> Callable[..., Path]:
    """Return a builder that creates a fake Ollama store under ``tmp_path / "ollama"``.

    The builder takes ``models`` (``(name, tag, model_size, options)`` tuples), ``missing_blob`` (a set of
    ``"name:tag"`` strings whose blobs are not written) and ``corrupt`` (``(name, tag)`` pairs written as
    invalid JSON), and returns the store root. It can be called repeatedly to add more models.

    Recognised ``options`` keys: ``projector`` / ``draft`` (bool, add the layer), ``no_model`` (bool, omit the model
    layer), ``registry`` and ``namespace`` (override the default ``registry.ollama.ai`` / ``library``).

    Blob files are tiny; the declared size lives only in the manifest.

    Returns:
        Callable[..., Path]: The store builder.
    """
    root = tmp_path / "ollama"

    def build(
        models: Iterable[ModelSpec] = (),
        missing_blob: Iterable[str] = (),
        corrupt: Iterable[tuple[str, str]] = (),
    ) -> Path:
        missing = set(missing_blob)
        blobs = root / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        for name, tag, size, options in models:
            registry = options.get("registry", DEFAULT_REGISTRY)
            namespace = options.get("namespace", DEFAULT_NAMESPACE)
            layers: list[dict[str, Any]] = []
            if not options.get("no_model"):
                layers.append(_layer("model", _digest(registry, namespace, name, tag, "model"), size))
            if options.get("projector"):
                layers.append(_layer("projector", _digest(registry, namespace, name, tag, "projector"), 1000))
            if options.get("draft"):
                layers.append(_layer("draft", _digest(registry, namespace, name, tag, "draft"), 2000))
            layers.append(_layer("template", _digest(registry, namespace, name, tag, "template"), 13))
            manifest = {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {
                    "mediaType": "application/vnd.docker.container.image.v1+json",
                    "digest": _digest(registry, namespace, name, tag, "config"),
                    "size": 242,
                },
                "layers": layers,
            }
            path = root / "manifests" / registry / namespace / name / tag
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            if f"{name}:{tag}" in missing:
                continue
            for layer in layers:
                (blobs / layer["digest"].replace(":", "-")).write_bytes(b"GGUF" + b"\0" * 12)
        for name, tag in corrupt:
            path = root / "manifests" / DEFAULT_REGISTRY / DEFAULT_NAMESPACE / name / tag
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
        return root

    return build


FAKE_SERVER = Path(__file__).parent / "fake_llama_server.py"
SERVER_WAIT = 15.0


@pytest.fixture
def reaper() -> Iterator[Callable[[proc.Handle], proc.Handle]]:
    """Track spawned processes and kill them (with children) at teardown.

    Yields:
        Callable: ``track(process)`` registers a process and returns it.
    """
    procs: list[proc.Handle] = []

    def track(process: proc.Handle) -> proc.Handle:
        procs.append(process)
        return process

    yield track
    for p in procs:
        try:
            for child in psutil.Process(p.pid).children(recursive=True):
                child.kill()
        except psutil.Error:
            pass
        if p.poll() is None:
            p.kill()
        p.wait(timeout=10)


def free_port() -> int:
    """Return a currently free local TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class FakeServer:
    """A running stand-in ``llama-server``."""

    process: proc.Handle
    host: str
    port: int


@pytest.fixture
def fake_server(tmp_path: Path, reaper: Callable[[proc.Handle], proc.Handle]) -> FakeServer:
    """Start a stand-in ``llama-server`` on a free port (healthy immediately)."""
    port_file = tmp_path / "fake.port"
    process = reaper(proc.spawn([sys.executable, str(FAKE_SERVER), "0", str(port_file)]))
    deadline = time.monotonic() + SERVER_WAIT
    while time.monotonic() < deadline:
        if port_file.is_file() and port_file.read_text(encoding="utf-8"):
            return FakeServer(process, "127.0.0.1", int(port_file.read_text(encoding="utf-8")))
        time.sleep(0.02)
    msg = "fake server did not start"
    raise RuntimeError(msg)
