"""Read-only access to an Ollama model store (manifests and blobs).

Nothing here ever writes to the Ollama directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama2cpp.constants import ENV_OLLAMA_MODELS

log = logging.getLogger(__name__)

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"
SYSTEM_MODELS_PATH = Path("/usr/share/ollama/.ollama/models")

MEDIA_MODEL = "application/vnd.ollama.image.model"
MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_DRAFT = "application/vnd.ollama.image.draft"

_MIN_MANIFEST_DEPTH = 3  # registry / name / tag
_DIGEST_RE = re.compile(r"[a-z0-9]+:[0-9a-f]+")


class ModelNotFoundError(LookupError):
    """Raised when a requested model is not present in the Ollama store."""


class _InvalidManifestError(ValueError):
    """Raised internally when a manifest does not have the expected shape."""


@dataclass(frozen=True)
class Layer:
    """One layer entry of an Ollama manifest."""

    media_type: str
    digest: str
    size: int


@dataclass(frozen=True)
class OllamaModel:
    """An installed Ollama model.

    ``blob_path`` and ``digest`` are ``None`` when the manifest has no model layer.
    """

    name: str
    manifest_path: Path
    layers: tuple[Layer, ...]
    blob_path: Path | None
    digest: str | None

    @property
    def size(self) -> int:
        """Size in bytes of the model layer, ``0`` when there is none."""
        return next((layer.size for layer in self.layers if layer.media_type == MEDIA_MODEL), 0)

    @property
    def has_projector(self) -> bool:
        """Whether the model ships a multimodal projector layer."""
        return any(layer.media_type == MEDIA_PROJECTOR for layer in self.layers)

    @property
    def has_draft(self) -> bool:
        """Whether the model ships a speculative-decoding draft layer."""
        return any(layer.media_type == MEDIA_DRAFT for layer in self.layers)

    @property
    def blob_exists(self) -> bool:
        """Whether the model blob is present on disk (``False`` for dangling manifests)."""
        return self.blob_path is not None and self.blob_path.is_file()


def find_ollama_path(explicit: Path | None = None, env: Mapping[str, str] = os.environ) -> Path | None:
    """Locate the Ollama models directory.

    Candidates are tried in order: ``explicit``, ``$OLLAMA_MODELS``, ``~/.ollama/models``,
    ``%LOCALAPPDATA%/Ollama/models`` and the Linux system-service path. The first one containing a
    ``manifests`` directory wins.

    Args:
        explicit: A user-provided path, tried first.
        env: Environment mapping to read ``OLLAMA_MODELS`` and ``LOCALAPPDATA`` from.

    Returns:
        Path | None: The models directory, or ``None`` when no candidate qualifies.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if env.get(ENV_OLLAMA_MODELS):
        candidates.append(Path(env[ENV_OLLAMA_MODELS]))
    candidates.append(Path.home() / ".ollama" / "models")
    if env.get("LOCALAPPDATA"):
        candidates.append(Path(env["LOCALAPPDATA"]) / "Ollama" / "models")
    candidates.append(SYSTEM_MODELS_PATH)
    return next((c for c in candidates if (c / "manifests").is_dir()), None)


def normalize_name(name: str) -> str:
    """Normalise a user-typed model name to the canonical form used by :func:`list_models`.

    Lowercases, drops the default ``registry.ollama.ai/`` and ``library/`` prefixes and appends ``:latest``
    when no tag is given.

    Args:
        name: Model name as typed, e.g. ``Gemma4`` or ``registry.ollama.ai/library/gemma4:12b``.

    Returns:
        str: The canonical name, e.g. ``gemma4:latest``.
    """
    result = name.strip().lower()
    result = result.removeprefix(f"{DEFAULT_REGISTRY}/").removeprefix(f"{DEFAULT_NAMESPACE}/")
    if ":" not in result.rsplit("/", 1)[-1]:
        result = f"{result}:{DEFAULT_TAG}"
    return result


def _parse_layers(data: object) -> tuple[Layer, ...]:
    """Validate the manifest JSON and extract its layers.

    Args:
        data: The decoded manifest JSON.

    Returns:
        tuple[Layer, ...]: The layers, empty when the manifest has none.

    Raises:
        _InvalidManifestError: If the structure is not a valid manifest.
    """
    if not isinstance(data, dict):
        raise _InvalidManifestError("manifest is not a JSON object")
    raw_layers: Any = data.get("layers", [])
    if not isinstance(raw_layers, list):
        raise _InvalidManifestError("'layers' is not a list")
    layers: list[Layer] = []
    for raw in raw_layers:
        if not isinstance(raw, dict):
            raise _InvalidManifestError("layer is not an object")
        media, digest, size = raw.get("mediaType"), raw.get("digest"), raw.get("size")
        if not isinstance(media, str) or not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise _InvalidManifestError(f"bad layer entry: {raw!r}")
        if not isinstance(size, int) or isinstance(size, bool):
            raise _InvalidManifestError(f"bad layer size: {raw!r}")
        layers.append(Layer(media, digest, size))
    return tuple(layers)


def _blob_path(root: Path, digest: str) -> Path:
    """Map a ``sha256:<hex>`` digest to its ``blobs/sha256-<hex>`` file.

    Args:
        root: The models directory.
        digest: A validated layer digest.

    Returns:
        Path: Where the blob lives (it may not exist).
    """
    return root / "blobs" / digest.replace(":", "-")


def _read_manifest(root: Path, path: Path) -> OllamaModel:
    """Build an :class:`OllamaModel` from one manifest file.

    Args:
        root: The models directory.
        path: The manifest file.

    Returns:
        OllamaModel: The parsed model.

    Raises:
        _InvalidManifestError: If the path is too shallow or the content is not a valid manifest.
    """
    parts = path.relative_to(root / "manifests").parts
    if len(parts) < _MIN_MANIFEST_DEPTH:
        raise _InvalidManifestError("manifest path is not registry/name/tag")
    layers = _parse_layers(json.loads(path.read_text(encoding="utf-8")))
    registry, repo, tag = parts[0], "/".join(parts[1:-1]), parts[-1]
    model_layer = next((layer for layer in layers if layer.media_type == MEDIA_MODEL), None)
    return OllamaModel(
        name=normalize_name(f"{registry}/{repo}:{tag}"),
        manifest_path=path,
        layers=layers,
        blob_path=_blob_path(root, model_layer.digest) if model_layer else None,
        digest=model_layer.digest if model_layer else None,
    )


def list_models(root: Path) -> list[OllamaModel]:
    """List every model whose manifest can be parsed.

    Unreadable or malformed manifests are logged and skipped.

    Args:
        root: The models directory (containing ``manifests/`` and ``blobs/``).

    Returns:
        list[OllamaModel]: Models sorted by name.
    """
    manifests = root / "manifests"
    if not manifests.is_dir():
        return []
    models: list[OllamaModel] = []
    for path in sorted(p for p in manifests.rglob("*") if p.is_file()):
        try:
            models.append(_read_manifest(root, path))
        except (OSError, ValueError) as exc:
            log.warning("Skipping unreadable manifest %s: %s", path, exc)
    return sorted(models, key=lambda m: m.name)


def resolve_model(root: Path, name: str) -> OllamaModel:
    """Find an installed model by (loosely typed) name.

    Args:
        root: The models directory.
        name: Model name as typed; case-insensitive, ``:latest`` implied, default registry optional.

    Returns:
        OllamaModel: The matching model.

    Raises:
        ModelNotFoundError: If no model matches; the message lists the available names.
    """
    wanted = normalize_name(name)
    models = list_models(root)
    for model in models:
        if model.name == wanted:
            return model
    available = ", ".join(m.name for m in models) or "no models installed"
    raise ModelNotFoundError(f"Model {name!r} not found. Available: {available}")
