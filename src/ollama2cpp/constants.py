"""Default configuration constants for ollama2cpp.

Every value can be overridden by a CLI option or an environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Local storage -----------------------------------------------------------
DEFAULT_PATH = Path.home() / ".ollama2cpp"
CONFIG_FILENAME = "config.json"
PID_FILENAME = "server.pid"

# --- llama-server defaults ---------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 8192

# --- Environment variable names ----------------------------------------------
ENV_OLLAMA_MODELS = "OLLAMA_MODELS"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_HOME = "OLLAMA2CPP_HOME"

# --- Installation ------------------------------------------------------------
WINGET_ID = "ggml.llamacpp"


def home() -> Path:
    """Return the ollama2cpp storage directory.

    The environment is read at call time, so tests and callers can override it.

    Returns:
        Path: ``$OLLAMA2CPP_HOME`` when set and non-empty, else ``DEFAULT_PATH``.
    """
    override = os.environ.get(ENV_HOME)
    return Path(override) if override else DEFAULT_PATH
