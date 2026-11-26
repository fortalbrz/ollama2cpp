"""Default configuration constants for llmopenfinance.

Every value can be overridden by a CLI option or an environment variable;
see ``docs/usage.md`` for the full list.
"""

from pathlib import Path

# --- LLM providers -----------------------------------------------------------
DEFAULT_OLLAMA_MODEL = "gemma4:12b"

# --- Environment variable names ----------------------------------------------
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"

# --- Local storage -----------------------------------------------------------
DEFAULT_PATH = Path.home() / ".ollama2cpp"  
