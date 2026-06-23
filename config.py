"""RaccoonLM v2 — Configuration (canonical, root-level)

This is the primary configuration module. All imports go through:
    from raccoonlm.config import settings, get_default_model

The copy in configuration/config.py is a backward-compatible re-export.
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 5555
    debug: bool = False

    # Ollama
    ollama_host: str = "http://localhost:11434"
    default_model: str = ""  # Empty = resolve dynamically via model registry

    # llama.cpp direct provider
    # RaccoonLM can talk directly to llama-server instead of routing through
    # Ollama or LM Studio. Point llama_cpp_command to your llama-server binary.
    llama_cpp_host: str = "http://localhost:8080"
    llama_cpp_command: str = "llama-server"
    llama_cpp_model_dirs: str = ""  # os.pathsep-separated extra GGUF scan dirs
    llama_cpp_gpu_layers: int = 999

    # Plugins
    internet_plugin: bool = True

    # Paths — resolved relative to THIS file (raccoonlm/) not configuration/
    db_path: str = str(Path(__file__).parent / "raccoonlm.db")

    class Config:
        env_prefix = "RACCOONLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure DB directory exists
os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)


def get_default_model() -> str:
    """
    Return the effective default model, resolving dynamically if needed.

    Priority chain:
    1. RACCOONLM_DEFAULT_MODEL env var (if set and non-empty)
    2. Last loaded model from registry
    3. First available Ollama model
    4. 'qwen3:4b' fallback

    Reference: RaccoonLM Issue #001 — default model should be dynamic
    """
    if settings.default_model and settings.default_model.strip():
        return settings.default_model
    # Lazy dynamic resolution
    from raccoonlm.core.models import resolve_default_model

    resolved = resolve_default_model()
    # Cache it back so concurrent calls don't re-resolve
    settings.default_model = resolved
    return resolved
