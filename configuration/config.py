"""RaccoonLM v2 — Configuration (standalone, llama.cpp only)

Import via:
    from raccoonlm.configuration.config import settings, get_default_model
"""

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    class BaseSettings(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 5555
    debug: bool = False

    # llama.cpp direct provider
    llama_cpp_host: str = "http://localhost:8080"
    llama_cpp_command: str = "llama-server"
    llama_cpp_model_dirs: str = ""
    llama_cpp_gpu_layers: int = 12

    # Plugins
    internet_plugin: bool = True

    # Paths — resolved relative to THIS file (raccoonlm/configuration/)
    db_path: str = str(Path(__file__).parent / "raccoonlm.db")

    class Config:
        env_prefix = "RACCOONLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure DB directory exists
os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)


def get_default_model() -> str:
    """Dynamic default model from env var or resolved from registry."""
    from raccoonlm.core.models import resolve_default_model
    resolved = resolve_default_model()
    return resolved
