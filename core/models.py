"""RaccoonLM v2 — Direct GGUF model registry (standalone, no Ollama/LM Studio)"""

import json, os, subprocess, logging, shutil, time
from pathlib import Path
from typing import Optional

from raccoonlm.config import settings
from raccoonlm.core.cache import invalidate_vram

log = logging.getLogger("uvicorn")


_LLAMA_CPP_PROC: subprocess.Popen | None = None
_LLAMA_CPP_MODEL_PATH: str | None = None


def _llamacpp_url(path: str) -> str:
    return settings.llama_cpp_host.rstrip("/") + path


def _common_gguf_dirs() -> list[Path]:
    """Directories likely to contain GGUF files for direct llama.cpp use."""
    home = Path.home()
    dirs = [
        home / "Downloads",
        home / "Desktop",
        home / ".cache" / "huggingface" / "hub",
        Path(settings.db_path).parent / "models",
        Path(settings.db_path).parent / "downloads",
    ]
    extra = (settings.llama_cpp_model_dirs or "").strip()
    if extra:
        dirs.extend(Path(x).expanduser() for x in extra.split(os.pathsep) if x.strip())
    return [d for d in dirs if d.exists()]


def _find_gguf_models(limit: int = 200) -> list[dict]:
    """Scan common local folders for GGUF files llama-server can load."""
    models: list[dict] = []
    seen: set[str] = set()
    for root in _common_gguf_dirs():
        try:
            for path in root.rglob("*.gguf"):
                spath = str(path.resolve())
                if spath in seen:
                    continue
                seen.add(spath)
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                models.append({
                    "name": path.stem,
                    "size": size,
                    "size_display": _fmt_bytes(size),
                    "modified": "",
                    "provider": "llamacpp",
                    "source": "llamacpp",
                    "path": spath,
                })
                if len(models) >= limit:
                    return models
        except Exception as e:
            log.debug(f"GGUF scan skipped {root}: {e}")
    models.sort(key=lambda m: m.get("name", "").lower())
    return models


def _resolve_llamacpp_model_path(model_name: str) -> str | None:
    """Resolve a UI-selected llama.cpp model name or direct path to a GGUF path."""
    candidate = Path(model_name).expanduser()
    if candidate.exists() and candidate.suffix.lower() == ".gguf":
        return str(candidate.resolve())
    for m in _find_gguf_models():
        if m["name"] == model_name or m.get("path") == model_name:
            return m.get("path")
    data = _MODEL_REGISTRY.get("models", {}).get(model_name) or {}
    gguf_path = data.get("gguf_path") or data.get("path")
    if gguf_path and Path(gguf_path).exists():
        return str(Path(gguf_path).resolve())
    return None


def get_llamacpp_models() -> list[dict]:
    """Get models from a running llama-server plus discovered local GGUF files."""
    models = []
    try:
        import httpx
        r = httpx.get(_llamacpp_url("/v1/models"), timeout=2.0)
        if r.status_code == 200:
            for m in r.json().get("data", []):
                mid = m.get("id", "llama.cpp")
                models.append({
                    "name": mid,
                    "size": 0,
                    "size_display": "llama.cpp server",
                    "modified": "",
                    "provider": "llamacpp",
                    "source": "llamacpp",
                })
    except Exception:
        pass

    seen = {m["name"] for m in models}
    for m in _find_gguf_models():
        if m["name"] not in seen:
            models.append(m)
            seen.add(m["name"])
    return models


def _llamacpp_chat_response_has_output(data: dict) -> bool:
    try:
        choices = data.get("choices") or []
        if not choices:
            return False
        msg = choices[0].get("message") or {}
        if (msg.get("content") or "").strip():
            return True
        usage = data.get("usage") or {}
        return (usage.get("completion_tokens") or 0) > 0
    except Exception:
        return False


def load_llamacpp_model(model_name: str) -> bool:
    """Load a GGUF directly with llama.cpp's llama-server and verify it."""
    global _LLAMA_CPP_PROC, _LLAMA_CPP_MODEL_PATH
    import httpx

    model_path = _resolve_llamacpp_model_path(model_name)
    if not model_path:
        # A server may already be running externally with this model id.
        try:
            r = httpx.post(
                _llamacpp_url("/v1/chat/completions"),
                json={"model": model_name, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1, "temperature": 0},
                timeout=30.0,
            )
            return r.status_code == 200 and _llamacpp_chat_response_has_output(r.json())
        except Exception as e:
            log.error(f"llama.cpp external server verify failed for {model_name}: {e}")
            return False

    if _LLAMA_CPP_PROC and _LLAMA_CPP_PROC.poll() is None and _LLAMA_CPP_MODEL_PATH == model_path:
        log.info(f"llama.cpp already running for {model_path}")
    else:
        unload_llamacpp_model()
        cmd = shutil.which(settings.llama_cpp_command) or settings.llama_cpp_command
        args = [
            cmd,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", settings.llama_cpp_host.rstrip("/").split(":")[-1],
            "--ctx-size", "8192",
            "--n-gpu-layers", str(settings.llama_cpp_gpu_layers),
        ]
        try:
            _LLAMA_CPP_PROC = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _LLAMA_CPP_MODEL_PATH = model_path
        except FileNotFoundError:
            log.error(f"llama-server not found. Set RACCOONLM_LLAMA_CPP_COMMAND to your llama-server path.")
            return False
        except Exception as e:
            log.error(f"Failed to start llama.cpp: {e}")
            return False

    for _ in range(60):
        try:
            r = httpx.get(_llamacpp_url("/health"), timeout=1.0)
            if r.status_code in (200, 503):
                break
        except Exception:
            pass
        time.sleep(0.5)

    try:
        # Retry loop: model may still be loading (HTTP 503)
        r = None
        for attempt in range(120):
            try:
                r = httpx.post(
                    _llamacpp_url("/v1/chat/completions"),
                    json={"model": model_name, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1, "temperature": 0},
                    timeout=30.0,
                )
                if r.status_code in (200, 503):
                    if r.status_code == 200:
                        break
                    # Still loading, wait and retry
                    time.sleep(2)
                    continue
            except httpx.TimeoutException:
                time.sleep(2)
                continue
            except Exception:
                time.sleep(1)
                continue
        if r and r.status_code == 200 and _llamacpp_chat_response_has_output(r.json()):
            invalidate_vram()
            register_model(model_name, "llamacpp", source="llamacpp", path=model_path, gguf_path=model_path, size=os.path.getsize(model_path), size_display=_fmt_bytes(os.path.getsize(model_path)))
            log.info(f"✅ llama.cpp model {model_name} verified")
            return True
        log.warning(f"llama.cpp verification failed for {model_name}: HTTP {r.status_code} {r.text[:300]}")
        return False
    except Exception as e:
        log.error(f"llama.cpp verify failed for {model_name}: {e}")
        return False


def unload_llamacpp_model() -> bool:
    """Stop the llama-server process started by RaccoonLM."""
    global _LLAMA_CPP_PROC, _LLAMA_CPP_MODEL_PATH
    if _LLAMA_CPP_PROC and _LLAMA_CPP_PROC.poll() is None:
        try:
            _LLAMA_CPP_PROC.terminate()
            _LLAMA_CPP_PROC.wait(timeout=10)
        except Exception:
            try:
                _LLAMA_CPP_PROC.kill()
            except Exception:
                pass
    _LLAMA_CPP_PROC = None
    _LLAMA_CPP_MODEL_PATH = None
    invalidate_vram()
    return True


# ── Model Registry DB ──
REGISTRY_PATH = Path(settings.db_path).parent / "model_registry.json"

_MODEL_REGISTRY: dict = {}


def _load_registry():
    global _MODEL_REGISTRY
    try:
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                _MODEL_REGISTRY = json.load(f)
        else:
            _MODEL_REGISTRY = {"providers": {}, "last_model": None, "models": {}}
    except:
        _MODEL_REGISTRY = {"providers": {}, "last_model": None, "models": {}}


def _save_registry():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(_MODEL_REGISTRY, f, indent=2)


def init_registry():
    """Initialize the model registry on startup."""
    _load_registry()
    if "models" not in _MODEL_REGISTRY:
        _MODEL_REGISTRY["models"] = {}
    if "last_model" not in _MODEL_REGISTRY:
        _MODEL_REGISTRY["last_model"] = None


def get_last_model() -> Optional[str]:
    """Get the last loaded model from registry."""
    return _MODEL_REGISTRY.get("last_model")


def set_last_model(name: str):
    """Persist the last loaded model name."""
    _MODEL_REGISTRY["last_model"] = name
    _save_registry()


def register_model(name: str, provider: str, **meta) -> dict:
    """Register a model in the local registry."""
    _MODEL_REGISTRY["models"][name] = {
        "name": name,
        "provider": provider,
        **meta,
    }
    _save_registry()
    return _MODEL_REGISTRY["models"][name]


def resolve_default_model() -> str:
    """Dynamic default model resolution.
    Priority:
    1. Last loaded model (from registry)
    2. RACCOONLM_DEFAULT_MODEL env
    3. First available GGUF model
    4. 'qwen3:4b' fallback
    """
    last = get_last_model()
    if last:
        return last
    env_default = os.environ.get("RACCOONLM_DEFAULT_MODEL", "").strip()
    if env_default:
        return env_default
    try:
        gguvs = _find_gguf_models(limit=1)
        if gguvs:
            return gguvs[0]["name"]
    except Exception:
        pass
    return "qwen3:4b"


def get_registered_models() -> list[dict]:
    """Get all models from the local registry (HF imports, manual)."""
    results = []
    for name, data in _MODEL_REGISTRY.get("models", {}).items():
        results.append({
            "name": name,
            "provider": data.get("provider", "unknown"),
            "source": data.get("source", "registered"),
            "size_display": data.get("size_display", ""),
        })
    return results


def get_all_models() -> list[dict]:
    """Get ALL models from all providers, deduplicated."""
    seen = set()
    all_models = []

    # llama.cpp direct GGUF models
    for m in get_llamacpp_models():
        key = f"{m.get('provider')}:{m.get('path') or m['name']}"
        if key not in seen:
            seen.add(key)
            all_models.append(m)

    # Registered models (HF imports, etc.)
    for m in get_registered_models():
        if m["name"] not in seen:
            seen.add(m["name"])
            all_models.append(m)

    return all_models


def check_model_loaded() -> dict:
    """Check if a model is loaded by checking the running llama-server health."""
    import httpx
    try:
        r = httpx.get(_llamacpp_url("/health"), timeout=2.0)
        if r.status_code in (200, 503):
            loaded = r.status_code == 200
            return {"loaded": loaded, "rss_mb": 0, "vram_gb": 0}
    except Exception:
        pass
    return {"loaded": False, "rss_mb": -1, "vram_gb": 0}


def _fmt_bytes(size: int) -> str:
    if size < 1024: return f"{size}B"
    if size < 1024**2: return f"{size/1024:.1f}KB"
    if size < 1024**3: return f"{size/1024**2:.1f}MB"
    return f"{size/1024**3:.1f}GB"
