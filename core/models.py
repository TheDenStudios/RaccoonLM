"""RaccoonLM v2 — Multi-provider model registry

Decouples model management from any single backend.
Supports: Ollama, HuggingFace GGUF imports, and future providers.
"""

import json, os, subprocess, logging
from pathlib import Path
from typing import Optional

from raccoonlm.config import settings
from raccoonlm.core.cache import invalidate_vram

log = logging.getLogger("uvicorn")

# ── Model Registry DB ──
REGISTRY_PATH = Path(settings.db_path).parent / "model_registry.json"

# Default: save at raccoonlm directory alongside the sqlite db
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
    2. First Ollama model available
    3. RACCOONLM_DEFAULT_MODEL env
    4. 'qwen3:4b' fallback
    """
    # 1. Last loaded
    last = get_last_model()
    if last:
        return last
    
    # 2. Check env override
    env_default = os.environ.get("RACCOONLM_DEFAULT_MODEL", "").strip()
    if env_default:
        return env_default
    
    # 3. Try to find any Ollama model
    try:
        import ollama
        result = ollama.list()
        if result and result.models and len(result.models) > 0:
            return result.models[0].model
    except:
        pass

    # 4. Hardcoded fallback
    return "qwen3:4b"


def get_ollama_models() -> list[dict]:
    """Get models from Ollama."""
    try:
        import ollama
        result = ollama.list()
        models = []
        for m in (result.models or []):
            name = getattr(m, 'model', 'unknown')
            size = getattr(m, 'size', 0) or 0
            modified = str(getattr(m, 'modified_at', '') or '')[:10]
            models.append({
                "name": name,
                "size": size,
                "size_display": _fmt_bytes(size),
                "modified": modified,
                "provider": "ollama",
                "source": "ollama",
            })
        return models
    except Exception as e:
        log.warning(f"Ollama list failed: {e}")
        return []


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


def get_lmstudio_models() -> list[dict]:
    """Get models from LM Studio local API (port 1234)."""
    try:
        import httpx
        r = httpx.get("http://localhost:1234/v1/models", timeout=3.0)
        if r.status_code != 200:
            return []
        data = r.json()
        models = []
        for m in data.get("data", []):
            mid = m.get("id", "unknown")
            models.append({
                "name": mid,
                "size": 0,
                "size_display": "LM Studio",
                "modified": "",
                "provider": "lmstudio",
                "source": "lmstudio",
            })
        return models
    except Exception:
        return []


def get_all_models() -> list[dict]:
    """Get ALL models from all providers, deduplicated."""
    seen = set()
    all_models = []

    # Ollama models
    for m in get_ollama_models():
        if m["name"] not in seen:
            seen.add(m["name"])
            all_models.append(m)

    # LM Studio models
    for m in get_lmstudio_models():
        if m["name"] not in seen:
            seen.add(m["name"])
            all_models.append(m)

    # Registered models (HF imports, etc.)
    for m in get_registered_models():
        if m["name"] not in seen:
            seen.add(m["name"])
            all_models.append(m)

    return all_models


def unload_ollama_model(model_name: str) -> bool:
    """Unload an Ollama model from VRAM by setting keep_alive=0."""
    try:
        import httpx
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=5.0,
        )
        log.info(f"Unloaded {model_name} from Ollama")
        return r.status_code == 200
    except Exception as e:
        log.warning(f"Failed to unload {model_name}: {e}")
        return False


def _lmstudio_chat_response_has_output(data: dict) -> bool:
    """Return True only when LM Studio actually generated something.

    LM Studio can accept /v1/models/load while the model is still not usable.
    The UI must not show "loaded" until a real chat completion succeeds.
    Some reasoning models put text in reasoning_content instead of content, so
    accept either visible content, reasoning content, or positive token usage.
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            return False
        msg = choices[0].get("message") or {}
        if (msg.get("content") or "").strip():
            return True
        if (msg.get("reasoning_content") or msg.get("reasoning") or "").strip():
            return True
        usage = data.get("usage") or {}
        return (usage.get("completion_tokens") or 0) > 0
    except Exception:
        return False


def load_lmstudio_model(model_name: str) -> bool:
    """Load a model via LM Studio local API (port 1234).
    
    LM Studio auto-loads models on first request, but also exposes
    a /v1/models/load endpoint for explicit loading in newer versions.
    
    Falls back to sending a dummy chat completion to trigger model load.
    """
    import httpx
    explicit_load_ok = False
    try:
        # Try explicit load endpoint (LM Studio 0.3+), but do NOT trust it by
        # itself. It can return success before the model is actually usable.
        r = httpx.post(
            "http://localhost:1234/v1/models/load",
            json={"model": model_name},
            timeout=10.0,
        )
        if r.status_code in (200, 201):
            explicit_load_ok = True
            log.info(f"LM Studio accepted load request for {model_name}; verifying with chat probe")
        else:
            log.warning(f"LM Studio explicit load failed for {model_name}: HTTP {r.status_code} {r.text[:300]}")
    except Exception as e:
        log.warning(f"LM Studio explicit load unavailable for {model_name}: {e}")

    # Fallback: send a tiny dummy chat to trigger auto-load
    try:
        r = httpx.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=30.0,
        )
        if r.status_code == 200 and _lmstudio_chat_response_has_output(r.json()):
            log.info(f"✅ LM Studio model {model_name} verified by dummy chat")
            invalidate_vram()
            return True
        log.warning(f"LM Studio load verification failed for {model_name}: HTTP {r.status_code} {r.text[:300]}")
        return False
    except Exception as e:
        log.error(f"LM Studio load failed for {model_name}: {e}")
        return False


def load_ollama_model(model_name: str) -> bool:
    """Load a model via Ollama — actually loads into VRAM.
    
    Uses a dummy chat request to force Ollama to load the model
    into VRAM. ollama.pull() alone only downloads the files.
    """
    try:
        import ollama
        # First ensure model files exist locally
        ollama.pull(model_name)
        
        # Force actual VRAM load with a tiny dummy generation
        # This is what actually makes Ollama allocate GPU memory
        result = ollama.generate(
            model=model_name,
            prompt="hello",
            options={"num_predict": 1, "temperature": 0},
            keep_alive=-1,  # Keep loaded indefinitely
        )
        # Invalidate VRAM cache after loading
        invalidate_vram()
        if result and hasattr(result, 'response'):
            log.info(f"✅ Model {model_name} loaded into VRAM successfully")
            return True
        else:
            log.warning(f"Model {model_name} pull OK but generate returned unexpected result")
            return False
    except Exception as e:
        log.error(f"Ollama load failed for {model_name}: {e}")
        return False


def check_model_loaded() -> dict:
    """Check if a model is actually loaded in VRAM by Ollama.
    
    Key detection: the 'ollama runner' process is the one that holds VRAM.
    Ollama base process uses ~0.7-0.9GB VRAM even with no model loaded.
    A model is only 'loaded' if an active runner process exists with >50MB RSS.
    Returns dict with 'loaded' bool, 'rss_mb', 'vram_gb'.
    """
    try:
        import subprocess
        runner_rss_mb = 0
        vram_gb = 0.0
        
        # Method 1 (primary): Check for Ollama runner process
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            # The runner process has --model flag and holds the actual VRAM
            if '--model' in line and 'ollama' in line.lower():
                parts = line.split()
                if len(parts) > 5:
                    try:
                        rss_kb = float(parts[5])  # RSS in KB on Linux
                        runner_rss_mb = rss_kb / 1024
                        break
                    except: pass
        
        # Method 2: Read VRAM from sysfs for reporting
        try:
            r2 = subprocess.run(
                ["cat", "/sys/class/drm/renderD128/device/mem_info_vram_used"],
                capture_output=True, text=True, timeout=2
            )
            if r2.returncode == 0:
                vram_gb = int(r2.stdout.strip()) / (1024**3)
        except: pass
        
        # Only consider 'loaded' if a runner process exists with significant RSS
        is_loaded = runner_rss_mb > 50
        
        return {
            "loaded": is_loaded,
            "rss_mb": round(runner_rss_mb, 1),
            "vram_gb": round(vram_gb, 1),
        }
    except Exception as e:
        log.warning(f"check_model_loaded error: {e}")
        return {"loaded": False, "rss_mb": -1, "vram_gb": 0, "error": str(e)}


def _fmt_bytes(size: int) -> str:
    if size < 1024: return f"{size}B"
    if size < 1024**2: return f"{size/1024:.1f}KB"
    if size < 1024**3: return f"{size/1024**2:.1f}MB"
    return f"{size/1024**3:.1f}GB"


def import_gguf(gguf_path: str, model_name: str = None) -> dict:
    """Import a GGUF file into Ollama and register it.
    
    Args:
        gguf_path: Path to the .gguf file
        model_name: Optional name (derived from filename if not given)
    
    Returns:
        dict with status and model name
    """
    if not model_name:
        import re
        base = os.path.basename(gguf_path)
        model_name = re.sub(r'\.gguf$', '', base, flags=re.I)
        model_name = model_name.lower().replace('_', '-').replace(' ', '-').strip('-')
        if len(model_name) > 60:
            model_name = model_name[:60]

    try:
        # Create Modelfile
        modelfile_content = f"FROM {gguf_path}\n"
        modelfile_path = Path.home() / ".ollama" / f"Modelfile-{model_name}"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        # Run ollama create
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300
        )
        
        # Clean up Modelfile
        try:
            modelfile_path.unlink()
        except:
            pass

        if result.returncode == 0:
            # Register in local registry
            size = os.path.getsize(gguf_path) if os.path.exists(gguf_path) else 0
            register_model(model_name, "ollama", source="hf_import",
                          gguf_path=gguf_path, size=size,
                          size_display=_fmt_bytes(size))
            return {"status": "ok", "model": model_name}
        else:
            return {"status": "error", "error": result.stderr}
    except Exception as e:
        return {"status": "error", "error": str(e)}
