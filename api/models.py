"""RaccoonLM v2 — Model management router (multi-provider)"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from raccoonlm.core.models import (
    get_all_models, get_last_model, set_last_model, load_ollama_model,
    load_lmstudio_model, unload_ollama_model, check_model_loaded
)
from raccoonlm.config import get_default_model, settings
from raccoonlm.core.cache import invalidate_vram
import raccoonlm.api.core as core_state

import logging
log = logging.getLogger("uvicorn")

models_router = APIRouter()


class ModelLoadRequest(BaseModel):
    model_name: str = ""
    provider: str = "ollama"


def get_current_model_name() -> str:
    """Get display model name without mutating _current_model."""
    if core_state._current_model and core_state._current_model.strip():
        return core_state._current_model
    last = get_last_model()
    if last:
        return last
    return get_default_model()


# ── List all models ──
@models_router.get("/api/models")
async def list_models():
    try:
        models = get_all_models()
        return {
            "models": models,
            "provider_count": len(set(m.get("provider", "unknown") for m in models)),
        }
    except Exception as e:
        raise HTTPException(503, f"Model registry error: {e}")


# ── Load model ──
@models_router.post("/api/models/load")
async def load_model(req: ModelLoadRequest):
    model_name = req.model_name or get_current_model_name()

    # Auto-unload previous model before loading a new one
    if core_state._current_model and core_state._current_model != model_name:
        log.info(f"Unloading previous model: {core_state._current_model}")
        unload_ollama_model(core_state._current_model)

    if req.provider in ("ollama", ""):
        success = load_ollama_model(model_name)
        if success:
            core_state._current_model = model_name
            core_state._current_provider = "ollama"
            set_last_model(model_name)
            return {"status": "ok", "model": model_name, "provider": "ollama"}
        raise HTTPException(500, f"Failed to load {model_name} via Ollama")

    if req.provider == "lmstudio":
        success = load_lmstudio_model(model_name)
        if success:
            core_state._current_model = model_name
            core_state._current_provider = "lmstudio"
            set_last_model(model_name)
            return {"status": "ok", "model": model_name, "provider": "lmstudio", "verified": True}
        raise HTTPException(500, f"Failed to load {model_name} via LM Studio. Is LM Studio running on port 1234?")

    raise HTTPException(400, f"Unknown provider: {req.provider}")


# ── Load model (query param, backward compat) ──
@models_router.post("/api/models/load/query")
async def load_model_query(model_name: str = ""):
    name = model_name or get_current_model_name()
    success = load_ollama_model(name)
    if success:
        core_state._current_model = name
        set_last_model(name)
        return {"status": "ok", "model": name, "provider": "ollama"}
    raise HTTPException(500, f"Failed to load {name}")


# ── Unload model ──
@models_router.post("/api/models/unload")
async def unload_model():
    try:
        if core_state._current_model:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={"model": core_state._current_model, "keep_alive": 0}
                )
        core_state._current_model = ""
        invalidate_vram()
        return {"status": "ok", "message": "Model unloaded"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Current model name ──
@models_router.get("/api/models/current")
async def current_model():
    return {"model": get_current_model_name(), "provider": core_state._current_provider}


# ── Load status (VRAM check) ──
@models_router.get("/api/models/load-status")
async def model_load_status():
    status = check_model_loaded()
    status["model"] = core_state._current_model if core_state._current_model else "—"
    return status


# ── Local models (all providers) ──
@models_router.get("/api/local-models")
async def local_models():
    try:
        models = get_all_models()
        # Filter to local-only (Ollama + LM Studio), exclude registered-only
        local = [m for m in models if m.get("source") in ("ollama", "lmstudio")]
        return {"models": local}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ── Delete local model ──
@models_router.delete("/api/local-models/{name}")
async def delete_local_model(name: str):
    try:
        import subprocess
        r = subprocess.run(["ollama", "rm", name], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return {"status": "deleted", "name": name}
        raise HTTPException(500, r.stderr or "Delete failed")
    except FileNotFoundError:
        raise HTTPException(500, "ollama command not found")
    except Exception as e:
        raise HTTPException(500, str(e))
