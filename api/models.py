"""RaccoonLM v2 — Model management router (llama.cpp direct GGUF only)"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from raccoonlm.core.models import (
    get_all_models, get_last_model, set_last_model,
    load_llamacpp_model, unload_llamacpp_model, check_model_loaded
)
from raccoonlm.config import get_default_model, settings
from raccoonlm.core.cache import invalidate_vram
import raccoonlm.api.core as core_state

import logging
log = logging.getLogger("uvicorn")

models_router = APIRouter()


class ModelLoadRequest(BaseModel):
    model_name: str = ""
    provider: str = "llamacpp"


def get_current_model_name() -> str:
    """Get display model name without mutating _current_model.

    Returns '—' if no model is loaded and no valid model exists on disk.
    """
    from raccoonlm.core.models import _resolve_llamacpp_model_path, _find_gguf_models
    if core_state._current_model and core_state._current_model.strip():
        return core_state._current_model
    last = get_last_model()
    if last:
        if _resolve_llamacpp_model_path(last):
            return last
        for m in _find_gguf_models():
            if m["name"] == last:
                return last
    # No loaded, last, or valid default model — show empty
    default = get_default_model()
    if default and default not in ("—", "qwen3:4b", "") and _resolve_llamacpp_model_path(default):
        return default
    for m in _find_gguf_models():
        return m["name"]
    return "—"


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
        if core_state._current_provider == "llamacpp":
            unload_llamacpp_model()

    success = load_llamacpp_model(model_name)
    if success:
        core_state._current_model = model_name
        core_state._current_provider = "llamacpp"
        set_last_model(model_name)
        return {"status": "ok", "model": model_name, "provider": "llamacpp", "verified": True}

    raise HTTPException(500, f"Failed to load {model_name} via llama.cpp. Is llama-server installed or RACCOONLM_LLAMA_CPP_COMMAND set?")


# ── Unload model ──
@models_router.post("/api/models/unload")
async def unload_model():
    try:
        if core_state._current_model:
            unload_llamacpp_model()
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


# ── Local models (llama.cpp only) ──
@models_router.get("/api/local-models")
async def local_models():
    try:
        models = get_all_models()
        local = [m for m in models if m.get("source") == "llamacpp" or m.get("provider") == "llamacpp"]
        return {"models": local}
    except Exception as e:
        return {"models": [], "error": str(e)}
