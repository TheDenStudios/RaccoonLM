"""RaccoonLM v2 — Router: entry point that registers all sub-routers"""

from fastapi import APIRouter

# ── Import sub-routers ──
from raccoonlm.api.core import core, init_plugins, shutdown_plugins
from raccoonlm.api.models import models_router
from raccoonlm.api.chat import chat
from raccoonlm.api.hub import hub
from raccoonlm.api.system import system

# ── Public API (re-export for main.py) ──
__all__ = ["router", "init_plugins", "shutdown_plugins"]

# ── Create root router ──
router = APIRouter()

# ── Include all sub-routers ──
router.include_router(core)          # /, /api/health
router.include_router(models_router) # /api/models/*, /api/local-models/*
router.include_router(chat)          # /api/chat*, /api/conversations*, /api/endpoint/*
router.include_router(hub)           # /api/hub/*
router.include_router(system)        # /api/hardware, /api/resources, /api/network, /api/prompts, /api/plugins
