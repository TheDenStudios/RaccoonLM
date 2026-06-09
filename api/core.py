"""RaccoonLM v2 — Core router: plugin registry, health, shared state"""

import time
from fastapi import APIRouter
from starlette.responses import RedirectResponse

from raccoonlm.config import settings
from raccoonlm.core.llm import ollama_list
from raccoonlm.plugins import InternetPlugin
from raccoonlm.plugins.base import Plugin
from raccoonlm.core.models import get_last_model

core = APIRouter()

# ── Plugin Registry ──
_plugins: dict[str, Plugin] = {}


def init_plugins():
    """Initialize all enabled plugins into the registry."""
    _plugins.clear()
    if settings.internet_plugin:
        p = InternetPlugin()
        _plugins[p.name] = p


async def shutdown_plugins():
    """Shutdown all registered plugins."""
    for p in _plugins.values():
        await p.shutdown()
    _plugins.clear()


def get_plugin(name: str) -> Plugin | None:
    return _plugins.get(name)


def get_all_tools() -> list[dict]:
    tools = []
    for p in _plugins.values():
        tools.extend(p.get_tool_definitions())
    return tools


def get_active_plugins() -> list[str]:
    return list(_plugins.keys())


# ── State ──
_start = time.time()
_current_model: str = ""
_current_provider: str = "ollama"


# ── Root ──
@core.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")


# ── Health ──
@core.get("/api/health")
async def health():
    plugin_list = get_active_plugins()
    model = _current_model if _current_model else "—"
    try:
        await ollama_list()
        ollama_ok = "connected"
    except Exception:
        ollama_ok = "unreachable"
    return {
        "status": "ok",
        "model": model,
        "ollama": ollama_ok,
        "uptime": time.time() - _start,
        "plugins": plugin_list,
        "mode": "online" if ollama_ok == "connected" else "auto-hosting",
    }
