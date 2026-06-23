"""RaccoonLM v2 — System: hardware, resources, prompts, plugins list"""

import subprocess
from fastapi import APIRouter, HTTPException

from raccoonlm.core import conversations as conv
from raccoonlm.api.core import _current_model, get_active_plugins

system = APIRouter()


# ── Hardware detection ──
@system.get("/api/hardware")
async def hardware():
    return conv.detect_hardware()


# ── Resource monitor (HUD data) ──
@system.get("/api/resources")
async def resources():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0)
        ram = psutil.virtual_memory()
        ram_pct = ram.percent
        ram_gb = round(ram.used / (1024**3), 1)
        ram_total = round(ram.total / (1024**3), 1)
    except Exception:
        cpu, ram_pct, ram_gb, ram_total = 0, 0, 0, 28

    from raccoonlm.core.cache import get_vram
    vram, vram_total = get_vram(gb=True)

    return {
        "cpu": cpu,
        "ram_pct": ram_pct,
        "ram_gb": ram_gb,
        "ram_total": ram_total,
        "vram": vram if vram else None,
        "vram_total": vram_total if vram_total else None,
        "model": _current_model or "—",
        "plugins": get_active_plugins(),
        "mode": "online",
    }


# ── Network status ──
@system.get("/api/network")
async def network_status():
    return {"online": True, "mode": "online"}


# ── Plugins list ──
@system.get("/api/plugins")
async def list_plugins():
    from raccoonlm.api.core import _plugins
    result = []
    for name, p in _plugins.items():
        tools = [{"name": d["function"]["name"], "description": d["function"]["description"]}
                 for d in p.get_tool_definitions()]
        result.append({"name": name, "tools": tools})
    return {"plugins": result}


# ── System Prompts ──
@system.get("/api/prompts")
async def list_prompts():
    return conv.list_prompts()


@system.get("/api/prompts/{pid}")
async def get_prompt(pid: str):
    p = conv.get_prompt(pid)
    if not p:
        raise HTTPException(404, "Not found")
    return p


@system.post("/api/prompts")
async def create_prompt(name: str = "", content: str = ""):
    if not content:
        raise HTTPException(400, "content required")
    return conv.save_prompt(name or "Sans titre", content)


@system.delete("/api/prompts/{pid}")
async def delete_prompt(pid: str):
    conv.delete_prompt(pid)
    return {"status": "ok"}
