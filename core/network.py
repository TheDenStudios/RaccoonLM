"""RaccoonLM v2 — Network connectivity & auto-hosting mode

Monitors Ollama connectivity and switches between online and offline modes.
Auto-hosting mode: when WiFi/Ollama is disconnected, RaccoonLM can still
serve responses from cached conversations and provide model management UI.
"""

import asyncio
import logging
import time
import httpx

from raccoonlm.config import settings

log = logging.getLogger("uvicorn")

# ── Connectivity State ──
_last_online_check: float = 0
_is_online: bool = True
_offline_since: float = 0
_CHECK_INTERVAL = 15  # seconds between connectivity checks


async def check_ollama_connectivity() -> bool:
    """Check if Ollama is reachable. Returns True if online."""
    global _last_online_check, _is_online, _offline_since

    now = time.time()
    if now - _last_online_check < _CHECK_INTERVAL:
        return _is_online

    _last_online_check = now
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
        if r.status_code == 200:
            if not _is_online:
                log.info("🔄 Ollama reconnected — switching to online mode")
            _is_online = True
            _offline_since = 0
            return True
        else:
            _set_offline()
            return False
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
        _set_offline()
        return False


def _set_offline():
    global _is_online, _offline_since
    if _is_online:
        _offline_since = time.time()
        log.warning("⚠️ Ollama unreachable — switching to auto-hosting mode")
    _is_online = False


def is_online() -> bool:
    return _is_online


def offline_duration() -> float:
    """Seconds since going offline. 0 if online."""
    return time.time() - _offline_since if not _is_online else 0


def get_connectivity_status() -> dict:
    """Get detailed connectivity status."""
    return {
        "online": _is_online,
        "offline_since": _offline_since,
        "offline_seconds": offline_duration(),
        "ollama_host": settings.ollama_host,
        "mode": "online" if _is_online else "auto-hosting",
    }


# ── Auto-Hosting: Simple canned responses when offline ──
AUTO_HOSTING_RESPONSES = {
    "default": (
        "⚠️ RaccoonLM est en mode **auto-hosting** — le backend Ollama est inaccessible.\n\n"
        "Fonctionnalités limitées:\n"
        "• ❌ Nouveaux messages IA (Ollama down)\n"
        "• ✅ Consulter les conversations passées\n"
        "• ✅ Gérer les modèles téléchargés\n"
        "• ✅ Interface de paramètres\n\n"
        "Dès qu'Ollama sera reconnecté, le chat reprendra normalement."
    ),
    "model_not_found": (
        "❌ Modèle introuvable. En mode auto-hosting, les modèles ne sont pas disponibles.\n"
        "Vérifie la connexion à Ollama ou télécharge un modèle via HuggingFace Hub."
    ),
}


def get_auto_response(key: str = "default") -> str:
    """Get a canned response for offline mode."""
    return AUTO_HOSTING_RESPONSES.get(key, AUTO_HOSTING_RESPONSES["default"])
