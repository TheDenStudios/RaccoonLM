"""RaccoonLM v2 — Connectivity
Simplified: RaccoonLM manages llama-server directly, no external provider to check.
"""

import logging
log = logging.getLogger("uvicorn")

_connected: bool = True


async def check_connectivity() -> bool:
    """Stub — always 'connected' since RaccoonLM manages llama-server directly."""
    return True


def is_online() -> bool:
    return _connected


def get_connectivity_status() -> dict:
    return {
        "online": _connected,
        "mode": "online",
    }
