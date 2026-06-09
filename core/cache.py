"""RaccoonLM v2 — Simple in-memory cache for expensive operations

Reduces redundant subprocess calls, HuggingFace API calls, and VRAM reads.
Calling check() returns cached value if within TTL, else runs fetcher function.
"""

import time
import logging

log = logging.getLogger("uvicorn")


class TTLCache:
    """Time-To-Live cache for a single value."""

    def __init__(self, ttl_seconds: int = 30):
        self._value = None
        self._ttl = ttl_seconds
        self._updated = 0

    def get(self, fetcher=None):
        """Return cached value or run fetcher() if expired."""
        now = time.time()
        if self._value is not None and (now - self._updated) < self._ttl:
            return self._value
        if fetcher:
            try:
                self._value = fetcher()
                self._updated = now
            except Exception as e:
                log.warning(f"Cache fetcher failed: {e}")
                # Return stale value if fetcher fails
        return self._value

    def invalidate(self):
        """Force next call to refresh."""
        self._value = None
        self._updated = 0

    def set(self, value):
        """Manually set cached value."""
        self._value = value
        self._updated = time.time()


# ── Shared cache instances ──

# VRAM reads: cache for 5 seconds (resMon calls every 3s)
vram_cache = TTLCache(ttl_seconds=5)

# Hardware detection (subprocess-heavy): cache for 60 seconds
hardware_cache = TTLCache(ttl_seconds=60)

# Hub search results: cache for 5 minutes (HF API has rate limits)
hub_search_cache: dict[str, dict] = {}
HUB_SEARCH_TTL = 300  # 5 minutes

# VRAM reading function (cached)
def get_vram(gb: bool = True) -> tuple:
    """Get (used_bytes, total_bytes). Cached for 5s."""
    import subprocess
    def _read():
        used, total = 0, 0
        try:
            r = subprocess.run(
                ["cat", "/sys/class/drm/renderD128/device/mem_info_vram_total"],
                capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                total = int(r.stdout.strip())
            r2 = subprocess.run(
                ["cat", "/sys/class/drm/renderD128/device/mem_info_vram_used"],
                capture_output=True, text=True, timeout=2
            )
            if r2.returncode == 0:
                used = int(r2.stdout.strip())
        except Exception:
            pass
        return used, total
    
    result = vram_cache.get(_read)
    if result is None:
        return (0, 0)
    if gb and result[0] > 0:
        return (round(result[0] / (1024**3), 1), round(result[1] / (1024**3), 1))
    return result


def invalidate_vram():
    vram_cache.invalidate()
    hardware_cache.invalidate()


def cache_hub_search(query: str, results: list) -> list:
    """Cache hub search results with TTL."""
    key = query.lower().strip()
    hub_search_cache[key] = {"results": results, "time": time.time()}
    # Clean old entries
    now = time.time()
    stale = [k for k, v in hub_search_cache.items() if now - v["time"] > HUB_SEARCH_TTL]
    for k in stale:
        del hub_search_cache[k]
    return results


def get_cached_hub_search(query: str) -> list | None:
    """Get cached hub search results if fresh."""
    key = query.lower().strip()
    entry = hub_search_cache.get(key)
    if entry and (time.time() - entry["time"]) < HUB_SEARCH_TTL:
        return entry["results"]
    return None
