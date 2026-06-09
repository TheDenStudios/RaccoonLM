# RaccoonLM v2 — Issue Tracking

## Issue #001: Default model should be dynamically resolved
**Status:** ✅ Fixed (2026-06-01)

### Description
The default model (`qwen3:4b`) was hardcoded in `config.py` as `settings.default_model = "qwen3:4b"`. This created several problems:
- First-time users without qwen3:4b installed would get errors
- The model name was not configurable at runtime
- No fallback chain existed

### Resolution
Implemented a dynamic model resolution system in `raccoonlm/models.py`:

**Priority chain for default model:**
1. Last loaded model (persisted in `model_registry.json`)
2. `RACCOONLM_DEFAULT_MODEL` environment variable
3. First available Ollama model (via `ollama.list()`)
4. `qwen3:4b` hardcoded fallback (rarely reached)

**Files changed:**
- `raccoonlm/config.py` — `get_default_model()` function
- `raccoonlm/models.py` — `resolve_default_model()`, `set_last_model()`, `get_last_model()`
- `raccoonlm/router.py` — All references to `settings.default_model` replaced with `get_default_model()`
- `raccoonlm/main.py` — Model registry initialization at startup

### Migration notes
- `RACCOONLM_DEFAULT_MODEL` env var still works but is now tier-2 priority (behind last-loaded)
- The `model_registry.json` file will be auto-created on first run
