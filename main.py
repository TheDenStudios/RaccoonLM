#!/usr/bin/env python3
"""RaccoonLM v2 — Lightweight Ollama API Wrapper

A minimal local AI model manager that wraps Ollama with:
- REST API on port 5555
- Model management (list/load/unload)
- Chat completion with tool support
- Internet plugin (web search + fetch)
- Health endpoint for watchdog monitoring

Environment variables (RACCOONLM_ prefix):
  HOST          — bind address (default: 0.0.0.0)
  PORT          — listen port (default: 5555)
  OLLAMA_HOST   — Ollama server URL (default: http://localhost:11434)
  DEFAULT_MODEL — default model name (default: qwen3:4b)
  N_CTX         — context window size (default: 8192)
  INTERNET_PLUGIN — enable internet plugin (default: True)
  DEBUG         — debug mode (default: False)
"""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from raccoonlm.config import settings
from raccoonlm.api.routes import router, init_plugins, shutdown_plugins
from raccoonlm.core.models import init_registry, set_last_model, get_last_model
from raccoonlm.config import get_default_model

# ── App ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log = logging.getLogger("uvicorn")
    log.info("╔═══════════════════════════════════════════╗")
    log.info("║     RaccoonLM v2 — Starting Up            ║")
    log.info("╠═══════════════════════════════════════════╣")
    log.info(f"║  Host: {settings.host:<24}       ║")
    log.info(f"║  Port: {settings.port:<24}       ║")
    log.info(f"║  Model: {get_default_model():<23} ║")
    log.info(f"║  Ollama: {settings.ollama_host:<22} ║")
    log.info(f"║  Internet Plugin: {'ON' if settings.internet_plugin else 'OFF':<14}    ║")
    log.info("╚═══════════════════════════════════════════╝")
    init_registry()
    init_plugins()

    # Auto-start OpenAI-compatible endpoint
    try:
        from raccoonlm.core.openai_endpoint import start as ep_start
        t = threading.Thread(target=ep_start, args=(5556, get_default_model()), daemon=True)
        t.start()
        log.info(f"║  OpenAI Endpoint: http://localhost:5556/v1  ║")
    except Exception as e:
        log.warning(f"OpenAI endpoint not started: {e}")

    yield
    await shutdown_plugins()

app = FastAPI(
    title="RaccoonLM v2",
    description="Local AI Model Manager — Ollama wrapper",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (chat UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(router)


# ── CLI ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)

    # Graceful shutdown
    def _signal_handler(sig, frame):
        logging.getLogger("uvicorn").info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    uvicorn.run(
        "raccoonlm.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
