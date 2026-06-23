# 🦝 RaccoonLM

**A standalone, private, local-first AI model manager.**  
*No Ollama. No LM Studio. Just direct GGUF loading via llama.cpp.*

RaccoonLM is a FastAPI-powered local AI model manager with a browser chat UI, direct GGUF model loading, streaming responses, conversation history, HuggingFace GGUF discovery, resource monitoring, plugin tools, and an OpenAI-compatible endpoint.

It is a self-hosted alternative to Ollama and LM Studio — designed to run entirely on your machine with no external dependencies.

![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-local%20API-green)
![llama.cpp](https://img.shields.io/badge/llama.cpp-direct%20GGUF-green)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Local chat UI** — dark neon Raccoon/Matrix-inspired browser interface.
- **Direct GGUF loading** — load any `.gguf` model via llama.cpp `llama-server` — no Ollama or LM Studio required.
- **Auto-discovery** — scans `~/Downloads`, `~/Desktop`, and HuggingFace cache for GGUF files automatically.
- **Streaming chat** — Server-Sent Events token streaming with thinking/reasoning support.
- **Conversation history** — persistent SQLite conversations with create, rename, delete, and reload.
- **HuggingFace GGUF hub** — search GGUF repositories, inspect quantizations, download files.
- **OpenAI-compatible endpoint** — optional `/v1/chat/completions` server for external tools.
- **Internet plugin** — DuckDuckGo search and page fetch tools.
- **Hardware/resource monitor** — CPU, RAM, VRAM, GPU detection, and live HUD.
- **System prompts** — built-in Raccoon preset and custom prompt saving.
- **Generation controls** — temperature, max tokens, GPU layer controls, stop button, token stats.
- **Private by default** — designed to run locally on your own machine/server.

---

## Why not Ollama or LM Studio?

RaccoonLM is an **alternative** to Ollama and LM Studio, not a wrapper.  
It loads GGUF models directly through `llama-server` without any intermediary:

- **No external daemon** — RaccoonLM starts and manages `llama-server` itself.
- **No API proxy** — direct communication with the model backend.
- **Lighter stack** — Python + llama.cpp, nothing else required.

---

## Screens / UI

RaccoonLM serves a single-page web app at:

```text
http://localhost:5555
```

Main UI areas:

- Left sidebar: conversations
- Center: chat stream
- Bottom-left gear: settings modal for model loading/inference
- Right sidebar: hardware, prompts, HuggingFace, downloads, inference params
- Floating HUD: system/resource monitor

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Model backend | llama.cpp `llama-server` (direct GGUF) |
| Storage | SQLite + JSON model registry |
| Frontend | Vanilla HTML/CSS/JS |
| Streaming | Server-Sent Events |
| Model hub | HuggingFace Hub |
| Plugins | Python plugin interface |
| Optional API bridge | OpenAI-compatible endpoint |

---

## Project Structure

```text
raccoonlm/
├── main.py                 # FastAPI app entrypoint
├── config.py               # Settings
├── pyproject.toml          # Project metadata
├── requirements.txt        # Runtime dependencies
├── static/
│   └── index.html          # Browser UI
├── api/
│   ├── routes.py           # Router aggregator
│   ├── core.py             # Health, root redirect, shared state, plugin registry
│   ├── models.py           # Model listing/loading/unloading endpoints
│   ├── chat.py             # Chat, streaming, conversations, OpenAI endpoint
│   ├── hub.py              # HuggingFace search/download endpoints
│   └── system.py           # Hardware, resources, prompts, plugins
├── core/
│   ├── models.py           # llama.cpp model registry & GGUF discovery
│   ├── llm.py              # llama.cpp chat wrapper
│   ├── streaming.py        # llama.cpp SSE streaming
│   ├── conversations.py    # SQLite persistence
│   ├── hub.py              # HuggingFace GGUF search/download logic
│   ├── network.py          # Connectivity (simplified)
│   ├── cache.py            # VRAM / hub search cache helpers
│   ├── schemas.py          # Pydantic schemas
│   └── openai_endpoint.py  # Optional OpenAI-compatible server
├── plugins/
│   ├── base.py             # Plugin base class
│   └── internet.py         # Web search + web fetch plugin
└── tests/
    └── test_llamacpp_loading_streaming.py
```

---

## Requirements

- Linux/macOS/Windows with **Python 3.11+**
- Recommended: **Python 3.12**
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` on your `PATH`, or set `RACCOONLM_LLAMA_CPP_COMMAND`

Optional:

- GPU acceleration for faster inference
- HuggingFace access for GGUF search/downloads

---

## Installation

```bash
# Clone
cd ~/Desktop
git clone https://github.com/TheDenStudios/RaccoonLM.git raccoonlm

# Create virtual environment inside the project
python3 -m venv raccoonlm/.venv
source raccoonlm/.venv/bin/activate

# Install dependencies
pip install -r raccoonlm/requirements.txt

# Make sure llama-server is available
# If you have llama.cpp built:
#   export RACCOONLM_LLAMA_CPP_COMMAND=/path/to/llama-server
# Or install via package manager (Linux):
#   sudo apt install llama.cpp
```

### Quickstart: using llama.cpp direct GGUF

1. Install or build `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp).
2. Ensure `llama-server` is on `PATH`, or set `RACCOONLM_LLAMA_CPP_COMMAND=/absolute/path/to/llama-server`.
3. Put `.gguf` files in `~/Downloads` or `~/Desktop` — RaccoonLM auto-discovers them.
4. Start RaccoonLM, open `http://localhost:5555`, choose `llama.cpp` provider, and load a GGUF.

---

## Running RaccoonLM

From the parent directory of the `raccoonlm/` folder:

```bash
raccoonlm/.venv/bin/python -m raccoonlm.main
```

Then open:

```text
http://localhost:5555
```

Environment variables use the `RACCOONLM_` prefix:

```bash
RACCOONLM_PORT=5555 \
RACCOONLM_HOST=0.0.0.0 \
RACCOONLM_LLAMA_CPP_COMMAND=/usr/local/bin/llama-server \
RACCOONLM_LLAMA_CPP_GPU_LAYERS=99 \
raccoonlm/.venv/bin/python -m raccoonlm.main
```

---

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `RACCOONLM_HOST` | `0.0.0.0` | Bind address |
| `RACCOONLM_PORT` | `5555` | Main FastAPI/UI port |
| `RACCOONLM_LLAMA_CPP_HOST` | `http://localhost:8080` | llama.cpp `llama-server` URL |
| `RACCOONLM_LLAMA_CPP_COMMAND` | `llama-server` | Path to `llama-server` binary |
| `RACCOONLM_LLAMA_CPP_MODEL_DIRS` | empty | Extra GGUF scan directories (`:` separated) |
| `RACCOONLM_LLAMA_CPP_GPU_LAYERS` | `999` | GPU layers passed to `llama-server` |
| `RACCOONLM_INTERNET_PLUGIN` | `True` | Enable internet search/fetch plugin |
| `RACCOONLM_DB_PATH` | `./raccoonlm.db` | SQLite database location |

---

## Development

```bash
# Activate venv
source raccoonlm/.venv/bin/activate

# Run tests
cd ~/Desktop  # from parent of raccoonlm/
python -m pytest raccoonlm/tests/ -v

# Run in debug mode
RACCOONLM_DEBUG=true raccoonlm/.venv/bin/python -m raccoonlm.main
```

---

## License

MIT
