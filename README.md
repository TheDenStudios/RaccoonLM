# 🦝 RaccoonLM

**A private, local-first AI model manager.**

RaccoonLM is a FastAPI-powered local first AI Model Manager with a browser chat UI, model loading, streaming responses, conversation history, HuggingFace GGUF discovery/downloads, resource monitoring, plugin tools, and an optional OpenAI-compatible endpoint.

It is designed for people who want a self-hosted alternative experience while keeping the stack lightweight.

![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-local%20API-green)
![Ollama](https://img.shields.io/badge/Ollama-supported-green)
![LM Studio](https://img.shields.io/badge/LM%20Studio-supported-green)
![llama.cpp](https://img.shields.io/badge/llama.cpp-direct%20GGUF-green)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Local chat UI** — dark neon Raccoon/Matrix-inspired browser interface.
- **Multi-provider model management** — list and load models from:
  - Ollama (`http://localhost:11434`)
  - LM Studio local server (`http://localhost:1234`)
  - llama.cpp `llama-server` direct GGUF loading (`http://localhost:8080`)
- **Provider-first model selector** — bottom-left settings modal separates provider selection from model selection.
- **Streaming chat** — Server-Sent Events token streaming with support for thinking/reasoning chunks.
- **Conversation history** — persistent SQLite conversations with create, rename, delete, and reload support.
- **HuggingFace GGUF hub** — search GGUF repositories, inspect quantizations, download files, and import into Ollama.
- **OpenAI-compatible endpoint** — optional `/v1/chat/completions` server on port `5556` for external tools.
- **Internet plugin** — DuckDuckGo search and page fetch tools for models that can use tool calls.
- **Hardware/resource monitor** — CPU, RAM, VRAM, GPU detection, and live HUD.
- **System prompts** — built-in Raccoon preset and custom prompt saving.
- **Generation controls** — temperature, max tokens, GPU layer controls, stop button, and token stats.
- **Private by default** — designed to run locally on your own machine/server.

---

## Screens / UI

RaccoonLM serves a single-page web app at:

```text
http://localhost:5555
```

Main UI areas:

- Left sidebar: conversations
- Center: chat stream
- Bottom-left gear: settings modal for provider/model loading
- Right sidebar: phase-1 settings sections still being migrated, including endpoint, hardware, prompts, HuggingFace, downloads, and inference parameters
- Floating HUD: system/resource monitor

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Local model backend | Ollama |
| LM Studio integration | OpenAI-compatible local API |
| llama.cpp integration | Direct `llama-server` OpenAI-compatible API + GGUF scanning |
| Storage | SQLite + JSON model registry |
| Frontend | Vanilla HTML/CSS/JS |
| Streaming | Server-Sent Events |
| Model hub | HuggingFace Hub |
| Plugins | Python plugin interface |
| Optional API bridge | Flask OpenAI-compatible endpoint |

---

## Project Structure

```text
raccoonlm/
├── main.py                 # FastAPI app entrypoint
├── config.py               # Settings and dynamic default model resolution
├── pyproject.toml          # Project metadata
├── requirements.txt        # Runtime dependencies
├── static/
│   └── index.html          # Browser UI
├── api/
│   ├── routes.py           # Router aggregator
│   ├── core.py             # Health, root redirect, shared state, plugin registry
│   ├── models.py           # Model listing/loading/unloading endpoints
│   ├── chat.py             # Chat, streaming, conversations, OpenAI endpoint controls
│   ├── hub.py              # HuggingFace search/download endpoints
│   └── system.py           # Hardware, resources, prompts, plugins, network status
├── core/
│   ├── models.py           # Ollama + LM Studio provider logic and model registry
│   ├── llm.py              # Ollama and LM Studio chat wrappers
│   ├── streaming.py        # Ollama + LM Studio SSE streaming
│   ├── conversations.py    # SQLite persistence
│   ├── hub.py              # HuggingFace GGUF search/download/import logic
│   ├── network.py          # Connectivity and fallback responses
│   ├── cache.py            # VRAM / hub search cache helpers
│   ├── schemas.py          # Pydantic schemas
│   └── openai_endpoint.py  # Optional OpenAI-compatible API server
├── plugins/
│   ├── base.py             # Plugin base class
│   └── internet.py         # Web search + web fetch tool plugin
└── tests/
    └── test_lmstudio_loading_streaming.py
```

---

## Requirements

- Linux/macOS/Windows with Python **3.11+**
- Recommended: Python **3.12**
- At least one local model provider:
  - [Ollama](https://ollama.com/) for Ollama models
  - [LM Studio](https://lmstudio.ai/) with the local server enabled for LM Studio models
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` for direct GGUF loading

Optional:

- GPU acceleration through your model backend
- HuggingFace access for GGUF search/downloads

---

## Installation

Because this repo is currently structured as a Python package directory, run it from the parent directory of the cloned folder.

```bash
# Clone
cd ~/Desktop
git clone https://github.com/TheDenStudios/RaccoonLM.git raccoonlm

# Create virtual environment inside the project
python3 -m venv raccoonlm/.venv
source raccoonlm/.venv/bin/activate

# Install dependencies
pip install -r raccoonlm/requirements.txt
```

Install and start Ollama if you want to use Ollama models:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
```

For LM Studio support:

1. Open LM Studio.
2. Download/load a model.
3. Start the local server.
4. Confirm it responds on `http://localhost:1234/v1/models`.

For direct llama.cpp support:

1. Install or build `llama-server` from llama.cpp.
2. Make sure `llama-server` is on `PATH`, or set `RACCOONLM_LLAMA_CPP_COMMAND=/absolute/path/to/llama-server`.
3. Put `.gguf` files in a scanned folder such as `~/Downloads`, `~/Desktop`, LM Studio cache, HuggingFace cache, or set `RACCOONLM_LLAMA_CPP_MODEL_DIRS` to extra directories separated by `:` on Linux/macOS.
4. In RaccoonLM settings, choose the `llama.cpp` provider and load a GGUF model. RaccoonLM starts `llama-server` on `http://localhost:8080` and verifies it with a tiny chat request.

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
RACCOONLM_OLLAMA_HOST=http://localhost:11434 \
raccoonlm/.venv/bin/python -m raccoonlm.main
```

---

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `RACCOONLM_HOST` | `0.0.0.0` | Bind address |
| `RACCOONLM_PORT` | `5555` | Main FastAPI/UI port |
| `RACCOONLM_OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `RACCOONLM_LLAMA_CPP_HOST` | `http://localhost:8080` | llama.cpp `llama-server` OpenAI-compatible URL |
| `RACCOONLM_LLAMA_CPP_COMMAND` | `llama-server` | Command/path used to launch llama.cpp directly |
| `RACCOONLM_LLAMA_CPP_MODEL_DIRS` | empty | Extra GGUF scan directories, separated by `:` |
| `RACCOONLM_LLAMA_CPP_GPU_LAYERS` | `999` | GPU layers passed to `llama-server --n-gpu-layers` |
| `RACCOONLM_DEFAULT_MODEL` | dynamic | Optional forced default model |
| `RACCOONLM_INTERNET_PLUGIN` | `true` | Enable web search/fetch plugin |
| `RACCOONLM_DEBUG` | `false` | Debug logging |

Default model resolution:

1. `RACCOONLM_DEFAULT_MODEL` if set
2. Last loaded model in `model_registry.json`
3. First available Ollama model
4. `qwen3:4b` fallback

---

## API Overview

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirects to `/static/index.html` |
| `/api/health` | GET | Service health, current model, plugin list |
| `/api/models` | GET | List local models from all providers |
| `/api/models/current` | GET | Current model and provider |
| `/api/models/load` | POST | Load/verify selected model |
| `/api/models/unload` | POST | Unload current model |
| `/api/models/load-status` | GET | Ollama runner/VRAM status |
| `/api/local-models` | GET | Local models list for UI |
| `/api/chat` | POST | Non-streaming chat |
| `/api/chat/stream` | POST | Streaming chat via SSE |
| `/api/conversations` | GET/POST | List/create conversations |
| `/api/conversations/{id}` | GET/PATCH/DELETE | Read/rename/delete conversation |
| `/api/conversations/{id}/chat` | POST | Chat inside a stored conversation |
| `/api/hub/search` | GET | Search HuggingFace GGUF models |
| `/api/hub/files` | GET | List GGUF files for a repo |
| `/api/hub/download` | POST | Download/import a GGUF file |
| `/api/hub/downloads` | GET | Download status list |
| `/api/hardware` | GET | Hardware detection |
| `/api/resources` | GET | CPU/RAM/VRAM monitor data |
| `/api/network` | GET | Connectivity status |
| `/api/prompts` | GET/POST | List/create prompts |
| `/api/plugins` | GET | Enabled plugins and tools |
| `/api/endpoint/start` | POST | Start OpenAI-compatible endpoint |
| `/api/endpoint/stop` | POST | Stop OpenAI-compatible endpoint |
| `/api/endpoint/status` | GET | Endpoint status |

Example load request:

```bash
curl -s -X POST http://localhost:5555/api/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"qwen3:4b","provider":"ollama"}'
```

Example chat request:

```bash
curl -s -X POST http://localhost:5555/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"temperature":0}'
```

---

## OpenAI-Compatible Endpoint

RaccoonLM can expose a minimal OpenAI-compatible endpoint, usually on:

```text
http://localhost:5556/v1
```

Supported routes:

- `GET /v1/models`
- `POST /v1/chat/completions`

This is useful for connecting local tools that expect an OpenAI-style API base URL.

---

## Provider Notes

### Ollama

- Model listing uses `ollama.list()`.
- Loading uses a tiny generation with `keep_alive=-1` to force the model into memory.
- Unloading uses Ollama `keep_alive=0`.
- VRAM/load-status detection is Ollama-specific.

### LM Studio

- Model listing uses `GET http://localhost:1234/v1/models`.
- Loading tries LM Studio's explicit load endpoint, then verifies with a tiny chat completion.
- RaccoonLM only marks LM Studio models as loaded after proof that the model can respond.
- Streaming routes through LM Studio's OpenAI-compatible SSE stream.

---

## Tests

Run from the parent directory:

```bash
raccoonlm/.venv/bin/python -m unittest discover -s raccoonlm/tests -v
```

Current regression coverage includes:

- rejecting empty LM Studio completion responses during load verification
- accepting visible content/reasoning/token output as proof of load
- routing streaming chat through LM Studio when the provider is `lmstudio`

---

## Development Notes

Useful checks:

```bash
# Python syntax compile
raccoonlm/.venv/bin/python -m compileall -q raccoonlm

# Unit tests
raccoonlm/.venv/bin/python -m unittest discover -s raccoonlm/tests -v

# API health
curl -s http://localhost:5555/api/health

# Model list
curl -s http://localhost:5555/api/models | python3 -m json.tool
```

Runtime files are intentionally ignored by Git:

- `.venv/`
- `__pycache__/`
- `raccoonlm.db*`
- `model_registry.json`
- `.env`

---

## Roadmap Ideas

- Finish migrating right-sidebar settings into the settings modal.
- Add a dedicated provider status/config page.
- Add configurable OpenAI-compatible providers beyond LM Studio.
- Improve OpenAI endpoint streaming fidelity.
- Add screenshots and packaged install command.
- Add systemd user-service template.

---

## License

MIT License. See `LICENSE`.

Built by TheDenStudios / Cyber.
