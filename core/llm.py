"""RaccoonLM v2 — Direct llama.cpp wrappers (standalone, no Ollama/LM Studio dependency)"""

import httpx
from raccoonlm.config import settings


async def llamacpp_chat_sync(model: str, messages: list,
                             temperature: float = None,
                             options: dict = None) -> dict:
    """Send a chat request directly to llama.cpp llama-server's OpenAI API."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature is not None else 0.7,
        "max_tokens": options.get("num_predict", 4096) if options else 4096,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(settings.llama_cpp_host.rstrip("/") + "/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage", {})
        return {
            "message": {"role": "assistant", "content": msg.get("content", "")},
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "done": True,
        }


def format_bytes(size: int) -> str:
    if size < 1024: return f"{size}B"
    if size < 1024**2: return f"{size/1024:.1f}KB"
    if size < 1024**3: return f"{size/1024**2:.1f}MB"
    return f"{size/1024**3:.1f}GB"
