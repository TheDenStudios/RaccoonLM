"""RaccoonLM v2 — SSE streaming (raw Ollama HTTP for full thinking token support)"""
import json
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
import httpx
from raccoonlm.config import settings

OLLAMA_CHAT_URL = f"{settings.ollama_host}/api/chat"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _raw_stream(model: str, messages: list, tools: list, options: dict = None) -> AsyncGenerator[dict, None]:
    """Stream raw JSON lines from Ollama /api/chat (bypasses Python lib)."""
    payload = {"model": model, "messages": messages, "stream": True}
    if tools: payload["tools"] = tools
    if options: payload["options"] = options
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_CHAT_URL, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Ollama stream failed: HTTP {resp.status_code} {body.decode(errors='ignore')[:300]}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line: continue
                try: yield json.loads(line)
                except json.JSONDecodeError: continue


async def _lmstudio_raw_stream(model: str, messages: list, options: dict = None) -> AsyncGenerator[dict, None]:
    """Stream OpenAI-compatible chunks from LM Studio as Ollama-like chunks."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": options.get("num_predict", 4096) if options else 4096,
    }
    if options and "temperature" in options:
        payload["temperature"] = options["temperature"]
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", "http://localhost:1234/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"LM Studio stream failed: HTTP {resp.status_code} {body.decode(errors='ignore')[:300]}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    yield {"done": True}
                    return
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                token = delta.get("content") or ""
                thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
                chunk = {"message": {"content": token, "thinking": thinking}, "done": False}
                usage = data.get("usage") or {}
                if usage:
                    chunk["prompt_eval_count"] = usage.get("prompt_tokens", 0)
                    chunk["eval_count"] = usage.get("completion_tokens", 0)
                if choice.get("finish_reason"):
                    chunk["done"] = True
                yield chunk


from raccoonlm.plugins.base import Plugin


def _resolve_plugins() -> dict[str, Plugin]:
    """Resolve plugins dynamically from the core router's registry."""
    try:
        from raccoonlm.api.core import _plugins
        return _plugins
    except ImportError:
        return {}


def _find_plugin_for_tool(tool_name: str, plugins: dict[str, Plugin]) -> Optional[Plugin]:
    """Find which plugin owns a given tool function name."""
    # Try prefix match (e.g., 'internet' in 'internet_web_search')
    prefix = tool_name.split("_")[0]
    if prefix in plugins:
        defs = plugins[prefix].get_tool_definitions()
        if any(d["function"]["name"] == tool_name for d in defs):
            return plugins[prefix]
    # Full scan
    for p in plugins.values():
        defs = p.get_tool_definitions()
        if any(d["function"]["name"] == tool_name for d in defs):
            return p
    return None


async def stream_chat(model: str, messages: list, tools: list,
                       system_prompt: str = "",
                       conv_store=None, conv_id: str = None,
                       plugins: dict[str, Plugin] = None,
                       options: dict = None,
                       provider: str = "ollama"
                       ) -> AsyncGenerator[str, None]:
    """SSE-streamed chat. Emits: thinking, token, tool_call, tool_result, done."""
    # Use passed plugins if not None, otherwise resolve from registry
    if plugins is None:
        plugins = _resolve_plugins()
    conv = conv_store.get(conv_id) if conv_store and conv_id else None

    full_msgs = []
    if system_prompt: full_msgs.append({"role": "system", "content": system_prompt})
    if conv: full_msgs.extend(conv["messages"])
    full_msgs.extend(messages)

    full_content = ""
    thinking_content = ""
    tool_calls = None

    # ── First pass ──
    try:
        if provider == "lmstudio":
            first_stream = _lmstudio_raw_stream(model, full_msgs, options)
        else:
            first_stream = _raw_stream(model, full_msgs, tools, options)

        async for chunk in first_stream:
            msg = chunk.get("message", {})
            if c := (msg.get("content", "") or ""):
                full_content += c
                yield sse("token", {"token": c, "done": False})
            if t := (msg.get("thinking", "") or ""):
                thinking_content += t
                yield sse("thinking", {"token": t, "done": False})
            # Check tool_calls in EVERY chunk — Ollama sends them
            # SEPARATELY from the done:true flag!
            tc = msg.get("tool_calls") or chunk.get("tool_calls")
            if tc:
                tool_calls = tc
            if chunk.get("done"):
                # Also check as fallback in case format differs
                if not tool_calls:
                    tool_calls = msg.get("tool_calls") or chunk.get("tool_calls")
                if conv:
                    conv["token_count"] += (chunk.get("prompt_eval_count", 0) or 0)
                    conv["token_count"] += (chunk.get("eval_count", 0) or 0)
    except Exception as e:
        yield sse("token", {"token": f"❌ {e}", "done": False})
        yield sse("done", {"done": True, "model": model, "conv_id": conv_id, "error": str(e)})
        return

    # ── Tool calls (multi-turn: loop until no more tool_calls, max 5 turns) ──
    max_turns = 5
    turn = 0
    while tool_calls and plugins and turn < max_turns:
        turn += 1
        for tc in tool_calls:
            fn_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
            fn_args = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments
            yield sse("tool_call", {"name": fn_name, "arguments": fn_args})
            full_msgs.append({
                "role": "assistant",
                "content": full_content or json.dumps({"tool_call": fn_name, "args": fn_args}),
            })
            plugin = _find_plugin_for_tool(fn_name, plugins)
            if plugin:
                result = await plugin.execute_tool(fn_name, fn_args)
            else:
                result = json.dumps({"error": f"No plugin found for tool: {fn_name}"})
            full_msgs.append({"role": "tool", "content": result})
            yield sse("tool_result", {"tool": fn_name, "result": result[:800]})
        
        # Reset for next pass
        tool_calls = None
        turn_content = ""
        
        async for chunk in _raw_stream(model, full_msgs, [], options):
            msg = chunk.get("message", {})
            if c := (msg.get("content", "") or ""):
                turn_content += c
                full_content += c
                yield sse("token", {"token": c, "done": False})
            if t := (msg.get("thinking", "") or ""):
                thinking_content += t
                yield sse("thinking", {"token": t, "done": False})
            tc = msg.get("tool_calls") or chunk.get("tool_calls")
            if tc:
                tool_calls = tc
            if chunk.get("done") and conv:
                conv["token_count"] += (chunk.get("prompt_eval_count", 0) or 0)
                conv["token_count"] += (chunk.get("eval_count", 0) or 0)
        
        if not turn_content:
            yield sse("token", {"token": "(Tool processed)", "done": False})

    # ── Store in conversation (memory + SQLite) ──
    if conv and messages:
        user_msg = messages[-1].get("content", "") if messages else ""
        if user_msg:
            conv["messages"].append({"role": "user", "content": user_msg})
        conv["messages"].append({"role": "assistant", "content": full_content})
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        if len(conv["messages"]) == 2 and conv["title"] == "Nouvelle conversation":
            conv["title"] = user_msg[:50] + ("…" if len(user_msg) > 50 else "")
        # Persist to SQLite
        from raccoonlm.core import conversations as _conv
        _conv.add_messages(conv_id, user_msg, full_content,
                           conv.get("token_count", 0), thinking_content)

    yield sse("done", {"done": True, "model": model, "conv_id": conv_id})
