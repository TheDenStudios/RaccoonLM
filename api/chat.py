"""RaccoonLM v2 — Chat, streaming, conversations, OpenAI endpoint"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from raccoonlm.config import get_default_model
from raccoonlm.core.schemas import ChatRequest
from raccoonlm.core.llm import chat_sync, lmstudio_chat_sync
from raccoonlm.core import conversations as conv
from raccoonlm.core.streaming import stream_chat
from raccoonlm.core.network import check_ollama_connectivity, get_auto_response
from raccoonlm.api.core import get_plugin, get_all_tools, _plugins
import raccoonlm.api.core as core_state

chat = APIRouter()


# ── Chat (non-streaming) ──
@chat.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    model = request.model or core_state._current_model or get_default_model()

    if core_state._current_provider != "lmstudio" and not await check_ollama_connectivity():
        return {
            "model": model,
            "message": {"role": "assistant", "content": get_auto_response("default")},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "done": True,
            "mode": "auto-hosting",
        }

    msgs = [m.model_dump() for m in request.messages]
    if request.system_prompt:
        msgs.insert(0, {"role": "system", "content": request.system_prompt})

    tools = get_all_tools()

    try:
        if core_state._current_provider == "lmstudio":
            # Route to LM Studio API
            response = await lmstudio_chat_sync(model, msgs, request.temperature, request.options)
            return {
                "model": model,
                "message": response["message"],
                "usage": response["usage"],
                "done": True,
                "provider": "lmstudio",
            }

        response = await chat_sync(model, msgs, tools, request.temperature, request.options)

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                plugin = get_plugin(tc.function.name.split("_")[0]) if "_" in tc.function.name else None
                if not plugin:
                    for p in _plugins.values():
                        defs = p.get_tool_definitions()
                        if any(d["function"]["name"] == tc.function.name for d in defs):
                            plugin = p
                            break
                if plugin:
                    result = await plugin.execute_tool(tc.function.name, tc.function.arguments)
                    msgs.append({"role": "assistant", "content": response.message.content or json.dumps(
                        {"tool_call": tc.function.name, "args": tc.function.arguments})})
                    msgs.append({"role": "tool", "content": result})
                    response = await chat_sync(model, msgs, [], request.temperature, request.options)

        return {
            "model": model,
            "message": {"role": "assistant", "content": response.message.content or ""},
            "usage": {"prompt_tokens": getattr(response, "prompt_eval_count", 0),
                       "completion_tokens": getattr(response, "eval_count", 0)},
            "done": True,
        }
    except Exception as e:
        if not await check_ollama_connectivity():
            return {
                "model": model,
                "message": {"role": "assistant", "content": get_auto_response("default")},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "done": True,
                "mode": "auto-hosting",
            }
        raise HTTPException(500, str(e))


# ── Streaming chat ──
@chat.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, conv_id: str = Query(None)):
    model = request.model or core_state._current_model or get_default_model()
    msgs = [m.model_dump() for m in request.messages]
    tools = get_all_tools()

    return StreamingResponse(
        stream_chat(model, msgs, tools, request.system_prompt or "",
                     conv, conv_id, _plugins, request.options,
                     provider=core_state._current_provider),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# ── Conversations ──
@chat.post("/api/conversations")
async def create_conv(title: str = "Nouvelle conversation",
                       model: str = "", system_prompt: str = ""):
    return conv.create(title, model or core_state._current_model or get_default_model(), system_prompt)


@chat.get("/api/conversations")
async def list_convs():
    return conv.list_all()


@chat.get("/api/conversations/{cid}")
async def get_conv(cid: str):
    c = conv.get(cid)
    if not c:
        raise HTTPException(404, "Not found")
    return c


@chat.delete("/api/conversations/{cid}")
async def delete_conv(cid: str):
    conv.delete(cid)
    return {"status": "ok"}


@chat.post("/api/conversations/{cid}/chat")
async def conv_chat(cid: str, request: ChatRequest):
    c = conv.get(cid)
    if not c:
        raise HTTPException(404, "Not found")

    model = request.model or c["model"] or core_state._current_model or get_default_model()
    user_msg = request.messages[-1].content if request.messages else ""

    msgs = []
    if c["system_prompt"]:
        msgs.append({"role": "system", "content": c["system_prompt"]})
    msgs.extend(c["messages"])
    if user_msg:
        msgs.append({"role": "user", "content": user_msg})

    tools = get_all_tools()

    try:
        if core_state._current_provider == "lmstudio":
            # Route to LM Studio API for conversation chat
            response = await lmstudio_chat_sync(model, msgs, request.temperature, request.options)
            conv.add_messages(cid, user_msg, response["message"]["content"],
                              response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"])
            return {
                "model": model, "conversation_id": cid,
                "message": response["message"],
                "usage": response["usage"],
                "done": True,
                "provider": "lmstudio",
            }

        response = await chat_sync(model, msgs, tools, request.temperature, request.options)
        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                plugin = get_plugin(tc.function.name.split("_")[0]) if "_" in tc.function.name else None
                if not plugin:
                    for p in _plugins.values():
                        defs = p.get_tool_definitions()
                        if any(d["function"]["name"] == tc.function.name for d in defs):
                            plugin = p
                            break
                if plugin:
                    result = await plugin.execute_tool(tc.function.name, tc.function.arguments)
                    msgs.append({"role": "assistant", "content": response.message.content or json.dumps(
                        {"tool_call": tc.function.name, "args": tc.function.arguments})})
                    msgs.append({"role": "tool", "content": result})
                    response = await chat_sync(model, msgs, [], request.temperature, request.options)

        conv.add_messages(cid, user_msg, response.message.content or "",
                          getattr(response, "prompt_eval_count", 0) + getattr(response, "eval_count", 0))

        return {
            "model": model, "conversation_id": cid,
            "message": {"role": "assistant", "content": response.message.content or ""},
            "usage": {"prompt_tokens": getattr(response, "prompt_eval_count", 0),
                       "completion_tokens": getattr(response, "eval_count", 0)},
            "done": True,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Conversation PATCH (rename + archive) ──
@chat.patch("/api/conversations/{cid}")
async def patch_conversation(cid: str, title: str = None, archived: bool = None):
    c = conv.get(cid)
    if not c:
        raise HTTPException(404, "Not found")
    conn = conv._conn()
    if title is not None:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
    if archived is not None:
        conn.execute("UPDATE conversations SET archived=? WHERE id=?", (1 if archived else 0, cid))
    conn.commit()
    return conv.get(cid)


# ── OpenAI Endpoint ──
_openai_endpoint = None


@chat.post("/api/endpoint/start")
async def endpoint_start(port: int = 5556):
    global _openai_endpoint
    from raccoonlm.core.openai_endpoint import start as ep_start
    result = ep_start(port, core_state._current_model or get_default_model())
    _openai_endpoint = result
    return result


@chat.post("/api/endpoint/stop")
async def endpoint_stop():
    global _openai_endpoint
    from raccoonlm.core.openai_endpoint import stop as ep_stop
    result = ep_stop()
    _openai_endpoint = None
    return result


@chat.get("/api/endpoint/status")
async def endpoint_status():
    from raccoonlm.core.openai_endpoint import is_running
    return {"running": is_running(), "port": 5556 if is_running() else None}
