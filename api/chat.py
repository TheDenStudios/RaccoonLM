"""RaccoonLM v2 — Chat, streaming, conversations, OpenAI endpoint"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from raccoonlm.config import get_default_model
from raccoonlm.core.schemas import ChatRequest
from raccoonlm.core.llm import llamacpp_chat_sync
from raccoonlm.core import conversations as conv
from raccoonlm.core.streaming import stream_chat
from raccoonlm.api.core import get_plugin, get_all_tools, _plugins
import raccoonlm.api.core as core_state

chat = APIRouter()


# ── Chat (non-streaming) ──
@chat.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    model = request.model or core_state._current_model or get_default_model()

    msgs = [m.model_dump() for m in request.messages]
    if request.system_prompt:
        msgs.insert(0, {"role": "system", "content": request.system_prompt})

    tools = get_all_tools()

    try:
        response = await llamacpp_chat_sync(model, msgs, request.temperature, request.options)
        return {
            "model": model,
            "message": response["message"],
            "usage": response["usage"],
            "done": True,
            "provider": core_state._current_provider,
        }
    except Exception as e:
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
        response = await llamacpp_chat_sync(model, msgs, request.temperature, request.options)
        conv.add_messages(cid, user_msg, response["message"]["content"],
                          response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"])
        return {
            "model": model, "conversation_id": cid,
            "message": response["message"],
            "usage": response["usage"],
            "done": True,
            "provider": core_state._current_provider,
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
    cur = conn.execute("UPDATE conversations SET title = COALESCE(?, title), archived = COALESCE(?, archived) WHERE id = ?",
                       (title, int(archived) if archived is not None else None, cid))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Not found")
    return {"status": "ok"}


# ── OpenAI-compatible endpoint toggle ──
@chat.post("/api/endpoint/start")
async def start_endpoint(port: int = 5556):
    try:
        from raccoonlm.core.openai_endpoint import start as ep_start
        import threading
        t = threading.Thread(target=ep_start, args=(port, get_default_model()), daemon=True)
        t.start()
        return {"status": "started", "url": f"http://localhost:{port}/v1"}
    except Exception as e:
        raise HTTPException(500, str(e))


@chat.post("/api/endpoint/stop")
async def stop_endpoint():
    try:
        from raccoonlm.core.openai_endpoint import stop as ep_stop
        ep_stop()
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
