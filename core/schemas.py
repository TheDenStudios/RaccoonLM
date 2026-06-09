"""RaccoonLM v2 — Pydantic schemas"""
from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    ollama: str = ""
    uptime: float = 0.0
    plugins: list[str] = []


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    options: Optional[dict] = None


class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
