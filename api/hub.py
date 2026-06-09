"""RaccoonLM v2 — HuggingFace Hub search, files, downloads"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from raccoonlm.core.hub import search_gguf, list_repo_files, group_by_quant, start_download, get_downloads
from raccoonlm.core.cache import cache_hub_search, get_cached_hub_search

hub = APIRouter()


class HubDownloadRequest(BaseModel):
    repo_id: str = ""
    filename: str = ""


# ── Search ──
@hub.get("/api/hub/search")
async def hub_search(q: str = "", limit: int = 15):
    if not q:
        return {"results": []}
    # Check cache first (TTL: 5 min)
    cached = get_cached_hub_search(q)
    if cached is not None:
        return {"results": cached, "cached": True}
    results = search_gguf(q, limit)
    cache_hub_search(q, results)
    return {"results": results}


# ── List files ──
@hub.get("/api/hub/files")
async def hub_files(repo_id: str = ""):
    if not repo_id:
        return {"files": []}
    files = list_repo_files(repo_id)
    return {"files": files, "groups": group_by_quant(files)}


# ── Download ──
@hub.post("/api/hub/download")
async def hub_download(req: HubDownloadRequest):
    if not req.repo_id or not req.filename:
        raise HTTPException(400, "repo_id and filename required")
    return start_download(req.repo_id, req.filename)


# ── Download list ──
@hub.get("/api/hub/downloads")
async def hub_downloads():
    return {"downloads": get_downloads()}
