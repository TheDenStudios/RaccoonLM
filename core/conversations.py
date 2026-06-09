"""RaccoonLM v2 — Conversation store (SQLite)"""
import sqlite3, uuid, os, threading, subprocess
from datetime import datetime, timezone
from raccoonlm.config import settings

DB_PATH = settings.db_path

_local = threading.local()

def _conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn

def _init():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT 'New Chat',
            model TEXT DEFAULT '', system_prompt TEXT DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, token_count INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, reasoning TEXT DEFAULT '',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_msgs_conv ON messages(conv_id);
        CREATE INDEX IF NOT EXISTS idx_msgs_timestamp ON messages(conv_id, id);
        CREATE TABLE IF NOT EXISTS system_prompts (
            id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
        );
    """)
    # Migration: add reasoning column if missing (safe re-run)
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT DEFAULT ''")
    except Exception:
        pass
    conn.commit()

def _id(): return uuid.uuid4().hex[:12]
def _now(): return datetime.now(timezone.utc).isoformat()

def create(title="Nouvelle conversation", model="", system_prompt="") -> dict:
    _init()
    cid, t = _id(), _now()
    conn = _conn()
    conn.execute("INSERT INTO conversations (id,title,model,system_prompt,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                 (cid, title, model, system_prompt, t, t))
    conn.commit()
    return get(cid)

def list_all() -> list:
    _init()
    rows = _conn().execute("SELECT id,title,model,created_at,updated_at,token_count FROM conversations ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]

def get(cid: str) -> dict | None:
    _init()
    row = _conn().execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if not row: return None
    c = dict(row)
    msgs = _conn().execute("SELECT role,content,reasoning FROM messages WHERE conv_id=? ORDER BY id", (cid,)).fetchall()
    c["messages"] = [{"role": m["role"], "content": m["content"], "reasoning": m["reasoning"] or ""} for m in msgs]
    return c

def delete(cid: str):
    _init()
    _conn().execute("DELETE FROM messages WHERE conv_id=?", (cid,))
    _conn().execute("DELETE FROM conversations WHERE id=?", (cid,))
    _conn().commit()

def add_messages(cid: str, user_msg: str, asst_msg: str, tokens: int = 0, reasoning: str = ""):
    _init()
    t = _now()
    conn = _conn()
    if user_msg:
        conn.execute("INSERT INTO messages (conv_id,role,content,timestamp) VALUES (?,?,?,?)",
                     (cid, "user", user_msg, t))
    conn.execute("INSERT INTO messages (conv_id,role,content,reasoning,timestamp) VALUES (?,?,?,?,?)",
                 (cid, "assistant", asst_msg, reasoning, t))
    conn.execute("UPDATE conversations SET updated_at=?, token_count=token_count+? WHERE id=?",
                 (t, tokens, cid))
    row = conn.execute("SELECT title FROM conversations WHERE id=?", (cid,)).fetchone()
    if row and row["title"] == "Nouvelle conversation" and user_msg:
        title = user_msg[:50] + ("…" if len(user_msg) > 50 else "")
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
    conn.commit()

# ── System Prompts ──
def list_prompts() -> list:
    _init()
    rows = _conn().execute("SELECT id,name,content,created_at FROM system_prompts ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_prompt(pid: str) -> dict | None:
    _init()
    row = _conn().execute("SELECT * FROM system_prompts WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None

def save_prompt(name: str, content: str) -> dict:
    _init()
    pid, t = _id(), _now()
    _conn().execute("INSERT INTO system_prompts (id,name,content,created_at) VALUES (?,?,?,?)",
                    (pid, name, content, t))
    _conn().commit()
    return {"id": pid, "name": name, "content": content}

def delete_prompt(pid: str):
    _init()
    _conn().execute("DELETE FROM system_prompts WHERE id=?", (pid,))
    _conn().commit()

# ── Hardware ──
def _read_vram_bytes() -> int:
    """Read VRAM total from sysfs. Returns 0 if not available."""
    try:
        r = subprocess.run(
            ["cat", "/sys/class/drm/renderD128/device/mem_info_vram_total"],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            return int(r.stdout.strip())
    except: pass
    return 0


def suggest_gpu_layers() -> dict:
    """
    Recommend Ollama num_gpu layers based on available VRAM.

    Heuristic — assumes ~200MB per layer for a 4B model, scales with model size.
    Returns safe defaults for CPU/GPU hybrid or full offload.
    """
    vram_bytes = _read_vram_bytes()
    if vram_bytes == 0:
        return {"layers": 0, "mode": "cpu", "vram_gb": 0, "note": "No VRAM detected"}

    vram_gb = vram_bytes / (1024**3)

    # Conservative: ~200MB per layer for 4B models, ~400MB for 7B
    # Leave 512MB headroom for Ollama + system
    usable_gb = max(0, vram_gb - 0.5)
    suggested = int(usable_gb / 0.2)  # ~200MB per layer

    if suggested <= 0:
        return {"layers": 0, "mode": "cpu", "vram_gb": round(vram_gb, 1),
                "note": f"{vram_gb:.1f}GB too small for GPU offload"}
    elif suggested >= 99:
        return {"layers": 99, "mode": "gpu-full", "vram_gb": round(vram_gb, 1),
                "note": f"Full GPU offload ({vram_gb:.1f}GB available)"}
    else:
        return {"layers": suggested, "mode": "gpu-auto", "vram_gb": round(vram_gb, 1),
                "note": f"{suggested} GPU layers ({vram_gb:.1f}GB VRAM)"}


def detect_hardware() -> dict:
    import shutil
    info = {"gpu": None, "ram_gb": 0, "cpu_cores": 0}
    try:
        info["cpu_cores"] = os.cpu_count() or 0
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except: pass
    try:
        import subprocess
        r = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            info["gpu"] = "amd"
            info["gpu_name"] = "AMD ROCm"
    except: pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            info["gpu"] = "nvidia"
            for line in r.stdout.split('\n'):
                if 'NVIDIA-SMI' in line:
                    info["gpu_name"] = line.strip()[:60]
                    break
    except: pass
    if not info.get("gpu") and os.path.exists("/sys/class/drm/renderD128"):
        info["gpu"] = "integrated"
        info["gpu_name"] = "AMD Radeon (iGPU)"

    # Add GPU layer recommendation
    gpu_layers = suggest_gpu_layers()
    info["gpu_layers"] = gpu_layers["layers"]
    info["gpu_mode"] = gpu_layers["mode"]
    info["vram_gb"] = gpu_layers["vram_gb"]
    info["vram_bytes"] = _read_vram_bytes()
    info["recommended_layers"] = gpu_layers["layers"]
    info["recommended_mode"] = gpu_layers["mode"]
    return info
