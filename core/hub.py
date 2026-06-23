"""RaccoonLM v2 — HuggingFace model hub & download"""
import os, json, threading, time, re, subprocess, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, list_models
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()
_downloads = {}  # {dl_id: {status, progress, filename, ...}}

# ── GGUF Quantization Detection ──
QUANT_PATTERNS = [
    (r'(?i)q4_k_m', 'Q4_K_M'), (r'(?i)q4_k_s', 'Q4_K_S'), (r'(?i)q4_0', 'Q4_0'), (r'(?i)q4_1', 'Q4_1'),
    (r'(?i)q5_k_m', 'Q5_K_M'), (r'(?i)q5_k_s', 'Q5_K_S'), (r'(?i)q5_0', 'Q5_0'), (r'(?i)q5_1', 'Q5_1'),
    (r'(?i)q2_k', 'Q2_K'), (r'(?i)q3_k_l', 'Q3_K_L'), (r'(?i)q3_k_m', 'Q3_K_M'), (r'(?i)q3_k_s', 'Q3_K_S'),
    (r'(?i)q6_k', 'Q6_K'), (r'(?i)q8_0', 'Q8_0'),
    (r'(?i)f16', 'F16'), (r'(?i)bf16', 'BF16'),
    (r'(?i)iq1_s', 'IQ1_S'), (r'(?i)iq2_xxs', 'IQ2_XXS'), (r'(?i)iq2_xs', 'IQ2_XS'),
    (r'(?i)iq2_s', 'IQ2_S'), (r'(?i)iq2_m', 'IQ2_M'),
    (r'(?i)iq3_xxs', 'IQ3_XXS'), (r'(?i)iq3_xs', 'IQ3_XS'), (r'(?i)iq3_s', 'IQ3_S'),
    (r'(?i)iq4_xs', 'IQ4_XS'), (r'(?i)iq4_nl', 'IQ4_NL'),
]


def parse_quant(filename: str) -> str:
    """Extract quantization type from GGUF filename."""
    for pattern, label in QUANT_PATTERNS:
        if re.search(pattern, filename):
            return label
    return "Unknown"


def parse_model_name(filename: str) -> str:
    """Extract readable model name from GGUF filename."""
    # Remove common suffixes and format
    name = re.sub(r'\.gguf$', '', filename, flags=re.I)
    name = re.sub(r'[-_]?(instruct|chat|base|merged|v\d+)', '', name, flags=re.I)
    # Remove quantization suffix
    for pattern, _ in QUANT_PATTERNS:
        name = re.sub(r'[-_]?' + pattern, '', name)
    name = name.replace('-', ' ').replace('_', ' ').strip()
    return name if name else filename


class QuantGroup:
    """Group of quantization files for one repo."""
    def __init__(self, quant: str, file: str):
        self.quant = quant
        self.files = [file]


def group_by_quant(files: list[str]) -> list[dict]:
    """Group GGUF files by quantization format."""
    groups = {}
    for f in files:
        quant = parse_quant(f)
        if quant not in groups:
            groups[quant] = []
        groups[quant].append(f)
    return [{"quant": q, "files": fs, "count": len(fs)} for q, fs in sorted(groups.items())]


def search_gguf(query: str, limit: int = 15) -> list[dict]:
    """Search HuggingFace for GGUF models with quantization info."""
    results = []
    seen = set()

    # Precise search with GGUF filter
    try:
        for m in list_models(search=query, filter="gguf", sort="downloads", limit=limit):
            if m.id not in seen:
                seen.add(m.id)
                # Fetch quant formats for this model
                quants = []
                try:
                    files = list_repo_files(m.id)
                    quants = group_by_quant(files)
                except:
                    pass
                results.append({
                    "id": m.id,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "tags": list(m.tags or []),
                    "pipeline": getattr(m, "pipeline_tag", ""),
                    "quants": quants,
                    "files": [],
                })
    except: pass

    # Broader search if few results
    if len(results) < 5:
        try:
            for m in list_models(search=query, sort="downloads", limit=limit):
                if m.id not in seen:
                    seen.add(m.id)
                    quants = []
                    try:
                        files = list_repo_files(m.id)
                        quants = group_by_quant(files)
                    except:
                        pass
                    results.append({
                        "id": m.id, "downloads": getattr(m, "downloads", 0) or 0,
                        "tags": list(m.tags or []), "pipeline": getattr(m, "pipeline_tag", ""),
                        "quants": quants, "files": [],
                    })
        except: pass

    return results


def list_repo_files(repo_id: str) -> list[str]:
    """List GGUF files in a repo."""
    try:
        files = api.list_repo_files(repo_id)
        gguvs = [f for f in files if f.endswith(".gguf")]
        return gguvs
    except: return []


def start_download(repo_id: str, filename: str) -> dict:
    """Start downloading a GGUF model file (threaded).
    After download, automatically creates an Ollama Modelfile and imports it.
    """
    dl_id = f"{repo_id}/{filename}"
    if dl_id in _downloads and _downloads[dl_id]["status"] == "downloading":
        return {"error": "Already downloading"}

    _downloads[dl_id] = {"repo_id": repo_id, "filename": filename,
                          "status": "starting", "progress": 0,
                          "started_at": time.time(), "error": None,
                          "path": None}

    def _run():
        try:
            _downloads[dl_id]["status"] = "downloading"
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename,
                local_files_only=False,
            )
            _downloads[dl_id]["path"] = local_path
            _downloads[dl_id]["progress"] = 100

            # Extract model name from filename
            model_name = re.sub(r'\.gguf$', '', os.path.basename(filename), flags=re.I)
            model_name = model_name.lower().replace('_', '-').replace(' ', '-')
            model_name = model_name.strip('-')
            if len(model_name) > 60:
                model_name = model_name[:60]

            # Register model for direct llama.cpp use
            try:
                from raccoonlm.core.models import register_model
                sz = os.path.getsize(local_path)
                def _fmt(n):
                    if n < 1024: return f"{n}B"
                    if n < 1024**2: return f"{n/1024:.1f}KB"
                    if n < 1024**3: return f"{n/1024**2:.1f}MB"
                    return f"{n/1024**3:.1f}GB"
                register_model(model_name, "llamacpp", source="llamacpp", path=local_path, gguf_path=local_path,
                               size=sz, size_display=_fmt(sz))
            except Exception:
                pass

            _downloads[dl_id]["status"] = "completed"

        except Exception as e:
            _downloads[dl_id]["status"] = "error"
            _downloads[dl_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "id": dl_id}


def _import_to_ollama(model_name: str, gguf_path: str) -> None:
    """Create an Ollama Modelfile and import the GGUF."""
    try:
        ollama_dir = Path.home() / ".ollama" / "models"
        blob_name = os.path.basename(gguf_path)
        models_dir = ollama_dir / "blobs"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create Modelfile
        modelfile_content = f"FROM {gguf_path}\n"
        modelfile_path = ollama_dir / f"Modelfile-{model_name}"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        # Run ollama create
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            _downloads.get(f"{Path(gguf_path).parent.name}/{os.path.basename(gguf_path)}", {})["import_error"] = result.stderr

        # Clean up Modelfile
        try:
            modelfile_path.unlink()
        except:
            pass

    except Exception as e:
        raise RuntimeError(f"Ollama import failed: {e}")


def get_downloads() -> list[dict]:
    return list(_downloads.values())


def cancel_download(repo_id: str, filename: str) -> dict:
    dl_id = f"{repo_id}/{filename}"
    if dl_id in _downloads:
        _downloads[dl_id]["status"] = "cancelled"
    return {"status": "cancelled"}
