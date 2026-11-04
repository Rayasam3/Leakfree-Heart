"""Print a diagnostic of the local setup (python, GPU, data, LLM, model artifact)."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def line(ok: bool | None, label: str, detail: str = ""):
    mark = {True: "[ OK ]", False: "[FAIL]", None: "[ -- ]"}[ok]
    print(f"{mark} {label:34s} {detail}")


print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
line(sys.version_info[:2] in [(3, 10), (3, 11), (3, 12)], "Python version 3.10 - 3.12", platform.python_version())
for mod in ["numpy", "pandas", "sklearn", "xgboost", "shap", "lime", "fastapi", "uvicorn", "sentence_transformers", "faiss", "pypdf"]:
    try:
        m = importlib.import_module(mod)
        line(True, mod, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        line(False, mod, f"missing ({type(e).__name__})")
try:
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "CPU only"
    line(True, "torch", torch.__version__)
    line(cuda, "CUDA GPU available to PyTorch", name)
    if cuda:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        line(True, "GPU memory", f"{gb:.1f} GB")
except Exception as e:  # noqa: BLE001
    line(False, "torch", f"missing ({type(e).__name__})  - dense/Conv1D models will be skipped")
try:
    import pytorch_tabnet  # noqa: F401
    line(True, "pytorch-tabnet (optional)", "installed")
except Exception:  # noqa: BLE001
    line(None, "pytorch-tabnet (optional)", "not installed; TabNet skipped")

raw = ROOT / "data" / "raw"
line((raw / "kaggle_heart.csv").exists(), "Kaggle copy (leakage audit)", str(raw / "kaggle_heart.csv"))
uci = all((raw / "uci" / f).exists() for f in ["processed.cleveland.data", "processed.hungarian.data", "processed.switzerland.data", "processed.va.data"])
line(uci, "UCI 4-site data", "run download_data.bat" if not uci else "found")
line((ROOT / "knowledge_base" / "index" / "chunks.json").exists(), "Knowledge-base index", "run build_kb.bat if missing")
line((ROOT / "models" / "artifact.joblib").exists(), "Trained model artifact", "run train_final.bat if missing")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    names = [m["name"] for m in r.json().get("models", [])]
    line(bool(names), "Ollama (local LLM)", ", ".join(names) if names else "running but no models: ollama pull llama3.2:3b")
except Exception:  # noqa: BLE001
    line(None, "Ollama (local LLM)", "not running; explanations use the fixed template")
