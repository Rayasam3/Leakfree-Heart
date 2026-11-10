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
line(uci, "UCI 4-site data", "run: python -m leakfree_heart download-uci" if not uci else "found")
kb_ok = (ROOT / "knowledge_base" / "index" / "chunks.json").exists()
line(kb_ok, "Knowledge-base index", "found" if kb_ok else "run: python -m leakfree_heart build-kb")
art_ok = (ROOT / "models" / "artifact.joblib").exists()
line(art_ok, "Trained model artifact", "found" if art_ok else "run: python -m leakfree_heart train-final")
ptb = ROOT / "data" / "raw" / "ptbxl"
n_rec = len(list((ptb / "records100").rglob("*.hea"))) if (ptb / "records100").exists() else 0
line(n_rec >= 21799, "PTB-XL ECG data", f"{n_rec}/21799 records" + ("" if n_rec >= 21799 else "  run: python -m leakfree_heart download-ptbxl"))
ec = ROOT / "results_ecg" / "cache"
ecg_ok = len(list(ec.glob("ecg_cnn_seed*.pt"))) > 0 if ec.exists() else False
line(ecg_ok, "Trained ECG models", "found" if ecg_ok else "run: python -m leakfree_heart run-ecg")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    names = [m["name"] for m in r.json().get("models", [])]
    line(bool(names), "Ollama (local LLM)", ", ".join(names) if names else "running but no models: ollama pull llama3.2:3b")
except Exception:  # noqa: BLE001
    line(None, "Ollama (local LLM)", "not running; explanations use the fixed template")
