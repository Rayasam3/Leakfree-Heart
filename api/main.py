"""FastAPI backend for LeakFree-Heart.

    uvicorn api.main:app --port 8000        (or start_api.bat)

Endpoints
  GET  /api/health      service status (model, LLM back-end, retriever)
  GET  /api/meta        feature schema + cross-validated performance + disclaimer
  POST /api/predict     calibrated risk + conformal decision (answer or DEFER)
  POST /api/explain     prediction + SHAP attributions for all 13 features
  POST /api/report      prediction + SHAP + retrieved guideline evidence + cited explanation + verification
  GET  /api/results     list of experiment tables; /api/results/{name} returns one as JSON
Research prototype - not a medical device, not for clinical use.
"""
from __future__ import annotations

import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakfree_heart.clinical_text import value_phrase  # noqa: E402
from leakfree_heart.config import load_config, path  # noqa: E402
from leakfree_heart.data import FEATURE_INFO, FEATURES  # noqa: E402
from leakfree_heart.explain import patient_attribution_dict, shap_attributions, top_features  # noqa: E402
from leakfree_heart.rag import Retriever, build_chunks  # noqa: E402
from leakfree_heart.rag.faithfulness import Verifier  # noqa: E402
from leakfree_heart.rag.generator import MockBackend, generate_explanation, get_backend, render_text  # noqa: E402
from leakfree_heart.uncertainty import decisions_from_sets, prediction_sets  # noqa: E402

DISCLAIMER = ("Research prototype built on small public datasets (UCI Heart Disease). It is NOT a medical device, "
              "does not diagnose, and must not guide clinical decisions. Discuss any health concern with a clinician. "
              "For new, severe or worsening chest pain call your local emergency number.")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="LeakFree-Heart API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class State:
    cfg = None
    art = None
    retr = None
    ver = None
    backend = None
    lock = threading.Lock()
    err = None


S = State()


def _load():
    S.cfg = load_config()
    f = path(S.cfg, "models") / "artifact.joblib"
    if not f.exists():
        S.err = "Model artifact missing. Run train_final.bat (python scripts/train_final.py)."
        return
    S.art = joblib.load(f)
    kb = path(S.cfg, "knowledge_base")
    idx = kb / "index" / "chunks.json"
    from leakfree_heart.rag.kb import load_chunks
    chunks = load_chunks(kb) if idx.exists() else build_chunks(kb)
    S.retr = Retriever(chunks, backend=S.cfg["rag"]["backend"], model_name=S.cfg["rag"]["embedding_model"])
    fc = S.cfg["faithfulness"]
    S.ver = Verifier(chunks, k_top=fc["k_top"], support_threshold=fc["support_threshold"])
    S.backend = get_backend(S.cfg["llm"])


def _need():
    if S.art is None:
        raise HTTPException(503, S.err or "model not loaded")


class Patient(BaseModel):
    age: float = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float | None = Field(None, ge=60, le=260)
    chol: float | None = Field(None, ge=50, le=700)
    fbs: int | None = Field(None, ge=0, le=1)
    restecg: int | None = Field(None, ge=0, le=2)
    thalach: float | None = Field(None, ge=50, le=230)
    exang: int | None = Field(None, ge=0, le=1)
    oldpeak: float | None = Field(None, ge=-3, le=8)
    slope: int | None = Field(None, ge=0, le=2)
    ca: int | None = Field(None, ge=0, le=3)
    thal: int | None = Field(None, ge=1, le=3)


class ReportRequest(Patient):
    condition: str = Field("auto", pattern="^(auto|template_rag|llm_rag|llm_rag_verified)$")


def _row(p: Patient) -> pd.DataFrame:
    d = p.model_dump(exclude={"condition"})
    return pd.DataFrame([{f: (np.nan if d.get(f) is None else float(d[f])) for f in FEATURES}])


def _decide(prob: float) -> dict:
    a = S.art
    thr = a["thresholds"][a["alpha"]]
    sets = prediction_sets(np.array([prob]), thr)
    dec = int(decisions_from_sets(sets)[0])
    label = {1: "higher estimated risk", 0: "lower estimated risk", -1: "uncertain - deferred to a clinician"}[dec]
    return {"decision": {1: "positive", 0: "negative", -1: "defer"}[dec], "decision_label": label,
            "conformal_set": [int(i) for i in np.where(sets[0])[0]], "alpha": a["alpha"],
            "target_coverage": 1 - a["alpha"]}


def _predict(p: Patient) -> tuple[pd.DataFrame, float]:
    row = _row(p)
    prob = float(S.art["model"].predict_proba(row)[0, 1])
    return row, prob


@app.get("/api/health")
def health():
    return {"status": "ok" if S.art is not None else "model_missing", "detail": S.err,
            "model": S.art["model_name"] if S.art else None,
            "llm_backend": getattr(S.backend, "name", None), "retriever": getattr(S.retr, "backend", None),
            "data_source": S.art["data_source"] if S.art else None}


@app.get("/api/meta")
def meta():
    _need()
    fi = {k: {**v, "options": ({str(a): b for a, b in v["options"].items()} if "options" in v else None)}
          for k, v in FEATURE_INFO.items()}
    return {"features": FEATURES, "feature_info": fi, "model": S.art["model_name"], "cv_metrics": S.art["cv_metrics"],
            "n_train": S.art["n_train"], "alpha": S.art["alpha"], "disclaimer": DISCLAIMER,
            "llm_backend": getattr(S.backend, "name", None), "created": S.art["created"]}


@app.post("/api/predict")
def predict(p: Patient):
    _need()
    _, prob = _predict(p)
    return {"probability": prob, **_decide(prob), "model": S.art["model_name"], "disclaimer": DISCLAIMER}


def _explain(p: Patient):
    row, prob = _predict(p)
    with S.lock:
        vals, base = shap_attributions(S.art["model"], S.art["X_background"], row, n_bg=40, seed=0,
                                       max_evals=10 * (2 * len(FEATURES) + 1) // 2)
    attr = patient_attribution_dict(vals[0])
    return row, prob, attr, base


@app.post("/api/explain")
def explain(p: Patient):
    _need()
    row, prob, attr, base = _explain(p)
    pat = {f: (None if pd.isna(v) else float(v)) for f, v in row.iloc[0].items()}
    items = [{"feature": f, "label": FEATURE_INFO[f]["label"], "value": pat[f], "phrase": value_phrase(f, pat[f]),
              "shap": attr[f]} for f in FEATURES]
    items.sort(key=lambda d: -abs(d["shap"]))
    return {"probability": prob, "base_value": base, **_decide(prob), "attributions": items}


@app.post("/api/report")
def report(req: ReportRequest):
    _need()
    row, prob, attr, base = _explain(req)
    pat = {f: (None if pd.isna(v) else float(v)) for f, v in row.iloc[0].items()}
    k = S.cfg["faithfulness"]["k_top"]
    top = top_features(attr, k)
    ev = S.retr.retrieve_for_explanation(pat, [f for f, _ in top], S.cfg["rag"]["per_feature_k"])
    real_llm = not isinstance(S.backend, MockBackend)
    cond = req.condition
    if cond == "auto":
        cond = "llm_rag_verified" if real_llm else "template_rag"
    if cond.startswith("llm") and not real_llm:
        cond = "template_rag"
    g = generate_explanation(cond, S.backend, S.ver, pat, prob, top, attr, ev, S.cfg["llm"]["max_repairs"])
    m = g.verification.metrics()
    checks = [{"feature": c.feature, "direction": c.direction, "citations": c.citations,
               "model_faithful": c.model_faithful, "evidence_grounded": c.evidence_grounded, "support": round(c.support, 3),
               "reasons": c.reasons(True)} for c in g.verification.checks]
    items = [{"feature": f, "label": FEATURE_INFO[f]["label"], "phrase": value_phrase(f, pat[f]), "shap": attr[f]} for f in FEATURES]
    items.sort(key=lambda d: -abs(d["shap"]))
    return {"probability": prob, "base_value": base, **_decide(prob), "attributions": items,
            "explanation": g.expl, "explanation_text": render_text(g.expl), "condition": cond, "generator": g.backend,
            "verification": {"metrics": {a: (None if (isinstance(b, float) and np.isnan(b)) else b) for a, b in m.items()},
                             "claims": checks, "n_repairs": g.n_repairs, "n_pruned": g.n_pruned},
            "evidence": [{"id": c.id, "title": c.title, "source": c.source, "year": c.year, "text": c.text,
                          "kind": c.kind, "verified": c.verified} for c in ev],
            "disclaimer": DISCLAIMER}


RES = ROOT / "results"


@app.get("/api/results")
def results_index():
    t = RES / "tables"
    return {"tables": sorted(p.stem for p in t.glob("*.csv")) if t.exists() else [],
            "figures": sorted(p.name for p in (RES / "figures").glob("*.png")) if (RES / "figures").exists() else []}


@app.get("/api/results/{name}")
def results_table(name: str):
    f = RES / "tables" / f"{Path(name).stem}.csv"
    if not f.exists():
        raise HTTPException(404, "no such table")
    df = pd.read_csv(f)
    return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="records"))}


(RES / "figures").mkdir(parents=True, exist_ok=True)
if True:
    app.mount("/figures", StaticFiles(directory=str(RES / "figures")), name="figures")
