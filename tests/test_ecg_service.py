"""ECG service and API tests. They use the tiny synthetic demo study (python scripts/run_ecg.py --demo)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO_RES, DEMO_ROOT = ROOT / "results_ecg_demo", ROOT / "data" / "raw" / "ptbxl_demo"
pytestmark = pytest.mark.skipif(not (DEMO_RES / "cache" / "ecg_preds.npz").exists(),
                                reason="run: python -m leakfree_heart run-ecg --demo")


@pytest.fixture(scope="module")
def svc():
    from leakfree_heart.config import load_config
    from leakfree_heart.ecg.service import ECGService
    s = ECGService(DEMO_ROOT, DEMO_RES, load_config()["ecg"])
    assert s.ready, s.error
    return s


def test_deferral_rates_match_study_table(svc):
    """The app must use exactly the study's calibration: answered fraction and accuracy equal the saved table."""
    tab = pd.read_csv(DEMO_RES / "tables" / "ecg_conformal.csv").query("condition == 'ecg_cnn'")
    for row in svc.status()["alphas"]:
        t = tab[np.isclose(tab.alpha, row["alpha"])].iloc[0]
        assert row["answered"] == pytest.approx(t.answered, abs=1e-3)
        if row["accuracy_answered"] is not None:
            assert row["accuracy_answered"] == pytest.approx(t.accuracy_answered, abs=1e-3)


def test_predict_shapes_and_explanation(svc):
    s = svc.samples()
    assert len(s) >= 8 and all({"ecg_id", "age", "sex"} == set(x) for x in s)
    r = svc.predict(None, s[0]["ecg_id"], None, None, 0.1)
    assert 0 <= r["probability"] <= 1 and r["decision"] in ("positive", "negative", "defer")
    assert len(r["signal"]) == 12 and len(r["signal"][0]) == 1000
    e = r["explanation"]
    assert len(e["lead_importance"]) == 12 and sum(e["lead_importance"]) == pytest.approx(1, abs=0.01)
    assert len(e["relevance"]) == 12 and len(e["relevance"][0]) == 100 and np.abs(e["relevance"]).max() <= 1.0 + 1e-6
    # integrated gradients should be (nearly) complete: attributions add up to the logit change from the baseline
    assert e["completeness_gap"] < 0.05
    assert r["recorded"] is not None


def test_training_ecgs_are_refused_and_bad_signals_rejected(svc):
    train_i = int(svc.split["train"][0])
    with pytest.raises(ValueError):
        svc.predict(None, int(svc.meta.index[train_i]), None, None, 0.05)
    with pytest.raises(ValueError):
        svc.predict(np.zeros((500, 12)), None, 60, 1, 0.05)
    with pytest.raises(ValueError):
        svc.predict(np.zeros((1000, 12)) + 100, None, 60, 1, 0.05)
    with pytest.raises(ValueError):
        svc.predict(np.zeros((1000, 12)), None, 60, 1, 0.07)


def test_custom_signal_matches_stored_signal(svc):
    i = svc.samples()[1]["ecg_id"]
    a = svc.predict(None, i, None, None, 0.05)
    row = int(svc.meta.index.get_loc(i))
    b = svc.predict(np.asarray(svc.X[row], dtype=np.float32), None, a["age"], a["sex"], 0.05)
    assert b["probability"] == pytest.approx(a["probability"], abs=1e-6)


def test_ecg_api_roundtrip(monkeypatch):
    monkeypatch.setenv("LEAKFREE_ECG_RESULTS", "results_ecg_demo")
    monkeypatch.setenv("LEAKFREE_ECG_ROOT", "data/raw/ptbxl_demo")
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        st = c.get("/api/ecg/status").json()
        assert st["ready"] and st["n_seeds"] >= 1
        sid = c.get("/api/ecg/samples").json()["samples"][0]["ecg_id"]
        r = c.post("/api/ecg/predict", json={"ecg_id": sid, "alpha": 0.1}).json()
        assert r["decision"] in ("positive", "negative", "defer") and len(r["signal"]) == 12
        assert c.post("/api/ecg/predict", json={"alpha": 0.1}).status_code == 422          # nothing to predict on
        assert c.post("/api/ecg/predict", json={"ecg_id": sid, "alpha": 0.07}).status_code == 422
        assert c.post("/api/ecg/predict", json={"signal": [[0.0] * 12] * 1000, "age": 5, "sex": 0}).status_code == 422
        t = c.get("/api/ecg/results").json()
        assert "ecg_benchmark" in t["tables"]
        assert c.get("/api/ecg/results/ecg_benchmark").json()["rows"]


def test_explain_stage_runs_on_demo():
    from leakfree_heart.ecg.study import STAGES, make_ecg_ctx
    ctx = make_ecg_ctx(demo=True)
    ctx.cfg["ecg"]["explain"] = {"n": 20, "ks": [1, 3], "random_draws": 3, "steps": 16}
    STAGES["explain"](ctx)
    fid = pd.read_csv(ctx.tables / "ecg_explanation_fidelity.csv")
    assert list(fid.k) == [1, 3] and (fid.n_ecg == 20).all()
    per = pd.read_csv(ctx.tables / "ecg_explanation_per_ecg.csv")
    assert np.allclose(per[[c for c in per.columns if c.startswith("imp_")]].sum(1), 1, atol=1e-6)
    assert (ctx.tables / "ecg_lead_importance.csv").exists()
