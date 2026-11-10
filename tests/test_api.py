import pytest
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "models" / "artifact.joblib"
pytestmark = pytest.mark.skipif(not ART.exists(), reason="run train_final first")


def test_api_roundtrip():
    from fastapi.testclient import TestClient
    from api.main import app
    pat = dict(age=57, sex=1, cp=3, trestbps=140, chol=260, fbs=0, restecg=1, thalach=118, exang=1,
               oldpeak=2.6, slope=1, ca=2, thal=3)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        r = c.post("/api/predict", json=pat).json()
        assert 0 <= r["probability"] <= 1 and r["decision"] in ("positive", "negative", "defer")
        rep = c.post("/api/report", json=pat).json()
        assert rep["explanation"]["claims"] and rep["evidence"]
        assert abs(sum(a["shap"] for a in rep["attributions"]) + rep["base_value"] - rep["probability"]) < 1e-6
        assert c.post("/api/predict", json=dict(pat, age=5)).status_code == 422
        assert c.post("/api/predict", json=dict(age=50, sex=0, cp=1)).json()["decision"] in ("defer", "positive", "negative")
