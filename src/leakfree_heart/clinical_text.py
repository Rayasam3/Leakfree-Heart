"""Turns raw feature values into short clinical phrases (used by prompts, templates and retrieval queries)."""
from __future__ import annotations

import math

from .data import FEATURE_INFO


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:  # noqa: BLE001
        return False


def label_of(feature: str, value) -> str:
    info = FEATURE_INFO[feature]
    if _isnan(value):
        return "not recorded"
    if info["kind"] == "categorical":
        return info["options"].get(int(value), str(value))
    return f"{value:g}"


def bp_category(sbp: float) -> str:
    if sbp < 120:
        return "normal range"
    if sbp < 130:
        return "elevated range"
    if sbp < 140:
        return "stage 1 hypertension range"
    return "stage 2 hypertension range"


def chol_category(c: float) -> str:
    if c < 200:
        return "desirable"
    if c < 240:
        return "borderline high"
    return "high"


def _lf(text: str) -> str:
    """Lower-case the first letter unless the word is an acronym such as 'ST-T' or 'LV'."""
    return text if len(text) > 1 and text[1].isupper() else text[:1].lower() + text[1:]


def value_phrase(feature: str, value) -> str:
    """Short phrase such as 'exercise ST depression of 2.3 mm'."""
    if _isnan(value):
        return f"{FEATURE_INFO[feature]['label'].lower()} (not recorded)"
    v = value
    if feature == "age":
        return f"age {v:g} years"
    if feature == "sex":
        return f"{label_of('sex', v).lower()} sex"
    if feature == "cp":
        return f"{label_of('cp', v).lower()} chest pain"
    if feature == "trestbps":
        return f"resting systolic blood pressure of {v:g} mmHg ({bp_category(v)})"
    if feature == "chol":
        return f"serum cholesterol of {v:g} mg/dL ({chol_category(v)})"
    if feature == "fbs":
        return "fasting blood sugar above 120 mg/dL" if int(v) == 1 else "fasting blood sugar at or below 120 mg/dL"
    if feature == "restecg":
        return f"resting ECG: {_lf(label_of('restecg', v))}"
    if feature == "thalach":
        return f"maximum exercise heart rate of {v:g} bpm"
    if feature == "exang":
        return "exercise-induced angina" if int(v) == 1 else "no exercise-induced angina"
    if feature == "oldpeak":
        return f"exercise ST depression of {v:g} mm"
    if feature == "slope":
        return f"{label_of('slope', v).lower()} peak-exercise ST slope"
    if feature == "ca":
        return f"{int(v)} major vessel(s) coloured by fluoroscopy"
    if feature == "thal":
        return f"thallium stress test: {label_of('thal', v).lower()}"
    return f"{feature}={v}"


# Retrieval query seeds per feature (kept simple and explicit for reproducibility).
QUERY_SEED = {
    "age": "age and cardiovascular risk",
    "sex": "sex differences in coronary disease risk",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure hypertension categories",
    "chol": "serum cholesterol levels categories",
    "fbs": "fasting blood sugar glucose diabetes cardiovascular risk",
    "restecg": "resting electrocardiogram abnormality left ventricular hypertrophy",
    "thalach": "maximum heart rate during exercise test",
    "exang": "exercise-induced angina ischaemia",
    "oldpeak": "exercise ST segment depression ischaemia",
    "slope": "slope of peak exercise ST segment",
    "ca": "number of major vessels coloured by fluoroscopy coronary disease extent",
    "thal": "thallium stress test fixed reversible defect",
}


def retrieval_query(feature: str, value) -> str:
    return f"{QUERY_SEED[feature]} {value_phrase(feature, value)}"
