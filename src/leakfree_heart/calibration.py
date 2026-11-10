"""Probability calibration helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


def calibrated(make_fn, method: str = "sigmoid", cv: int = 5):
    """Wrap a model factory so the model is calibrated with internal cross-fitting (no test-fold leakage)."""
    def _make(seed: int):
        return CalibratedClassifierCV(estimator=make_fn(seed), method=method, cv=cv, ensemble=True)
    return _make


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability of P(disease): mean predicted vs observed frequency in equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                         "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)
