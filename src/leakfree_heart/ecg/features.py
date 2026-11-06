"""Hand-crafted ECG features for a strong non-deep baseline (gradient boosting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from .data import FS, LEAD_NAMES


def _heart_rate(sig: np.ndarray) -> tuple[float, float, float]:
    """Return (heart rate bpm, RR std in s, number of beats) from a lead with clear R waves."""
    x = sig - np.median(sig)
    thr = 0.5 * np.percentile(np.abs(x), 99)
    if thr <= 0:
        return 0.0, 0.0, 0.0
    peaks, _ = find_peaks(x, height=thr, distance=int(0.3 * FS))
    if len(peaks) < 2:
        return 0.0, 0.0, float(len(peaks))
    rr = np.diff(peaks) / FS
    return float(60.0 / rr.mean()), float(rr.std()), float(len(peaks))


def ecg_features(X: np.ndarray) -> pd.DataFrame:
    """X: (N, T, 12) -> DataFrame with per-lead statistics and rhythm features (no labels used)."""
    rows = []
    for i in range(len(X)):
        s = np.asarray(X[i], dtype=np.float32)
        f = {}
        for j, name in enumerate(LEAD_NAMES):
            v = s[:, j]
            f[f"{name}_mean"], f[f"{name}_std"] = float(v.mean()), float(v.std())
            f[f"{name}_min"], f[f"{name}_max"] = float(v.min()), float(v.max())
            f[f"{name}_skew"], f[f"{name}_kurt"] = float(stats.skew(v)), float(stats.kurtosis(v))
            f[f"{name}_absdiff"] = float(np.abs(np.diff(v)).mean())
        hr = [_heart_rate(s[:, j]) for j in (1, 6, 10)]          # leads II, V1, V5
        best = max(hr, key=lambda h: h[2])
        f["hr_bpm"], f["rr_std"], f["n_beats"] = best
        rows.append(f)
    return pd.DataFrame(rows)
