"""Uncertainty-aware prediction with deferral.

Two complementary mechanisms:

1. Split-conformal classification (LAC score, optionally class-conditional / Mondrian).
   The prediction *set* is {0}, {1}, {0,1} or {} . A singleton is an answer; anything else is DEFERRED to a
   clinician. Marginal (or per-class) coverage >= 1 - alpha holds under exchangeability.
2. Confidence-based selective prediction (risk-coverage curve), used for the accuracy-vs-coverage analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


# ------------------------------------------------------------------- conformal
def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:                         # too few calibration points for this alpha -> no pruning
        return 1.0
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalCalibration:
    scores_by_class: dict[int, np.ndarray]
    scores_all: np.ndarray

    def thresholds(self, alpha: float, mondrian: bool) -> dict[int, float]:
        if mondrian:
            return {k: _conformal_quantile(self.scores_by_class[k], alpha) for k in (0, 1)}
        q = _conformal_quantile(self.scores_all, alpha)
        return {0: q, 1: q}


def calibrate_conformal(p_cal: np.ndarray, y_cal: np.ndarray) -> ConformalCalibration:
    """LAC nonconformity score: s = 1 - P(true class)."""
    p_true = np.where(y_cal == 1, p_cal, 1 - p_cal)
    s = 1 - p_true
    return ConformalCalibration({k: s[y_cal == k] for k in (0, 1)}, s)


def prediction_sets(p: np.ndarray, thr: dict[int, float]) -> np.ndarray:
    """Boolean (n, 2): class k is in the set iff its nonconformity score 1 - P(k) <= q_k.
    class 0: 1 - (1 - p) = p <= q_0      class 1: 1 - p <= q_1  <=>  p >= 1 - q_1"""
    p = np.asarray(p)
    return np.column_stack([p <= thr[0], p >= 1 - thr[1]])


def decisions_from_sets(sets: np.ndarray) -> np.ndarray:
    """-1 = defer (empty or two-class set); 0/1 = confident class."""
    size = sets.sum(1)
    out = np.full(len(sets), -1)
    single = size == 1
    out[single] = sets[single].argmax(1)
    return out


def conformal_fold(make_fn, Xtr, ytr, Xte, alphas, calib_fraction: float = 0.25, mondrian: bool = True,
                   seed: int = 0):
    """Fit once, return P(disease) on Xte and the decision vector for every alpha."""
    ytr = np.asarray(ytr).astype(int)
    idx = np.arange(len(ytr))
    tr, ca = train_test_split(idx, test_size=calib_fraction, stratify=ytr, random_state=seed)
    model = make_fn(seed)
    model.fit(Xtr.iloc[tr], ytr[tr])
    cal = calibrate_conformal(model.predict_proba(Xtr.iloc[ca])[:, 1], ytr[ca])
    p_te = model.predict_proba(Xte)[:, 1]
    dec = {}
    for a in alphas:
        dec[a] = decisions_from_sets(prediction_sets(p_te, cal.thresholds(a, mondrian)))
    return p_te, dec


def run_conformal_cv(make_fn, X: pd.DataFrame, y, alphas, n_splits: int = 10, n_repeats: int = 5,
                     calib_fraction: float = 0.25, mondrian: bool = True, seed: int = 42):
    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    P = np.zeros((n_repeats, len(y)))
    D = {a: np.full((n_repeats, len(y)), -2) for a in alphas}
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        r = k // n_splits
        p, dec = conformal_fold(make_fn, X.iloc[tr], y[tr], X.iloc[te], alphas, calib_fraction, mondrian, seed + k)
        P[r, te] = p
        for a in alphas:
            D[a][r, te] = dec[a]
    return P, D


def conformal_summary(y, P: np.ndarray, D: dict[float, np.ndarray], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-alpha deferral statistics averaged over CV repeats, with patient-bootstrap 95% CIs."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    R, n = P.shape
    rows = []
    for a, dec in D.items():
        def stats_for(idx):
            res = []
            for r in range(R):
                d, yy, pp = dec[r][idx], y[idx], P[r][idx]
                ans = d >= 0
                covered = (d == yy) | (d == -1)       # set contains truth: singleton correct, or defer (>=1 class)
                cov = np.mean(covered)
                acc_ans = np.mean(d[ans] == yy[ans]) if ans.any() else np.nan
                forced_acc = np.mean((pp >= 0.5).astype(int) == yy)
                acc_def = np.mean((pp[~ans] >= 0.5).astype(int) == yy[~ans]) if (~ans).any() else np.nan
                sens = np.mean(d[ans & (yy == 1)] == 1) if (ans & (yy == 1)).any() else np.nan
                spec = np.mean(d[ans & (yy == 0)] == 0) if (ans & (yy == 0)).any() else np.nan
                res.append([ans.mean(), acc_ans, forced_acc, acc_def, sens, spec])
            return np.nanmean(np.array(res), axis=0)
        point = stats_for(np.arange(n))
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(stats_for(idx))
        lo, hi = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
        names = ["answered_fraction", "accuracy_on_answered", "accuracy_if_forced_all",
                 "accuracy_on_deferred_if_forced", "sensitivity_on_answered", "specificity_on_answered"]
        row = {"alpha": a, "target_coverage": 1 - a}
        for i, nme in enumerate(names):
            row[nme] = point[i]; row[nme + "_lo"] = lo[i]; row[nme + "_hi"] = hi[i]
        # empirical conformal coverage (truth in set) per class
        cov1, cov0 = [], []
        for r in range(R):
            d = dec[r]
            covered = (d == y) | (d == -1)
            cov1.append(covered[y == 1].mean()); cov0.append(covered[y == 0].mean())
        row["coverage_disease"] = float(np.mean(cov1)); row["coverage_healthy"] = float(np.mean(cov0))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------ selective prediction
def risk_coverage(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy as a function of coverage when abstaining on the least confident cases."""
    y = np.asarray(y).astype(int)
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    correct = ((np.asarray(p) >= 0.5).astype(int) == y)[order].astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return coverage, cum_acc


def aurc(y, p) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, acc = risk_coverage(y, p)
    return float(np.mean(1 - acc))


def oracle_aurc(y, p) -> float:
    y = np.asarray(y).astype(int)
    err = ((np.asarray(p) >= 0.5).astype(int) != y).mean()
    n = len(y)
    k = int(round((1 - err) * n))
    risk = np.concatenate([np.zeros(k), (np.arange(1, n - k + 1)) / (k + np.arange(1, n - k + 1))])
    return float(risk.mean())


def accuracy_at_coverage(y, p, coverage: float) -> float:
    cov, acc = risk_coverage(y, p)
    i = min(len(cov) - 1, int(np.ceil(coverage * len(cov))) - 1)
    return float(acc[i])


def max_coverage_at_accuracy(y, p, target: float, min_cases: int = 20) -> float:
    """Largest coverage whose cumulative accuracy is still >= target (0 if never reached)."""
    cov, acc = risk_coverage(y, p)
    ok = np.where((acc >= target) & (np.arange(1, len(cov) + 1) >= min_cases))[0]
    return float(cov[ok.max()]) if len(ok) else 0.0


def selective_table(y, P: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), target=0.95) -> pd.DataFrame:
    R = P.shape[0]
    rows = []
    for c in coverages:
        rows.append({"coverage": c, "accuracy": float(np.mean([accuracy_at_coverage(y, P[r], c) for r in range(R)]))})
    out = pd.DataFrame(rows)
    out.attrs["aurc"] = float(np.mean([aurc(y, P[r]) for r in range(R)]))
    out.attrs["oracle_aurc"] = float(np.mean([oracle_aurc(y, P[r]) for r in range(R)]))
    out.attrs["max_cov_at_target"] = float(np.mean([max_coverage_at_accuracy(y, P[r], target) for r in range(R)]))
    return out
