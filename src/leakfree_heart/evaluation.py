"""Evaluation utilities: repeated stratified CV, bootstrap confidence intervals, calibration metrics and
corrected statistical tests for comparing models."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold


# ---------------------------------------------------------------- point metrics
def ece_top_label(y, p, n_bins: int = 10) -> float:
    """Expected calibration error of the predicted class (equal-width bins)."""
    y, p = np.asarray(y), np.asarray(p)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def binary_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return {
        "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": f1_score(y, pred, zero_division=0), "auc": auc,
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "brier": brier_score_loss(y, np.clip(p, 0, 1)), "ece": ece_top_label(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ------------------------------------------------------------ repeated CV runner
@dataclass
class CVResult:
    name: str
    oof: np.ndarray                      # (n_repeats, n) out-of-fold P(disease)
    y: np.ndarray
    fold_metrics: pd.DataFrame           # one row per (repeat, fold)
    n_train: float = 0.0
    n_test: float = 0.0
    index: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def pooled_metrics(self) -> pd.DataFrame:
        return pd.DataFrame([binary_metrics(self.y, self.oof[r]) for r in range(self.oof.shape[0])])


def _one_split(make_fn, X, y, tr, te, seed, k):
    m = make_fn(seed + k)
    m.fit(X.iloc[tr], y[tr])
    return te, m.predict_proba(X.iloc[te])[:, 1]


def run_repeated_cv(name: str, make_fn, X: pd.DataFrame, y, n_splits: int = 10, n_repeats: int = 5,
                    seed: int = 42, n_jobs: int = 1) -> CVResult:
    """`make_fn(seed) -> unfitted sklearn estimator`. All preprocessing lives inside the estimator."""
    y = np.asarray(y).astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(X, y))
    out = Parallel(n_jobs=n_jobs)(delayed(_one_split)(make_fn, X, y, tr, te, seed, k)
                                  for k, (tr, te) in enumerate(splits))
    oof = np.zeros((n_repeats, len(y)))
    rows = []
    for k, ((tr, te), (te_, p)) in enumerate(zip(splits, out)):
        r, f = divmod(k, n_splits)
        oof[r, te] = p
        m = binary_metrics(y[te], p)
        m.update(repeat=r, fold=f)
        rows.append(m)
    n_te = np.mean([len(te) for _, te in splits])
    return CVResult(name, oof, y, pd.DataFrame(rows), n_train=len(y) - n_te, n_test=n_te)


# ---------------------------------------------------------------- bootstrap CIs
def bootstrap_ci(y, oof: np.ndarray, metrics=("accuracy", "auc", "sensitivity", "specificity", "brier"),
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Patient-level percentile bootstrap on pooled out-of-fold predictions (averaged over CV repeats)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n, R = len(y), oof.shape[0]
    point = {m: np.mean([binary_metrics(y, oof[r])[m] for r in range(R)]) for m in metrics}
    boots = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals = [binary_metrics(yb, oof[r][idx]) for r in range(R)]
        for m in metrics:
            boots[m].append(np.nanmean([v[m] for v in vals]))
    rows = []
    for m in metrics:
        lo, hi = np.nanpercentile(boots[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({"metric": m, "estimate": point[m], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def fmt_ci(row) -> str:
    return f"{row['estimate'] * 100:.1f} [{row['ci_low'] * 100:.1f}, {row['ci_high'] * 100:.1f}]"


# -------------------------------------------------------------- statistical tests
def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV scores.
    The naive paired t-test is anti-conservative because training sets overlap."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def pairwise_comparison(results: dict[str, CVResult], metric: str = "auc") -> pd.DataFrame:
    """Holm-adjusted corrected-t p-values for every model pair on fold-level `metric`."""
    names = list(results)
    pairs, raw = [], []
    for a, b in combinations(names, 2):
        fa = results[a].fold_metrics.sort_values(["repeat", "fold"])[metric].values
        fb = results[b].fold_metrics.sort_values(["repeat", "fold"])[metric].values
        t, p = corrected_resampled_ttest(fa, fb, results[a].n_train, results[a].n_test)
        pairs.append((a, b, float(np.mean(fa) - np.mean(fb)), t))
        raw.append(p)
    adj = holm(raw) if raw else []
    return pd.DataFrame([{"model_a": a, "model_b": b, "mean_diff": d, "t": t, "p_raw": p, "p_holm": q,
                          "significant_0.05": q < 0.05}
                         for (a, b, d, t), p, q in zip(pairs, raw, adj)])


def summarize_cv(results: dict[str, CVResult], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        ci = bootstrap_ci(r.y, r.oof, n_boot=n_boot, seed=seed).set_index("metric")
        fm = r.fold_metrics
        row = {"model": name}
        for m in ci.index:
            row[m] = fmt_ci(ci.loc[m]) if m != "brier" else \
                f"{ci.loc[m, 'estimate']:.3f} [{ci.loc[m, 'ci_low']:.3f}, {ci.loc[m, 'ci_high']:.3f}]"
        row["fold_acc_mean±sd"] = f"{fm['accuracy'].mean() * 100:.1f} ± {fm['accuracy'].std() * 100:.1f}"
        row["ece"] = float(np.mean([binary_metrics(r.y, r.oof[i])["ece"] for i in range(r.oof.shape[0])]))
        row["_auc"] = ci.loc["auc", "estimate"]
        row["_acc"] = ci.loc["accuracy", "estimate"]
        rows.append(row)
    return pd.DataFrame(rows)
