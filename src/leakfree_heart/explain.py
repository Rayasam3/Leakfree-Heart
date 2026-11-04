"""Model-agnostic explanations in *clinical feature space* (the 13 raw inputs), in probability units.

SHAP (permutation/Shapley sampling) and LIME explain the whole fitted pipeline, so imputation and
one-hot encoding never leak into the explanation. Includes agreement, stability and deletion-fidelity metrics.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import CATEGORICAL, FEATURES, NUMERIC


def _proba_fn(model):
    return lambda A: model.predict_proba(pd.DataFrame(np.asarray(A, dtype=float), columns=FEATURES))[:, 1]


def reference_values(X_train: pd.DataFrame) -> pd.Series:
    """Median for numeric features, mode for categorical ones (the 'typical patient')."""
    ref = {}
    for c in FEATURES:
        ref[c] = X_train[c].median() if c in NUMERIC else X_train[c].mode(dropna=True).iloc[0]
    return pd.Series(ref)


def shap_attributions(model, X_background: pd.DataFrame, X_explain: pd.DataFrame, n_bg: int = 50,
                      seed: int = 0, max_evals: int | None = None):
    """Returns (values[n, 13], base_value). values sum to P(patient) - base_value."""
    import shap
    bg = X_background.sample(min(n_bg, len(X_background)), random_state=seed)
    masker = shap.maskers.Independent(bg[FEATURES].values.astype(float), max_samples=len(bg))
    m = len(FEATURES)
    expl = shap.explainers.Permutation(_proba_fn(model), masker, max_evals=max_evals or 10 * (2 * m + 1),
                                       seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = expl(X_explain[FEATURES].values.astype(float))
    return np.asarray(res.values), float(np.mean(res.base_values))


def lime_attributions(model, X_train: pd.DataFrame, X_explain: pd.DataFrame, seed: int = 0,
                      num_samples: int = 1500) -> np.ndarray:
    from lime.lime_tabular import LimeTabularExplainer
    ref = reference_values(X_train)
    Xi = X_train[FEATURES].fillna(ref)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    expl = LimeTabularExplainer(Xi.values.astype(float), feature_names=FEATURES, categorical_features=cat_idx,
                                class_names=["healthy", "disease"], discretize_continuous=True,
                                mode="classification", random_state=seed)

    def f2(A):
        p = _proba_fn(model)(A)
        return np.column_stack([1 - p, p])

    out = np.zeros((len(X_explain), len(FEATURES)))
    Xe = X_explain[FEATURES].fillna(ref)
    for i in range(len(Xe)):
        e = expl.explain_instance(Xe.iloc[i].values.astype(float), f2, num_features=len(FEATURES),
                                  num_samples=num_samples, labels=(1,))
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


# ----------------------------------------------------------- agreement metrics
def topk_set(a: np.ndarray, k: int) -> set:
    return set(np.argsort(-np.abs(a))[:k].tolist())


def agreement(A: np.ndarray, B: np.ndarray, k: int = 5) -> dict:
    """Mean per-patient agreement between two attribution matrices."""
    rho, jac, sgn = [], [], []
    for a, b in zip(A, B):
        r = spearmanr(np.abs(a), np.abs(b)).statistic
        rho.append(0.0 if np.isnan(r) else r)
        ta, tb = topk_set(a, k), topk_set(b, k)
        jac.append(len(ta & tb) / len(ta | tb))
        common = list(ta & tb)
        if common:
            sgn.append(np.mean(np.sign(a[common]) == np.sign(b[common])))
    return {"spearman_abs": float(np.mean(rho)), f"top{k}_jaccard": float(np.mean(jac)),
            f"top{k}_sign_agreement": float(np.mean(sgn)) if sgn else np.nan}


def shap_stability(model, X_train: pd.DataFrame, X_explain: pd.DataFrame, runs: int = 5, n_bg: int = 50,
                   k: int = 5, seed: int = 0) -> dict:
    """Re-run SHAP with different background samples / permutations; report top-k Jaccard to run 0."""
    mats = [shap_attributions(model, X_train, X_explain, n_bg=n_bg, seed=seed + r)[0] for r in range(runs)]
    jac, rho = [], []
    for r in range(1, runs):
        a = agreement(mats[0], mats[r], k)
        jac.append(a[f"top{k}_jaccard"]); rho.append(a["spearman_abs"])
    return {"top%d_jaccard_vs_run0" % k: float(np.mean(jac)), "spearman_vs_run0": float(np.mean(rho))}


# --------------------------------------------------------------- fidelity tests
def deletion_fidelity(model, X_explain: pd.DataFrame, A: np.ndarray, ref: pd.Series, k: int = 3,
                      seed: int = 0) -> dict:
    """ERASER-style tests. Replace features by the typical-patient reference value.

    comprehensiveness: probability drop of the predicted class when the top-k attributed features are removed
                       (higher = the attribution found features the model really uses)
    sufficiency      : probability drop when ONLY the top-k are kept (lower = they are sufficient)"""
    rng = np.random.default_rng(seed)
    X = X_explain[FEATURES].copy()
    X = X.fillna(ref)
    p0 = model.predict_proba(X)[:, 1]
    pred = (p0 >= 0.5)

    def conf(p):  # probability of the originally predicted class
        return np.where(pred, p, 1 - p)

    def masked(keep_sets):
        Xm = X.copy()
        for i, keep in enumerate(keep_sets):
            for j, f in enumerate(FEATURES):
                if j not in keep:
                    Xm.iloc[i, Xm.columns.get_loc(f)] = ref[f]
        return Xm

    top = [topk_set(a, k) for a in A]
    all_idx = set(range(len(FEATURES)))
    comp = conf(p0) - conf(model.predict_proba(masked([all_idx - t for t in top]))[:, 1])
    suff = conf(p0) - conf(model.predict_proba(masked(top))[:, 1])
    rnd = [set(rng.choice(len(FEATURES), k, replace=False).tolist()) for _ in A]
    comp_r = conf(p0) - conf(model.predict_proba(masked([all_idx - t for t in rnd]))[:, 1])
    return {"comprehensiveness": float(comp.mean()), "sufficiency": float(suff.mean()),
            "comprehensiveness_random": float(comp_r.mean())}


def patient_attribution_dict(values: np.ndarray) -> dict[str, float]:
    return {f: float(v) for f, v in zip(FEATURES, values)}


def top_features(attr: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    return sorted(attr.items(), key=lambda kv: -abs(kv[1]))[:k]
