"""E6 SHAP-guided feature reduction (nested), E7 robustness to missing data (tests the base paper's untested claim),
E8 subgroup fairness audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..data import FEATURE_INFO, FEATURES, split_xy
from ..evaluation import binary_metrics, bootstrap_ci, run_repeated_cv
from ..models import DISPLAY, make_model
from . import plots
from .common import Ctx, primary_model, timer


# ------------------------------------------------------------------ E6 feature reduction
def run_feature_reduction(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    X, y = split_xy(ctx.df[ctx.df.site == cfg["data"]["primary_site"]])
    y = y.values
    rows = []
    names = [n for n in ("logreg", "xgb", "rf") if n in ctx.models] or ["logreg"]
    for name in names:
        for k in (3, 5, 8, 13):
            r = run_repeated_cv(f"{name}-top{k}", ctx.factory(name, top_k=None if k == 13 else k), X, y,
                                cfg["cv"]["n_splits"], max(1, cfg["cv"]["n_repeats"] // 2), ctx.seed, cfg["cv"]["n_jobs"])
            ci = bootstrap_ci(y, r.oof, metrics=("accuracy", "auc"), n_boot=min(300, cfg["cv"]["n_bootstrap"]), seed=ctx.seed).set_index("metric")
            rows.append(dict(model=name, n_features=k, accuracy=ci.loc["accuracy", "estimate"],
                             acc_lo=ci.loc["accuracy", "ci_low"], acc_hi=ci.loc["accuracy", "ci_high"],
                             auc=ci.loc["auc", "estimate"], auc_lo=ci.loc["auc", "ci_low"], auc_hi=ci.loc["auc", "ci_high"]))
            ctx.say(f"[E6]   {name} top-{k}: acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['auc']:.3f}")
    tab = pd.DataFrame(rows)
    ctx.save_table(tab, "feature_reduction")
    # which features get selected (fit on full data for display only)
    sel = make_model(names[0], cfg, ctx.seed, top_k=8).fit(X, y)
    ctx.save_json({"top8_features_full_data_for_display": list(sel.named_steps["select"].selected_)}, "feature_reduction_selected")
    ctx.say(f"[E6] done in {tm()}")
    return {}


# --------------------------------------------------------------------- E7 missing data
class MCARMasker(BaseEstimator, TransformerMixin):
    """Randomly sets feature values to NaN (missing completely at random).
    where = "test"  : only at prediction time (a model trained on complete data meets missing inputs)
            "both"  : in training and test (the model is trained under the same missingness)"""

    def __init__(self, rate: float = 0.0, where: str = "test", random_state: int = 0):
        self.rate, self.where, self.random_state = rate, where, random_state
        self._calls = 0

    def fit(self, X, y=None):
        return self

    def _mask(self, X, salt):
        X = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.random_state * 1000 + salt)
        m = rng.random(X.shape) < self.rate
        return X.mask(m)

    def fit_transform(self, X, y=None):
        return self._mask(X, 0) if self.where == "both" and self.rate > 0 else pd.DataFrame(X)

    def transform(self, X):
        self._calls += 1
        return self._mask(X, 1 + self._calls) if self.rate > 0 else pd.DataFrame(X)


def run_missing(ctx: Ctx) -> dict:
    from sklearn.pipeline import Pipeline
    cfg = ctx.cfg
    tm = timer()
    X, y = split_xy(ctx.df[ctx.df.site == cfg["data"]["primary_site"]])
    y = y.values
    rates = [0.0, 0.1, 0.2, 0.3, 0.4] if not ctx.quick else [0.0, 0.2, 0.4]
    names = [n for n in ("logreg", "rf", "xgb", "dense_paper") if n in ctx.models]
    rows = []
    for name in names:
        for where in ("test", "both"):
            for r_ in rates:
                if r_ == 0 and where == "both":
                    continue

                def mk(seed, name=name, r_=r_, where=where):
                    base = make_model(name, cfg, seed, missing_indicator=True)
                    return Pipeline([("mask", MCARMasker(r_, where, seed))] + list(base.steps))

                res = run_repeated_cv(name, mk, X, y, cfg["cv"]["n_splits"], max(1, cfg["cv"]["n_repeats"] // 2),
                                      ctx.seed, cfg["cv"]["n_jobs"] if name != "dense_paper" else 1)
                rows.append(dict(model=name, masked_in=where, missing_rate=r_, auc=res.fold_metrics.auc.mean(),
                                 auc_sd=res.fold_metrics.auc.std(), accuracy=res.fold_metrics.accuracy.mean()))
        ctx.say(f"[E7]   {name} done")
    tab = pd.DataFrame(rows)
    ctx.save_table(tab, "missing_data_robustness")
    _plot_missing(ctx, tab)
    ctx.say(f"[E7] done in {tm()}")
    return {}


def _plot_missing(ctx: Ctx, tab: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, where, title in zip(axes, ("test", "both"), ("Missing at prediction time only", "Missing in training and prediction")):
        d = tab[(tab.masked_in == where) | (tab.missing_rate == 0)]
        for i, (name, g) in enumerate(d.groupby("model", sort=False)):
            g = g.sort_values("missing_rate")
            ax.plot(g.missing_rate * 100, g.auc, marker="o", ms=6, mec=plots.C["surface"], mew=1.5,
                    color=plots.SERIES[i % 5], label=DISPLAY.get(name, name))
        ax.set_title(title)
        ax.set_xlabel("Features randomly missing (%)")
    axes[0].set_ylabel("ROC AUC")
    axes[1].legend(fontsize=8)
    plots.save(fig, ctx.figures / "fig_missing_data.png")


# ----------------------------------------------------------------------- E8 fairness
def run_fairness(ctx: Ctx) -> dict:
    import pickle
    cfg = ctx.cfg
    tm = timer()
    f = ctx.cache / "benchmark_oof.pkl"
    name = primary_model(ctx)
    if not f.exists():
        ctx.say("[E8] needs the benchmark first (run E2)")
        return {}
    d = pickle.load(open(f, "rb"))
    if name not in d:
        name = next(iter(d))
    oof, y = d[name]["oof"], d[name]["y"]
    df = ctx.df[ctx.df.site == cfg["data"]["primary_site"]].reset_index(drop=True)
    groups = {"sex": df["sex"].map({0: "female", 1: "male"}).values,
              "age": np.where(df["age"] >= df["age"].median(), f"age>={int(df['age'].median())}", f"age<{int(df['age'].median())}")}
    rows = []
    for gname, gv in groups.items():
        for g in np.unique(gv):
            idx = np.where(gv == g)[0]
            pm = [binary_metrics(y[idx], oof[r][idx]) for r in range(oof.shape[0])]
            ci = bootstrap_ci(y[idx], oof[:, idx], metrics=("auc", "sensitivity", "specificity"), n_boot=300, seed=ctx.seed).set_index("metric")
            rows.append(dict(model=name, attribute=gname, group=g, n=len(idx), prevalence=float(y[idx].mean()),
                             accuracy=np.mean([m["accuracy"] for m in pm]), auc=np.mean([m["auc"] for m in pm]),
                             auc_lo=ci.loc["auc", "ci_low"], auc_hi=ci.loc["auc", "ci_high"],
                             sensitivity=np.mean([m["sensitivity"] for m in pm]),
                             specificity=np.mean([m["specificity"] for m in pm])))
    tab = pd.DataFrame(rows)
    gaps = []
    for a, g in tab.groupby("attribute"):
        gaps.append(dict(attribute=a, sensitivity_gap=g.sensitivity.max() - g.sensitivity.min(),
                         specificity_gap=g.specificity.max() - g.specificity.min(),
                         equalised_odds_gap=max(g.sensitivity.max() - g.sensitivity.min(), g.specificity.max() - g.specificity.min()),
                         auc_gap=g.auc.max() - g.auc.min()))
    ctx.save_table(tab, "fairness_subgroups")
    ctx.save_table(pd.DataFrame(gaps), "fairness_gaps")
    ctx.say("[E8] subgroup performance:\n" + tab.round(3).to_string(index=False))
    ctx.say(f"[E8] done in {tm()}")
    return {}
