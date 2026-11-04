"""E5a - Explanation quality: SHAP vs LIME agreement, SHAP stability and deletion fidelity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data import FEATURES, split_xy
from ..explain import (agreement, deletion_fidelity, lime_attributions, reference_values, shap_attributions,
                       shap_stability)
from . import plots
from .common import Ctx, primary_model, timer


def fit_primary(ctx: Ctx):
    """Hold out 25% of the primary cohort; fit the primary model on the rest. Used by E5/E6."""
    cfg = ctx.cfg
    X, y = split_xy(ctx.df[ctx.df.site == cfg["data"]["primary_site"]])
    Xtr, Xte, ytr, yte = train_test_split(X, y.values, test_size=0.25, stratify=y.values, random_state=ctx.seed)
    name = primary_model(ctx)
    model = ctx.factory(name)(ctx.seed).fit(Xtr, ytr)
    return name, model, Xtr, Xte, ytr, yte


def run(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    e = cfg["explain"]
    name, model, Xtr, Xte, ytr, yte = fit_primary(ctx)
    n = min(e["n_patients"], len(Xte))
    Xe = Xte.iloc[:n]
    ctx.say(f"[E5a] model={name}; explaining {n} held-out patients")
    S, base = shap_attributions(model, Xtr, Xe, n_bg=e["shap_background"], seed=ctx.seed)
    L = lime_attributions(model, Xtr, Xe, seed=ctx.seed)
    k = e["k_top"]
    ag = agreement(S, L, k)
    stab = shap_stability(model, Xtr, Xe.iloc[: min(n, 20)], runs=e["stability_runs"], n_bg=e["shap_background"], k=k, seed=ctx.seed)
    ref = reference_values(Xtr)
    fid = {}
    for lab, A in (("SHAP", S), ("LIME", L)):
        fid[lab] = deletion_fidelity(model, Xe, A, ref, k=3, seed=ctx.seed)
    out = {"model": name, "n_patients": n, "shap_vs_lime": ag, "shap_stability": stab, "fidelity": fid,
           "base_value": base}
    ctx.save_json(out, "explanation_quality")
    glob = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(S).mean(0), "mean_abs_lime": np.abs(L).mean(0)})
    glob["rank_shap"] = glob.mean_abs_shap.rank(ascending=False).astype(int)
    glob = glob.sort_values("mean_abs_shap", ascending=False)
    ctx.save_table(glob, "global_importance")
    np.savez(ctx.cache / "attributions.npz", shap=S, lime=L, idx=Xe.index.values)
    ctx.say(f"[E5a] SHAP~LIME {ag}\n      stability {stab}\n      fidelity {fid}")
    _plot_global(ctx, glob)
    ctx.say(f"[E5a] done in {tm()}")
    return out


def _plot_global(ctx: Ctx, glob: pd.DataFrame):
    import matplotlib.pyplot as plt
    from ..data import FEATURE_INFO
    g = glob.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh([FEATURE_INFO[f]["label"] for f in g.feature], g.mean_abs_shap * 100, color=plots.C["blue"], height=0.62)
    for i, v in enumerate(g.mean_abs_shap * 100):
        ax.text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8, color=plots.C["ink2"])
    ax.set_xlabel("Mean |SHAP| (percentage points of predicted probability)")
    ax.set_title("Global feature importance (held-out patients)")
    ax.grid(axis="y", visible=False)
    plots.save(fig, ctx.figures / "fig_global_shap.png")
