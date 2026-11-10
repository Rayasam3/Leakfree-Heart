"""E2 - Leakage-free benchmark: repeated stratified CV on the de-duplicated primary cohort, bootstrap CIs,
Holm-corrected Nadeau-Bengio tests, and E3 - external validation across hospitals when UCI sites are available."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..data import FEATURES, TARGET, split_xy
from ..evaluation import (binary_metrics, bootstrap_ci, fmt_ci, pairwise_comparison, run_repeated_cv, summarize_cv)
from ..models import DISPLAY
from . import plots
from .common import Ctx, timer


def run(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    site = cfg["data"]["primary_site"]
    d = ctx.df[ctx.df.site == site]
    X, y = split_xy(d)
    y = y.values
    ctx.say(f"[E2] primary cohort '{site}': n={len(y)}, prevalence={y.mean():.3f}; models={ctx.models}")
    results = {}
    for name in ctx.models:
        t = timer()
        results[name] = run_repeated_cv(name, ctx.factory(name), X, y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"],
                                        ctx.seed, n_jobs=cfg["cv"]["n_jobs"] if name in ("logreg", "svm", "rf", "xgb") else 1)
        fm = results[name].fold_metrics
        ctx.say(f"[E2]   {name:12s} acc={fm.accuracy.mean():.3f}±{fm.accuracy.std():.3f}  auc={fm.auc.mean():.3f}  ({t()})")

    summ = summarize_cv(results, n_boot=cfg["cv"]["n_bootstrap"], seed=ctx.seed)
    summ["model_name"] = summ["model"].map(DISPLAY)
    ctx.save_table(summ, "benchmark_summary")
    pair = pairwise_comparison(results, "auc")
    ctx.save_table(pair, "benchmark_pairwise_auc")
    pair_acc = pairwise_comparison(results, "accuracy")
    ctx.save_table(pair_acc, "benchmark_pairwise_accuracy")
    folds = pd.concat([r.fold_metrics.assign(model=n) for n, r in results.items()])
    ctx.save_table(folds, "benchmark_fold_metrics")
    with open(ctx.cache / "benchmark_oof.pkl", "wb") as fh:
        pickle.dump({n: dict(oof=r.oof, y=r.y) for n, r in results.items()}, fh)
    # per-model confusion at 0.5 (mean over repeats)
    cm = []
    for n, r in results.items():
        pm = r.pooled_metrics()
        cm.append(dict(model=n, tp=pm.tp.mean(), tn=pm.tn.mean(), fp=pm.fp.mean(), fn=pm.fn.mean()))
    ctx.save_table(pd.DataFrame(cm), "benchmark_confusion")
    _forest(ctx, results, cfg["cv"]["n_bootstrap"])
    ctx.say(f"[E2] done in {tm()}")
    best = summ.sort_values("_auc", ascending=False).iloc[0]
    return {"best_by_auc": best["model"], "best_auc": float(best["_auc"]), "best_acc": float(best["_acc"])}


def _forest(ctx: Ctx, results, n_boot):
    import matplotlib.pyplot as plt
    rows = []
    for n, r in results.items():
        ci = bootstrap_ci(r.y, r.oof, metrics=("accuracy", "auc"), n_boot=min(n_boot, 500), seed=ctx.seed).set_index("metric")
        rows.append((n, ci))
    rows.sort(key=lambda t: t[1].loc["auc", "estimate"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "auc"), ("Accuracy", "ROC AUC")):
        for i, (n, ci) in enumerate(rows):
            e, lo, hi = ci.loc[metric, ["estimate", "ci_low", "ci_high"]]
            ax.plot([lo, hi], [i, i], color=plots.C["blue"], lw=2)
            ax.plot(e, i, "o", color=plots.C["blue"], ms=8, mec=plots.C["surface"], mew=2)
            ax.text(hi + 0.004, i, f"{e:.3f}", va="center", fontsize=8, color=plots.C["ink2"])
        ax.set_title(f"{title} (95% bootstrap CI)")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([DISPLAY.get(n, n) for n, _ in rows])
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"De-duplicated '{ctx.cfg['data']['primary_site']}' cohort, repeated {ctx.cfg['cv']['n_splits']}-fold CV",
                 x=0.01, ha="left", fontsize=10, color=plots.C["ink2"])
    plots.save(fig, ctx.figures / "fig_benchmark_forest.png")


# ======================================================================= E3 external validation
def run_external(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    sites = sorted(ctx.df.site.unique())
    if len(sites) < 2:
        msg = ("External validation needs the other UCI sites. Run `python scripts/download_data.py` "
               "and set data.source: auto/uci.")
        ctx.say("[E3] SKIPPED - " + msg)
        ctx.save_json({"skipped": True, "reason": msg}, "external_validation")
        return {"skipped": True}
    tm = timer()
    train_site = cfg["data"]["primary_site"]
    tr = ctx.df[ctx.df.site == train_site]
    Xtr, ytr = split_xy(tr)
    rows = []
    names = [n for n in ctx.models if n in ("logreg", "svm", "rf", "xgb", "dense_paper", "conv1d")]
    for name in names:
        # missing-indicator pipeline: external sites miss whole variables (e.g. ca/thal in Hungarian data)
        m = ctx.factory(name, missing_indicator=True)(ctx.seed).fit(Xtr, ytr)
        for s in sites:
            if s == train_site:
                continue
            te = ctx.df[ctx.df.site == s]
            Xte, yte = split_xy(te)
            p = m.predict_proba(Xte)[:, 1]
            bm = binary_metrics(yte.values, p)
            ci = bootstrap_ci(yte.values, p[None, :], n_boot=500, seed=ctx.seed).set_index("metric")
            rows.append(dict(model=name, train=train_site, test=s, n=len(yte), prevalence=float(yte.mean()),
                             missing_frac=float(te[FEATURES].isna().mean().mean()),
                             accuracy=bm["accuracy"], auc=bm["auc"], sensitivity=bm["sensitivity"],
                             specificity=bm["specificity"], brier=bm["brier"],
                             auc_ci=fmt_ci(ci.loc["auc"]), acc_ci=fmt_ci(ci.loc["accuracy"])))
        ctx.say(f"[E3]   {name} done")
    tab = pd.DataFrame(rows)
    # leave-one-site-out over all sites (pooled training), reports the realistic cross-hospital gap
    loso = []
    for held in sites:
        trn = ctx.df[ctx.df.site != held]
        tst = ctx.df[ctx.df.site == held]
        Xa, ya = split_xy(trn); Xb, yb = split_xy(tst)
        for name in names:
            m = ctx.factory(name, missing_indicator=True)(ctx.seed).fit(Xa, ya)
            p = m.predict_proba(Xb)[:, 1]
            bm = binary_metrics(yb.values, p)
            loso.append(dict(model=name, held_out=held, n=len(yb), accuracy=bm["accuracy"], auc=bm["auc"],
                             sensitivity=bm["sensitivity"], specificity=bm["specificity"]))
    ctx.save_table(tab, "external_validation")
    ctx.save_table(pd.DataFrame(loso), "leave_one_site_out")
    ctx.say(f"[E3] done in {tm()}")
    return {"skipped": False}
