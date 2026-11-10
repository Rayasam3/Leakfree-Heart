"""E4 - Calibration and uncertainty-aware deferral (selective prediction + split-conformal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration import calibrated, reliability_table
from ..data import split_xy
from ..evaluation import binary_metrics, run_repeated_cv
from ..uncertainty import (aurc, conformal_summary, max_coverage_at_accuracy, oracle_aurc, risk_coverage,
                           run_conformal_cv, selective_table)
from . import plots
from .common import Ctx, primary_model, timer


def run(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    site = cfg["data"]["primary_site"]
    X, y = split_xy(ctx.df[ctx.df.site == site])
    y = y.values
    name = primary_model(ctx)
    ctx.say(f"[E4] primary model = {name}")
    base = ctx.factory(name)
    cal = calibrated(base, "sigmoid", cv=5)
    nj = cfg["cv"]["n_jobs"] if name in ("logreg", "svm", "rf", "xgb") else 1
    raw_cv = run_repeated_cv(name, base, X, y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], ctx.seed, nj)
    cal_cv = run_repeated_cv(name + "+sigmoid", cal, X, y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], ctx.seed, nj)

    # --- calibration table
    rows = []
    for lab, r in (("uncalibrated", raw_cv), ("sigmoid-calibrated", cal_cv)):
        pm = r.pooled_metrics()
        P = r.oof
        rows.append(dict(model=name, variant=lab, accuracy=pm.accuracy.mean(), auc=pm.auc.mean(),
                         brier=pm.brier.mean(), ece=pm.ece.mean(),
                         aurc=np.mean([aurc(y, P[i]) for i in range(P.shape[0])]),
                         oracle_aurc=np.mean([oracle_aurc(y, P[i]) for i in range(P.shape[0])])))
    calib_tab = pd.DataFrame(rows)
    ctx.save_table(calib_tab, "calibration_comparison")
    ctx.say("[E4] calibration:\n" + calib_tab.round(4).to_string(index=False))

    # --- selective prediction by confidence (calibrated OOF)
    tgt = cfg["uncertainty"]["target_accuracy"]
    sel = selective_table(y, cal_cv.oof, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4), target=tgt)
    sel_info = {"aurc": sel.attrs["aurc"], "oracle_aurc": sel.attrs["oracle_aurc"],
                f"max_coverage_at_{int(tgt * 100)}pct_accuracy": sel.attrs["max_cov_at_target"]}
    ctx.save_table(sel, "selective_accuracy_vs_coverage")
    ctx.save_json(sel_info, "selective_summary")
    ctx.say("[E4] accuracy vs coverage:\n" + sel.round(3).to_string(index=False) + f"\n     {sel_info}")

    # --- conformal
    alphas = cfg["uncertainty"]["alphas"]
    P, D = run_conformal_cv(cal, X, y, alphas, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"],
                            cfg["uncertainty"]["calib_fraction"], cfg["uncertainty"]["mondrian"], ctx.seed)
    conf = conformal_summary(y, P, D, n_boot=min(cfg["cv"]["n_bootstrap"], 500), seed=ctx.seed)
    ctx.save_table(conf, "conformal_deferral")
    cols = ["alpha", "answered_fraction", "accuracy_on_answered", "accuracy_if_forced_all",
            "accuracy_on_deferred_if_forced", "sensitivity_on_answered", "specificity_on_answered",
            "coverage_disease", "coverage_healthy"]
    ctx.say("[E4] conformal deferral:\n" + conf[cols].round(3).to_string(index=False))

    _plot_risk_coverage(ctx, y, raw_cv.oof, cal_cv.oof, tgt)
    _plot_conformal(ctx, conf)
    _plot_reliability(ctx, y, raw_cv.oof, cal_cv.oof)
    ctx.say(f"[E4] done in {tm()}")
    return {"calibration": calib_tab.to_dict("records"), "selective": sel_info}


def _plot_risk_coverage(ctx, y, P_raw, P_cal, tgt):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for P, lab, col in ((P_cal, "Calibrated confidence", plots.C["blue"]), (P_raw, "Uncalibrated confidence", plots.C["orange"])):
        curves = [risk_coverage(y, P[i]) for i in range(P.shape[0])]
        cov = curves[0][0]
        acc = np.mean([c[1] for c in curves], axis=0)
        keep = cov >= 0.2
        ax.plot(cov[keep], acc[keep], color=col, label=lab)
    overall = np.mean([((P_cal[i] >= .5).astype(int) == y).mean() for i in range(P_cal.shape[0])])
    ax.axhline(overall, color=plots.C["ink2"], ls=":", lw=1.2, label="Random deferral (= overall accuracy)")
    ax.axhline(tgt, color=plots.C["violet"], ls="--", lw=1.2, label=f"{int(tgt * 100)}% target")
    ax.set_xlabel("Coverage (fraction of patients the model answers)")
    ax.set_ylabel("Accuracy on answered patients")
    ax.set_title("Accuracy vs coverage when deferring the least confident cases")
    ax.set_xlim(0.2, 1.0)
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="lower left")
    plots.save(fig, ctx.figures / "fig_risk_coverage.png")


def _plot_conformal(ctx, conf):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = conf["alpha"].values
    for col, lab, c in (("answered_fraction", "Fraction answered", plots.C["blue"]),
                        ("accuracy_on_answered", "Accuracy on answered", plots.C["aqua"]),
                        ("accuracy_on_deferred_if_forced", "Accuracy on deferred if forced", plots.C["orange"])):
        ax.plot(x, conf[col], marker="o", color=c, ms=6, mec=plots.C["surface"], mew=1.5, label=lab)
        ax.fill_between(x, conf[col + "_lo"], conf[col + "_hi"], color=c, alpha=0.12, lw=0)
    ax.set_xlabel("Conformal error level alpha (target coverage = 1 - alpha)")
    ax.set_ylabel("Proportion")
    ax.set_title("Split-conformal deferral (class-conditional)")
    ax.legend(fontsize=8, loc="center right")
    plots.save(fig, ctx.figures / "fig_conformal_deferral.png")


def _plot_reliability(ctx, y, P_raw, P_cal):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=plots.C["ink2"], ls=":", lw=1.2, label="Perfect")
    for P, lab, c in ((P_raw, "Uncalibrated", plots.C["orange"]), (P_cal, "Sigmoid-calibrated", plots.C["blue"])):
        t = reliability_table(np.tile(y, P.shape[0]), P.ravel(), 8)
        ax.plot(t.mean_pred, t.frac_pos, marker="o", color=c, ms=6, mec=plots.C["surface"], mew=1.5, label=lab)
    ax.set_xlabel("Predicted probability of disease")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (out-of-fold)")
    ax.legend(fontsize=8)
    plots.save(fig, ctx.figures / "fig_reliability.png")
