"""E1 - Leakage audit of the base paper's protocol on the public 1025-row Kaggle copy of the Cleveland data.

Questions answered
  Q1  How many patients are really in the 1025 rows?                      -> duplicate statistics
  Q2  How many test rows of the paper's 80/20 split have an exact copy in the training set?
  Q3  How much of the reported accuracy comes from those copies?          -> accuracy on overlapping vs unseen rows
  Q4  What is the accuracy of the same models after de-duplication?       -> hold-out and repeated CV
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import path
from ..data import FEATURES, KAGGLE_CP_MAP, KAGGLE_THAL_MAP, TARGET, duplicate_stats, kaggle_label_direction, load_kaggle_raw
from ..evaluation import run_repeated_cv
from ..models import make_model, make_paper_replica
from ..torch_models import TORCH_AVAILABLE
from . import plots
from .common import Ctx, timer

PROTOCOL_ORDER = ["Paper protocol (duplicated data, one 80/20 split)",
                  "Duplicated data, 10-fold CV",
                  "De-duplicated, one 80/20 split",
                  "De-duplicated, 10-fold CV"]


def _row_keys(X: pd.DataFrame) -> list[tuple]:
    return [tuple(np.round(r, 6)) for r in X[FEATURES].values.astype(float)]


def _fit_eval(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return p, ((p >= 0.5).astype(int) == yte)


def _models_for_audit(ctx: Ctx):
    names = ["logreg", "rf", "xgb"]
    if TORCH_AVAILABLE:
        names.append("dense_paper")
    return names


def run(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    kag = load_kaggle_raw(path(cfg, "data_raw"))
    Xk, yk = kag[FEATURES].copy(), kag[TARGET].values.astype(int)
    out: dict = {"duplicate_stats": duplicate_stats(kag)}
    ctx.say(f"[E1] Kaggle copy: {out['duplicate_stats']['n_rows']} rows, "
            f"{out['duplicate_stats']['n_unique']} unique patients "
            f"({out['duplicate_stats']['duplicate_fraction']:.1%} duplicate rows)")

    # coding audit -----------------------------------------------------------------
    out["coding_audit"] = {
        "label_direction_detected": "target=1 means DISEASE" if kaggle_label_direction(kag) == 1
        else "target is INVERTED: target=1 means NO disease",
        "cp_recode_kaggle_to_uci0": KAGGLE_CP_MAP, "thal_recode_kaggle_to_canonical": KAGGLE_THAL_MAP,
        "slope_recode": "canonical = 2 - kaggle", "ca_null": "kaggle ca==4 is a null", "thal_null": "kaggle thal==0 is a null",
        "note": "Category order differs from UCI, so notebooks that label Kaggle cp/thal/slope with the UCI "
                "descriptions (and target=1 as disease) attach wrong clinical meanings.",
    }

    rows = []
    # ---------- P1: the paper's protocol (random 80/20, seed 42, no stratification) ----------
    Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.20, random_state=42)
    train_keys = set(_row_keys(Xtr))
    overlap = np.array([k in train_keys for k in _row_keys(Xte)])
    out["overlap_P1"] = {"n_test": int(len(Xte)), "n_test_with_exact_copy_in_train": int(overlap.sum()),
                         "fraction": float(overlap.mean())}
    ctx.say(f"[E1] paper split: {overlap.sum()}/{len(Xte)} test rows ({overlap.mean():.1%}) have an exact copy in the training set")

    split_acc = {}
    for name in _models_for_audit(ctx):
        m = make_paper_replica(cfg, 42) if name == "dense_paper" else make_model(name, cfg, 42)
        if name != "dense_paper":
            from ..preprocessing import make_preprocessor
            m.steps[0] = ("pre", make_preprocessor("ordinal"))     # same preprocessing as the paper
        p, ok = _fit_eval(m, Xtr, ytr, Xte, yte)
        acc = ok.mean()
        rows.append(dict(protocol=PROTOCOL_ORDER[0], model=name, n_test=len(yte), accuracy=acc))
        split_acc[name] = dict(overall=float(acc),
                               on_rows_with_copy_in_train=float(ok[overlap].mean()) if overlap.any() else np.nan,
                               on_unseen_rows=float(ok[~overlap].mean()) if (~overlap).any() else np.nan,
                               n_unseen=int((~overlap).sum()))
        ctx.say(f"[E1]   {name:12s} paper-protocol acc={acc:.3f}  copy-rows={split_acc[name]['on_rows_with_copy_in_train']:.3f}"
                f"  unseen-rows={split_acc[name]['on_unseen_rows']:.3f} (n={split_acc[name]['n_unseen']})")
    out["paper_protocol_breakdown"] = split_acc

    # ---------- P3: de-duplicated single split ----------
    ded = kag.drop_duplicates().reset_index(drop=True)
    Xd, yd = ded[FEATURES].copy(), ded[TARGET].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.20, random_state=42, stratify=yd)
    for name in _models_for_audit(ctx):
        m = make_paper_replica(cfg, 42) if name == "dense_paper" else make_model(name, cfg, 42)
        if name != "dense_paper":
            from ..preprocessing import make_preprocessor
            m.steps[0] = ("pre", make_preprocessor("ordinal"))
        p, ok = _fit_eval(m, Xtr, ytr, Xte, yte)
        rows.append(dict(protocol=PROTOCOL_ORDER[2], model=name, n_test=len(yte), accuracy=ok.mean()))

    # ---------- repeated single splits: distribution of the inflated accuracy (cheap models) ----------
    seeds = range(10 if not ctx.quick else 3)
    infl, clean = {n: [] for n in ("logreg", "rf", "xgb")}, {n: [] for n in ("logreg", "rf", "xgb")}
    for s in seeds:
        a, b, c, d = train_test_split(Xk, yk, test_size=0.20, random_state=s)
        a2, b2, c2, d2 = train_test_split(Xd, yd, test_size=0.20, random_state=s, stratify=yd)
        for n in infl:
            from ..preprocessing import make_preprocessor
            m1 = make_model(n, cfg, s); m1.steps[0] = ("pre", make_preprocessor("ordinal"))
            m2 = make_model(n, cfg, s); m2.steps[0] = ("pre", make_preprocessor("ordinal"))
            infl[n].append(_fit_eval(m1, a, c, b, d)[1].mean())
            clean[n].append(_fit_eval(m2, a2, c2, b2, d2)[1].mean())
    out["repeated_splits"] = {n: {"duplicated_mean": float(np.mean(infl[n])), "duplicated_sd": float(np.std(infl[n])),
                                  "dedup_mean": float(np.mean(clean[n])), "dedup_sd": float(np.std(clean[n]))} for n in infl}

    # ---------- repeated CV on duplicated vs de-duplicated ----------
    nsp, nrep = cfg["cv"]["n_splits"], max(1, cfg["cv"]["n_repeats"] // 2) if not ctx.quick else 2
    cv_models = [n for n in ctx.models if n in ("logreg", "rf", "xgb", "dense_paper", "conv1d")]
    for name in cv_models:
        for label, X_, y_, proto in (("dup", Xk, yk, PROTOCOL_ORDER[1]), ("dedup", Xd, yd, PROTOCOL_ORDER[3])):
            r = run_repeated_cv(name, ctx.factory(name), X_, y_, nsp, nrep, ctx.seed,
                                n_jobs=cfg["cv"]["n_jobs"] if name in ("logreg", "rf", "xgb") else 1)
            rows.append(dict(protocol=proto, model=name, n_test=len(y_), accuracy=r.fold_metrics["accuracy"].mean(),
                             accuracy_sd=r.fold_metrics["accuracy"].std(), auc=r.fold_metrics["auc"].mean()))
            ctx.say(f"[E1]   CV {label:5s} {name:12s} acc={rows[-1]['accuracy']:.3f}")

    tab = pd.DataFrame(rows)
    ctx.save_table(tab, "leakage_audit")
    ctx.save_json(out, "leakage_audit_details")
    _plot(ctx, tab)
    ctx.say(f"[E1] done in {tm()}")
    return out


def _plot(ctx: Ctx, tab: pd.DataFrame):
    import matplotlib.pyplot as plt
    protos = [p for p in PROTOCOL_ORDER if p in set(tab.protocol)]
    models = list(dict.fromkeys(tab.model))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.8 / len(protos)
    cols = {PROTOCOL_ORDER[0]: plots.C["orange"], PROTOCOL_ORDER[1]: plots.C["magenta"],
            PROTOCOL_ORDER[2]: plots.C["aqua"], PROTOCOL_ORDER[3]: plots.C["blue"]}
    for i, pr in enumerate(protos):
        vals = [tab[(tab.protocol == pr) & (tab.model == m)].accuracy.mean() for m in models]
        x = np.arange(len(models)) + i * w - 0.4 + w / 2
        ax.bar(x, vals, w * 0.88, color=cols[pr], label=pr)
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v + 0.006, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=8, color=plots.C["ink"])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("Accuracy")
    ax.set_title("Same models, different evaluation protocols: duplicates inflate accuracy")
    ax.legend(loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.32))
    plots.save(fig, ctx.figures / "fig_leakage_audit.png")
