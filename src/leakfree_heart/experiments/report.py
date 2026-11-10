"""Assemble results/RESULTS.md from the saved tables (numbers are copied from files, never typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import Ctx


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"


def _read(ctx: Ctx, name: str):
    p = ctx.tables / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _json(ctx: Ctx, name: str):
    p = ctx.tables / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_report(ctx: Ctx) -> Path:
    L = ["# LeakFree-Heart - experiment results\n",
         f"Data source: `{ctx.df.attrs.get('source')}`; patients after de-duplication: {len(ctx.df)}; "
         f"sites: {', '.join(sorted(ctx.df.site.unique()))}. "
         f"Run mode: {'QUICK (smoke test, numbers not for publication)' if ctx.quick else 'full'}.\n"]
    d = _json(ctx, "leakage_audit_details")
    if d:
        ds = d["duplicate_stats"]
        L.append("## E1 Leakage audit of the base-paper protocol\n")
        L.append(f"- The public Kaggle copy has **{ds['n_rows']} rows but only {ds['n_unique']} unique patients** "
                 f"({ds['duplicate_fraction']:.1%} duplicate rows; one record appears up to {ds['max_copies']} times).\n"
                 f"- On the paper's 80/20 split, **{d['overlap_P1']['n_test_with_exact_copy_in_train']} of {d['overlap_P1']['n_test']} test rows "
                 f"({d['overlap_P1']['fraction']:.1%}) have an identical copy in the training set.**\n")
        bd = pd.DataFrame(d["paper_protocol_breakdown"]).T.reset_index().rename(columns={"index": "model"})
        L.append("Accuracy under the paper protocol, split by whether the test row has a copy in training:\n\n" + _md(bd))
        t = _read(ctx, "leakage_audit")
        if t is not None:
            L.append("All protocols:\n\n" + _md(t.round(3)))
        L.append("![leakage](figures/fig_leakage_audit.png)\n")
    t = _read(ctx, "benchmark_summary")
    if t is not None:
        L.append("## E2 Leakage-free benchmark (primary cohort, repeated stratified CV)\n")
        cols = [c for c in ["model_name", "accuracy", "auc", "sensitivity", "specificity", "brier", "ece"] if c in t.columns]
        L.append("Mean [95% bootstrap CI], %, except Brier/ECE.\n\n" + _md(t[cols]))
        for nm, title in (("benchmark_pairwise_auc", "Pairwise AUC comparison (Nadeau-Bengio corrected t, Holm)"),):
            p = _read(ctx, nm)
            if p is not None:
                L.append(f"{title}:\n\n" + _md(p.round(4)))
        L.append("![forest](figures/fig_benchmark_forest.png)\n")
    e = _json(ctx, "external_validation")
    t = _read(ctx, "external_validation")
    if e and e.get("skipped"):
        L.append("## E3 External validation\n\nSkipped: " + e["reason"] + "\n")
    elif t is not None:
        L.append("## E3 External validation (train Cleveland -> test other hospitals)\n\n" + _md(t.round(3)))
        l = _read(ctx, "leave_one_site_out")
        if l is not None:
            L.append("Leave-one-site-out:\n\n" + _md(l.round(3)))
    for nm, title in (("calibration_comparison", "E4a Calibration"), ("selective_accuracy_vs_coverage", "E4b Accuracy vs coverage"),
                      ("conformal_deferral", "E4c Conformal deferral")):
        t = _read(ctx, nm)
        if t is not None:
            keep = [c for c in t.columns if not c.endswith(("_lo", "_hi"))]
            L.append(f"## {title}\n\n" + _md(t[keep].round(3)))
    s = _json(ctx, "selective_summary")
    if s:
        L.append("Selective summary: " + json.dumps(s) + "\n")
    L.append("![rc](figures/fig_risk_coverage.png)\n![conf](figures/fig_conformal_deferral.png)\n![rel](figures/fig_reliability.png)\n")
    q = _json(ctx, "explanation_quality")
    if q:
        L.append("## E5a Explanation quality\n\n```json\n" + json.dumps(q, indent=2) + "\n```\n![shap](figures/fig_global_shap.png)\n")
    t = _read(ctx, "faithfulness_conditions")
    if t is not None:
        keep = [c for c in t.columns if not c.endswith(("_lo", "_hi"))]
        L.append("## E5b Faithfulness by condition\n\n" + _md(t[keep].round(3)))
        v = _read(ctx, "verifier_fault_injection")
        if v is not None:
            L.append("Verifier validation by fault injection:\n\n" + _md(v.round(3)))
        L.append("![faith](figures/fig_faithfulness.png)\n")
    for nm, title in (("feature_reduction", "E6 SHAP-guided feature reduction (nested)"),
                      ("missing_data_robustness", "E7 Missing-data robustness"), ("fairness_subgroups", "E8 Subgroup audit"),
                      ("fairness_gaps", "E8 Fairness gaps")):
        t = _read(ctx, nm)
        if t is not None:
            L.append(f"## {title}\n\n" + _md(t.round(3)))
    out = ctx.tables.parent / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    ctx.say(f"[report] wrote {out}")
    return out
