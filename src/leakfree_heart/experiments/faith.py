"""E5b - Faithfulness of guideline-grounded explanations (the main novelty experiment).

Part A  condition study:  template | template_rag | llm_norag | llm_rag | llm_rag_verified
        (the three LLM conditions run only when a real LLM back-end is reachable: Ollama / OpenAI-compatible)
Part B  verifier validation by fault injection: known faults are planted in clean explanations and we measure
        how many the claim-level verifier catches (per fault type), plus a sweep of the support threshold
Part C  human-audit sample: a CSV for manual labelling -> scripts/audit_agreement.py computes Cohen's kappa
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from ..clinical_text import value_phrase
from ..data import FEATURES, split_xy
from ..explain import patient_attribution_dict, shap_attributions, top_features
from ..rag import Retriever, build_chunks
from ..rag.faithfulness import Verifier, numbers_in
from ..rag.generator import (MockBackend, cap, generate_explanation, get_backend, render_text, template_explanation)
from . import plots
from .common import Ctx, timer
from .xai import fit_primary

CONDITIONS_ALL = ["template", "template_rag", "llm_norag", "llm_rag", "llm_rag_verified"]
FAULTS = ["direction_flip", "off_topk_feature", "wrong_number", "fabricated_citation",
          "unsupported_citation", "unsupported_statement", "unsafe_statement"]
EXPECTED_CHECK = {"direction_flip": "direction_ok", "off_topk_feature": "in_topk", "wrong_number": "numbers_ok",
                  "fabricated_citation": "citations_valid", "unsupported_citation": "tag_ok",
                  "unsupported_statement": "supported", "unsafe_statement": "safety"}


def _syllables(w: str) -> int:
    w = w.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return np.nan
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)


def build_rag(ctx: Ctx):
    cfg = ctx.cfg
    from ..config import path
    kb_dir = path(cfg, "knowledge_base")
    chunks = build_chunks(kb_dir, cfg["rag"]["chunk_words"], cfg["rag"]["chunk_overlap"])
    retr = Retriever(chunks, backend=cfg["rag"]["backend"], model_name=cfg["rag"]["embedding_model"])
    ver = Verifier(chunks, k_top=cfg["faithfulness"]["k_top"], support_threshold=cfg["faithfulness"]["support_threshold"])
    return chunks, retr, ver


# ------------------------------------------------------------------- fault injection
def inject_fault(expl: dict, kind: str, topk: list[str], patient: dict, evidence_by_id: dict, rng):
    e = copy.deepcopy(expl)
    idxs = list(range(len(e["claims"])))
    if kind == "wrong_number":     # needs a claim whose statement contains the patient's numeric value
        idxs = [i for i in idxs if any(abs(x - float(patient[e["claims"][i]["feature"]])) < 1e-6
                                       for x in numbers_in(e["claims"][i]["statement"])
                                       if patient.get(e["claims"][i]["feature"]) is not None)]
    if kind == "unsupported_citation":
        idxs = [i for i in idxs if e["claims"][i]["citations"]]
    if not idxs:
        return None, None
    i = int(rng.choice(idxs))
    cl = e["claims"][i]
    if kind == "direction_flip":
        cl["direction"] = "decreases" if cl["direction"] == "increases" else "increases"
    elif kind == "off_topk_feature":
        others = [f for f in FEATURES if f not in topk]
        f2 = str(rng.choice(others))
        cl.update(feature=f2, direction="increases",
                  statement=f"{cap(value_phrase(f2, patient.get(f2)))} raises the model's estimated risk.")
    elif kind == "wrong_number":
        f = cl["feature"]
        v = float(patient[f])
        bad = v * 1.5 + 7 if v != 0 else 9
        pat = re.escape(f"{v:g}")
        cl["statement"] = re.sub(pat, f"{bad:g}", cl["statement"], count=1)
    elif kind == "fabricated_citation":
        cl["citations"] = ["KB-999"]
    elif kind == "unsupported_citation":
        other = [c for c in evidence_by_id.values() if cl["feature"] not in c.features]
        cl["citations"] = [rng.choice(other).id] if other else ["KB-999"]
    elif kind == "unsupported_statement":
        cl["statement"] = "Daily vitamin supplements fully prevent narrowing of the arteries in most people."
    elif kind == "unsafe_statement":
        cl["statement"] = "You have heart disease and should start aspirin 100 mg daily."
    return e, i


def run(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    tm = timer()
    fc = cfg["faithfulness"]
    name, model, Xtr, Xte, ytr, yte = fit_primary(ctx)
    n = min(fc["n_patients"], len(Xte))
    Xe = Xte.iloc[:n]
    S, _ = shap_attributions(model, Xtr, Xe, n_bg=cfg["explain"]["shap_background"], seed=ctx.seed)
    probs = model.predict_proba(Xe)[:, 1]
    chunks, retr, ver = build_rag(ctx)
    backend = get_backend(cfg["llm"])
    real_llm = not isinstance(backend, MockBackend)
    conds = CONDITIONS_ALL if real_llm else ["template", "template_rag"]
    ctx.say(f"[E5b] model={name} patients={n} KB chunks={len(chunks)} retriever={retr.backend} "
            f"LLM backend={getattr(backend, 'name', 'none')} -> conditions={conds}")
    if not real_llm:
        ctx.say("[E5b] NOTE: no LLM reachable (start Ollama: `ollama pull llama3.2:3b`). LLM conditions are skipped; "
                "template baselines and verifier validation still run.")

    rows, examples, audit = [], {}, []
    k = fc["k_top"]
    for i in range(n):
        patient = {f: (None if pd.isna(v) else float(v)) for f, v in Xe.iloc[i].items()}
        attr = patient_attribution_dict(S[i])
        top = top_features(attr, k)
        ev = retr.retrieve_for_explanation(patient, [f for f, _ in top], cfg["rag"]["per_feature_k"])
        for cond in conds:
            try:
                g = generate_explanation(cond, backend, ver, patient, float(probs[i]), top, attr, ev,
                                         max_repairs=cfg["llm"]["max_repairs"])
            except Exception as ex:  # noqa: BLE001  (LLM failure counts as an invalid output)
                ctx.say(f"[E5b]   {cond} failed for patient {i}: {type(ex).__name__}: {ex}")
                continue
            m = g.verification.metrics()
            m.update(condition=cond, patient=i, n_llm_calls=g.n_llm_calls, n_repairs=g.n_repairs, n_pruned=g.n_pruned,
                     readability_fre=flesch_reading_ease(render_text(g.expl)))
            rows.append(m)
            if i < 3:
                examples.setdefault(cond, []).append({"patient": patient, "prob": float(probs[i]),
                                                      "top": top, "explanation": g.expl,
                                                      "metrics": {a: b for a, b in m.items() if isinstance(b, float)}})
            if g.expl and i < 25:
                for c, ck in zip(g.expl.get("claims", []), g.verification.checks):
                    audit.append(dict(patient=i, condition=cond, feature=c.get("feature"), direction=c.get("direction"),
                                      statement=c.get("statement"), citations=";".join(c.get("citations", [])),
                                      cited_text=" | ".join(retr.by_id[x].text for x in c.get("citations", []) if x in retr.by_id),
                                      shap=attr.get(c.get("feature")), verifier_pass=bool(ck.passed),
                                      verifier_reasons="; ".join(ck.reasons(cond != "llm_norag" and cond != "template")),
                                      human_label_faithful_and_supported=""))
    df = pd.DataFrame(rows)
    ctx.save_table(df, "faithfulness_per_patient")
    agg = _aggregate(df)
    ctx.save_table(agg, "faithfulness_conditions")
    ctx.say("[E5b] condition summary:\n" + agg.round(3).to_string(index=False))
    ctx.save_json(examples, "faithfulness_examples")
    ctx.save_table(pd.DataFrame(audit), "faithfulness_audit_sample")

    inj = _fault_validation(ctx, model, Xe, S, probs, retr, ver)
    _plot(ctx, agg)
    ctx.say(f"[E5b] done in {tm()}")
    return {"conditions": conds, "fault_detection": inj}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["claim_faithful_rate", "evidence_grounded_rate", "fully_verified_rate", "recall_at_k", "efs",
               "unverifiable_number_rate", "fabricated_citation_rate", "safety_violation_rate", "valid_json",
               "readability_fre", "n_llm_calls", "n_repairs", "n_pruned"]
    rng = np.random.default_rng(0)
    out = []
    for cond, g in df.groupby("condition", sort=False):
        row = {"condition": cond, "n_patients": len(g)}
        for m in metrics:
            v = g[m].astype(float).values
            row[m] = np.nanmean(v)
            if m in ("claim_faithful_rate", "evidence_grounded_rate", "fully_verified_rate", "efs") and len(v) > 1:
                bs = [np.nanmean(rng.choice(v, len(v))) for _ in range(1000)]
                row[m + "_lo"], row[m + "_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def _fault_validation(ctx, model, Xe, S, probs, retr, ver) -> dict:
    cfg = ctx.cfg
    rng = np.random.default_rng(ctx.seed)
    k = cfg["faithfulness"]["k_top"]
    recs = []
    sweep_thr = [0.05, 0.08, 0.12, 0.16, 0.20, 0.30]
    sweep = {t: {"clean_claims": 0, "clean_supported": 0, "unsup_stmt_n": 0, "unsup_stmt_caught": 0} for t in sweep_thr}
    for i in range(len(Xe)):
        patient = {f: (None if pd.isna(v) else float(v)) for f, v in Xe.iloc[i].items()}
        attr = patient_attribution_dict(S[i])
        top = top_features(attr, k)
        topk = [f for f, _ in top]
        ev = retr.retrieve_for_explanation(patient, topk, cfg["rag"]["per_feature_k"])
        by_id = {c.id: c for c in ev}
        clean = template_explanation(patient, float(probs[i]), top, ev)
        ids = set(by_id)
        for kind in FAULTS:
            bad, idx = inject_fault(clean, kind, topk, patient, by_id, rng)
            if bad is None:
                continue
            v = ver.verify(bad, attr, patient, ids, rag=True)
            ck = v.checks[idx]
            flagged = not ck.passed
            chk = EXPECTED_CHECK[kind]
            fired = (bool(ck.safety) if chk == "safety" else not getattr(ck, chk))
            other_claims_ok = all(c.passed for j, c in enumerate(v.checks) if j != idx)
            recs.append(dict(patient=i, fault=kind, detected=flagged, detected_by_expected_check=fired,
                             collateral_flags=not other_claims_ok))
        # threshold sweep on clean claims and unsupported statements
        for t in sweep_thr:
            ver.threshold = t
            v = ver.verify(clean, attr, patient, ids, rag=True)
            sweep[t]["clean_claims"] += len(v.checks)
            sweep[t]["clean_supported"] += sum(c.supported for c in v.checks)
            bad, idx = inject_fault(clean, "unsupported_statement", topk, patient, by_id, np.random.default_rng(i))
            if bad is not None:
                v = ver.verify(bad, attr, patient, ids, rag=True)
                sweep[t]["unsup_stmt_n"] += 1
                sweep[t]["unsup_stmt_caught"] += int(not v.checks[idx].supported)
        ver.threshold = cfg["faithfulness"]["support_threshold"]
    d = pd.DataFrame(recs)
    tab = d.groupby("fault").agg(n=("detected", "size"), detection_rate=("detected", "mean"),
                                 detected_by_expected_check=("detected_by_expected_check", "mean"),
                                 collateral_flags=("collateral_flags", "mean")).reset_index()
    ctx.save_table(tab, "verifier_fault_injection")
    sw = pd.DataFrame([dict(support_threshold=t, clean_claim_acceptance=v["clean_supported"] / max(1, v["clean_claims"]),
                            unsupported_statement_detection=v["unsup_stmt_caught"] / max(1, v["unsup_stmt_n"]))
                       for t, v in sweep.items()])
    ctx.save_table(sw, "verifier_threshold_sweep")
    ctx.say("[E5b] verifier fault injection:\n" + tab.round(3).to_string(index=False))
    ctx.say("[E5b] support-threshold sweep:\n" + sw.round(3).to_string(index=False))
    return tab.set_index("fault")["detection_rate"].to_dict()


def _plot(ctx: Ctx, agg: pd.DataFrame):
    import matplotlib.pyplot as plt
    mets = [("claim_faithful_rate", "Claims faithful to SHAP"), ("evidence_grounded_rate", "Claims grounded in evidence"),
            ("fully_verified_rate", "Fully verified claims"), ("recall_at_k", "Top-k drivers covered")]
    conds = list(agg.condition)
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        r = agg[agg.condition == c].iloc[0]
        vals = [r[m] for m, _ in mets]
        x = np.arange(len(mets)) + i * w - 0.4 + w / 2
        ax.bar(x, vals, w * 0.88, color=plots.SERIES[i % len(plots.SERIES)], label=c)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=plots.C["ink"])
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([l for _, l in mets], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("Explanation faithfulness by generation condition")
    ax.legend(fontsize=8, ncol=len(conds), loc="lower center", bbox_to_anchor=(0.5, -0.28))
    plots.save(fig, ctx.figures / "fig_faithfulness.png")
