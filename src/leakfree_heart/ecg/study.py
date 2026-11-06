"""ECG study: leakage audit, six conditions (clinical / ECG / fusion), calibration, conformal deferral, subgroups.

Stages: audit  train  evaluate  leakage_train
Everything is evaluated on PTB-XL fold 10 (patient-wise); fold 8 stops training; fold 9 is split by patient into
cal_a (temperature scaling, stacking) and cal_b (conformal calibration). No stage ever selects anything on the test fold.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..config import ROOT, load_config
from ..evaluation import binary_metrics
from ..uncertainty import calibrate_conformal, decisions_from_sets, prediction_sets
from . import data as D
from .features import ecg_features

CONDITIONS = ["tab_logreg", "feat_xgb", "feat_xgb_tab", "ecg_cnn", "fusion_e2e", "fusion_stack"]
NICE = {"tab_logreg": "Clinical only (age, sex)", "feat_xgb": "ECG features + XGBoost",
        "feat_xgb_tab": "ECG features + age/sex + XGBoost", "ecg_cnn": "ECG ResNet (signal only)",
        "fusion_e2e": "Fusion: ECG ResNet + age/sex (end to end)", "fusion_stack": "Fusion: ECG ResNet logit + age/sex (stacked)"}
SETS = ["train", "stop", "cal_a", "cal_b", "test"]


@dataclass
class ECGCtx:
    cfg: dict
    root: Path
    out: Path
    quick: bool = False
    meta: pd.DataFrame = None
    X: np.ndarray = None
    split: dict = None
    tab: np.ndarray = None
    y: np.ndarray = None
    tables: Path = None
    figures: Path = None
    cache: Path = None
    log: list = field(default_factory=list)

    def say(self, m: str) -> None:
        print(m, flush=True)
        self.log.append(m)

    def save_table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.tables / f"{name}.csv", index=False)

    def save_json(self, obj, name: str) -> None:
        (self.tables / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


# ------------------------------------------------------------------ setup
def make_ecg_ctx(config_path=None, quick=False, demo=False, out_name=None, device=None) -> ECGCtx:
    cfg = load_config(config_path)
    ecfg = cfg["ecg"]
    if device:
        ecfg["device"] = device
    root = ROOT / ecfg["root"]
    out = ROOT / (out_name or ("results_ecg_demo" if demo else "results_ecg"))
    if demo:
        from .synthetic import make_synthetic_ptbxl
        root = ROOT / "data" / "raw" / "ptbxl_demo"
        if not (root / "ptbxl_database.csv").exists():
            print("[demo] writing a small SYNTHETIC PTB-XL-shaped dataset (not real data) ...")
            make_synthetic_ptbxl(root, 400)
    if quick or demo:
        ecfg.update(epochs=3, patience=3, seeds=[0], n_boot=100)
    ctx = ECGCtx(cfg=cfg, root=root, out=out, quick=quick or demo)
    ctx.tables, ctx.figures, ctx.cache = (out / "tables", out / "figures", out / "cache")
    for p in (ctx.tables, ctx.figures, ctx.cache):
        p.mkdir(parents=True, exist_ok=True)
    meta = D.load_metadata(root, ecfg["min_likelihood"])
    if quick and not demo:
        meta = meta.sample(n=min(3000, len(meta)), random_state=0).sort_index()
    ctx.meta = meta
    ctx.X = D.load_signals(root, meta, ctx.cache / ("signals_quick.npy" if quick else "signals_100hz.npy"))
    ctx.split = D.official_split(meta, cfg["project"]["seed"])
    ctx.y = meta["label"].to_numpy().astype(int)
    ctx.tab = np.column_stack([(meta["age_years"].to_numpy() - 60.0) / 20.0, meta["sex_f"].to_numpy()]).astype(np.float32)
    ctx.say(f"[ecg] root={root}  ECGs={len(meta)}  patients={meta['patient_id'].nunique()}  "
            f"abnormal={ctx.y.mean():.3f}  device={_resolved_device(ecfg['device'])}")
    ctx.say("[ecg] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in ctx.split.items()))
    return ctx


def _resolved_device(pref: str) -> str:
    from . import nets
    if not nets.TORCH:
        return "none (PyTorch missing)"
    d = nets.device_name(pref)
    if d == "cuda":
        import torch
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "CPU  <-- training will be slow; install the CUDA build of PyTorch"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_temperature(logit, y) -> float:
    def nll(T):
        p = np.clip(_sigmoid(logit / T), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


# ------------------------------------------------------------------ stage 1: audit
def run_audit(ctx: ECGCtx) -> None:
    summ = D.dataset_summary(ctx.meta)
    ctx.save_table(summ, "ecg_dataset_summary")
    ctx.say(summ.to_string(index=False))
    sizes = {k: len(v) for k, v in ctx.split.items()}
    rows = []
    for name, sp in (("Official patient-wise folds", ctx.split),
                     ("Random record-wise split (same sizes)", D.random_record_split(ctx.meta, sizes, 42))):
        o = D.patient_overlap(ctx.meta, sp)
        rows.append({"protocol": name, **o})
    df = pd.DataFrame(rows)
    ctx.save_table(df, "ecg_patient_leakage")
    ctx.say("[audit] patient overlap between train and test:\n" + df.to_string(index=False))


# ------------------------------------------------------------------ stage 2: train
def _tabular_models(ctx: ECGCtx, feats: pd.DataFrame, seed: int) -> dict:
    """Non-deep conditions. Returns {cond: {set: logit}}."""
    import xgboost as xgb
    sp, y, tab = ctx.split, ctx.y, ctx.tab
    out = {}
    T3 = np.column_stack([tab, tab[:, 0] * tab[:, 1]])
    lr = LogisticRegression(max_iter=1000).fit(T3[sp["train"]], y[sp["train"]])
    out["tab_logreg"] = {k: lr.decision_function(T3[sp[k]]) for k in SETS}

    def boosted(F):
        m = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              eval_metric="logloss", early_stopping_rounds=30, random_state=seed, n_jobs=2,
                              tree_method="hist")
        m.fit(F[sp["train"]], y[sp["train"]], eval_set=[(F[sp["stop"]], y[sp["stop"]])], verbose=False)
        return {k: _logit(m.predict_proba(F[sp[k]])[:, 1]) for k in SETS}
    Fv = feats.to_numpy(dtype=np.float32)
    out["feat_xgb"] = boosted(Fv)
    out["feat_xgb_tab"] = boosted(np.column_stack([Fv, tab]))
    return out


def run_train(ctx: ECGCtx) -> None:
    from . import nets
    e = ctx.cfg["ecg"]
    sp, y, X, tab = ctx.split, ctx.y, ctx.X, ctx.tab
    fcache = ctx.cache / ("ecg_features_quick.pkl" if ctx.quick else "ecg_features.pkl")
    if fcache.exists():
        feats = pd.read_pickle(fcache)
    else:
        ctx.say("[train] computing hand-crafted ECG features ...")
        feats = ecg_features(X)
        feats.to_pickle(fcache)
    preds: dict[str, dict] = {}
    t0 = time.time()
    for seed in e["seeds"]:
        ctx.say(f"[train] seed {seed}: clinical and feature baselines")
        for c, d in _tabular_models(ctx, feats, seed).items():
            preds.setdefault(c, {})[seed] = d
        if not nets.TORCH:
            ctx.say("[train] PyTorch not installed: skipping the neural conditions")
            continue
        for cond, use_tab in (("ecg_cnn", False), ("fusion_e2e", True)):
            ctx.say(f"[train] seed {seed}: {NICE[cond]}  ({time.time() - t0:.0f}s so far)")
            model, mean, std = nets.train_net(
                X[sp["train"]], y[sp["train"]], X[sp["stop"]], y[sp["stop"]],
                tab[sp["train"]] if use_tab else None, tab[sp["stop"]] if use_tab else None,
                seed=seed, epochs=e["epochs"], batch_size=e["batch_size"], lr=e["lr"], patience=e["patience"],
                device=e["device"], width=e["width"])
            d = {k: nets.predict_logits(model, X[sp[k]], tab[sp[k]] if use_tab else None, mean, std, e["device"])
                 for k in SETS}
            preds.setdefault(cond, {})[seed] = d
            if cond == "ecg_cnn" and ctx.cfg["ecg"].get("save_models", True):
                import torch
                torch.save({"state": model.state_dict(), "mean": mean, "std": std, "seed": seed, "tab": use_tab},
                           ctx.cache / f"{cond}_seed{seed}.pt")
            elif cond == "fusion_e2e":
                import torch
                torch.save({"state": model.state_dict(), "mean": mean, "std": std, "seed": seed, "tab": use_tab},
                           ctx.cache / f"{cond}_seed{seed}.pt")
        # stacked fusion: logistic regression on [ECG logit, age, sex] fitted on cal_a only
        ec = preds["ecg_cnn"][seed]
        feat = lambda k: np.column_stack([ec[k], tab[sp[k]]])  # noqa: E731
        st = LogisticRegression(max_iter=1000).fit(feat("cal_a"), y[sp["cal_a"]])
        preds.setdefault("fusion_stack", {})[seed] = {k: st.decision_function(feat(k)) for k in SETS}
    flat = {f"{c}|{s}|{k}": v for c, ds in preds.items() for s, d in ds.items() for k, v in d.items()}
    np.savez_compressed(ctx.cache / "ecg_preds.npz", **flat)
    ctx.say(f"[train] saved predictions for {len(preds)} conditions ({time.time() - t0:.0f}s)")


def _load_preds(ctx: ECGCtx) -> dict[str, dict[str, np.ndarray]]:
    """-> {cond: {set: ensemble-mean logit over seeds}} and per-seed logits."""
    z = np.load(ctx.cache / "ecg_preds.npz")
    per: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for key in z.files:
        c, s, k = key.split("|")
        per.setdefault(c, {}).setdefault(int(s), {})[k] = z[key]
    ens = {c: {k: np.mean([per[c][s][k] for s in per[c]], axis=0) for k in SETS} for c in per}
    return ens, per


# ------------------------------------------------------------------ stage 3: evaluate
def _patient_boot(pid: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    groups = pd.Series(np.arange(len(pid))).groupby(pid).apply(lambda s: s.to_numpy()).tolist()
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]


def _ci(vals, nd=3):
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), nd), round(float(hi), nd)


def run_evaluate(ctx: ECGCtx) -> None:
    e = ctx.cfg["ecg"]
    sp, y, meta = ctx.split, ctx.y, ctx.meta
    ens, per = _load_preds(ctx)
    conds = [c for c in CONDITIONS if c in ens]
    yt, yc, ya = y[sp["test"]], y[sp["cal_b"]], y[sp["cal_a"]]
    boots = _patient_boot(meta["patient_id"].to_numpy()[sp["test"]], e["n_boot"], 7)

    # temperature scaling on cal_a (all conditions, uniform pipeline)
    prob = {c: {} for c in conds}
    temps = {}
    for c in conds:
        T = fit_temperature(ens[c]["cal_a"], ya)
        temps[c] = T
        for k in SETS:
            prob[c][k] = _sigmoid(ens[c][k] / T)
    # ------------------------------ benchmark table
    rows, aucs = [], {}
    for c in conds:
        p = prob[c]["test"]
        m = binary_metrics(yt, p)
        aucs[c] = np.array([roc_auc_score(yt[b], p[b]) for b in boots if len(np.unique(yt[b])) == 2])
        bm = [binary_metrics(yt[b], p[b]) for b in boots[: min(len(boots), 300)]]
        seed_auc = [roc_auc_score(yt, per[c][s]["test"]) for s in per[c]]
        row = {"condition": c, "name": NICE[c], "n_test": len(yt), "n_seeds": len(per[c]),
               "auc": round(m["auc"], 4), "auc_ci": _ci(aucs[c], 3),
               "accuracy": round(m["accuracy"], 4), "acc_ci": _ci([b["accuracy"] for b in bm]),
               "sensitivity": round(m["sensitivity"], 4), "specificity": round(m["specificity"], 4),
               "brier": round(m["brier"], 4), "ece": round(m["ece"], 4),
               "auc_seed_mean": round(float(np.mean(seed_auc)), 4), "auc_seed_sd": round(float(np.std(seed_auc)), 4),
               "temperature": round(temps[c], 3)}
        rows.append(row)
    bench = pd.DataFrame(rows)
    ctx.save_table(bench, "ecg_benchmark")
    ctx.say("[evaluate] benchmark (test fold, patient-bootstrap CIs):\n" +
            bench[["name", "auc", "auc_ci", "accuracy", "sensitivity", "specificity", "ece"]].to_string(index=False))

    # ------------------------------ paired AUC differences
    pairs = [("fusion_e2e", "ecg_cnn"), ("fusion_stack", "ecg_cnn"), ("fusion_e2e", "tab_logreg"),
             ("ecg_cnn", "feat_xgb"), ("fusion_stack", "feat_xgb_tab"), ("fusion_e2e", "fusion_stack")]
    prow = []
    for a, b in pairs:
        if a in aucs and b in aucs:
            n = min(len(aucs[a]), len(aucs[b]))
            d = aucs[a][:n] - aucs[b][:n]
            prow.append({"model_a": a, "model_b": b, "auc_a": round(roc_auc_score(yt, prob[a]["test"]), 4),
                         "auc_b": round(roc_auc_score(yt, prob[b]["test"]), 4),
                         "diff": round(float(roc_auc_score(yt, prob[a]["test"]) - roc_auc_score(yt, prob[b]["test"])), 4),
                         "diff_ci": _ci(d), "p_boot": round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)})
    ctx.save_table(pd.DataFrame(prow), "ecg_pairwise_auc")
    ctx.say("[evaluate] paired AUC differences:\n" + pd.DataFrame(prow).to_string(index=False))

    # ------------------------------ calibration before / after temperature scaling
    crow = []
    for c in conds:
        raw = _sigmoid(ens[c]["test"])
        for tag, p in (("raw", raw), ("temperature-scaled", prob[c]["test"])):
            m = binary_metrics(yt, p)
            crow.append({"condition": c, "variant": tag, "brier": round(m["brier"], 4), "ece": round(m["ece"], 4)})
    ctx.save_table(pd.DataFrame(crow), "ecg_calibration")

    # ------------------------------ conformal deferral (calibration = cal_b) + selective accuracy
    conf_rows, sel_rows = [], []
    for c in conds:
        cal = calibrate_conformal(prob[c]["cal_b"], yc)
        pt = prob[c]["test"]
        for a in e["alphas"]:
            dec = decisions_from_sets(prediction_sets(pt, cal.thresholds(a, True)))

            def stats(idx):
                d, yy, pp = dec[idx], yt[idx], pt[idx]
                ans = d >= 0
                return [ans.mean(), np.mean(d[ans] == yy[ans]) if ans.any() else np.nan,
                        np.mean((pp[~ans] >= .5).astype(int) == yy[~ans]) if (~ans).any() else np.nan,
                        np.mean(((d == yy) | (d == -1))[yy == 1]), np.mean(((d == yy) | (d == -1))[yy == 0])]
            pt_est = stats(np.arange(len(yt)))
            bs = np.array([stats(b) for b in boots[: min(len(boots), 300)]])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            conf_rows.append({"condition": c, "alpha": a, "answered": round(pt_est[0], 4),
                              "answered_ci": (round(float(lo[0]), 3), round(float(hi[0]), 3)),
                              "accuracy_answered": round(pt_est[1], 4), "acc_answered_ci": (round(float(lo[1]), 3), round(float(hi[1]), 3)),
                              "accuracy_deferred_if_forced": round(pt_est[2], 4),
                              "coverage_abnormal": round(pt_est[3], 4), "coverage_normal": round(pt_est[4], 4)})
        conf = np.maximum(pt, 1 - pt)
        order = np.argsort(-conf)
        corr = ((pt >= .5).astype(int) == yt)[order]
        for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            k = max(1, int(round(cov * len(yt))))
            sel_rows.append({"condition": c, "coverage": cov, "accuracy": round(float(corr[:k].mean()), 4)})
    ctx.save_table(pd.DataFrame(conf_rows), "ecg_conformal")
    ctx.save_table(pd.DataFrame(sel_rows), "ecg_selective")
    cdf = pd.DataFrame(conf_rows)
    ctx.say("[evaluate] conformal deferral (alpha = 0.05 / 0.10):\n" +
            cdf[cdf.alpha.isin([0.05, 0.10])][["condition", "alpha", "answered", "accuracy_answered",
                                                "coverage_abnormal", "coverage_normal"]].to_string(index=False))

    # ------------------------------ subgroups and diagnostic superclasses for the primary condition
    primary = bench.assign(sel=[roc_auc_score(ya, prob[c]["cal_a"]) for c in bench.condition]).sort_values(
        "sel", ascending=False).iloc[0]["condition"]       # chosen on cal_a, never on the test fold
    ctx.say(f"[evaluate] primary condition (best AUC on cal_a): {primary}")
    mt = meta.iloc[sp["test"]]
    p = prob[primary]["test"]
    srows = []
    groups = [("sex", "male", mt["sex_f"] == 0), ("sex", "female", mt["sex_f"] == 1),
              ("age", "<50", mt["age_years"] < 50), ("age", "50-69", (mt["age_years"] >= 50) & (mt["age_years"] < 70)),
              ("age", ">=70", mt["age_years"] >= 70)]
    for attr, name, mask in groups:
        mk = mask.to_numpy()
        if mk.sum() > 20 and len(np.unique(yt[mk])) == 2:
            m = binary_metrics(yt[mk], p[mk])
            srows.append({"attribute": attr, "group": name, "n": int(mk.sum()), "prevalence": round(float(yt[mk].mean()), 3),
                          "auc": round(m["auc"], 4), "accuracy": round(m["accuracy"], 4),
                          "sensitivity": round(m["sensitivity"], 4), "specificity": round(m["specificity"], 4)})
    ctx.save_table(pd.DataFrame(srows), "ecg_subgroups")
    sup = []
    for c in D.ABNORMAL:
        mk = mt["superclasses"].apply(lambda s, c=c: c in s and s != {"NORM"}).to_numpy()
        if mk.sum():
            sup.append({"superclass": c, "n_test": int(mk.sum()), "recall_at_0.5": round(float((p[mk] >= .5).mean()), 4)})
    sup.append({"superclass": "NORM (specificity)", "n_test": int((yt == 0).sum()),
                "recall_at_0.5": round(float((p[yt == 0] < .5).mean()), 4)})
    ctx.save_table(pd.DataFrame(sup), "ecg_superclass_recall")
    ctx.say("[evaluate] recall by diagnostic superclass:\n" + pd.DataFrame(sup).to_string(index=False))
    ctx.save_json({"primary": primary, "temperatures": temps, "n_test": int(len(yt)),
                   "n_test_patients": int(mt["patient_id"].nunique())}, "ecg_summary")
    _plots(ctx, conds, prob, yt, pd.DataFrame(sel_rows), cdf)


def _plots(ctx, conds, prob, yt, sel, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, c in enumerate(conds):
        f, t, _ = roc_curve(yt, prob[c]["test"])
        ax.plot(f, t, ls=styles[i % 6], lw=1.2, color="k" if i % 2 == 0 else "0.4",
                label=f"{NICE[c]} ({roc_auc_score(yt, prob[c]['test']):.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.6)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout(); fig.savefig(ctx.figures / "ecg_roc.png", dpi=250); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for i, c in enumerate(conds):
        s = sel[sel.condition == c]
        ax.plot(s.coverage * 100, s.accuracy * 100, ls=styles[i % 6], marker="o", ms=2.5, lw=1.1,
                color="k" if i % 2 == 0 else "0.4", label=NICE[c])
    ax.invert_xaxis(); ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Accuracy on answered (%)"); ax.legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(ctx.figures / "ecg_risk_coverage.png", dpi=250); plt.close(fig)


# ------------------------------------------------------------------ stage 4: leakage training experiment
def run_leakage_train(ctx: ECGCtx) -> None:
    """Train the ECG ResNet with the official patient-wise split and with a random record-wise split; compare test AUC."""
    from . import nets
    if not nets.TORCH:
        ctx.say("[leakage_train] PyTorch not installed: skipped")
        return
    e = ctx.cfg["ecg"]
    sizes = {k: len(v) for k, v in ctx.split.items()}
    rows = []
    for name, sp in (("Official patient-wise folds", ctx.split),
                     ("Random record-wise split", D.random_record_split(ctx.meta, sizes, 42))):
        ctx.say(f"[leakage_train] training ECG ResNet on: {name}")
        model, mean, std = nets.train_net(ctx.X[sp["train"]], ctx.y[sp["train"]], ctx.X[sp["stop"]], ctx.y[sp["stop"]],
                                          seed=0, epochs=e["epochs"], batch_size=e["batch_size"], lr=e["lr"],
                                          patience=e["patience"], device=e["device"], width=e["width"])
        lg = nets.predict_logits(model, ctx.X[sp["test"]], None, mean, std, e["device"])
        yt = ctx.y[sp["test"]]
        m = binary_metrics(yt, _sigmoid(lg))
        ov = D.patient_overlap(ctx.meta, sp)
        rows.append({"protocol": name, "test_patient_overlap": round(ov["fraction"], 4),
                     "test_auc": round(m["auc"], 4), "test_accuracy": round(m["accuracy"], 4)})
    df = pd.DataFrame(rows)
    ctx.save_table(df, "ecg_leakage_training")
    ctx.say("[leakage_train]\n" + df.to_string(index=False))


STAGES = {"audit": run_audit, "train": run_train, "evaluate": run_evaluate, "leakage_train": run_leakage_train}
