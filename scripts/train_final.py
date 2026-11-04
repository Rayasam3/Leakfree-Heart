"""Train the deployable model (calibrated primary model + split-conformal thresholds) and save models/artifact.joblib.

    python scripts/train_final.py [--model logreg] [--alpha 0.10]

The artifact holds: the calibrated pipeline, conformal thresholds for several alphas, a SHAP background sample,
reference ('typical patient') values and the cross-validated benchmark numbers for display in the app.
Note: 25% of the cohort is held back to calibrate the conformal thresholds, so the deployed model is fitted on 75%.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402

from leakfree_heart.calibration import calibrated  # noqa: E402
from leakfree_heart.config import ensure_dir, path  # noqa: E402
from leakfree_heart.data import FEATURE_INFO, FEATURES, split_xy  # noqa: E402
from leakfree_heart.experiments.common import make_ctx, primary_model  # noqa: E402
from leakfree_heart.explain import reference_values  # noqa: E402
from leakfree_heart.models import make_model  # noqa: E402
from leakfree_heart.uncertainty import calibrate_conformal  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    ctx = make_ctx(args.config)
    cfg = ctx.cfg
    name = args.model or primary_model(ctx)
    X, y = split_xy(ctx.df[ctx.df.site == cfg["data"]["primary_site"]])
    y = y.values
    seed = ctx.seed
    idx = np.arange(len(y))
    tr, ca = train_test_split(idx, test_size=cfg["uncertainty"]["calib_fraction"], stratify=y, random_state=seed)
    print(f"[train_final] model={name} train={len(tr)} conformal-calibration={len(ca)}")
    model = calibrated(lambda s: make_model(name, cfg, s), "sigmoid", 5)(seed).fit(X.iloc[tr], y[tr])
    cal = calibrate_conformal(model.predict_proba(X.iloc[ca])[:, 1], y[ca])
    thr = {a: cal.thresholds(a, cfg["uncertainty"]["mondrian"]) for a in sorted(set(cfg["uncertainty"]["alphas"] + [args.alpha]))}

    metrics = {}
    bs = ctx.tables / "benchmark_summary.csv"
    if bs.exists():
        s = pd.read_csv(bs)
        row = s[s.model == name]
        if len(row):
            metrics = {k: row.iloc[0][k] for k in ("accuracy", "auc", "sensitivity", "specificity", "brier") if k in row.columns}
    art = dict(version="1.0", model_name=name, model=model, thresholds=thr, alpha=args.alpha,
               mondrian=cfg["uncertainty"]["mondrian"], X_background=X.iloc[tr].sample(min(100, len(tr)), random_state=seed),
               reference=reference_values(X.iloc[tr]).to_dict(), feature_info=FEATURE_INFO, features=FEATURES,
               cv_metrics=metrics, data_source=ctx.df.attrs.get("source"), n_train=len(tr), n_calibration=len(ca),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = ensure_dir(path(cfg, "models")) / "artifact.joblib"
    joblib.dump(art, out)
    print(f"[train_final] saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
