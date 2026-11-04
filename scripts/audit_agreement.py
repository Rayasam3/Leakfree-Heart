"""Agreement between the automatic verifier and a human annotator.

1. Run the faithfulness experiment: it writes results/tables/faithfulness_audit_sample.csv
2. Open that CSV and fill the column human_label_faithful_and_supported with 1 (claim is faithful to the SHAP
   direction AND supported by the cited passage) or 0.  Use a second annotator for inter-rater kappa if possible.
3. python scripts/audit_agreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
f = ROOT / "results" / "tables" / "faithfulness_audit_sample.csv"
if not f.exists():
    sys.exit("Run: python scripts/run_experiments.py --only faith   first")
d = pd.read_csv(f)
col = "human_label_faithful_and_supported"
d = d[d[col].isin([0, 1])]
if len(d) < 20:
    sys.exit(f"Only {len(d)} labelled rows in {f.name}. Label at least 100 claims for a meaningful kappa.")
y, v = d[col].astype(int), d["verifier_pass"].astype(int)
print(f"labelled claims: {len(d)}")
print("verifier vs human accuracy:", round((y == v).mean(), 3))
print("Cohen's kappa:", round(cohen_kappa_score(y, v), 3))
print("confusion [rows=human 0/1, cols=verifier 0/1]:\n", confusion_matrix(y, v))
