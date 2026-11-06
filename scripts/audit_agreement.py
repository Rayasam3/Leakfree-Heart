"""Human audit of explanation claims -> agreement with the automatic verifier (Cohen's kappa).

1. python scripts/run_experiments.py --only faith report
   -> results/tables/audit_blinded.csv (shuffled LLM claims; condition and verifier verdict are hidden)
      results/tables/audit_key.csv     (do NOT open before labelling)
2. Open audit_blinded.csv in Excel. For each row decide:
      1 = the statement is faithful to the SHAP direction (feature + 'direction' column) AND supported by
          the text in 'cited_text'  (if there is no cited text, 1 only if the statement is clearly true and numeric
          values are consistent)
      0 = otherwise
   Rater A fills `rater_A`. A second person fills `rater_B` (needed for inter-rater kappa).
3. python scripts/audit_agreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

T = Path(__file__).resolve().parents[1] / "results" / "tables"
if not (T / "audit_blinded.csv").exists():
    sys.exit("Run: python scripts/run_experiments.py --only faith report   first")
d = pd.read_csv(T / "audit_blinded.csv").merge(pd.read_csv(T / "audit_key.csv").drop(columns=["verifier_pass"], errors="ignore"), on="audit_id")
# Verifier verdict per claim, taken from the full sample file. For the no-retrieval condition no citation can exist,
# so the verdict there is "faithful to the model" (no failure reasons); with retrieval it is "faithful AND supported".
S = pd.read_csv(T / "faithfulness_audit_sample.csv")
S["verifier_pass"] = [(not isinstance(r, str) or r == "") if c == "llm_norag" else bool(p)
                      for c, r, p in zip(S.condition, S.verifier_reasons, S.verifier_pass)]
d = d.merge(S[["condition", "patient", "feature", "statement", "verifier_pass"]].drop_duplicates(["condition", "patient", "feature", "statement"]),
            on=["condition", "patient", "feature", "statement"], how="left")
d = d[d.verifier_pass.notna()]
A = d[d.rater_A.isin([0, 1])]
if len(A) < 30:
    sys.exit(f"Only {len(A)} rows labelled by rater_A. Label at least 100 for a meaningful kappa.")
y, v = A.rater_A.astype(int), A.verifier_pass.astype(int)
print(f"rows labelled by A: {len(A)}")
print("verifier vs rater A  accuracy:", round((y == v).mean(), 3), " Cohen's kappa:", round(cohen_kappa_score(y, v), 3))
print("confusion [rows=human 0/1, cols=verifier 0/1]:\n", confusion_matrix(y, v, labels=[0, 1]))
print("\nhuman-judged faithful rate by condition:")
print(A.groupby("condition").rater_A.mean().round(3).to_string())
B = d[d.rater_A.isin([0, 1]) & d.rater_B.isin([0, 1])]
if len(B) >= 30:
    print(f"\nrater A vs rater B (n={len(B)}): Cohen's kappa =", round(cohen_kappa_score(B.rater_A.astype(int), B.rater_B.astype(int)), 3))
    yb = B.rater_B.astype(int)
    print("verifier vs rater B kappa =", round(cohen_kappa_score(yb, B.verifier_pass.astype(int)), 3))
else:
    print("\n(no/too few rater_B labels: inter-rater kappa skipped)")
