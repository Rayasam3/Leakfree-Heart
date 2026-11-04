# LeakFree-Heart
**Trustworthy Heart Disease Prediction: A Leakage-Free Benchmark with Guideline-Grounded, Faithfulness-Verified Explanations and Uncertainty-Aware Deferral**

A response to Alshayeji & Abed (2025). Three contributions:
1. **Leakage audit + benchmark**: the 1025-row Kaggle data holds 302 unique patients; duplicates inflate the reported 98.5%. We evaluate with de-duplicated, repeated, patient-level CV.
2. **Guideline-grounded RAG explanations with a claim-level faithfulness verifier** (Explanation Faithfulness Score, verify-and-repair, fault-injection validation).
3. **Calibrated conformal prediction with deferral**: accuracy versus coverage, including the coverage at which accuracy reaches 95%.

Also: missing-data robustness, fairness audit, SHAP-guided feature reduction, external validation (UCI sites).

**Honest note:** leakage-free overall accuracy is about 80-85%. 95% is reported only as selective accuracy at a stated coverage.

Start with `SETUP_AND_RUN_GUIDE.md`. Layout: `src/` library, `scripts/` entry points, `api/` FastAPI, `frontend/` React, `configs/config.yaml`, `results/`, `paper/`, `tests/`.
