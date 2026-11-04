# Paper outline (numbers copied from results/RESULTS.md; re-run on your machine and update)

**Title:** Trustworthy Heart Disease Prediction: A Leakage-Free Benchmark with Guideline-Grounded, Faithfulness-Verified Explanations and Uncertainty-Aware Deferral

## Abstract (draft)
The public 1025-row heart-disease dataset contains only 302 unique patients. Under the 80/20 protocol of the base paper, 97.1% of test rows have an identical copy in training, so accuracy reflects memorisation (random forest 97.1% on the duplicated split vs 83.9% de-duplicated 10-fold CV). We provide a leakage-free benchmark (best: logistic regression, 85.0% accuracy, 91.4% AUC), calibrated split-conformal deferral (94.7% accuracy on the 42% of patients answered at alpha=0.05; 95% accuracy reachable at 54.6% coverage), and guideline-grounded explanations checked by a claim-level verifier (100% detection on 7 injected fault types).

## Contributions
1. Leakage audit + de-duplicated repeated-CV benchmark with CIs and corrected paired tests (no model differs significantly, Holm p>0.59).
2. Claim-level faithfulness verifier and EFS metric, validated by fault injection.
3. Conformal deferral with risk-coverage analysis.
4. Missing-data, fairness, feature-reduction studies.

## Sections
Introduction | Related work | Data and leakage audit (E1) | Methods | Experiments E2-E8 | Discussion | Limitations | Conclusion

## Must be done before submission (not yet done)
- Run external validation (E3): needs UCI Hungary/Switzerland/VA download on your machine.
- Run LLM conditions (no-RAG, RAG, RAG+verified) with Ollama; compare EFS. Only template baselines ran in the sandbox.
- Run torch models (paper-style dense net, Conv1D) and the dense retriever (BGE) on your machine.
- Verify every entry in knowledge_base/seed/seed_kb.json and the guideline citations.
- Human audit of ~100 explanations (scripts/audit_agreement.py) for Cohen's kappa.
- Template baselines scored 0.99 fully verified / 0.01 safety violation: inspect that single flagged case before reporting.
