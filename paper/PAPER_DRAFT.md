# Trustworthy Heart Disease Prediction: A Leakage-Free Benchmark with Guideline-Grounded, Faithfulness-Verified Explanations and Uncertainty-Aware Deferral

> DRAFT. All audit numbers filled. Sections 4.1-4.3, 4.5-4.7 use your full-run results. Every other number below comes from your full run (UCI data, primary cohort Cleveland, n=303; 302 unique patients in the Kaggle copy).

## Abstract
Tabular deep-learning studies on the public Kaggle heart-disease file report accuracies above 98%. We show that this file contains 1,025 rows but only 302 unique patients, and that 97.1% of the test rows in the usual 80/20 split have an identical copy in the training set. With duplicates kept, 10-fold cross-validation gives 99.6% for random forest and 98.9% for a dense network; with duplicates removed the same models reach 82.9% and 82.6%. We therefore build a leakage-free benchmark (repeated stratified 10-fold CV, patient-level bootstrap CIs, corrected resampled t-tests) on which six models, including a dense and a Conv1D network, reach 80.4-84.2% accuracy and 0.889-0.915 AUC with no significant differences between them. On top of the best model we add (i) calibrated split-conformal prediction with deferral, which answers 40% of patients at 94.2% accuracy (alpha=0.05) and reaches 95% accuracy at about 51% coverage, and (ii) guideline-grounded explanations produced by a 3B local LLM and checked claim by claim against the model's SHAP attributions and the cited source. Without retrieval only 14% of the LLM's claims are faithful to the model (83% contain numbers that cannot be verified); with retrieval 79%; with retrieval plus our verify-and-repair loop 100%, at a small cost in coverage of the model's top drivers (recall 0.96). The Explanation Faithfulness Score rises from 0.15 to 0.83 to 0.98. The verifier detects all seven injected fault types. We also report missing-data robustness, subgroup performance, feature reduction and external validation on three other hospitals, where ranking ability transfers (AUC 0.70-0.89) but fixed thresholds do not.

## 1. Introduction
Contributions:
1. A reproducible audit of the duplicated Kaggle heart-disease file and of the evaluation protocol that produces near-perfect accuracy; a leakage-free benchmark with uncertainty.
2. Uncertainty-aware deferral (calibration + class-conditional split-conformal) with accuracy-coverage analysis.
3. A claim-level faithfulness verifier and the Explanation Faithfulness Score (EFS) for RAG explanations, validated by fault injection and a blinded claim-level audit (AI pre-labelled, human reviewed; agreement with the verifier 78.9%, Cohen's kappa 0.20).
4. Robustness (missing data), fairness and external-validation analyses.

## 2. Data and leakage audit (E1)
- Kaggle copy: 1,025 rows, 302 unique patients (70.5% duplicate rows), one record appearing up to 8 times.
- Paper-style split: 199/205 test rows (97.1%) have an exact copy in training.
- Paper protocol accuracy: logistic regression 79.5%, random forest 97.1%, XGBoost 91.7%, dense network 97.6%.
- 10-fold CV, duplicated vs de-duplicated: logistic regression 84.9 vs 83.8; random forest 99.6 vs 82.9; XGBoost 95.5 vs 82.1; dense 98.9 vs 82.6; Conv1D 91.7 vs 79.5.
- Models that can memorise (RF, deep nets) gain 13-16 points from duplicates; a linear model gains about 1. Figure: results/figures/fig_leakage_audit.png.
- Primary cohort for all later experiments: UCI Cleveland (n=303, prevalence 0.459). Other UCI sites (Hungarian, Switzerland, VA) are used for external validation (n=918 in total after de-duplication).

## 3. Methods
Pipelines (imputation, scaling, encoding, optional feature selection) are fitted inside each training fold. Models: logistic regression, RBF-SVM, random forest, XGBoost, dense network (paper-style), Conv1D. Repeated stratified 10-fold CV; bootstrap CIs over patients; Nadeau-Bengio corrected t-test with Holm correction. Calibration: sigmoid; ECE and Brier. Conformal: LAC score, class-conditional (Mondrian) split conformal; a singleton prediction set is an answer, anything else is a deferral. Explanations: SHAP (permutation, raw features, probability units) and LIME; RAG with hybrid TF-IDF + BGE dense retrieval (reciprocal rank fusion), 27 curated passages plus cleaned guideline/health-agency pages; a 3B LLM (llama3.2:3b, local) returns structured JSON claims; the verifier checks (a) direction and value against SHAP, (b) citation existence, (c) lexical/semantic support by the cited passage, (d) safety rules; repair loop (max 2) then pruning. EFS combines claim-faithfulness and evidence-grounding.

## 4. Results
### 4.1 Benchmark (E2, primary cohort) and external validation (E3)
Repeated stratified CV on the Cleveland cohort (n=303), mean with 95% bootstrap CI:
| model | accuracy % | AUC % | sensitivity % | specificity % | Brier |
|---|---|---|---|---|---|
| Logistic regression | 84.2 [80.0, 88.2] | 91.1 [87.7, 94.2] | 79.4 [72.4, 85.7] | 88.3 [83.4, 92.6] | 0.116 |
| SVM (RBF) | 83.2 [79.1, 87.1] | 89.9 [86.2, 93.4] | 79.1 [72.4, 85.4] | 86.7 [81.5, 91.3] | 0.123 |
| Random forest | 82.7 [78.5, 86.5] | 90.8 [87.4, 93.8] | 78.6 [71.5, 84.8] | 86.2 [81.0, 90.9] | 0.123 |
| XGBoost | 82.1 [78.0, 86.0] | 89.6 [85.8, 92.8] | 77.4 [70.4, 83.6] | 86.1 [81.1, 90.9] | 0.131 |
| Dense NN (base paper) | 83.5 [79.5, 87.3] | 88.7 [84.9, 92.1] | 78.8 [72.1, 85.0] | 87.4 [82.8, 91.7] | 0.130 |
| Conv1D CNN | 80.5 [76.2, 84.4] | 88.6 [84.8, 91.9] | 76.3 [69.7, 82.2] | 84.0 [78.6, 88.9] | 0.135 |

No pairwise AUC difference is significant after Holm correction (smallest raw p = 0.035, logistic regression vs Conv1D; Holm p = 0.52; all other Holm p = 1.0). The simple linear model is as good as any deep model.

External validation (trained on Cleveland, tested on the other UCI hospitals; 20-29% of feature values missing in these sites):
| test site | n | prevalence | logreg AUC [95% CI] | RF AUC | XGBoost AUC | logreg accuracy |
|---|---|---|---|---|---|---|
| Hungarian | 293 | 0.36 | 0.887 [0.846, 0.928] | 0.894 | 0.884 | 79.5% |
| Switzerland | 123 | 0.93 | 0.734 [0.476, 0.932] | 0.757 | 0.771 | 45.5% |
| VA | 199 | 0.74 | 0.696 [0.602, 0.790] | 0.732 | 0.720 | 48.7% |
Ranking ability transfers well to Hungary and only moderately to VA and Switzerland. Accuracy falls sharply in the last two sites because disease prevalence differs greatly (74% and 93%) and the decision threshold does not transfer (logistic-regression sensitivity 0.36-0.46). Thresholds and calibration must be re-fitted per site; this supports using calibrated probabilities with deferral rather than a fixed cut-off. The Switzerland CI is very wide (n=123, 93% positive). Leave-one-site-out AUCs for logistic regression: Cleveland 0.873, Hungarian 0.895, Switzerland 0.753, VA 0.726 (results/tables/leave_one_site_out.csv). The deep networks transfer worst in specificity (0.29-0.38 in leave-one-site-out), i.e. they over-call disease.

### 4.2 Calibration and deferral (E4)
Logistic regression, sigmoid-calibrated: accuracy 84.8%, AUC 0.912, Brier 0.117, ECE 0.055.
Selective accuracy by confidence: 87.2% at 90% coverage, 91.7% at 70%, 95.3% at 50%; maximum coverage at 95% accuracy = 51.3%.
Conformal deferral (class-conditional):
| alpha | answered | accuracy on answered | accuracy if forced on deferred |
|---|---|---|---|
| 0.05 | 40.2% | 94.2% | 78.4% |
| 0.10 | 61.5% | 91.1% | 74.3% |
| 0.15 | 78.2% | 89.3% | 67.9% |
| 0.20 | 91.4% | 85.3% | 64.5% |
Deferred patients are indeed harder: the model is only 65-78% accurate on them. Note: 95% is accuracy on the answered subset, not overall accuracy.

### 4.3 Explanation quality (E5a, 60 held-out patients)
SHAP-LIME agreement: Spearman 0.84, top-5 Jaccard 0.72, sign agreement 100%. SHAP stability over reruns: top-5 Jaccard 0.77, Spearman 0.94. Deletion fidelity: removing the top-3 SHAP features changes the prediction by 0.218 (random features: 0.043); LIME 0.139.

### 4.4 Faithfulness of RAG explanations (E5b)
40 held-out patients, top-5 SHAP drivers each, llama3.2:3b (local), 77-chunk knowledge base, all conditions completed for all 40 patients:
| condition | claims faithful to model | evidence-grounded | fully verified | recall of top drivers | EFS [95% CI] | unverifiable numbers |
|---|---|---|---|---|---|---|
| Template (no retrieval) | 99.0% | n/a | 98.5% | 1.00 | 0.994 | 1.0% |
| Template + retrieval | 100% | 100% | 99.5% | 1.00 | 1.000 | 0% |
| LLM, no retrieval | 14.1% | n/a | 14.1% | 1.00 | 0.145 [0.048, 0.263] | 82.9% |
| LLM + retrieval | 79.1% | 95.2% | 78.6% | 0.97 | 0.825 [0.727, 0.912] | 3.0% |
| LLM + retrieval + verify-and-repair | 100% | 100% | 100% | 0.96 | 0.978 [0.964, 0.992] | 0% |
Retrieval alone removes most of the invented numbers (82.9% to 3.0%) and raises faithful claims from 14% to 79%. The verify-and-repair loop closes the rest: on average 1.7 LLM calls, 0.7 repairs and 0.43 pruned claims per patient; a small loss in coverage of the top drivers (recall 0.96) is the price. No fabricated citations occurred in any condition. One LLM+retrieval output (2.5%) was not valid JSON. Safety-rule violations: 0.5% for the template and no-retrieval conditions, 0% for the retrieval and verified conditions.
Verifier validation by fault injection (40 patients, 7 fault types: direction flip, off-top-k feature, wrong number, fabricated citation, unsupported citation, unsupported statement, unsafe statement): detection 100% for every type, detected by the expected check; collateral flags 0-2.5%. Support-threshold sweep: clean-claim acceptance 100% for thresholds 0.05-0.20 and 99.5% at 0.30, unsupported-statement detection 100% throughout.
The 100% for the verified condition is a property of the gated pipeline (a claim that fails is repaired or removed), not evidence of clinical correctness. Human audit of 180 shuffled claims (60 per LLM condition), agreement with the verifier: accuracy 78.9%, Cohen's kappa 0.20. Labels were AI pre-labelled and human reviewed by the author (single rater, so no inter-rater kappa). Judged faithful and supported by the rater: 95.0% (no retrieval), 68.3% (RAG), 66.7% (RAG + verifier); the verifier passed 93.3%, 90.0% and 100% respectively. The verifier rarely rejected a claim the rater accepted (3 of 138), but it passed 35 of 42 claims the rater rejected, mostly semantic errors (a normal thallium result described as ischaemia, generic statements pointing against the SHAP sign, fragments). Automatic verification therefore guarantees structure, direction, numbers and citation validity, not full semantic correctness; a stronger entailment check is future work.

### 4.5 Feature reduction (E6)
Logistic regression: top-3 features 81.5% / AUC 0.870; top-5 84.0% / 0.903; top-8 82.8% / 0.897; all 13 84.7% / 0.910. Five features recover the full-model performance.

### 4.6 Missing data (E7)
Features were removed completely at random (MCAR). ROC AUC (mean over folds) when missing values appear only at prediction time:
| model | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Random forest | 0.916 | 0.908 | 0.892 | 0.860 | 0.849 |
| XGBoost | 0.898 | 0.891 | 0.874 | 0.848 | 0.836 |
| Logistic regression | 0.912 | 0.888 | 0.865 | 0.824 | 0.798 |
| Dense NN (base paper) | 0.888 | 0.848 | 0.810 | 0.785 | 0.750 |
The base paper's deep network is the least robust (-14 AUC points at 40% missing) and random forest the most (-7). Training under the same missingness recovers much of the loss: at 40% missing the dense network reaches 0.856, logistic regression 0.847, random forest 0.847, XGBoost 0.844. So a model trained only on complete data degrades when inputs are missing, and missing-aware training is needed. (Test-only accuracy at 40% missing: random forest 76.7%, logistic regression 72.9%, dense 70.4%.)

### 4.7 Subgroup performance (E8, logistic regression)
Women: accuracy 89.7%, sensitivity 69.6%. Men: 81.7%, 81.6%. Age <56: 89.0%, sensitivity 79.6%. Age >=56: 79.5%, 79.3%. Sensitivity is lowest in women, driven by low prevalence (25.8%).

## 5. Discussion
Leakage explains the gap between published near-perfect accuracy and realistic ~84%. Differences between models are within noise on 303 patients, so model choice matters less than evaluation design. Deferral makes 95% accuracy reachable only on a subset. Retrieval removes most unsupported numbers; the verifier removes the rest at a small recall cost.

## 6. Limitations
Small data; external accuracy drops sharply at sites with very different prevalence (45-49% at two hospitals); the LLM is small (3B) and one model was tested; verifier support check is lexical/semantic, not clinical; KB entries are paraphrases that must be verified against sources; dataset is from 1980s-1990s; no prospective validation; not a medical device.

## 7. Conclusion
(2-3 sentences summarising the three contributions.)

## References to cite
Alshayeji & Abed (2025) base paper; Detrano et al. (1989); Nadeau & Bengio (2003); Angelopoulos & Bates (2021) conformal; Lundberg & Lee (2017) SHAP; Ribeiro et al. (2016) LIME; Lewis et al. (2020) RAG; Guo et al. (2017) calibration; the guidelines listed in knowledge_base/sources.yaml.
