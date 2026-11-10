# LeakFree-Heart

**Leakage-aware heart disease prediction from clinical data and 12-lead ECGs, with calibrated "not sure" decisions and tested explanations.**

LeakFree-Heart is a research framework and web app. It applies the same five steps to two kinds of patient data:

1. **Leakage check:** no patient appears in both training and test data.
2. **Prediction:** classical models and deep networks.
3. **Calibration:** predicted probabilities that can be trusted.
4. **Answer or defer:** conformal prediction lets the model say "not sure" and refer the case to a clinician.
5. **Tested explanations:** every explanation is checked against the model itself.

| Data | Source | What the model sees |
|---|---|---|
| Clinical records | UCI Heart Disease (Cleveland, Hungary, Switzerland, VA) | 13 clinical variables per patient |
| 12-lead ECG | PTB-XL 1.0.3 (PhysioNet) | 10 s raw ECG signal, plus age and sex |

> **Research prototype.** Not a medical device. Do not use it for clinical decisions.

---

## Key results

**ECG study.** Held-out fold of 2,050 ECGs from 1,792 patients the model never saw.

| Model | AUC (95% CI) | Accuracy |
|---|---|---|
| Age and sex only | 0.758 (0.734–0.779) | 66.3% |
| ECG features + XGBoost | 0.877 (0.862–0.892) | 79.0% |
| **1-D ResNet on the raw ECG** | **0.951 (0.941–0.959)** | **87.8%** |
| ResNet + age and sex (fusion) | 0.952 (0.942–0.960) | 87.9% |

- At the 95% confidence setting the ECG model answers **79.5%** of ECGs with **94.2%** accuracy and refers the rest to a clinician.
- Adding age and sex to the ECG model gives no significant gain (AUC +0.001, p = 0.40).
- Flat-lining the 3 leads the explanation ranks highest removes about twice as much model evidence as flat-lining 3 random leads (1.09 vs 0.52).

**Clinical-data study.** Cleveland cohort, duplicates removed, repeated 10-fold cross-validation.

- The best model is logistic regression: 84.2% accuracy, AUC 0.911. The six models tested do not differ significantly.
- The widely used Kaggle `heart.csv` file has 1,025 rows but only 302 distinct patients. With the duplicates kept, random-forest accuracy is 99.6%; with them removed, it is 82.9%.
- The claim verifier raises faithful language-model explanations from 76.7% to 95.0%.

All result tables are in `results/tables` (clinical) and `results_ecg/tables` (ECG).

---

## Quick start

### 1. Requirements

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.10 – 3.12 | everything |
| NVIDIA GPU | optional (a 4 GB GTX 1650 is enough) | faster ECG training |
| Node.js | 18 or newer | rebuilding the web app (optional) |
| Ollama | latest | language-model explanations (optional) |

### 2. Install

Run these from the project folder (Windows Command Prompt shown; on macOS/Linux use `source venv/bin/activate`):

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -m leakfree_heart doctor
```

Without an NVIDIA GPU, install PyTorch from `https://download.pytorch.org/whl/cpu` instead. `doctor` lists anything missing and the command that fixes it.

### 3. Download the data

```
python -m leakfree_heart download-uci
python -m leakfree_heart download-ptbxl
python -m leakfree_heart download-guidelines
```

PTB-XL is about 0.7 GB. If a download stops, run the command again; it resumes.

### 4. Run the studies

```
python -m leakfree_heart run-ecg --quick           # 2-minute pipeline check
python -m leakfree_heart run-ecg                   # full ECG study (about 2 hours on a GTX 1650)
python -m leakfree_heart run-experiments           # full clinical study (1–2 hours)
python -m leakfree_heart train-final               # model used by the clinical tab of the app
```

Run a single stage with `--only`, for example `python -m leakfree_heart run-ecg --only explain`. ECG stages: `audit`, `train`, `evaluate`, `leakage_train`, `explain`.

### 5. Start the web app

```
python -m leakfree_heart build-ui                  # once (needs Node.js)
python -m leakfree_heart serve --open
```

Open http://127.0.0.1:8000. The API documentation is at `/docs`.

| Tab | What it does |
|---|---|
| **ECG** | Pick one of 20 held-out ECGs or upload your own (1000 rows × 12 leads, 100 Hz, mV). Shows the probability, the answer or "not sure" decision, and a per-lead importance strip. |
| **Clinical data** | Enter 13 clinical values. Shows the risk, the decision and a written explanation checked by the verifier (uses Ollama if running, otherwise a template). |
| **Study results** | All tables and figures from both studies. |

---

## All commands

| Command | Purpose |
|---|---|
| `doctor` | Check Python, packages, GPU, data and models |
| `download-uci`, `download-ptbxl`, `download-guidelines` | Download the datasets and guideline sources |
| `build-kb` | Rebuild the knowledge-base index |
| `run-experiments` | Clinical study |
| `run-ecg` | ECG study (`--quick`, `--demo`, `--only`, `--skip`, `--device`) |
| `train-final` | Train the clinical-tab model |
| `serve` | Start the API and web app (`--port`, `--host`, `--open`) |
| `build-ui`, `dev-ui` | Build the React app / run its dev server |
| `setup-llm` | Download the local language model (`llama3.2:3b`) with Ollama |
| `kb-sheet`, `audit-agreement` | Knowledge-base check sheet / human-audit agreement |
| `test` | Run the unit tests |

Use them as `python -m leakfree_heart <command>`.

---

## Project structure

```
src/leakfree_heart/   core library: data, models, calibration, conformal prediction,
                      explanations, ecg/ (PTB-XL study and inference), rag/ (knowledge base, verifier)
api/                  FastAPI backend (also serves the built web app)
frontend/             React (Vite) web app
configs/config.yaml   all settings
scripts/              scripts behind the commands
results/              clinical-study tables and figures
results_ecg/          ECG-study tables and figures
knowledge_base/       guideline passages used by the explanations
tests/                unit tests
```

---

## Limitations

- The ECG model classifies normal vs abnormal; it does not name the specific abnormality.
- PTB-XL comes from a single source; external validation on other ECG data is still needed.
- ECG explanations (integrated gradients) have not been reviewed by cardiologists.
- The knowledge-base passages are summaries of guidelines and still need checking against the originals (`python -m leakfree_heart kb-sheet`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named leakfree_heart` | Activate the venv, then run `pip install -e .` |
| ECG training shows `device=CPU` | Install the CUDA build of PyTorch |
| ECG tab says the model is not available | Run `download-ptbxl` and the full `run-ecg`, then restart `serve` |
| Browser shows only the API page | Run `build-ui`, then restart `serve` |
| Port 8000 is busy | `python -m leakfree_heart serve --port 8001` |
| Clinical tab says the model is missing | Run `train-final` |

---

## Data and licences

- **PTB-XL 1.0.3**: Wagner et al., *Scientific Data* 7:154 (2020). Available on PhysioNet under CC BY 4.0.
- **UCI Heart Disease**: Detrano et al., *American Journal of Cardiology* 64:304–310 (1989). UCI Machine Learning Repository.
