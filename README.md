# LeakFree-Heart

Leakage-aware heart models that explain themselves and say "not sure" when they should.

The project has two studies and a web app. In both studies no patient appears in both training and test data.

1. **Clinical-data study (UCI Heart Disease).** Audit of the 1,025-row Kaggle file (it holds only 302 unique patients, so a random split leaks), a de-duplicated benchmark, guideline-grounded explanations checked claim by claim, and conformal deferral.
2. **ECG study (PTB-XL, 12-lead ECG plus age and sex).** Official patient-wise folds, a 1-D ResNet, fusion with age and sex, calibration, conformal deferral, subgroup results, and a per-lead explanation in the web app.
3. **Web app (React + FastAPI).** An ECG tab, a clinical-data tab and a results tab, served from one port.

> Research prototype. It is not a medical device and must not be used for clinical decisions.

## Main results (ECG study, held-out fold of 2,050 ECGs from unseen patients)

| Model | AUC (95% CI) | Accuracy |
|---|---|---|
| Age and sex only | 0.758 (0.734-0.779) | 66.3% |
| ECG features + XGBoost | 0.877 (0.862-0.892) | 79.0% |
| ECG ResNet (signal only) | 0.951 (0.941-0.959) | 87.8% |
| ECG ResNet + age/sex (end-to-end fusion) | 0.952 (0.942-0.960) | 87.9% |

- Adding age and sex to the ECG model gives no significant gain (paired AUC difference +0.001, p = 0.40).
- At the 95% confidence level the model answers 79.5% of ECGs at 94.2% accuracy and defers the rest to a clinician.
- Training with a random record-wise split puts 16.2% of test ECGs in patients also seen in training and gives a slightly higher AUC (0.957 against 0.947 with patient-wise folds, one seed).

All tables are in `results_ecg/tables`, and the clinical-study tables are in `results/tables` (summary in `results/RESULTS.md`).

## Requirements

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.10, 3.11 or 3.12 | everything |
| NVIDIA GPU + driver | any recent (a 4 GB GTX 1650 is enough) | fast ECG training (optional; CPU works but is slow) |
| Node.js | 18 or newer | rebuilding the web app (optional; only for development) |
| Ollama | latest | language-model explanations in the clinical tab (optional) |

## Setup

Run these from the project folder. On Windows use Command Prompt; on macOS or Linux replace `venv\Scripts\activate` with `source venv/bin/activate`.

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -m leakfree_heart doctor
```

- Without an NVIDIA GPU, use `https://download.pytorch.org/whl/cpu` instead.
- `doctor` lists what is installed and what is missing, with the command that fixes each item.
- If `pip install -e .` is not possible, set `PYTHONPATH=src` and use the same `python -m leakfree_heart ...` commands.

## Download the data

```
python -m leakfree_heart download-uci
python -m leakfree_heart download-ptbxl
python -m leakfree_heart download-guidelines
```

- PTB-XL (100 Hz, about 0.7 GB, 21,799 ECGs) comes from PhysioNet. If the download is interrupted, run it again; it resumes.
- Some guideline PDFs cannot be fetched automatically. The command lists them; save them into `knowledge_base/guidelines` and run `python -m leakfree_heart build-kb`.
- Do not use `data/raw/heart.csv` as a primary cohort: it contains 723 duplicate rows.

## Run the studies

```
python -m leakfree_heart run-ecg --quick           # about 2 minutes, checks the pipeline (numbers are not for the paper)
python -m leakfree_heart run-ecg                   # full ECG study, about 2 hours on a GTX 1650
python -m leakfree_heart run-experiments --quick   # about 5 minutes
python -m leakfree_heart run-experiments           # full clinical study, 1-2 hours
python -m leakfree_heart train-final               # model behind the clinical-data tab
```

A stage can be repeated on its own, for example `python -m leakfree_heart run-ecg --only evaluate`. Results are written to `results_ecg/` and `results/`.

## Run the web app

```
python -m leakfree_heart build-ui                  # once, needs Node.js
python -m leakfree_heart serve --open
```

The app and the API run on one port: http://127.0.0.1:8000 (API documentation at `/docs`). Stop with Ctrl+C.

- **ECG tab:** needs the full ECG study (`run-ecg`). Choose one of 20 held-out example ECGs or upload your own file (12 leads in the order I, II, III, aVR, aVL, aVF, V1-V6, 100 Hz, 10 s, millivolts, as 1000 rows by 12 columns). It shows the probability, an answer or "not sure" decision at the confidence level you pick, and a lead-by-lead explanation.
- **Clinical data tab:** needs `train-final`. It uses Ollama if it is running (`python -m leakfree_heart setup-llm` downloads `llama3.2:3b`); otherwise a fixed template that passes through the same checker.
- **Study results tab:** shows the tables and figures of both studies.
- For interface work, run `serve` in one window and `python -m leakfree_heart dev-ui` in another, then open http://localhost:5173.

## Tests

```
python -m leakfree_heart test
```

The ECG tests use a tiny synthetic dataset created by `python -m leakfree_heart run-ecg --demo`. Tests that need files you have not created yet are skipped.

## All commands

| Command | What it does |
|---|---|
| `doctor` | check Python, packages, GPU, data and model files |
| `download-uci`, `download-guidelines`, `download-ptbxl` | download the data |
| `build-kb` | rebuild the knowledge-base index |
| `run-experiments` | clinical study |
| `run-ecg` | ECG study (`--quick`, `--demo`, `--only`, `--skip`, `--device`) |
| `train-final` | train the clinical-tab model |
| `serve` | start the API and web app (`--port`, `--host`, `--open`) |
| `build-ui`, `dev-ui` | build the React app / run its development server |
| `setup-llm` | pull the local language model with Ollama |
| `kb-sheet`, `audit-agreement` | knowledge-base verification sheet / human-audit agreement |
| `test` | run the unit tests |

Every command also exists as a plain script in `scripts/`, for example `python scripts/run_ecg.py`.

## Project layout

| Folder | Contents |
|---|---|
| `src/leakfree_heart/` | library: data, models, calibration, uncertainty, explanations, `ecg/` (PTB-XL study and inference service), `rag/` (knowledge base and verifier), `cli.py` |
| `scripts/` | the scripts behind the commands |
| `api/` | FastAPI backend; also serves the built web app |
| `frontend/` | React app (Vite) |
| `configs/config.yaml` | every setting |
| `results/`, `results_ecg/` | tables and figures written by the studies |
| `knowledge_base/`, `models/`, `paper/`, `tests/` | guideline passages, trained clinical model, paper material, unit tests |

## Limitations

- The ECG explanation is a model attribution (integrated gradients). It is not checked claim by claim like the clinical explanations and has not been reviewed by cardiologists.
- The ECG label is binary (normal against abnormal, from PTB-XL diagnostic superclasses), so the model does not say which abnormality is present.
- The knowledge-base passages for the clinical explanations still need checking against their sources (`python -m leakfree_heart kb-sheet`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named leakfree_heart` | Activate the venv, then run `pip install -e .` in the project folder |
| ECG training prints `device=CPU` | Install the CUDA build of PyTorch, then check `python -c "import torch; print(torch.cuda.is_available())"` |
| ECG tab says the model is not available | Run `download-ptbxl` and the full `run-ecg` (not `--quick`), then restart `serve` |
| Browser shows only the API page | Run `build-ui`, then restart `serve` |
| Port 8000 is busy | `python -m leakfree_heart serve --port 8001` |
| Clinical tab says the model artifact is missing | Run `train-final` |

## Data and licences

PTB-XL 1.0.3 (PhysioNet, CC BY 4.0): Wagner et al., Scientific Data 7:154 (2020). UCI Heart Disease: Detrano et al. (1989). This work responds to Alshayeji and Abed (2025), "Heart disease prediction by tabular modeling with deep learning network and interpretability".
