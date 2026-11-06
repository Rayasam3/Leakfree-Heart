# LeakFree-Heart: setup and run guide

Every step is one command, run from the project folder. On Windows use Command Prompt; on macOS/Linux replace `venv\Scripts\activate` with `source venv/bin/activate`.

## 0. Install once (outside the project)

| Tool | Version | Check |
|---|---|---|
| Python | 3.10, 3.11 or 3.12 | `python --version` |
| Node.js (only to build the web app) | 18 or newer | `node --version` |
| NVIDIA driver (optional, speeds up ECG training) | any recent | `nvidia-smi` |
| Ollama (optional, for the language-model explanations) | https://ollama.com/download | `ollama --version` |

## 1. Set up the project (about 10-20 minutes, once)

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -m leakfree_heart doctor
```

- No NVIDIA GPU? Use `--index-url https://download.pytorch.org/whl/cpu` instead. Training on CPU works but is much slower.
- `doctor` prints what is installed and what is still missing, with the command that fixes each item.

## 2. Download the data

```
python -m leakfree_heart download-uci
python -m leakfree_heart download-ptbxl
python -m leakfree_heart download-guidelines
```

- `download-ptbxl` fetches the 100 Hz PTB-XL data (about 0.7 GB, 21,799 ECGs). If the connection drops, run it again: it resumes.
- Some guideline PDFs cannot be fetched automatically. The command lists them; download them by hand into `knowledge_base\guidelines` and run `python -m leakfree_heart build-kb`.
- Do not use `data\raw\heart.csv` as a primary cohort: it has 723 duplicate rows.

## 3. Run the studies (these produce every number in the paper)

```
python -m leakfree_heart run-ecg --quick           # about 2 minutes, checks the pipeline (numbers are NOT for the paper)
python -m leakfree_heart run-ecg                   # full ECG study: about 2 hours on a GTX 1650
python -m leakfree_heart run-experiments --quick   # about 5 minutes
python -m leakfree_heart run-experiments           # full clinical study: 1-2 hours
```

Results go to `results_ecg\tables`, `results_ecg\figures`, `results\tables`, `results\figures` and `results\RESULTS.md`. A run that stops can be started again; add `--only stage_name` to repeat one stage.

## 4. Train the model behind the clinical-data tab

```
python -m leakfree_heart train-final
```

## 5. Start the app

```
python -m leakfree_heart build-ui                  # once, needs Node.js
python -m leakfree_heart serve --open
```

The web app and the API run on one port: http://127.0.0.1:8000 (API documentation at `/docs`). Stop with Ctrl+C.

- The **ECG** tab needs the full ECG study to have been run (step 3). It loads the three trained ECG networks and the same calibration the study used.
- The **Clinical data** tab needs step 4. The explanation writer uses Ollama if it is running (`python -m leakfree_heart setup-llm` pulls `llama3.2:3b`); otherwise it uses a fixed template that goes through the same verifier.
- To work on the interface: run `serve` in one window and `python -m leakfree_heart dev-ui` in another, then open http://localhost:5173.

## 6. Tests

```
python -m leakfree_heart test
```

The ECG tests use a tiny synthetic dataset: `python -m leakfree_heart run-ecg --demo` creates it (it takes a minute). Tests that need files you have not created yet are skipped, not failed.

## 7. Before you submit the paper

1. **Verify the knowledge base.** `python -m leakfree_heart kb-sheet` writes `knowledge_base\kb_verification.csv`. Check each row against its source and set `verified` to 1 or 0. Then run `python -m leakfree_heart kb-sheet --apply` and `python -m leakfree_heart build-kb`.
2. **Re-run the explanation-faithfulness stage** if you changed anything: `python -m leakfree_heart run-experiments --only faith report`.
3. **Human audit.** Fill `rater_A` (and `rater_B`, by a second person) in `results\tables\audit_blinded.csv`, then run `python -m leakfree_heart audit-agreement`.
4. Check every number in the paper against the files in `results\tables` and `results_ecg\tables`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python and tick "Add python.exe to PATH" |
| `No module named leakfree_heart` | Activate the venv, then run `pip install -e .` in the project folder |
| ECG training says `device=CPU` | Install the CUDA build of PyTorch (step 1); check with `python -c "import torch; print(torch.cuda.is_available())"` |
| ECG tab says the model is not available | Run `download-ptbxl` and the full `run-ecg` (not `--quick`), then restart `serve` |
| Web page shows a plain API page, not the app | Run `build-ui`, then restart `serve` |
| Port 8000 busy | `python -m leakfree_heart serve --port 8001` |
| Clinical tab says "model artifact missing" | Run `train-final` |
| Out of GPU memory | Close other GPU programs; the app itself runs the ECG model on the CPU |
