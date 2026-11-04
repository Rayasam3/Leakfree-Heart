# LeakFree-Heart: Setup and Run Guide (Windows, CMD only)

Open **Command Prompt** (not PowerShell). Go to the project folder first:

```
cd C:\path\to\leakfree-heart
```

## 0. Install once (outside the project)
| Tool | Version | Check |
|---|---|---|
| Python | 3.10, 3.11 or 3.12 (tick "Add to PATH") | `python --version` |
| Node.js | 18 or newer | `node --version` |
| NVIDIA driver | any recent (for GTX 1650) | `nvidia-smi` |
| Ollama (for the LLM) | https://ollama.com/download | `ollama --version` |

## 1. Setup (about 10-20 min, one time)
```
setup.bat
check_env.bat
```
`setup.bat` makes the `.venv`, installs packages, installs PyTorch (CUDA build if an NVIDIA GPU is found, otherwise CPU) and runs `npm install`.

## 2. Download data
```
download_data.bat
```
Downloads the 4 UCI heart-disease sites (Cleveland, Hungary, Switzerland, VA) into `data\raw`. They are the basis for external validation. If UCI is unreachable, the project falls back to the Kaggle-coded Cleveland data. Do not use `data\raw\heart.csv` as a primary cohort: it has 723 duplicate rows.

## 3. Knowledge base for the RAG explanations
```
download_guidelines.bat
build_kb.bat
```
- Some guideline PDFs cannot be fetched automatically. The script prints the list. Download them by hand into `knowledge_base\guidelines` and run `build_kb.bat` again.
- **Before you publish:** open `knowledge_base\seed\seed_kb.json`. Every entry is `"verified": false`. Check each against its source and set it to `true`. Also verify the citations in `knowledge_base\sources.yaml`.

## 4. Local LLM (GPU, 4 GB is enough)
```
setup_llm.bat
```
Pulls `llama3.2:3b`. If Ollama is not running, the app uses a deterministic template fallback, which still passes through the verifier.

## 5. Run the experiments (for the paper)
```
run_quick.bat          (about 5 min, sanity check)
run_experiments.bat    (full, 1-2 hours)
```
Outputs: `results\tables`, `results\figures`, `results\RESULTS.md`. Every number in the paper must come from these files.

## 6. Train the app model and start the app
```
train_final.bat
start_app.bat
```
Open http://localhost:5173 . The API runs at http://localhost:8000/docs .
Separate windows: `start_api.bat` and `start_frontend.bat`.

## 7. Tests
```
run_tests.bat
```

## One command for everything
```
run_all.bat
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python with "Add to PATH" |
| PyTorch install is slow | Normal (about 2.5 GB for CUDA). Let it finish |
| Out of GPU memory | Use `llama3.2:3b` only; close other GPU apps |
| Port 8000 or 5173 busy | Close the old window, or `taskkill /F /IM python.exe` |
| API says "no artifact" | Run `train_final.bat` |
| Model results differ slightly | Seeds are fixed; different library versions can shift decimals |
