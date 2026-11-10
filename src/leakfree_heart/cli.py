"""One command-line entry point for the whole project:   python -m leakfree_heart <command>   (or:  leakfree <command>)

    doctor               check Python, packages, GPU, data and model files
    download-uci         download the 4 UCI heart-disease sites
    download-guidelines  fetch guideline pages and build the knowledge-base index
    build-kb             rebuild the knowledge-base index
    download-ptbxl       download the PTB-XL ECG data (resumable)
    run-experiments      heart-disease study (leakage audit, benchmark, trust, explanations ...)
    run-ecg              ECG (multimodal) study on PTB-XL
    train-final          train the model behind the clinical-data tab
    serve                start the API and the web app on http://127.0.0.1:8000
    build-ui / dev-ui    build the React app once / run its development server
    setup-llm            pull the local language model with Ollama
    test                 run the unit tests

Extra arguments are passed to the underlying script, e.g.  python -m leakfree_heart run-ecg --only audit
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import webbrowser

from .config import ROOT

SCRIPTS = {
    "doctor": "check_env.py",
    "download-uci": "download_data.py",
    "download-guidelines": "download_guidelines.py",
    "build-kb": "build_kb.py",
    "download-ptbxl": "download_ptbxl.py",
    "run-experiments": "run_experiments.py",
    "run-ecg": "run_ecg.py",
    "train-final": "train_final.py",
    "kb-sheet": "kb_verification_sheet.py",
    "audit-agreement": "audit_agreement.py",
}
OTHER = ["serve", "build-ui", "dev-ui", "setup-llm", "test"]


def _script(name: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / SCRIPTS[name]), *args], cwd=ROOT)


def _npm(*args: str) -> int:
    npm = shutil.which("npm")
    if not npm:
        print("Node.js (npm) was not found. Install the LTS version from https://nodejs.org and try again.")
        return 1
    return subprocess.call([npm, *args], cwd=ROOT / "frontend")


def _serve(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="leakfree serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args(args)
    import uvicorn
    sys.path.insert(0, str(ROOT))
    url = f"http://{a.host}:{a.port}"
    if (ROOT / "frontend" / "dist" / "index.html").exists():
        print(f"Web app: {url}     API docs: {url}/docs     Stop with Ctrl+C")
        if a.open:
            webbrowser.open(url)
    else:
        print(f"API: {url}/docs   (the web app is not built yet: run  python -m leakfree_heart build-ui  then restart)")
    uvicorn.run("api.main:app", host=a.host, port=a.port)
    return 0


def _setup_llm() -> int:
    if not shutil.which("ollama"):
        print("Ollama is not installed. Download it from https://ollama.com/download , install it, then run this again.")
        return 1
    print("Pulling a 3-billion-parameter model that fits in 4 GB of GPU memory ...")
    return subprocess.call(["ollama", "pull", "llama3.2:3b"])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = [*SCRIPTS, *OTHER]
    if not argv or argv[0] in ("-h", "--help", "help") or argv[0] not in cmds:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help", "help"):
            print(f"Unknown command: {argv[0]}")
            return 2
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd in SCRIPTS:
            return _script(cmd, rest)
        if cmd == "serve":
            return _serve(rest)
        if cmd == "build-ui":
            return _npm("install", "--no-audit", "--no-fund") or _npm("run", "build")
        if cmd == "dev-ui":
            return _npm("run", "dev")
        if cmd == "setup-llm":
            return _setup_llm()
        if cmd == "test":
            return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests", *rest], cwd=ROOT)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
