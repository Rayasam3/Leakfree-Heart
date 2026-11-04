@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Quick smoke test with reduced settings (a few minutes). Numbers are NOT for the paper.
python scripts\run_experiments.py --quick %*
