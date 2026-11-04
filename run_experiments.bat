@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
rem Examples:   run_experiments.bat            (everything)
rem             run_experiments.bat --only leakage benchmark
python scripts\run_experiments.py %*
