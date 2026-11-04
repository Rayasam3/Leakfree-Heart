@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo API at http://127.0.0.1:8000  (interactive docs at /docs). Press Ctrl+C to stop.
uvicorn api.main:app --host 127.0.0.1 --port 8000
