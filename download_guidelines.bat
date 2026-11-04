@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Fetching public guideline and health-agency pages for the RAG knowledge base...
python scripts\download_guidelines.py
echo.
echo Building the knowledge-base index...
python scripts\build_kb.py
