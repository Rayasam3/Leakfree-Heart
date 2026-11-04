@echo off
cd /d "%~dp0"
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed. Download it from https://ollama.com/download/windows , install it,
  echo then run this file again.
  exit /b 1
)
echo Pulling a 3-billion-parameter model that fits in 4 GB of GPU memory...
ollama pull llama3.2:3b
echo.
echo Done. Ollama runs in the background; the project detects it automatically.
