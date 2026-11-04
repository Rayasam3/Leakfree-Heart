@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  LeakFree-Heart setup (Windows, cmd)
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python 3.10 or 3.11 from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
  exit /b 1
)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10, 3.11 or 3.12 is required. Your version is:
  python --version
  exit /b 1
)

if not exist venv (
  echo [1/5] Creating virtual environment in .\venv
  python -m venv venv
  if errorlevel 1 exit /b 1
) else (
  echo [1/5] Virtual environment already exists
)
call venv\Scripts\activate.bat

echo [2/5] Upgrading pip
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/5] Installing project requirements
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] requirements installation failed. See the messages above.
  exit /b 1
)

echo [4/5] Installing PyTorch
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo        No NVIDIA GPU detected - installing the CPU build of PyTorch
  pip install torch --index-url https://download.pytorch.org/whl/cpu
) else (
  echo        NVIDIA GPU detected - installing the CUDA 12.1 build of PyTorch
  pip install torch --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 (
  echo [ERROR] PyTorch installation failed.
  exit /b 1
)

echo [5/5] Installing the React app (needs Node.js 18 or newer)
where npm >nul 2>nul
if errorlevel 1 (
  echo [WARN] Node.js was not found. Install it from https://nodejs.org LTS version and run:  cd frontend ^& npm install
) else (
  pushd frontend
  call npm install --no-audit --no-fund
  popd
)

echo.
echo Setup finished. Next step:  check_env.bat
endlocal
