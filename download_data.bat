@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Downloading the UCI Heart Disease data (4 hospitals)...
python scripts\download_data.py
if errorlevel 1 (
  echo.
  echo Download failed. Download heart+disease.zip manually from
  echo   https://archive.ics.uci.edu/dataset/45/heart+disease
  echo and extract the four processed.*.data files into data\raw\uci\
  exit /b 1
)
