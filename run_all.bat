@echo off
setlocal
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo === 1/4 Download data ===
python scripts\download_data.py
if errorlevel 1 echo [WARN] UCI download failed - continuing with the Cleveland data rebuilt from the Kaggle copy
echo === 2/4 Build knowledge base ===
python scripts\download_guidelines.py
python scripts\build_kb.py
echo === 3/4 Run all experiments (this takes a while) ===
python scripts\run_experiments.py
if errorlevel 1 echo [WARN] some stages failed - see the messages above
echo === 4/4 Train the deployable model ===
python scripts\train_final.py
echo.
echo Finished. Results: results\RESULTS.md   Start the app with: start_app.bat
endlocal
