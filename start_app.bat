@echo off
cd /d "%~dp0"
start "LeakFree-Heart API" cmd /k start_api.bat
timeout /t 6 /nobreak >nul
start "LeakFree-Heart UI" cmd /k start_frontend.bat
timeout /t 6 /nobreak >nul
start http://localhost:5173
