@echo off
cd /d "%~dp0frontend"
echo React app at http://localhost:5173  (the API must be running: start_api.bat)
call npm run dev
