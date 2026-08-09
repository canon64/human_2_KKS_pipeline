@echo off
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ==========================================================
echo  Grok history update
echo   1. put the exported ZIP into: grok_history\incoming\
echo   2. Ollama must be running (embedding step)
echo ==========================================================
echo.

if exist "python\python.exe" (
    "python\python.exe" "%~dp0run_grok_history_update.py"
) else (
    python "%~dp0run_grok_history_update.py"
)
set "_RC=%ERRORLEVEL%"

echo.
if not "%_RC%"=="0" (
    echo [ERROR] update failed.
) else (
    echo [OK] done.
)
echo Press any key to close...
pause >nul
exit /b %_RC%
