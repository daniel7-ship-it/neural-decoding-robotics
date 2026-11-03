@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv || exit /b 1
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
)
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" -m neuralduck.demo --gui %*
if errorlevel 1 pause
