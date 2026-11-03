@echo off
setlocal
cd /d "%~dp0"
if not defined NEURALDUCK_PYTHON set "NEURALDUCK_PYTHON=.venv\Scripts\python.exe"
if not exist "%NEURALDUCK_PYTHON%" (
  echo Missing local Python environment. See README.md.
  exit /b 1
)
set "PYTHONPATH=src"
"%NEURALDUCK_PYTHON%" -m neuralduck.completion_report
if errorlevel 1 exit /b 1
if /I "%~1"=="--no-open" exit /b 0
start "" "artifacts\consolidated-report\report.html"
