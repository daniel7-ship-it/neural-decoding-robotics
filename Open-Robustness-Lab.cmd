@echo off
cd /d "%~dp0"
set PYTHONPATH=src
".venv\Scripts\python.exe" -m neuralduck.robustness_dashboard
if errorlevel 1 pause
