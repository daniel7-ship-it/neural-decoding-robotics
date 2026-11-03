@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -m neuralduck.dashboard --seconds 3600
if errorlevel 1 pause
