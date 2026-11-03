@echo off
cd /d "%~dp0"
set PYTHONPATH=src
".venv\Scripts\python.exe" -m neuralduck.closed_loop
pause
