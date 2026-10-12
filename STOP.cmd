@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
python launch.py stop
if errorlevel 1 pause
