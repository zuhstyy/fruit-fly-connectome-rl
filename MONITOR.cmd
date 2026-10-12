@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
python launch.py monitor
if errorlevel 1 pause
