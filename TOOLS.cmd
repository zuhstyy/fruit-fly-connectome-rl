@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "%~dp0launch.py" (
  echo Extract the whole ZIP first. launch.py must be beside this file.
  pause
  exit /b 1
)
python launch.py tools
pause
