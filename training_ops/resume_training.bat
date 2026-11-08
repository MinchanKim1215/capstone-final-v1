@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%~dp0..\drone_forecast_joint_training\.venv\Scripts\python.exe" -u "%~dp0run_training.py"
echo Training supervisor exited with code %errorlevel%.
pause
