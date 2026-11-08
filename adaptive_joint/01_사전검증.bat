@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m adaptive_joint.run preflight
pause
