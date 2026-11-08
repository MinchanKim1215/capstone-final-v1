@echo off
chcp 65001 >nul
cd /d "%~dp0"
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m forecast_causal.status
pause
