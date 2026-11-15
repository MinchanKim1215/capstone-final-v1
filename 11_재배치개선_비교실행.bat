@echo off
chcp 65001 >nul
pushd "%~dp0"
set PYTHONIOENCODING=utf-8
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m final_v1_relocation.campaign %*
popd
pause
