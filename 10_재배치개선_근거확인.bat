@echo off
chcp 65001 >nul
pushd "%~dp0"
set PYTHONIOENCODING=utf-8
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m unittest final_v1_relocation.tests.test_relocation -v
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m final_v1_relocation.replay_audit
popd
pause
