@echo off
chcp 65001 >nul
pushd "%~dp0"
"drone_forecast_joint_training\.venv\Scripts\python.exe" -B -m forecast_graph.status
popd
pause
