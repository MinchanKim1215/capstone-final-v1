@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run 01_setup_nvidia.bat first. & pause & exit /b 1)
set "target_updates=256"
set "worker_count=4"
if not "%~1"=="" set "target_updates=%~1"
if not "%~2"=="" set "worker_count=%~2"
".venv\Scripts\python.exe" transformer_run.py pipeline --run transformer_main --episodes 192 --pretrain-steps 3000 --updates "%target_updates%" --workers "%worker_count%" --device cuda
set "result_code=%errorlevel%"
if not "%result_code%"=="0" echo Completed checkpoints preserved. Read the error and repeat the same command after resolution.
pause
exit /b %result_code%
