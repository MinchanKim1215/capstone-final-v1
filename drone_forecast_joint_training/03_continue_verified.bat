@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup first. & pause & exit /b 1)
set "target_updates=16"
set "worker_count=4"
if not "%~1"=="" set "target_updates=%~1"
if not "%~2"=="" set "worker_count=%~2"
".venv\Scripts\python.exe" transformer_run.py train --run verified_joint --mode joint --updates "%target_updates%" --workers "%worker_count%" --device cuda
set "result_code=%errorlevel%"
pause
exit /b %result_code%
