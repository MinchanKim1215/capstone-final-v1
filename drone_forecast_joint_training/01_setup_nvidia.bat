@echo off
setlocal
cd /d "%~dp0"
py -3.12 setup_environment.py --torch cu128
set "result_code=%errorlevel%"
if not "%result_code%"=="0" echo Installation or CUDA verification failed. Read the first error above.
pause
exit /b %result_code%
