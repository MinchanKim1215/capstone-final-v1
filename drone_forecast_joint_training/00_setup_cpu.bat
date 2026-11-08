@echo off
setlocal
cd /d "%~dp0"
py -3.12 setup_environment.py --torch cpu
set "result_code=%errorlevel%"
pause
exit /b %result_code%
