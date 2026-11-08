@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" transformer_run.py select --runs transformer_main --workers 4
set "result_code=%errorlevel%"
pause
exit /b %result_code%
