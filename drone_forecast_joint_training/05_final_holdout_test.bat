@echo off
setlocal
cd /d "%~dp0"
echo Finish full validation and model selection before opening final holdout.
".venv\Scripts\python.exe" transformer_run.py finaltest --workers 4
set "result_code=%errorlevel%"
pause
exit /b %result_code%
