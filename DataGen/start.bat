@echo off
title mini-AGI DataGen Studio
cd /d "%~dp0"

echo ========================================================
echo          mini-AGI DataGen Studio Launcher
echo ========================================================
echo.

IF EXIST "..\.venv\Scripts\python.exe" (
    SET "PYTHON=..\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON=python"
)

echo Starting DataGen Studio at http://127.0.0.1:5050 ...
start "" "http://127.0.0.1:5050"
"%PYTHON%" server.py
pause
