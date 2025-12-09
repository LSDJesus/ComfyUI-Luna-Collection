@echo off
REM Luna Daemon - System Tray Launcher (Batch version)

echo Luna Daemon - System Tray Launcher
echo ==================================================
echo.

REM Find Python
set PYTHON_EXE=D:\AI\ComfyUI\venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo Error: Python not found at %PYTHON_EXE%
    echo Please edit this script with the correct Python path
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo.

REM Install dependencies
echo Checking dependencies...
"%PYTHON_EXE%" -m pip install --quiet pystray pillow 2>nul

REM Launch tray app in background
echo.
echo Launching Luna Daemon tray app...
echo Check your system tray for the Luna icon!
echo (You can close this window - the daemon runs in the background)
echo.

cd /d "%~dp0"
REM Start in background using START command with /B flag (no new window)
start /B "" "%PYTHON_EXE%" tray_app.py

echo Daemon launched! You can close this window.
echo.

