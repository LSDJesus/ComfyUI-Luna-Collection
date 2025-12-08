@echo off
REM Luna Daemon Launcher
REM Starts the Luna Daemon in a separate window

setlocal enabledelayedexpansion

REM Get the script directory
set SCRIPT_DIR=%~dp0

REM Navigate up to ComfyUI root and capture the path
pushd "%SCRIPT_DIR%..\..\.."
set COMFYUI_ROOT=%CD%
popd

REM Get Luna Collection root
set LUNA_ROOT=%COMFYUI_ROOT%\custom_nodes\ComfyUI-Luna-Collection

REM Get Python executable from venv (in ComfyUI root)
set PYTHON_EXE=%COMFYUI_ROOT%\venv\Scripts\python.exe

REM Check if Python exists
if not exist "%PYTHON_EXE%" (
    echo Error: Python executable not found at %PYTHON_EXE%
    echo Checked path: %PYTHON_EXE%
    echo ComfyUI Root: %COMFYUI_ROOT%
    echo.
    echo Please ensure ComfyUI venv is set up properly
    pause
    exit /b 1
)

echo ===============================================
echo Luna Daemon Launcher
echo ===============================================
echo Python: %PYTHON_EXE%
echo Luna Root: %LUNA_ROOT%
echo ComfyUI Root: %COMFYUI_ROOT%
echo.

REM Start daemon directly (this way we see any errors)
cd /d "%LUNA_ROOT%"
echo Starting daemon from: %CD%
echo Command: %PYTHON_EXE% luna_daemon\__main__.py
echo.

REM Run daemon - if it crashes, the window will show the error before closing
%PYTHON_EXE% luna_daemon\__main__.py

REM If we get here, the daemon exited
echo.
echo WARNING: Daemon process exited!
echo Check the output above for error messages
echo.
pause
