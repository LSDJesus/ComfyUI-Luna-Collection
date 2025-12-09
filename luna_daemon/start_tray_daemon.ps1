# Luna Daemon - System Tray Launcher
# Installs dependencies and launches the tray application

Write-Host "Luna Daemon - System Tray Launcher" -ForegroundColor Cyan
Write-Host "=" * 50

# Find Python
$pythonExe = "D:\AI\ComfyUI\venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: Python not found at $pythonExe" -ForegroundColor Red
    Write-Host "Please edit this script with the correct Python path"
    pause
    exit 1
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Green

# Install dependencies if needed
Write-Host "`nChecking dependencies..." -ForegroundColor Yellow
& $pythonExe -m pip install --quiet pystray pillow 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Could not install dependencies" -ForegroundColor Yellow
}

# Launch tray app in background
Write-Host "`nLaunching Luna Daemon tray app..." -ForegroundColor Green
Write-Host "Check your system tray for the Luna icon!" -ForegroundColor Cyan
Write-Host "(You can close this window - the daemon runs in the background)" -ForegroundColor Gray
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trayApp = Join-Path $scriptDir "tray_app.py"

# Launch in background so terminal doesn't stay active
Start-Process -FilePath $pythonExe -ArgumentList $trayApp -WindowStyle Hidden

Write-Host "Daemon launched! You can close this window." -ForegroundColor Green
Write-Host ""
