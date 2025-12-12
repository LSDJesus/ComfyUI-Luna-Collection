<#
.SYNOPSIS
    Start the Luna Daemon System Tray App
.DESCRIPTION
    Starts the Luna Daemon tray app which manages shared VAE and CLIP models.
    The tray app enforces single-instance and runs in your system tray.
    Run this once before starting any ComfyUI instances.
.PARAMETER Device
    GPU device to load models on (default: cuda:1)
.PARAMETER NewWindow
    Launch tray app in a separate terminal window (default: true)
.PARAMETER NoWindow
    Run in current terminal instead of spawning a new window
#>

param(
    [string]$Device = "cuda:1",
    [switch]$NewWindow,
    [switch]$NoWindow
)

$ComfyUIPath = "D:\AI\ComfyUI"
$ScriptPath = $MyInvocation.MyCommand.Path

# If NewWindow is set (or neither flag specified and not already spawned), spawn in new terminal
if ($NewWindow -or (-not $NoWindow -and -not $env:LUNA_DAEMON_SPAWNED)) {
    $env:LUNA_DAEMON_SPAWNED = "1"
    
    # Build arguments for the new window
    $args = "-NoExit -File `"$ScriptPath`" -Device `"$Device`" -NoWindow"
    
    Start-Process pwsh -ArgumentList $args -WindowStyle Normal
    Write-Host "Luna Daemon launched in new window" -ForegroundColor Green
    exit 0
}

# Running in target terminal (either -NoWindow or spawned)
$Host.UI.RawUI.WindowTitle = "Luna Daemon Tray [$Device]"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Luna Daemon System Tray" -ForegroundColor Cyan
Write-Host "  Device: $Device" -ForegroundColor Cyan
Write-Host "  Single-instance enforced" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Look for the Luna icon in your system tray!" -ForegroundColor Green
Write-Host ""

Set-Location $ComfyUIPath
& .\venv\Scripts\Activate.ps1

# Set environment variables
$env:LUNA_DAEMON_DEVICE = $Device
$env:LUNA_DAEMON_MODE = "1"  # Suppress irrelevant warnings from other modules

# Start the tray app (enforces single instance automatically)
python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.tray_app
