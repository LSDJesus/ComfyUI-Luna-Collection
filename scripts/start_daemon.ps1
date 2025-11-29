<#
.SYNOPSIS
    Start the Luna VAE/CLIP Daemon
.DESCRIPTION
    Starts the daemon that holds shared VAE and CLIP models.
    Run this once before starting any ComfyUI instances.
.PARAMETER Device
    GPU device to load models on (default: cuda:1)
.PARAMETER Dynamic
    Use dynamic scaling version (v2) with auto-scaling workers
.EXAMPLE
    .\start_daemon.ps1
    .\start_daemon.ps1 -Dynamic
    .\start_daemon.ps1 -Device "cuda:0"
#>

param(
    [string]$Device = "cuda:1",
    [switch]$Dynamic
)

$Host.UI.RawUI.WindowTitle = "Luna-CLIP-VAE-Daemon"

$ComfyUIPath = "D:\AI\ComfyUI"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Luna VAE/CLIP Daemon" -ForegroundColor Cyan
Write-Host "  Device: $Device" -ForegroundColor Cyan
if ($Dynamic) {
    Write-Host "  Mode: Dynamic Scaling (v2)" -ForegroundColor Yellow
} else {
    Write-Host "  Mode: Static (v1)" -ForegroundColor Gray
}
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ComfyUIPath
& .\venv\Scripts\Activate.ps1

# Set device via environment variable if needed
$env:LUNA_DAEMON_DEVICE = $Device

$DaemonPath = ".\custom_nodes\ComfyUI-Luna-Collection\luna_daemon"

if ($Dynamic) {
    python "$DaemonPath\server_v2.py"
} else {
    python "$DaemonPath\server.py"
}
