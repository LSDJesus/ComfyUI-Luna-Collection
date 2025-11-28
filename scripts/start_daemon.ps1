<#
.SYNOPSIS
    Start only the Luna VAE/CLIP Daemon
.DESCRIPTION
    Starts the daemon that holds shared VAE and CLIP models.
    Run this once before starting any ComfyUI instances.
.PARAMETER Device
    GPU device to load models on (default: cuda:1)
#>

param(
    [string]$Device = "cuda:1"
)

$Host.UI.RawUI.WindowTitle = "Luna-CLIP-VAE-Daemon"

$ComfyUIPath = "D:\AI\ComfyUI"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Luna VAE/CLIP Daemon" -ForegroundColor Cyan
Write-Host "  Device: $Device" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ComfyUIPath
& .\venv\Scripts\Activate.ps1

# Set device via environment variable if needed
$env:LUNA_DAEMON_DEVICE = $Device

python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
