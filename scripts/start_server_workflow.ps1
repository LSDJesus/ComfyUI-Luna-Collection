<#
.SYNOPSIS
    Start Luna VAE/CLIP Daemon and ComfyUI workflow
.DESCRIPTION
    Checks if the Luna daemon is running, starts it if not, then launches ComfyUI.
    The daemon holds shared VAE and CLIP models in VRAM, allowing multiple ComfyUI
    instances to share them instead of each loading their own copy.
.PARAMETER Port
    ComfyUI listen port (default: 8188)
.PARAMETER DaemonDevice
    GPU device for daemon (default: cuda:1)
.EXAMPLE
    .\start_server_workflow.ps1 -Port 8188
    .\start_server_workflow.ps1 -Port 8189
#>

param(
    [int]$Port = 8188,
    [string]$DaemonDevice = "cuda:1"
)

$DaemonTitle = "Luna-CLIP-VAE-Daemon"
$ComfyUIPath = "D:\AI\ComfyUI"
$DaemonPort = 19283

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Luna ComfyUI Launcher" -ForegroundColor Cyan
Write-Host "  Port: $Port | Daemon Device: $DaemonDevice" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if daemon is responding
function Test-DaemonHealth {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $DaemonPort)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

# Check if daemon terminal is already open by window title
$daemonProc = Get-Process pwsh -ErrorAction SilentlyContinue | 
    Where-Object { $_.MainWindowTitle -eq $DaemonTitle }

if (-not $daemonProc) {
    # Also check if daemon port is in use (daemon might be running without window title)
    if (Test-DaemonHealth) {
        Write-Host "[OK] Daemon already running on port $DaemonPort" -ForegroundColor Green
    } else {
        Write-Host "[STARTING] Luna VAE/CLIP Daemon..." -ForegroundColor Yellow
        
        # Start daemon in new terminal with explicit title
        $daemonScript = @"
`$Host.UI.RawUI.WindowTitle = '$DaemonTitle'
cd '$ComfyUIPath'
.\venv\Scripts\Activate.ps1
Write-Host 'Starting Luna VAE/CLIP Daemon...' -ForegroundColor Cyan
python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
"@
        
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", $daemonScript
        
        # Wait for daemon to be ready
        Write-Host "[WAITING] Daemon loading models..." -ForegroundColor Yellow
        $ready = $false
        $maxWait = 60  # seconds
        $waited = 0
        
        while (-not $ready -and $waited -lt $maxWait) {
            Start-Sleep -Seconds 2
            $waited += 2
            Write-Host "." -NoNewline
            
            if (Test-DaemonHealth) {
                $ready = $true
            }
        }
        
        Write-Host ""
        
        if ($ready) {
            Write-Host "[OK] Daemon ready!" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Daemon may still be loading, continuing..." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[OK] Daemon terminal already open" -ForegroundColor Green
}

Write-Host ""
Write-Host "[STARTING] ComfyUI on port $Port..." -ForegroundColor Cyan

# Start ComfyUI
Set-Location $ComfyUIPath
& .\venv\Scripts\Activate.ps1
python -s main.py --listen 0.0.0.0 --port $Port
