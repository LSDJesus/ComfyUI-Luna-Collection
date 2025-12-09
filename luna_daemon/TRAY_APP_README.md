# Luna Daemon - System Tray Application

Run the Luna Daemon as a persistent background service with system tray control.

## Features

✅ **Single Instance** - Only one daemon can run on your PC (prevents port conflicts)  
✅ **System Tray Icon** - Easy control from Windows taskbar  
✅ **Visual Status** - Icon changes color based on daemon state:
   - 🟢 Green: Running and responsive
   - 🟡 Yellow: Starting up
   - 🔴 Red: Error or stopped
   - ⚫ Gray: Not started

✅ **Right-Click Menu**:
   - Start Daemon
   - Stop Daemon
   - Check Status
   - View Logs
   - Exit

✅ **Auto-Start** - Daemon starts automatically when tray app launches  
✅ **Persistent** - Stays running even when ComfyUI closes

## Installation

1. **Install dependencies:**
   ```bash
   pip install pystray pillow
   ```

2. **Launch the tray app:**
   - **Windows:** Double-click `start_tray_daemon.bat`
   - **PowerShell:** `.\start_tray_daemon.ps1`
   - **Manual:** `python tray_app.py`

## Usage

### First Time Setup

1. Double-click `start_tray_daemon.bat`
2. Look for the Luna icon in your system tray (near the clock)
3. The daemon will start automatically after 1 second
4. Right-click the icon to control the daemon

### Daily Use

Just keep the tray app running in the background! It will:
- Start the daemon on launch
- Keep it running persistently
- Let you check status anytime
- Prevent duplicate instances

### Stopping the Daemon

To stop the daemon cleanly:
1. Right-click the tray icon
2. Click "Stop Daemon"

This will:
- Complete any in-flight requests
- Unload all models  
- Close all connections
- Exit cleanly

The tray icon will turn **gray** when stopped. You can restart it by clicking "Start Daemon" from the menu.

## Auto-Start with Windows (Optional)

To have the daemon start automatically when Windows boots:

1. Press `Win + R`
2. Type `shell:startup` and press Enter
3. Create a shortcut to `start_tray_daemon.bat` in that folder

## Troubleshooting

### "Luna Daemon is already running!"

Another instance is already running. Check your system tray for the Luna icon.

If you can't find it but still get this message:
1. Open Task Manager
2. Look for `python.exe` process on port 19299
3. End that process
4. Try launching again

### Dependencies not installed

Run manually:
```bash
cd luna_daemon
pip install pystray pillow
python tray_app.py
```

### Port conflicts (19283, 19285)

Make sure no other Luna Daemon instances are running:
```powershell
Get-NetTCPConnection -LocalPort 19283 -ErrorAction SilentlyContinue | 
  Where-Object { $_.OwningProcess -ne 0 } | 
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## Technical Details

- **Lock Port:** 19299 (for single-instance enforcement)
- **Daemon Ports:** 19283 (socket), 19285 (WebSocket)
- **Single Instance:** Uses socket binding to prevent duplicates
- **Thread-based:** Daemon runs in background thread of tray app

## Integration with ComfyUI

The tray daemon can run **independently** of ComfyUI:

1. Start the tray app once (manually or at boot)
2. Launch ComfyUI normally
3. ComfyUI nodes will connect to the running daemon
4. Close ComfyUI - daemon stays running
5. Next ComfyUI session connects to existing daemon instantly

This gives you:
- ✅ Faster ComfyUI startup (no daemon loading time)
- ✅ Persistent LoRA cache across sessions
- ✅ No port conflicts from multiple launches
- ✅ Better resource management

## Future Enhancements

- [ ] Clean shutdown command from tray menu
- [ ] Log file viewer window
- [ ] Auto-update status polling
- [ ] Configuration dialog (device, precision, etc.)
- [ ] Multi-daemon support (CLIP-only, VAE-only instances)
- [ ] System notifications for errors/warnings
