# Luna Daemon Auto-Start Implementation

## Problem Solved
Previously, daemon loader nodes (`LunaDaemonVAELoader`, `LunaDaemonCLIPLoader`) would raise an error if the daemon wasn't running, forcing users to manually start it with a script. This created friction in the workflow.

## Solution Implemented

### 1. New Function: `start_daemon()` in `luna_daemon/client.py`
Added a new convenience function that spawns the daemon as a subprocess:

```python
def start_daemon() -> bool:
    """Start the Luna Daemon subprocess if not running"""
    if is_daemon_running():
        return True
    
    # Spawns: python -m luna_daemon.server
    # Windows: Uses CREATE_NEW_CONSOLE for separate window
    # Unix: Uses start_new_session for proper detachment
```

**Features:**
- Returns immediately (doesn't block)
- Cross-platform (Windows/Unix)
- Graceful handling if already running
- Proper subprocess isolation

### 2. Helper Function: `ensure_daemon_running()` in `nodes/luna_daemon_loader.py`
Consolidates the auto-start logic for all loader nodes:

```python
def ensure_daemon_running(node_name: str) -> None:
    """
    Auto-start daemon if not running.
    Called by loader nodes before use.
    """
    ensure_daemon_available()
    if daemon_client.is_daemon_running():
        return
    
    # Try auto-start
    daemon_client.start_daemon()
    time.sleep(2)  # Wait for daemon to listen
    
    if not daemon_client.is_daemon_running():
        raise RuntimeError("Daemon failed to start")
```

**Error Handling:**
- Clear error messages with manual start instructions
- Catches subprocess failures gracefully
- Provides fallback options

### 3. Updated Loader Nodes
- **LunaDaemonVAELoader.load_vae()**
- **LunaDaemonCLIPLoader.load_clip()**

Both now call `ensure_daemon_running()` instead of just checking the status.

## User Experience

### Before
```
User loads VAE → Error: "Luna Daemon is not running!"
User must manually: python -m luna_daemon.server
User retries workflow
```

### After
```
User loads VAE → Auto-start triggers → 2-second wait → Workflow runs
(Daemon starts in background/separate window)
```

## Technical Details

### Process Spawning
- **Windows**: Uses `CREATE_NEW_CONSOLE` flag → daemon runs in new cmd window
- **Unix**: Uses `start_new_session=True` → daemon detached from parent terminal
- **Stdout/Stderr**: Redirected to DEVNULL (clean ComfyUI output)

### Timeout Handling
- 2-second sleep allows daemon to bind to socket
- Connection check verifies daemon is actually listening
- Clear error message if startup fails

### Backward Compatibility
- Existing code that calls `is_daemon_running()` unchanged
- Existing code that starts daemon manually still works
- Error messages provide manual start instructions as fallback

## No Longer Needed
Users no longer need to run:
```powershell
python -m luna_daemon.server
```

Unless they explicitly want daemon in same terminal for debugging.

## Files Modified
1. `luna_daemon/client.py` - Added `start_daemon()` function
2. `nodes/luna_daemon_loader.py` - Added `ensure_daemon_running()` helper + updated loaders

## Testing
```python
# Can be tested by:
# 1. Opening ComfyUI with daemon not running
# 2. Loading a VAE or CLIP from daemon loader
# 3. Observing auto-start in background
# 4. Workflow should proceed automatically
```
