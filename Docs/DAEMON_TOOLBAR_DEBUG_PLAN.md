# Luna Daemon Toolbar - Systematic Debug Plan

**Last Updated:** December 12, 2025  
**Status:** Investigation Phase  
**Problem:** ComfyUI daemon panel/toolbar not fully functional

---

## Problem Statement

The daemon toolbar exists and loads, but interaction is broken:
- ✅ Daemon is running (confirmed: port 19283, 19285, 19299 listening)
- ✅ Daemon imports work (fixed relative import issues)
- ✅ Health check works manually (`is_daemon_running()` returns True)
- ❌ ComfyUI panel still shows "not available" during workflow execution
- ❌ Toolbar buttons may not be responding

---

## Current Architecture

### Backend (Python)
- **luna_daemon_api.py** - Web endpoints exposed to ComfyUI UI
  - Path: `nodes/utilities/luna_daemon_api.py`
  - Exposes API endpoints for panel control
  - Must have proper routing registered with PromptServer

### Frontend (JavaScript)
- **luna_daemon_panel.js** - UI panel for daemon control
  - Path: `js/luna_daemon_panel.js`
  - Calls API endpoints from backend
  - Renders status, controls, etc.
  - Auto-refresh loop every 3 seconds

### Communication Chain
```
ComfyUI UI (JS) 
  ↓
fetch() to /luna/* endpoints
  ↓
luna_daemon_api.py handlers
  ↓
daemon_client (socket to daemon)
  ↓
Daemon (port 19283)
```

---

## Investigation Checklist

### Phase 1: Backend API Registration ✓
- [ ] Verify `luna_daemon_api.py` is being imported
- [ ] Check if web routes are registered with PromptServer
- [ ] Confirm endpoints exist: `/luna/status`, `/luna/start`, `/luna/stop`, `/luna/unload`
- [ ] Test endpoints manually with curl/browser

**How to test:**
```bash
# While ComfyUI is running, test these URLs:
curl http://localhost:8188/luna/status
curl -X POST http://localhost:8188/luna/start
```

### Phase 2: Frontend Panel Loading
- [ ] Verify `luna_daemon_panel.js` is being loaded by ComfyUI
- [ ] Check browser console for JavaScript errors
- [ ] Confirm panel HTML is rendered in DOM
- [ ] Verify event listeners are attached to buttons

**How to test:**
```javascript
// In browser DevTools console:
document.querySelector(".luna-daemon-panel")  // Should exist
document.getElementById("luna-refresh")        // Should exist
document.getElementById("luna-toggle")         // Should exist
```

### Phase 3: API Communication
- [ ] Verify fetch() calls from JS reach backend
- [ ] Check if responses return correct data
- [ ] Verify daemon client can connect in API handler context
- [ ] Test health check in API handler

**How to test:**
- Open browser Network tab in DevTools
- Refresh panel
- Watch for `/luna/status` requests
- Check response payload

### Phase 4: Daemon Connection in Handler
- [ ] Verify daemon_client imports work in `luna_daemon_api.py`
- [ ] Check if `is_daemon_running()` works when called from API handler
- [ ] Verify socket connection timeout settings are correct
- [ ] Check if daemon responds to health requests from API context

**How to test:**
- Add logging to `luna_daemon_api.py` handlers
- Watch ComfyUI console for output
- Restart ComfyUI and trigger panel refresh

---

## Known Issues & Fixes Applied

### Issue 1: Relative Import Collisions ✅ FIXED
**Problem:** `from ..luna_daemon` failed with "no module named nodes.luna_daemon"  
**Root Cause:** ComfyUI's own `nodes` module interfered with relative imports  
**Solution:** Changed to absolute imports with sys.path manipulation
**Files Fixed:**
- `nodes/loaders/luna_model_router.py`
- `nodes/loaders/luna_daemon_loader.py`

### Issue 2: Daemon Health Check Timeout
**Problem:** Health check used 120-second timeout (CLIENT_TIMEOUT)  
**Solution:** Added retry logic with 2-second timeout for health checks  
**Files Fixed:** `luna_daemon/client.py` (is_running method)

### Issue 3: Import Error Reporting
**Problem:** Silent failures made debugging impossible  
**Solution:** Added detailed logging to daemon import blocks  
**Files Fixed:** `nodes/loaders/luna_model_router.py`

### Issue 4: Daemon API Not Registered ✅ FIXED
**Problem:** Web routes in `luna_daemon_api.py` were never registered  
**Root Cause:** Module was never imported in `nodes/__init__.py`  
**Solution:** Added import to `nodes/__init__.py` to ensure routes are registered at startup
**Files Fixed:** `nodes/__init__.py`
**Details:** The `register_routes()` function in `luna_daemon_api.py` only runs when the module is imported

---

## Next Steps - Specific Actions

1. **Check if API is registered**
   - Open `nodes/utilities/luna_daemon_api.py`
   - Search for `@PromptServer.instance.routes`
   - Verify all endpoints are decorated correctly
   - Check if module is imported in `nodes/__init__.py`

2. **Check if JS is loaded**
   - Search `js/` directory for all .js files
   - Check if `luna_daemon_panel.js` is included in ComfyUI
   - Verify in `__init__.py` that JS files are registered

3. **Add debug logging**
   - Add console.log to every function in `luna_daemon_panel.js`
   - Add print() to every handler in `luna_daemon_api.py`
   - Restart ComfyUI and check console output

4. **Test API endpoints directly**
   - Use curl or Postman
   - Make direct HTTP requests to `/luna/status`
   - Check response payload

5. **Check daemon socket connection**
   - In API handler, add print statements to daemon_client calls
   - Verify socket connects to 127.0.0.1:19283
   - Check if health check succeeds in API context

---

## Validation Criteria (Success Condition)

✅ When fixed, these should all be true:

1. ComfyUI console shows daemon import success messages
2. Browser console shows no errors loading `luna_daemon_panel.js`
3. Panel renders in ComfyUI sidebar with status visible
4. Network tab shows successful `/luna/status` API calls
5. Panel buttons respond (Start/Stop/Unload)
6. Daemon status shows "Running" in panel
7. Workflow execution still works with daemon mode

---

## Files to Investigate (in order)

### Critical (Check first)
1. `nodes/utilities/luna_daemon_api.py` - Does it register routes?
2. `nodes/__init__.py` - Is daemon_api imported?
3. `js/luna_daemon_panel.js` - Is it being loaded?
4. `__init__.py` (root) - Are JS files registered?

### Secondary
1. `luna_daemon/client.py` - Health check working?
2. `luna_daemon/server.py` - Daemon responding?
3. `luna_daemon/config.py` - Correct ports/devices?

### Reference
- `luna_daemon/TRAY_APP_README.md` - Expected behavior
- `luna_daemon/README.md` - Architecture docs
- `NODES_DOCUMENTATION.md` - Node reference

---

## How to Use This Document

1. **Work through Investigation Checklist** - Don't skip items
2. **Test each phase** - Confirm before moving to next
3. **Document findings** - Update this doc with results
4. **Debug methodically** - Add logging, don't guess
5. **Validate fixes** - Confirm all success criteria met

**Do NOT:** Try multiple fixes at once. Test one thing, verify it works, document it, then move on.

