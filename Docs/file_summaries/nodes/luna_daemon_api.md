# luna_daemon_api.py

## Purpose
HTTP endpoints for the Luna Daemon panel in ComfyUI. Provides web API for daemon control, status monitoring, and model management.

## Exports
**Classes:**
- None

**Functions:**
- `register_routes()` - Registers API routes with ComfyUI's PromptServer

**Constants:**
- None

## Key Imports
- `aiohttp` - Asynchronous web framework for API endpoints
- `server` - ComfyUI's PromptServer for web integration
- `luna_daemon.client` - Daemon communication client
- `luna_daemon.config` - Daemon configuration constants (DAEMON_HOST, DAEMON_PORT, etc.)
- `subprocess`, `asyncio` - Process management and async operations

## ComfyUI Node Configuration
- No ComfyUI nodes - this file only registers web routes

## Input Schema
- N/A (web API endpoints, no node inputs)

## Key Methods/Functions
- `register_routes() -> None`
  - Registers all API endpoints with PromptServer.instance.routes
  - Called lazily when PromptServer becomes available
- `get_daemon_status(request) -> web.json_response`
  - GET /luna/daemon/status: Returns daemon running status, loaded models, VRAM usage, uptime
- `start_daemon(request) -> web.json_response`
  - POST /luna/daemon/start: Starts daemon process as background subprocess
  - Monitors startup and confirms daemon is responding
- `stop_daemon(request) -> web.json_response`
  - POST /luna/daemon/stop: Sends shutdown command to daemon
- `reconnect_daemon(request) -> web.json_response`
  - POST /luna/daemon/reconnect: Reloads daemon client module and resets connections
- `unload_models(request) -> web.json_response`
  - POST /luna/daemon/unload: Unloads all models from daemon to allow loading different ones

## Dependencies
**Internal:**
- Requires: `luna_daemon.client`, `luna_daemon.config`

**External:**
- Required: `aiohttp`, ComfyUI `server` (PromptServer)
- Optional: None

## Integration Points
**Input:** HTTP requests from ComfyUI web interface (daemon panel)
**Output:** JSON responses with daemon status, control results, and error messages
**Side Effects:** Starts/stops daemon subprocess, manages daemon model loading state, reloads client modules

## Notes
- Routes registered lazily when PromptServer.instance becomes available
- Handles daemon process lifecycle using subprocess.Popen with platform-specific flags
- Provides comprehensive status monitoring including VRAM usage and loaded model details
- Supports daemon reconnection and model unloading for workflow switching
- All endpoints return structured JSON responses for frontend integration