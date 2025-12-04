"""
Luna Daemon Web API
HTTP endpoints for the Luna Daemon panel in ComfyUI.
"""

import os
import json

# Try to import aiohttp
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Try to import PromptServer for web endpoints
PromptServer = None
try:
    from server import PromptServer as _PromptServer
    PromptServer = _PromptServer
except ImportError:
    pass

# Try to import daemon client
try:
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.config import (
        DAEMON_HOST, DAEMON_PORT, SHARED_DEVICE, MAX_WORKERS
    )
    DAEMON_AVAILABLE = True
except ImportError:
    daemon_client = None
    DAEMON_AVAILABLE = False
    DAEMON_HOST = "127.0.0.1"
    DAEMON_PORT = 19283
    SHARED_DEVICE = "cuda:1"
    MAX_WORKERS = 4


# Track if routes have been registered
_routes_registered = False


def register_routes():
    """Register API routes - called when PromptServer.instance is available"""
    global _routes_registered
    
    if _routes_registered:
        return
    
    if not PromptServer or not hasattr(PromptServer, 'instance') or PromptServer.instance is None:
        return
    
    if not HAS_AIOHTTP:
        return
    
    _routes_registered = True
    
    @PromptServer.instance.routes.get("/luna/daemon/status")
    async def get_daemon_status(request):
        """Get current daemon status including loaded models"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({
                "running": False,
                "error": "Daemon client not available"
            })
        
        try:
            if not daemon_client.is_daemon_running():
                return web.json_response({
                    "running": False,
                })
            
            # Get detailed info from daemon
            info = daemon_client.get_daemon_info()
            
            # Build models list for display
            models_loaded = []
            if info.get("loaded_vae"):
                models_loaded.append(f"VAE: {info['loaded_vae']}")
            if info.get("loaded_clip"):
                clip_names = info['loaded_clip']
                if isinstance(clip_names, list):
                    models_loaded.append(f"CLIP: {', '.join(clip_names)}")
                else:
                    models_loaded.append(f"CLIP: {clip_names}")
            
            return web.json_response({
                "running": True,
                "device": info.get("device", SHARED_DEVICE),
                "vram_used_gb": info.get("vram_used_gb", 0),
                "vram_total_gb": info.get("vram_total_gb", 0),
                "request_count": info.get("request_count", 0),
                "uptime_seconds": int(info.get("uptime_seconds", 0)),
                "models_loaded": models_loaded,
                "vae_loaded": info.get("vae_loaded", False),
                "clip_loaded": info.get("clip_loaded", False),
                "loaded_vae": info.get("loaded_vae"),
                "loaded_vae_path": info.get("loaded_vae_path"),
                "loaded_clip": info.get("loaded_clip"),
                "loaded_clip_paths": info.get("loaded_clip_paths"),
            })
            
        except Exception as e:
            return web.json_response({
                "running": False,
                "error": str(e),
            })
    
    
    @PromptServer.instance.routes.post("/luna/daemon/start")
    async def start_daemon(request):
        """Start the daemon process"""
        import subprocess
        import sys
        
        try:
            # Get the path to the daemon server module
            daemon_module = "custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server"
            
            # Start daemon in background
            python_exe = sys.executable
            
            # Build command
            cmd = [python_exe, "-m", daemon_module]
            
            # Start as subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Give it a moment to start
            import asyncio
            await asyncio.sleep(1)
            
            # Check if it started
            if DAEMON_AVAILABLE and daemon_client.is_daemon_running():
                return web.json_response({"status": "ok", "message": "Daemon started"})
            else:
                return web.json_response({"status": "error", "message": "Daemon failed to start"})
                
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)})
    
    
    @PromptServer.instance.routes.post("/luna/daemon/stop")
    async def stop_daemon(request):
        """Stop the daemon process"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"status": "error", "message": "Daemon client not available"})
        
        try:
            if daemon_client.is_daemon_running():
                client = daemon_client.get_client()
                client.shutdown()
                
                # Wait for shutdown
                import asyncio
                await asyncio.sleep(0.5)
                
                return web.json_response({"status": "ok", "message": "Daemon stopped"})
            else:
                return web.json_response({"status": "ok", "message": "Daemon was not running"})
                
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)})
    
    
    @PromptServer.instance.routes.post("/luna/daemon/unload")
    async def unload_models(request):
        """Unload all models from daemon to allow loading different ones"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"status": "error", "message": "Daemon client not available"})
        
        try:
            if not daemon_client.is_daemon_running():
                return web.json_response({"status": "error", "message": "Daemon is not running"})
            
            result = daemon_client.unload_daemon_models()
            return web.json_response({
                "status": "ok",
                "message": "Models unloaded",
                "unloaded_vae": result.get("unloaded_vae"),
                "unloaded_clip": result.get("unloaded_clip"),
            })
                
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)})


# Try to register routes immediately if PromptServer.instance exists
register_routes()


# Empty mappings since this file just registers routes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
