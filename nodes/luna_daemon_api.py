"""
Luna Daemon Web API
HTTP endpoints for the Luna Daemon panel in ComfyUI.
"""

import os
import json
import importlib
import logging

# Set up logging for this module
logger = logging.getLogger("Luna.DaemonAPI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
        DAEMON_HOST, DAEMON_PORT, CLIP_DEVICE, MAX_WORKERS
    )
    DAEMON_AVAILABLE = True
except ImportError:
    # Fallback: try absolute import path
    try:
        import os
        import sys
        # Add parent directory to path for direct import
        _daemon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'luna_daemon'))
        if _daemon_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(_daemon_dir))
        
        from luna_daemon import client as daemon_client
        from luna_daemon.config import (
            DAEMON_HOST, DAEMON_PORT, CLIP_DEVICE, MAX_WORKERS
        )
        DAEMON_AVAILABLE = True
    except ImportError as e:
        logger.error(f"Failed to import daemon client: {e}")
        daemon_client = None
        DAEMON_AVAILABLE = False
        DAEMON_HOST = "127.0.0.1"
        DAEMON_PORT = 19283
        CLIP_DEVICE = "cuda:1"
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
                "device": info.get("device", CLIP_DEVICE),
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
        import asyncio
        
        try:
            # Check if already running
            if DAEMON_AVAILABLE and daemon_client and daemon_client.is_daemon_running():
                logger.info("Daemon already running")
                return web.json_response({"status": "ok", "message": "Daemon already running"})
            
            logger.info("Starting Luna Daemon...")
            
            # Get path to daemon package
            daemon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'luna_daemon'))
            
            # Python executable
            python_exe = sys.executable
            
            # Build command to run daemon as module
            # Use the daemon directory as the starting point
            cmd = [python_exe, "-m", "luna_daemon"]
            
            # Set working directory to ComfyUI root (where custom_nodes is)
            comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Working directory: {comfyui_root}")
            
            # Start as subprocess in background
            process = subprocess.Popen(
                cmd,
                cwd=comfyui_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True if os.name != 'nt' else False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            logger.info(f"Daemon process started with PID {process.pid}")
            
            # Wait for daemon to start with retries
            max_retries = 15
            retry_delay = 0.5  # Start with 0.5s, increase as we wait
            
            for attempt in range(max_retries):
                await asyncio.sleep(retry_delay)
                
                # Increase delay for later attempts
                if attempt > 5:
                    retry_delay = 1.0
                
                if DAEMON_AVAILABLE and daemon_client and daemon_client.is_daemon_running():
                    logger.info(f"Daemon connected successfully (attempt {attempt + 1})")
                    return web.json_response({
                        "status": "ok", 
                        "message": f"Daemon started successfully (attempt {attempt + 1})"
                    })
                
                # Check if process died
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Process exited"
                    logger.error(f"Daemon process died: {error_msg[:200]}")
                    return web.json_response({
                        "status": "error", 
                        "message": f"Daemon process exited: {error_msg[:200]}"
                    })
            
            # Timeout
            logger.warning("Daemon startup timeout - process running but not responding")
            return web.json_response({
                "status": "error", 
                "message": "Daemon process started but failed to respond within timeout (7.5s). Check luna_daemon/server.py logs."
            })
                
        except Exception as e:
            logger.exception("Failed to start daemon")
            return web.json_response({"status": "error", "message": f"Failed to start daemon: {str(e)}"})
    
    
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
    
    
    @PromptServer.instance.routes.post("/luna/daemon/reconnect")
    async def reconnect_daemon(request):
        """Force reconnect to daemon and reload client"""
        global daemon_client, DAEMON_AVAILABLE
        
        try:
            # If client was never loaded, try to load it
            if daemon_client is None:
                try:
                    from ..luna_daemon import client as dc
                    daemon_client = dc
                    DAEMON_AVAILABLE = True
                except ImportError:
                    # Try fallback path
                    try:
                        import sys
                        _daemon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'luna_daemon'))
                        if _daemon_dir not in sys.path:
                            sys.path.insert(0, os.path.dirname(_daemon_dir))
                        from luna_daemon import client as dc
                        daemon_client = dc
                        DAEMON_AVAILABLE = True
                    except Exception as e:
                        return web.json_response({"status": "error", "message": f"Import failed: {str(e)}"})
            
            # Reload the module to pick up any config changes
            if daemon_client:
                importlib.reload(daemon_client)
                if hasattr(daemon_client, 'reset_clients'):
                    daemon_client.reset_clients()
            
            # Check connection
            if daemon_client and daemon_client.is_daemon_running():
                return web.json_response({"status": "ok", "message": "Reconnected successfully"})
            else:
                return web.json_response({"status": "error", "message": "Client reloaded but daemon not reachable"})
                
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
