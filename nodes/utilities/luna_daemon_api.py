"""
Luna Daemon Web API
HTTP endpoints for the Luna Daemon panel in ComfyUI.
"""

import os
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
    from ...luna_daemon import client as daemon_client
    from ...luna_daemon.config import (
        DAEMON_HOST, DAEMON_PORT, CLIP_DEVICE, MAX_WORKERS
    )
    DAEMON_AVAILABLE = True
except ImportError:
    # Fallback: try absolute import path
    # NOTE: sys.path is configured centrally in __init__.py
    # All necessary directories should already be in sys.path
    try:
        import os
        import sys
        from pathlib import Path
        
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
            
            # Get detailed info from daemon using the module-level function
            info = daemon_client.get_daemon_info()
            
            # Build models list for display
            models_loaded = []
            vae_pool = info.get("vae_pool", {})
            clip_pool = info.get("clip_pool", {})
            
            # Collect loaded models from pool stats
            if vae_pool.get("loaded_models"):
                for model_name in vae_pool.get("loaded_models", []):
                    models_loaded.append(f"VAE: {model_name}")
            
            if clip_pool.get("loaded_models"):
                for model_name in clip_pool.get("loaded_models", []):
                    models_loaded.append(f"CLIP: {model_name}")
            
            # Get weight registry models (detailed info with VRAM)
            weight_registry_models = info.get("weight_registry_models", [])
            
            # Get VRAM info
            vram_info = info.get("vram", {})
            total_vram_gb = sum(v.get("total_gb", 0) for v in vram_info.values())
            used_vram_gb = sum(v.get("used_gb", 0) for v in vram_info.values())
            
            # Build GPU array for multi-GPU display
            gpus = []
            daemon_clip_device = info.get("devices", {}).get("clip", "cuda:0")
            for device_id, vram in vram_info.items():
                used = vram.get("used_gb", 0)
                total = vram.get("total_gb", 0)
                reserved = used  # For daemon, used = reserved
                percent = (used / total * 100) if total > 0 else 0
                
                # Determine if this is the daemon device
                is_daemon_device = (device_id == daemon_clip_device)
                
                # Get GPU name from device ID
                gpu_id_num = device_id.split(":")[-1] if ":" in device_id else "0"
                
                gpus.append({
                    "id": gpu_id_num,
                    "name": f"GPU {gpu_id_num}",
                    "used_gb": round(used, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 1),
                    "percent": round(percent, 1),
                    "is_daemon_device": is_daemon_device
                })
            
            # Get ComfyUI VRAM usage if available
            comfyui_vram = {}
            try:
                import comfy.model_management as mm
                if hasattr(mm, 'get_total_memory'):
                    # Get main inference device info
                    device = mm.get_torch_device()
                    total_mem = mm.get_total_memory(device) / (1024**3)  # Convert to GB
                    free_mem = mm.get_free_memory(device) / (1024**3)
                    used_mem = total_mem - free_mem
                    
                    comfyui_vram = {
                        "device": str(device),
                        "total_gb": round(total_mem, 2),
                        "used_gb": round(used_mem, 2),
                        "free_gb": round(free_mem, 2),
                        "percent": round((used_mem / total_mem * 100) if total_mem > 0 else 0, 1)
                    }
                    
                    # Try to get loaded models info
                    if hasattr(mm, 'current_loaded_models'):
                        loaded_models = mm.current_loaded_models
                        comfyui_vram["loaded_models_count"] = len(loaded_models)
            except Exception as e:
                logger.debug(f"Could not get ComfyUI VRAM info: {e}")
            
            return web.json_response({
                "running": True,
                "device": info.get("devices", {}).get("clip", CLIP_DEVICE),
                "vram_used_gb": round(used_vram_gb, 2),
                "vram_allocated_gb": round(used_vram_gb, 2),  # For daemon, allocated = used
                "vram_total_gb": round(total_vram_gb, 1),
                "vram_percent": round((used_vram_gb / total_vram_gb * 100) if total_vram_gb > 0 else 0, 1),
                "request_count": info.get("request_count", 0),
                "clip_request_count": info.get("clip_request_count", 0),
                "vae_request_count": info.get("vae_request_count", 0),
                "uptime_seconds": int(info.get("uptime_sec", 0)),
                "models_loaded": models_loaded,
                "checkpoints": [],  # TODO: Track checkpoints in daemon
                "gpus": gpus,
                "vae_loaded": vae_pool.get("workers_count", 0) > 0,
                "clip_loaded": clip_pool.get("workers_count", 0) > 0,
                "weight_registry_models": weight_registry_models,
                "comfyui_vram": comfyui_vram,
            })
            
        except Exception as e:
            import traceback
            logger.error(f"[Luna.DaemonAPI] Error getting daemon status: {e}")
            logger.error(traceback.format_exc())
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
                
                # Even though it's running, sync the attention mode
                try:
                    import comfy.cli_args  # type: ignore
                    args = comfy.cli_args.args
                    attention_mode = "auto"
                    if getattr(args, 'use_sage_attention', False):
                        attention_mode = "sage"
                    elif getattr(args, 'use_flash_attention', False):
                        attention_mode = "flash"
                    elif getattr(args, 'use_quad_cross_attention', False) or getattr(args, 'use_split_cross_attention', False):
                        attention_mode = "split"
                    elif getattr(args, 'use_pytorch_cross_attention', False):
                        attention_mode = "pytorch"
                    
                    logger.info(f"Updating daemon attention mode to: {attention_mode}")
                    result = daemon_client.set_attention_mode(attention_mode)
                    if result.get("success"):
                        logger.info(f"Successfully updated daemon to {attention_mode} attention")
                    else:
                        logger.warning(f"Failed to update daemon attention mode: {result.get('message')}")
                except Exception as e:
                    logger.warning(f"Could not sync attention mode: {e}")
                
                return web.json_response({"status": "ok", "message": "Daemon already running, attention mode synced"})
            
            logger.info("Starting Luna Daemon Tray...")
            
            # Detect ComfyUI's attention mode and pass to daemon via environment variable
            attention_mode = "auto"
            try:
                import comfy.cli_args  # type: ignore
                args = comfy.cli_args.args
                if getattr(args, 'use_sage_attention', False):
                    attention_mode = "sage"
                elif getattr(args, 'use_flash_attention', False):
                    attention_mode = "flash"
                elif getattr(args, 'use_quad_cross_attention', False) or getattr(args, 'use_split_cross_attention', False):
                    attention_mode = "split"
                elif getattr(args, 'use_pytorch_cross_attention', False):
                    attention_mode = "pytorch"
                logger.info(f"Detected ComfyUI attention mode: {attention_mode}")
            except Exception as e:
                logger.warning(f"Could not detect attention mode: {e}, using auto")
            
            # Get path to daemon package and tray_app.py
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[2]
            daemon_dir = repo_root / 'luna_daemon'
            tray_app = os.path.join(daemon_dir, 'tray_app.py')
            
            # Python executable
            python_exe = sys.executable
            
            # Build command to run tray app (single-instance enforced)
            cmd = [python_exe, tray_app]
            
            # Prepare environment with attention mode
            env = os.environ.copy()
            env['LUNA_ATTENTION_MODE'] = attention_mode
            
            # Set working directory to the daemon directory so relative imports work
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Working directory: {daemon_dir}")
            logger.info(f"Environment: LUNA_ATTENTION_MODE={attention_mode}")
            
            # Start tray app as subprocess
            # The tray app enforces single-instance via port lock
            # Launch detached without capturing output so GUI can initialize properly
            if os.name == 'nt':
                # Windows: Use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                process = subprocess.Popen(
                    cmd,
                    cwd=daemon_dir,
                    env=env,
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True
                )
            else:
                # Unix: Use start_new_session
                process = subprocess.Popen(
                    cmd,
                    cwd=daemon_dir,
                    env=env,
                    start_new_session=True,
                    close_fds=True
                )
            
            logger.info(f"Daemon tray app launched with PID {process.pid}")
            
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
                        # NOTE: sys.path is configured centrally in __init__.py
                        # No need for manual path manipulation here
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


    @PromptServer.instance.routes.get("/luna/daemon/devices")
    async def get_devices(request):
        """Get current device configuration and available GPUs"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"success": False, "error": "Daemon client not available"})
        
        try:
            if not daemon_client.is_daemon_running():
                return web.json_response({"success": False, "error": "Daemon is not running"})
            
            result = daemon_client.get_devices()
            return web.json_response(result)
                
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})


    @PromptServer.instance.routes.post("/luna/daemon/set-clip-device")
    async def set_clip_device(request):
        """Change CLIP device"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"success": False, "error": "Daemon client not available"})
        
        try:
            data = await request.json()
            device = data.get("device")
            
            if not device:
                return web.json_response({"success": False, "error": "device parameter required"})
            
            if not daemon_client.is_daemon_running():
                return web.json_response({"success": False, "error": "Daemon is not running"})
            
            result = daemon_client.set_clip_device(device)
            return web.json_response(result)
                
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})


    @PromptServer.instance.routes.post("/luna/daemon/set-vae-device")
    async def set_vae_device(request):
        """Change VAE device"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"success": False, "error": "Daemon client not available"})
        
        try:
            data = await request.json()
            device = data.get("device")
            
            if not device:
                return web.json_response({"success": False, "error": "device parameter required"})
            
            if not daemon_client.is_daemon_running():
                return web.json_response({"success": False, "error": "Daemon is not running"})
            
            result = daemon_client.set_vae_device(device)
            return web.json_response(result)
                
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})


    @PromptServer.instance.routes.post("/luna/daemon/set-llm-device")
    async def set_llm_device(request):
        """Change LLM device"""
        if not DAEMON_AVAILABLE or daemon_client is None:
            return web.json_response({"success": False, "error": "Daemon client not available"})
        
        try:
            data = await request.json()
            device = data.get("device")
            
            if not device:
                return web.json_response({"success": False, "error": "device parameter required"})
            
            if not daemon_client.is_daemon_running():
                return web.json_response({"success": False, "error": "Daemon is not running"})
            
            result = daemon_client.set_llm_device(device)
            return web.json_response(result)
                
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})


# Try to register routes immediately if PromptServer.instance exists
register_routes()


# Empty mappings since this file just registers routes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
