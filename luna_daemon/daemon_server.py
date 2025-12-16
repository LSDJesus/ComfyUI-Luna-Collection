"""
Luna Daemon Server - Main Orchestrator

Simplified daemon server that coordinates:
- VAE/CLIP worker pools
- WebSocket monitoring
- LoRA RAM cache
- Socket communication with ComfyUI clients

This is the main entry point for the daemon.
"""

import os
import sys
import socket
import pickle
import threading
import time
import logging
from typing import Any, Dict, Optional, Set

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LunaDaemon")

# =============================================================================
# Import Configuration
# =============================================================================

# Handle both package import and direct script execution
config_loaded = False
try:
    # Try relative imports first (package context)
    from .config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_WS_PORT,
        CLIP_DEVICE, VAE_DEVICE,
        VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
        MODEL_PRECISION,
        MAX_VAE_WORKERS, MAX_CLIP_WORKERS, MIN_VAE_WORKERS, MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD, SCALE_UP_DELAY_SEC, IDLE_TIMEOUT_SEC,
        ServiceType, SERVICE_TYPE
    )
    try:
        from .config import CLIP_PRECISION, VAE_PRECISION
    except ImportError:
        CLIP_PRECISION = MODEL_PRECISION
        VAE_PRECISION = MODEL_PRECISION
    config_loaded = True
except (ImportError, ValueError):
    # Fallback: direct import when run as __main__
    try:
        import importlib.util
        daemon_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(daemon_dir, "config.py")
        spec = importlib.util.spec_from_file_location("luna_daemon_config", config_path)
        if spec and spec.loader:
            config_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_mod)
            
            DAEMON_HOST = getattr(config_mod, "DAEMON_HOST", "127.0.0.1")
            DAEMON_PORT = getattr(config_mod, "DAEMON_PORT", 19283)
            DAEMON_WS_PORT = getattr(config_mod, "DAEMON_WS_PORT", 19284)
            CLIP_DEVICE = getattr(config_mod, "CLIP_DEVICE", "cuda:0")
            VAE_DEVICE = getattr(config_mod, "VAE_DEVICE", "cuda:0")
            VAE_PATH = getattr(config_mod, "VAE_PATH", "")
            CLIP_L_PATH = getattr(config_mod, "CLIP_L_PATH", "")
            CLIP_G_PATH = getattr(config_mod, "CLIP_G_PATH", "")
            EMBEDDINGS_DIR = getattr(config_mod, "EMBEDDINGS_DIR", "")
            MODEL_PRECISION = getattr(config_mod, "MODEL_PRECISION", "fp16")
            CLIP_PRECISION = getattr(config_mod, "CLIP_PRECISION", MODEL_PRECISION)
            VAE_PRECISION = getattr(config_mod, "VAE_PRECISION", MODEL_PRECISION)
            MAX_VAE_WORKERS = getattr(config_mod, "MAX_VAE_WORKERS", 2)
            MAX_CLIP_WORKERS = getattr(config_mod, "MAX_CLIP_WORKERS", 2)
            MIN_VAE_WORKERS = getattr(config_mod, "MIN_VAE_WORKERS", 0)
            MIN_CLIP_WORKERS = getattr(config_mod, "MIN_CLIP_WORKERS", 0)
            QUEUE_THRESHOLD = getattr(config_mod, "QUEUE_THRESHOLD", 2)
            SCALE_UP_DELAY_SEC = getattr(config_mod, "SCALE_UP_DELAY_SEC", 1.0)
            IDLE_TIMEOUT_SEC = getattr(config_mod, "IDLE_TIMEOUT_SEC", 30.0)
            ServiceType = getattr(config_mod, "ServiceType", None)
            SERVICE_TYPE = getattr(config_mod, "SERVICE_TYPE", None)
            config_loaded = True
    except:
        pass

if not config_loaded:
    # Fallback defaults
    DAEMON_HOST = "127.0.0.1"
    DAEMON_PORT = 19283
    DAEMON_WS_PORT = 19284
    CLIP_DEVICE = "cuda:0"
    VAE_DEVICE = "cuda:0"
    VAE_PATH = ""
    CLIP_L_PATH = ""
    CLIP_G_PATH = ""
    EMBEDDINGS_DIR = ""
    MODEL_PRECISION = "fp16"
    CLIP_PRECISION = "fp16"
    VAE_PRECISION = "fp16"
    MAX_VAE_WORKERS = 2
    MAX_CLIP_WORKERS = 2
    MIN_VAE_WORKERS = 0
    MIN_CLIP_WORKERS = 0
    QUEUE_THRESHOLD = 2
    SCALE_UP_DELAY_SEC = 1.0
    IDLE_TIMEOUT_SEC = 30.0
    
    from enum import Enum, auto
    class ServiceType(Enum):
        FULL = auto()
        VAE_ONLY = auto()
        CLIP_ONLY = auto()
    SERVICE_TYPE = ServiceType.FULL

# =============================================================================
# Import Modules
# =============================================================================

# Handle imports - try relative first, then absolute
try:
    from .core import ServiceType as CoreServiceType
    from .workers import WorkerPool, WorkerType, ScalingConfig
    from .monitoring import WebSocketServer
    from .lora_cache import LoRACache, get_lora_cache
except (ImportError, ValueError):
    # Fallback: load modules directly
    import importlib.util
    daemon_dir = os.path.dirname(os.path.abspath(__file__))
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        raise ImportError(f"Could not load {name}")
    
    core_mod = load_module("luna_daemon_core", os.path.join(daemon_dir, "core.py"))
    CoreServiceType = getattr(core_mod, "ServiceType", ServiceType)
    
    workers_mod = load_module("luna_daemon_workers", os.path.join(daemon_dir, "workers.py"))
    WorkerPool = workers_mod.WorkerPool
    WorkerType = workers_mod.WorkerType
    ScalingConfig = workers_mod.ScalingConfig
    
    monitoring_mod = load_module("luna_daemon_monitoring", os.path.join(daemon_dir, "monitoring.py"))
    WebSocketServer = monitoring_mod.WebSocketServer
    
    cache_mod = load_module("luna_daemon_cache", os.path.join(daemon_dir, "lora_cache.py"))
    LoRACache = cache_mod.LoRACache
    get_lora_cache = cache_mod.get_lora_cache


# =============================================================================
# Main Daemon Server
# =============================================================================

class LunaDaemon:
    """
    Main daemon server with VAE/CLIP worker pools and monitoring.
    
    Simplified architecture:
    - Workers handle VAE encode/decode and CLIP encoding
    - WebSocket broadcasts status to JS panel
    - LoRA cache stores state_dicts in RAM
    - Socket server handles client requests
    """
    
    def __init__(
        self,
        host: str = DAEMON_HOST,
        port: int = DAEMON_PORT,
        ws_port: int = DAEMON_WS_PORT,
        clip_device: str = CLIP_DEVICE,
        vae_device: str = VAE_DEVICE,
        service_type: ServiceType = SERVICE_TYPE,
        # Backward compatibility with old tray app parameters
        device: str = None,
        clip_precision: str = None,
        vae_precision: str = None,
        **kwargs
    ):
        # Handle old tray app parameter names
        if device is not None:
            clip_device = device
            vae_device = device
        
        self.host = host
        self.port = port
        self.ws_port = ws_port
        self.clip_device = clip_device
        self.vae_device = vae_device
        self.service_type = service_type
        
        # Scaling configuration
        self.scaling_config = ScalingConfig(
            min_vae_workers=MIN_VAE_WORKERS,
            max_vae_workers=MAX_VAE_WORKERS,
            min_clip_workers=MIN_CLIP_WORKERS,
            max_clip_workers=MAX_CLIP_WORKERS,
            queue_threshold=QUEUE_THRESHOLD,
            scale_up_delay_sec=SCALE_UP_DELAY_SEC,
            idle_timeout_sec=IDLE_TIMEOUT_SEC
        )
        
        # Config paths for workers
        self.config_paths = {
            'vae': VAE_PATH,
            'clip_l': CLIP_L_PATH,
            'clip_g': CLIP_G_PATH,
            'embeddings': EMBEDDINGS_DIR
        }
        
        # Worker pools
        self.vae_pool: Optional[WorkerPool] = None
        self.clip_pool: Optional[WorkerPool] = None
        
        # WebSocket monitoring
        self.ws_server: Optional[WebSocketServer] = None
        
        # LoRA cache
        self.lora_cache = get_lora_cache()
        
        # Socket server
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._client_threads: Set[threading.Thread] = set()
        
        # Stats
        self._start_time = time.time()
        self._request_count = 0
        self._clip_request_count = 0
        self._vae_request_count = 0
    
    def _on_scale_event(self, event_type: str, data: dict):
        """Callback for worker pool scaling events."""
        if self.ws_server:
            self.ws_server.broadcast("scaling", data)
    
    def get_info(self) -> Dict[str, Any]:
        """Get daemon status info for monitoring."""
        info = {
            "status": "running" if self._running else "stopped",
            "uptime_sec": time.time() - self._start_time,
            "request_count": self._request_count,
            "clip_request_count": self._clip_request_count,
            "vae_request_count": self._vae_request_count,
            "service_type": self.service_type.name if hasattr(self.service_type, 'name') else str(self.service_type),
            "devices": {
                "clip": self.clip_device,
                "vae": self.vae_device
            }
        }
        
        if self.vae_pool:
            info["vae_pool"] = self.vae_pool.get_stats()
        if self.clip_pool:
            info["clip_pool"] = self.clip_pool.get_stats()
        
        # VRAM info for ALL GPUs
        if torch.cuda.is_available():
            info["vram"] = {}
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                info["vram"][f"cuda:{i}"] = {
                    "used_gb": round((total - free) / 1024**3, 2),
                    "total_gb": round(total / 1024**3, 2)
                }
        
        return info
    
    def _handle_request(self, cmd: str, data: dict) -> Any:
        """Route request to appropriate handler."""
        # Skip counting ping/health check requests
        if cmd != "ping":
            self._request_count += 1
        
        # VAE commands
        if cmd == "vae_encode":
            self._vae_request_count += 1
            if self.vae_pool is None:
                return {"error": "VAE pool not available"}
            return self.vae_pool.submit(cmd, data)
        
        elif cmd == "vae_decode":
            self._vae_request_count += 1
            if self.vae_pool is None:
                return {"error": "VAE pool not available"}
            return self.vae_pool.submit(cmd, data)
        
        # CLIP commands
        elif cmd == "clip_encode":
            self._clip_request_count += 1
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        elif cmd == "clip_encode_sdxl":
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        # LoRA cache commands
        elif cmd == "lora_cache_get":
            lora_name = data.get("lora_name")
            if not lora_name:
                return {"error": "lora_name required"}
            state_dict = self.lora_cache.get(lora_name)
            if state_dict is None:
                return {"cached": False}
            return {"cached": True, "state_dict": state_dict}
        
        elif cmd == "lora_cache_put":
            lora_name = data.get("lora_name")
            state_dict = data.get("state_dict")
            if not lora_name or state_dict is None:
                return {"error": "lora_name and state_dict required"}
            success = self.lora_cache.put(lora_name, state_dict)
            return {"success": success}
        
        elif cmd == "lora_cache_check":
            lora_name = data.get("lora_name")
            if not lora_name:
                return {"error": "lora_name required"}
            return {"cached": self.lora_cache.contains(lora_name)}
        
        elif cmd == "lora_cache_stats":
            return self.lora_cache.get_stats()
        
        # Status/info commands
        elif cmd == "ping":
            return {"pong": True, "time": time.time()}
        
        elif cmd == "get_info":
            return self.get_info()
        
        elif cmd == "get_status":
            return self.get_info()
        
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    def _handle_client(self, conn: socket.socket, addr):
        """Handle a client connection."""
        logger.debug(f"Client connected: {addr}")
        
        try:
            while self._running:
                # Receive length prefix
                length_data = conn.recv(8)
                if not length_data or len(length_data) < 8:
                    break
                
                msg_length = int.from_bytes(length_data, 'big')
                
                # Receive message
                data = b''
                while len(data) < msg_length:
                    chunk = conn.recv(min(msg_length - len(data), 65536))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < msg_length:
                    break
                
                # Deserialize request
                try:
                    request = pickle.loads(data)
                except Exception as e:
                    logger.error(f"Failed to deserialize request: {e}")
                    continue
                
                cmd = request.get("cmd", "")
                
                # Process request
                try:
                    result = self._handle_request(cmd, request)
                except Exception as e:
                    logger.error(f"Error handling {cmd}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    result = {"error": str(e)}
                
                # Serialize and send response
                try:
                    response_data = pickle.dumps(result)
                    length_prefix = len(response_data).to_bytes(8, 'big')
                    conn.sendall(length_prefix + response_data)
                except Exception as e:
                    logger.error(f"Failed to send response: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
            logger.debug(f"Client disconnected: {addr}")
    
    def start(self):
        """Start the daemon server."""
        logger.info("=" * 60)
        logger.info("Luna Daemon Starting...")
        logger.info("=" * 60)
        
        self._running = True
        self._start_time = time.time()
        
        # Start worker pools based on service type
        service = self.service_type
        if hasattr(service, 'value'):
            service = service  # Already enum
        
        if service in (ServiceType.FULL, CoreServiceType.FULL) or str(service) == "ServiceType.FULL":
            # Start VAE pool
            self.vae_pool = WorkerPool(
                worker_type=WorkerType.VAE,
                device=self.vae_device,
                precision=VAE_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.vae_pool.start()
            logger.info(f"[VAE] Worker pool started on {self.vae_device}")
            
            # Start CLIP pool
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=self.clip_device,
                precision=CLIP_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.clip_pool.start()
            logger.info(f"[CLIP] Worker pool started on {self.clip_device}")
        
        elif service in (ServiceType.VAE_ONLY, CoreServiceType.VAE_ONLY) or "VAE" in str(service):
            self.vae_pool = WorkerPool(
                worker_type=WorkerType.VAE,
                device=self.vae_device,
                precision=VAE_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.vae_pool.start()
            logger.info(f"[VAE] Worker pool started on {self.vae_device}")
        
        elif service in (ServiceType.CLIP_ONLY, CoreServiceType.CLIP_ONLY) or "CLIP" in str(service):
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=self.clip_device,
                precision=CLIP_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.clip_pool.start()
            logger.info(f"[CLIP] Worker pool started on {self.clip_device}")
        
        # Start WebSocket monitoring
        self.ws_server = WebSocketServer(
            status_provider=self.get_info,
            host=self.host,
            port=self.ws_port
        )
        self.ws_server.start()
        logger.info(f"[WS] Monitoring server started on ws://{self.host}:{self.ws_port}")
        
        # Start socket server
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._server_socket.settimeout(1.0)
        
        logger.info(f"[Socket] Server listening on {self.host}:{self.port}")
        logger.info("=" * 60)
        logger.info("Luna Daemon Ready!")
        logger.info("=" * 60)
        
        # Accept loop
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
                self._client_threads.add(thread)
                
                # Clean up finished threads
                self._client_threads = {t for t in self._client_threads if t.is_alive()}
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def stop(self):
        """Stop the daemon server."""
        logger.info("Luna Daemon stopping...")
        self._running = False
        
        # Stop pools
        if self.vae_pool:
            self.vae_pool.stop()
        if self.clip_pool:
            self.clip_pool.stop()
        
        # Stop WebSocket
        if self.ws_server:
            self.ws_server.stop()
        
        # Close socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
        
        logger.info("Luna Daemon stopped")
    
    def run(self):
        """Alias for start() - for backward compatibility with tray app."""
        self.start()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    daemon = LunaDaemon()
    
    try:
        daemon.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
