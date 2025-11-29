"""
Luna VAE/CLIP Daemon Server v2 - Dynamic Worker Scaling
Intelligently scales VAE and CLIP workers based on demand and available VRAM.

Features:
- Starts with 1 CLIP worker and 1 VAE worker
- Automatically spins up additional workers when queue backs up
- Spins down idle workers after configurable timeout
- VRAM-aware: won't scale beyond available memory
- Separate scaling for CLIP (fast) vs VAE (slower, needs more workers)

Usage:
    python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server_v2
"""

import os
import sys
import socket
import pickle
import threading
import time
import queue
import logging
import json
import asyncio
import hashlib
import base64
import struct
from typing import Any, Dict, Tuple, Optional, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

# Add ComfyUI to path if needed
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

# Add daemon folder to path for direct script execution
daemon_path = os.path.dirname(os.path.abspath(__file__))
if daemon_path not in sys.path:
    sys.path.insert(0, daemon_path)

# Try relative import first (when used as module), fall back to direct import
try:
    from .config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_WS_PORT, SHARED_DEVICE,
        VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
        MAX_WORKERS, LOG_LEVEL, MODEL_PRECISION,
        VRAM_LIMIT_GB, VRAM_SAFETY_MARGIN_GB,
        MAX_VAE_WORKERS, MAX_CLIP_WORKERS, MIN_VAE_WORKERS, MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD, SCALE_UP_DELAY_SEC, IDLE_TIMEOUT_SEC
    )
except ImportError:
    from config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_WS_PORT, SHARED_DEVICE,
        VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
        MAX_WORKERS, LOG_LEVEL, MODEL_PRECISION,
        VRAM_LIMIT_GB, VRAM_SAFETY_MARGIN_GB,
        MAX_VAE_WORKERS, MAX_CLIP_WORKERS, MIN_VAE_WORKERS, MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD, SCALE_UP_DELAY_SEC, IDLE_TIMEOUT_SEC
    )

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [Daemon] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration for Dynamic Scaling
# ============================================================================

@dataclass
class ScalingConfig:
    """Configuration for dynamic worker scaling"""
    # VRAM limits (GB)
    vram_limit_gb: float = VRAM_LIMIT_GB
    vram_safety_margin_gb: float = VRAM_SAFETY_MARGIN_GB
    
    # Model sizes in VRAM (bf16) - will be adjusted at runtime
    vae_size_gb: float = 0.082  # ~82 MB for VAE
    clip_size_gb: float = 1.6   # ~1.6 GB for CLIP-L + CLIP-G combined
    
    # Worker limits
    max_vae_workers: int = MAX_VAE_WORKERS
    max_clip_workers: int = MAX_CLIP_WORKERS
    min_vae_workers: int = MIN_VAE_WORKERS
    min_clip_workers: int = MIN_CLIP_WORKERS
    
    # Scaling triggers
    queue_threshold: int = QUEUE_THRESHOLD
    scale_up_delay_sec: float = SCALE_UP_DELAY_SEC
    
    # Idle timeout (seconds)
    idle_timeout_sec: float = IDLE_TIMEOUT_SEC
    
    # How often to check for scaling decisions
    scaling_check_interval_sec: float = 0.25


class WorkerType(Enum):
    VAE = "vae"
    CLIP = "clip"


# ============================================================================
# Worker Classes
# ============================================================================

class ModelWorker:
    """A single worker that holds a model and processes requests"""
    
    def __init__(
        self,
        worker_id: int,
        worker_type: WorkerType,
        device: str,
        precision: str,
        request_queue: queue.Queue,
        result_queues: Dict[int, queue.Queue]
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.request_queue = request_queue
        self.result_queues = result_queues
        
        self.model = None
        self.is_running = False
        self.is_loaded = False
        self.last_active = time.time()
        self.request_count = 0
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # For CLIP, we also need paths
        self.clip_l_path = CLIP_L_PATH
        self.clip_g_path = CLIP_G_PATH
        self.embeddings_dir = EMBEDDINGS_DIR
    
    @property
    def dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        return torch.float32
    
    def _convert_state_dict_precision(self, sd: dict) -> dict:
        """Convert state dict tensors to target precision"""
        if self.precision == "fp32":
            return sd
        
        converted = {}
        for key, value in sd.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                converted[key] = value.to(self.dtype)
            else:
                converted[key] = value
        return converted
    
    def load_model(self):
        """Load the model for this worker"""
        import comfy.sd
        import comfy.utils
        
        if self.worker_type == WorkerType.VAE:
            logger.info(f"[VAE-{self.worker_id}] Loading VAE model...")
            sd = comfy.utils.load_torch_file(VAE_PATH)
            if self.precision != "fp32":
                sd = self._convert_state_dict_precision(sd)
            self.model = comfy.sd.VAE(sd=sd)
            logger.info(f"[VAE-{self.worker_id}] VAE loaded ({self.precision})")
            
        elif self.worker_type == WorkerType.CLIP:
            logger.info(f"[CLIP-{self.worker_id}] Loading CLIP model...")
            clip_paths = []
            if os.path.exists(self.clip_l_path):
                clip_paths.append(self.clip_l_path)
            if os.path.exists(self.clip_g_path):
                clip_paths.append(self.clip_g_path)
            
            emb_dir = self.embeddings_dir if os.path.exists(self.embeddings_dir) else None
            self.model = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=emb_dir
            )
            
            # Convert to target precision
            if self.precision != "fp32" and hasattr(self.model, 'cond_stage_model'):
                self.model.cond_stage_model.to(self.dtype)
            
            logger.info(f"[CLIP-{self.worker_id}] CLIP loaded ({self.precision})")
        
        self.is_loaded = True
        torch.cuda.empty_cache()
    
    def unload_model(self):
        """Unload the model to free VRAM"""
        if self.model is not None:
            logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Unloading model...")
            del self.model
            self.model = None
            self.is_loaded = False
            torch.cuda.empty_cache()
    
    def process_vae_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode image pixels to latent space"""
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        latents = self.model.encode(pixels)
        return latents.cpu()
    
    def process_vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent space to image pixels"""
        pixels = self.model.decode(latents)
        return pixels.cpu()
    
    def process_clip_encode(self, positive: str, negative: str = "") -> Tuple:
        """Encode text to CLIP conditioning"""
        tokens_pos = self.model.tokenize(positive)
        cond, pooled = self.model.encode_from_tokens(tokens_pos, return_pooled=True)
        
        tokens_neg = self.model.tokenize(negative if negative else "")
        uncond, pooled_neg = self.model.encode_from_tokens(tokens_neg, return_pooled=True)
        
        return (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
    
    def process_clip_encode_sdxl(
        self,
        positive: str,
        negative: str = "",
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024
    ) -> Tuple[list, list]:
        """Encode text with SDXL-specific size conditioning"""
        tokens_pos = self.model.tokenize(positive)
        tokens_neg = self.model.tokenize(negative if negative else "")
        
        cond, pooled = self.model.encode_from_tokens(tokens_pos, return_pooled=True)
        uncond, pooled_neg = self.model.encode_from_tokens(tokens_neg, return_pooled=True)
        
        positive_out = [[
            cond.cpu(),
            {
                "pooled_output": pooled.cpu(),
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height
            }
        ]]
        
        negative_out = [[
            uncond.cpu(),
            {
                "pooled_output": pooled_neg.cpu(),
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height
            }
        ]]
        
        return (positive_out, negative_out)
    
    def run(self):
        """Main worker loop - process requests from queue"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Wait for request with timeout (allows checking is_running)
                try:
                    request_id, cmd, data = self.request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.last_active = time.time()
                self.request_count += 1
                
                try:
                    # Process based on command type
                    if self.worker_type == WorkerType.VAE:
                        if cmd == "vae_encode":
                            result = self.process_vae_encode(data["pixels"])
                        elif cmd == "vae_decode":
                            result = self.process_vae_decode(data["latents"])
                        else:
                            result = {"error": f"Unknown VAE command: {cmd}"}
                    
                    elif self.worker_type == WorkerType.CLIP:
                        if cmd == "clip_encode":
                            result = self.process_clip_encode(
                                data["positive"],
                                data.get("negative", "")
                            )
                        elif cmd == "clip_encode_sdxl":
                            result = self.process_clip_encode_sdxl(
                                data["positive"],
                                data.get("negative", ""),
                                data.get("width", 1024),
                                data.get("height", 1024),
                                data.get("crop_w", 0),
                                data.get("crop_h", 0),
                                data.get("target_width", 1024),
                                data.get("target_height", 1024)
                            )
                        else:
                            result = {"error": f"Unknown CLIP command: {cmd}"}
                    
                except Exception as e:
                    logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Error: {e}")
                    result = {"error": str(e)}
                
                # Send result back
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker error: {e}")
    
    def start(self):
        """Start the worker thread"""
        if not self.is_loaded:
            self.load_model()
        
        self.thread = threading.Thread(
            target=self.run,
            name=f"{self.worker_type.value}-worker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker started")
    
    def stop(self):
        """Stop the worker thread and unload model"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.unload_model()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker stopped")


# ============================================================================
# Worker Pool Manager
# ============================================================================

class WorkerPool:
    """Manages a pool of workers with dynamic scaling"""
    
    def __init__(
        self,
        worker_type: WorkerType,
        device: str,
        precision: str,
        config: ScalingConfig,
        on_scale_event: Optional[Callable[[str, dict], None]] = None
    ):
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.config = config
        self.on_scale_event = on_scale_event  # Callback for scaling events
        
        self.workers: List[ModelWorker] = []
        self.request_queue: queue.Queue = queue.Queue()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.next_request_id = 0
        self.lock = threading.Lock()
        
        self._next_worker_id = 0
        self._scaling_thread: Optional[threading.Thread] = None
        self._running = False
    
    @property
    def min_workers(self) -> int:
        if self.worker_type == WorkerType.VAE:
            return self.config.min_vae_workers
        return self.config.min_clip_workers
    
    @property
    def max_workers(self) -> int:
        if self.worker_type == WorkerType.VAE:
            return self.config.max_vae_workers
        return self.config.max_clip_workers
    
    @property
    def model_size_gb(self) -> float:
        if self.worker_type == WorkerType.VAE:
            return self.config.vae_size_gb
        return self.config.clip_size_gb
    
    def get_available_vram_gb(self) -> float:
        """Get available VRAM in GB"""
        if 'cuda' not in self.device:
            return float('inf')
        
        device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        used = torch.cuda.memory_allocated(device_idx) / 1024**3
        return total - used
    
    def can_scale_up(self) -> bool:
        """Check if we can add another worker"""
        if len(self.workers) >= self.max_workers:
            return False
        
        available = self.get_available_vram_gb()
        needed = self.model_size_gb + self.config.vram_safety_margin_gb
        return available >= needed
    
    def scale_up(self) -> Optional[ModelWorker]:
        """Add a new worker to the pool"""
        with self.lock:
            if not self.can_scale_up():
                return None
            
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            
            worker = ModelWorker(
                worker_id=worker_id,
                worker_type=self.worker_type,
                device=self.device,
                precision=self.precision,
                request_queue=self.request_queue,
                result_queues=self.result_queues
            )
            
            worker.start()
            self.workers.append(worker)
            
            vram_available = self.get_available_vram_gb()
            logger.info(
                f"[{self.worker_type.value.upper()}] Scaled UP to {len(self.workers)} workers "
                f"(VRAM available: {vram_available:.2f} GB)"
            )
            
            # Broadcast scaling event
            if self.on_scale_event:
                self.on_scale_event("scale_up", {
                    "pool": self.worker_type.value,
                    "worker_id": worker_id,
                    "active_workers": len(self.workers),
                    "vram_available_gb": round(vram_available, 2)
                })
            
            return worker
    
    def scale_down(self, worker: ModelWorker):
        """Remove an idle worker from the pool"""
        with self.lock:
            if len(self.workers) <= self.min_workers:
                return
            
            if worker in self.workers:
                worker_id = worker.worker_id
                worker.stop()
                self.workers.remove(worker)
                
                vram_available = self.get_available_vram_gb()
                logger.info(
                    f"[{self.worker_type.value.upper()}] Scaled DOWN to {len(self.workers)} workers "
                    f"(VRAM available: {vram_available:.2f} GB)"
                )
                
                # Broadcast scaling event
                if self.on_scale_event:
                    self.on_scale_event("scale_down", {
                        "pool": self.worker_type.value,
                        "worker_id": worker_id,
                        "active_workers": len(self.workers),
                        "vram_available_gb": round(vram_available, 2)
                    })
    
    def _scaling_loop(self):
        """Background thread that monitors and scales workers"""
        last_scale_up_check = 0
        queue_was_backed_up = False
        
        while self._running:
            time.sleep(self.config.scaling_check_interval_sec)
            
            now = time.time()
            queue_depth = self.request_queue.qsize()
            
            # Scale UP check
            if queue_depth > self.config.queue_threshold:
                if not queue_was_backed_up:
                    queue_was_backed_up = True
                    last_scale_up_check = now
                elif now - last_scale_up_check >= self.config.scale_up_delay_sec:
                    if self.can_scale_up():
                        self.scale_up()
                    last_scale_up_check = now
            else:
                queue_was_backed_up = False
            
            # Scale DOWN check - find idle workers
            with self.lock:
                idle_workers = [
                    w for w in self.workers
                    if now - w.last_active > self.config.idle_timeout_sec
                ]
            
            # Only scale down one at a time, and keep minimum
            for worker in idle_workers:
                if len(self.workers) > self.min_workers:
                    self.scale_down(worker)
                    break  # One at a time
    
    def submit(self, cmd: str, data: dict) -> Any:
        """Submit a request and wait for result"""
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            self.result_queues[request_id] = queue.Queue()
        
        # Submit to work queue
        self.request_queue.put((request_id, cmd, data))
        
        # Wait for result
        try:
            result = self.result_queues[request_id].get(timeout=60.0)
        finally:
            with self.lock:
                del self.result_queues[request_id]
        
        return result
    
    def start(self):
        """Start the pool with minimum workers"""
        self._running = True
        
        # Start minimum workers
        for _ in range(self.min_workers):
            self.scale_up()
        
        # Start scaling monitor thread
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            name=f"{self.worker_type.value}-scaler",
            daemon=True
        )
        self._scaling_thread.start()
    
    def stop(self):
        """Stop all workers"""
        self._running = False
        
        with self.lock:
            for worker in self.workers:
                worker.stop()
            self.workers.clear()
    
    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self.lock:
            return {
                "type": self.worker_type.value,
                "active_workers": len(self.workers),
                "queue_depth": self.request_queue.qsize(),
                "total_requests": sum(w.request_count for w in self.workers),
                "worker_ids": [w.worker_id for w in self.workers],
            }


# ============================================================================
# WebSocket Server for Status Monitoring (LUNA-Narrates Compatible)
# ============================================================================

class WebSocketServer:
    """
    Simple WebSocket server for daemon status monitoring.
    Compatible with LUNA-Narrates monitoring pattern.
    
    Message Types (similar to ComfyUI):
    - {"type": "status", "data": {...}}      - Periodic status updates
    - {"type": "scaling", "data": {...}}     - Worker scale up/down events  
    - {"type": "request", "data": {...}}     - Request started/completed
    - {"type": "error", "data": {...}}       - Error events
    """
    
    def __init__(self, daemon: 'DynamicDaemon', host: str, port: int):
        self.daemon = daemon
        self.host = host
        self.port = port
        self.clients: Set[socket.socket] = set()
        self.clients_lock = threading.Lock()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._broadcast_thread: Optional[threading.Thread] = None
    
    def _create_accept_key(self, key: str) -> str:
        """Create WebSocket accept key from client key"""
        GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        sha1 = hashlib.sha1((key + GUID).encode()).digest()
        return base64.b64encode(sha1).decode()
    
    def _handshake(self, conn: socket.socket) -> bool:
        """Perform WebSocket handshake"""
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                return False
            
            # Parse headers
            headers = {}
            lines = data.split('\r\n')
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Check for WebSocket upgrade
            if headers.get('upgrade', '').lower() != 'websocket':
                return False
            
            # Get client key
            client_key = headers.get('sec-websocket-key', '')
            if not client_key:
                return False
            
            # Send handshake response
            accept_key = self._create_accept_key(client_key)
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n"
                "\r\n"
            )
            conn.sendall(response.encode())
            return True
            
        except Exception as e:
            logger.error(f"WebSocket handshake error: {e}")
            return False
    
    def _encode_frame(self, data: str) -> bytes:
        """Encode data as WebSocket text frame"""
        payload = data.encode('utf-8')
        length = len(payload)
        
        if length <= 125:
            frame = bytes([0x81, length]) + payload
        elif length <= 65535:
            frame = bytes([0x81, 126]) + struct.pack('>H', length) + payload
        else:
            frame = bytes([0x81, 127]) + struct.pack('>Q', length) + payload
        
        return frame
    
    def _decode_frame(self, conn: socket.socket) -> Optional[str]:
        """Decode incoming WebSocket frame"""
        try:
            header = conn.recv(2)
            if len(header) < 2:
                return None
            
            opcode = header[0] & 0x0F
            
            # Close frame
            if opcode == 0x08:
                return None
            
            # Ping - send pong
            if opcode == 0x09:
                conn.sendall(bytes([0x8A, 0]))
                return ""
            
            masked = (header[1] & 0x80) != 0
            length = header[1] & 0x7F
            
            if length == 126:
                length = struct.unpack('>H', conn.recv(2))[0]
            elif length == 127:
                length = struct.unpack('>Q', conn.recv(8))[0]
            
            if masked:
                mask = conn.recv(4)
                data = bytearray(conn.recv(length))
                for i in range(length):
                    data[i] ^= mask[i % 4]
                return data.decode('utf-8')
            else:
                return conn.recv(length).decode('utf-8')
                
        except Exception:
            return None
    
    def broadcast(self, message_type: str, data: dict):
        """Broadcast message to all connected clients"""
        message = json.dumps({"type": message_type, "data": data})
        frame = self._encode_frame(message)
        
        with self.clients_lock:
            dead_clients = []
            for client in self.clients:
                try:
                    client.sendall(frame)
                except Exception:
                    dead_clients.append(client)
            
            # Clean up dead connections
            for client in dead_clients:
                self.clients.discard(client)
                try:
                    client.close()
                except:
                    pass
    
    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle a single WebSocket client connection"""
        if not self._handshake(conn):
            conn.close()
            return
        
        with self.clients_lock:
            self.clients.add(conn)
        
        logger.info(f"WebSocket client connected: {addr}")
        
        # Send initial status
        try:
            status = self.daemon.get_info()
            message = json.dumps({"type": "status", "data": status})
            conn.sendall(self._encode_frame(message))
        except Exception as e:
            logger.error(f"Error sending initial status: {e}")
        
        # Keep connection alive and handle incoming messages
        try:
            while self._running:
                conn.settimeout(1.0)
                try:
                    data = self._decode_frame(conn)
                    if data is None:  # Connection closed
                        break
                    
                    # Handle client messages (e.g., explicit status request)
                    if data:
                        try:
                            msg = json.loads(data)
                            if msg.get("type") == "get_status":
                                status = self.daemon.get_info()
                                response = json.dumps({"type": "status", "data": status})
                                conn.sendall(self._encode_frame(response))
                        except json.JSONDecodeError:
                            pass
                            
                except socket.timeout:
                    continue
                except Exception:
                    break
                    
        finally:
            with self.clients_lock:
                self.clients.discard(conn)
            try:
                conn.close()
            except:
                pass
            logger.info(f"WebSocket client disconnected: {addr}")
    
    def _broadcast_loop(self):
        """Periodically broadcast status to all clients"""
        while self._running:
            time.sleep(1.0)  # Broadcast every second
            
            if self.clients:
                try:
                    status = self.daemon.get_info()
                    self.broadcast("status", status)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
    
    def start(self):
        """Start the WebSocket server"""
        self._running = True
        
        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start broadcast thread
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="ws-broadcast",
            daemon=True
        )
        self._broadcast_thread.start()
        
        # Accept loop in separate thread
        def accept_loop():
            while self._running:
                try:
                    conn, addr = self._server_socket.accept()
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"WebSocket accept error: {e}")
        
        threading.Thread(target=accept_loop, name="ws-accept", daemon=True).start()
        logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the WebSocket server"""
        self._running = False
        
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass


# ============================================================================
# Main Daemon Server
# ============================================================================

class DynamicDaemon:
    """Main daemon server with dynamic worker scaling"""
    
    def __init__(self, device: str = SHARED_DEVICE, precision: str = MODEL_PRECISION):
        self.device = device
        self.precision = precision
        self.config = ScalingConfig()
        
        # Adjust model sizes based on precision
        if precision == "bf16" or precision == "fp16":
            # Half the fp32 sizes
            self.config.vae_size_gb = 0.082  # 164MB / 2
            self.config.clip_size_gb = 1.6   # (2.72GB + 483MB) / 2
        else:
            self.config.vae_size_gb = 0.164
            self.config.clip_size_gb = 3.2
        
        self.vae_pool: Optional[WorkerPool] = None
        self.clip_pool: Optional[WorkerPool] = None
        self.ws_server: Optional[WebSocketServer] = None
        
        self.start_time = time.time()
        self.request_count = 0
    
    def _on_scale_event(self, event_type: str, data: dict):
        """Callback for worker pool scaling events - broadcasts to WebSocket clients"""
        if self.ws_server:
            self.ws_server.broadcast("scaling", {
                "event": event_type,
                **data
            })
    
    def start_pools(self):
        """Initialize and start worker pools"""
        logger.info(f"Starting worker pools on {self.device} ({self.precision})...")
        
        self.vae_pool = WorkerPool(
            worker_type=WorkerType.VAE,
            device=self.device,
            precision=self.precision,
            config=self.config,
            on_scale_event=self._on_scale_event
        )
        
        self.clip_pool = WorkerPool(
            worker_type=WorkerType.CLIP,
            device=self.device,
            precision=self.precision,
            config=self.config,
            on_scale_event=self._on_scale_event
        )
        
        # Start pools (loads initial workers)
        self.clip_pool.start()  # CLIP first (larger)
        self.vae_pool.start()   # Then VAE
        
        # Report VRAM usage
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            used = torch.cuda.memory_allocated(device_idx) / 1024**3
            total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
            logger.info(f"Initial VRAM usage: {used:.2f} / {total:.2f} GB")
    
    def get_info(self) -> dict:
        """Get daemon status info"""
        info = {
            "status": "ok",
            "version": "2.0-dynamic",
            "device": self.device,
            "precision": self.precision,
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_count,
        }
        
        if self.vae_pool:
            info["vae_pool"] = self.vae_pool.get_stats()
        if self.clip_pool:
            info["clip_pool"] = self.clip_pool.get_stats()
        
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            info["vram_used_gb"] = torch.cuda.memory_allocated(device_idx) / 1024**3
            info["vram_total_gb"] = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        
        return info
    
    def handle_request(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming request from ComfyUI node"""
        try:
            # Receive data with end marker
            data = b""
            while True:
                chunk = conn.recv(1048576)  # 1MB chunks
                if not chunk:
                    break
                data += chunk
                if b"<<END>>" in data:
                    data = data.replace(b"<<END>>", b"")
                    break
            
            if not data:
                return
            
            request = pickle.loads(data)
            cmd = request.get("cmd", "unknown")
            
            self.request_count += 1
            logger.debug(f"Request #{self.request_count}: {cmd}")
            
            # Route command
            if cmd == "health":
                result = {"status": "ok"}
            elif cmd == "info":
                result = self.get_info()
            elif cmd in ("vae_encode", "vae_decode"):
                result = self.vae_pool.submit(cmd, request)
            elif cmd in ("clip_encode", "clip_encode_sdxl"):
                result = self.clip_pool.submit(cmd, request)
            else:
                result = {"error": f"Unknown command: {cmd}"}
            
            # Send response
            response = pickle.dumps(result)
            conn.sendall(response)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                conn.sendall(pickle.dumps({"error": str(e)}))
            except:
                pass
        finally:
            conn.close()
    
    def run(self):
        """Main server loop"""
        # Start worker pools
        self.start_pools()
        
        # Start WebSocket server for monitoring
        self.ws_server = WebSocketServer(self, DAEMON_HOST, DAEMON_WS_PORT)
        self.ws_server.start()
        
        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((DAEMON_HOST, DAEMON_PORT))
            server.listen(MAX_WORKERS * 2)
            
            logger.info(f"Socket server: {DAEMON_HOST}:{DAEMON_PORT}")
            logger.info(f"WebSocket monitor: ws://{DAEMON_HOST}:{DAEMON_WS_PORT}")
            logger.info("Dynamic scaling enabled:")
            logger.info(f"  VAE: {self.config.min_vae_workers}-{self.config.max_vae_workers} workers")
            logger.info(f"  CLIP: {self.config.min_clip_workers}-{self.config.max_clip_workers} workers")
            logger.info(f"  Idle timeout: {self.config.idle_timeout_sec}s")
            logger.info("Ready to accept connections!")
            logger.info("Press Ctrl+C to stop")
            
            while True:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=self.handle_request,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.ws_server:
                self.ws_server.stop()
            if self.vae_pool:
                self.vae_pool.stop()
            if self.clip_pool:
                self.clip_pool.stop()
            server.close()


def main():
    """Entry point"""
    print("=" * 60)
    print("  Luna VAE/CLIP Daemon v2")
    print("  Dynamic Worker Scaling + WebSocket Monitoring")
    print(f"  Device: {SHARED_DEVICE} | Precision: {MODEL_PRECISION}")
    print(f"  Socket: {DAEMON_HOST}:{DAEMON_PORT}")
    print(f"  WebSocket: ws://{DAEMON_HOST}:{DAEMON_WS_PORT}")
    print("=" * 60)
    print()
    
    daemon = DynamicDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
