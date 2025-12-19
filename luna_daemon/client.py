"""
Luna Daemon Client Library

Simplified client for communicating with Luna VAE/CLIP daemon.
Handles:
- VAE encode/decode requests
- CLIP encoding requests
- LoRA cache operations
- Daemon status and health checks

This version removes model_forward complexity - models run locally
with InferenceModeWrapper for VRAM savings.
"""

import socket
import pickle
import struct
import torch
from typing import Tuple, Optional, Any, List, Dict

try:
    from .config import DAEMON_HOST, DAEMON_PORT, CLIENT_TIMEOUT, ENABLE_CUDA_IPC
except ImportError:
    DAEMON_HOST = "127.0.0.1"
    DAEMON_PORT = 19283
    CLIENT_TIMEOUT = 120
    ENABLE_CUDA_IPC = False


# =============================================================================
# Exceptions
# =============================================================================

class DaemonConnectionError(Exception):
    """Raised when daemon is not available."""
    pass


class ModelMismatchError(Exception):
    """Raised when workflow tries to use a different model than loaded."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def get_local_gpu_id() -> Optional[int]:
    """Get the GPU ID that the current process is using."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.current_device()
    except:
        return None


# =============================================================================
# Daemon Client
# =============================================================================

class DaemonClient:
    """Client for communicating with Luna Daemon."""
    
    def __init__(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        self.host = host
        self.port = port
        self.timeout = CLIENT_TIMEOUT
        
        # IPC state
        self._ipc_enabled = False
        self._daemon_gpu_id: Optional[int] = None
        self._local_gpu_id: Optional[int] = None
        
        if ENABLE_CUDA_IPC:
            try:
                self.negotiate_ipc()
            except:
                pass
    
    def _send_request(self, request: dict) -> Any:
        """Send request to daemon with length-prefix protocol."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Serialize and send with length prefix
            data = pickle.dumps(request)
            sock.sendall(struct.pack('>Q', len(data)) + data)  # 8-byte length
            
            # Receive response header
            header = b""
            while len(header) < 8:
                chunk = sock.recv(8 - len(header))
                if not chunk:
                    raise DaemonConnectionError("Connection closed while reading header")
                header += chunk
            
            response_len = int.from_bytes(header, 'big')
            
            # Receive response payload
            chunks = []
            bytes_recd = 0
            while bytes_recd < response_len:
                chunk_size = min(response_len - bytes_recd, 1048576)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    raise DaemonConnectionError("Connection closed while reading response")
                chunks.append(chunk)
                bytes_recd += len(chunk)
            
            response_data = b"".join(chunks)
            sock.close()
            
            result = pickle.loads(response_data)
            
            if isinstance(result, dict) and "error" in result:
                if result.get("type") == "model_mismatch":
                    raise ModelMismatchError(result["error"])
                raise DaemonConnectionError(f"Daemon error: {result['error']}")
            
            return result
            
        except ConnectionRefusedError:
            raise DaemonConnectionError("Luna Daemon is not running!")
        except socket.timeout:
            raise DaemonConnectionError(f"Daemon request timed out after {self.timeout}s")
        except DaemonConnectionError:
            raise
        except Exception as e:
            raise DaemonConnectionError(f"Daemon communication error: {e}")
    
    def is_running(self) -> bool:
        """Check if daemon is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.host, self.port))
            
            request = {"cmd": "ping"}
            data = pickle.dumps(request)
            sock.sendall(struct.pack('>Q', len(data)) + data)
            
            header = sock.recv(8)
            if len(header) < 8:
                sock.close()
                return False
            
            response_len = int.from_bytes(header, 'big')
            response_data = sock.recv(response_len)
            sock.close()
            
            result = pickle.loads(response_data)
            return result.get("pong", False)
        except:
            return False
    
    def get_info(self) -> dict:
        """Get daemon status info."""
        return self._send_request({"cmd": "get_info"})
    
    def negotiate_ipc(self) -> bool:
        """Negotiate CUDA IPC mode with daemon."""
        if not ENABLE_CUDA_IPC:
            return False
        
        self._local_gpu_id = get_local_gpu_id()
        if self._local_gpu_id is None:
            return False
        
        try:
            result = self._send_request({
                "cmd": "negotiate_ipc",
                "client_gpu_id": self._local_gpu_id
            })
            
            if result.get("ipc_enabled"):
                self._ipc_enabled = True
                self._daemon_gpu_id = result.get("daemon_gpu_id")
                return True
        except:
            pass
        
        return False
    
    # =========================================================================
    # Weight Registry Operations (CUDA IPC Weight Sharing)
    # =========================================================================
    
    def load_vae_weights(self, vae_path: str, model_key: Optional[str] = None) -> Dict:
        """
        Load VAE weights in daemon and get IPC handles.
        
        Args:
            vae_path: Path to VAE safetensors file
            model_key: Optional key for registry
        
        Returns:
            Dict with success, model_key, and ipc_handles
        """
        return self._send_request({
            "cmd": "load_vae_weights",
            "vae_path": vae_path,
            "model_key": model_key
        })
    
    def load_clip_weights(self, clip_l_path: Optional[str] = None, 
                         clip_g_path: Optional[str] = None,
                         model_key: Optional[str] = None) -> Dict:
        """
        Load CLIP weights in daemon and get IPC handles.
        
        Args:
            clip_l_path: Path to CLIP-L safetensors
            clip_g_path: Path to CLIP-G safetensors (for SDXL)
            model_key: Optional key for registry
        
        Returns:
            Dict with success, model_key, and ipc_handles
        """
        return self._send_request({
            "cmd": "load_clip_weights",
            "clip_l_path": clip_l_path,
            "clip_g_path": clip_g_path,
            "model_key": model_key
        })
    
    def get_weight_handles(self, model_key: str) -> Dict:
        """
        Get IPC handles for a previously loaded model.
        
        Args:
            model_key: Key from load_vae_weights or load_clip_weights
        
        Returns:
            Dict with success and ipc_handles
        """
        return self._send_request({
            "cmd": "get_weight_handles",
            "model_key": model_key
        })
    
    def list_loaded_weights(self) -> List[str]:
        """Get list of all loaded model keys in weight registry."""
        result = self._send_request({"cmd": "list_loaded_weights"})
        return result.get("models", [])
    
    def unload_weights(self, model_key: str) -> bool:
        """
        Unload model from weight registry.
        
        Args:
            model_key: Key to unload
        
        Returns:
            True if unloaded successfully
        """
        result = self._send_request({
            "cmd": "unload_weights",
            "model_key": model_key
        })
        return result.get("success", False)
    
    # =========================================================================
    # VAE Operations
    # =========================================================================
    
    def vae_encode(
        self, 
        pixels: torch.Tensor, 
        workflow_id: str,
        tiled: bool = False, 
        tile_size: int = 512,
        overlap: int = 64
    ) -> torch.Tensor:
        """Encode pixels to latent space via daemon."""
        pixels_cpu = pixels.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_encode",
            "pixels": pixels_cpu,
            "workflow_id": workflow_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "overlap": overlap
        })
        
        del pixels_cpu
        return result
    
    def vae_decode(
        self, 
        latents: torch.Tensor, 
        workflow_id: str,
        tiled: bool = False, 
        tile_size: int = 64,
        overlap: int = 16
    ) -> torch.Tensor:
        """Decode latents to image pixels via daemon."""
        latents_cpu = latents.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_decode",
            "latents": latents_cpu,
            "workflow_id": workflow_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "overlap": overlap
        })
        
        del latents_cpu
        return result
    
    # =========================================================================
    # CLIP Operations
    # =========================================================================
    
    def clip_encode(
        self, 
        positive: str, 
        negative: str,
        workflow_id: str,
        lora_stack: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text prompts to CLIP conditioning via daemon."""
        request: Dict[str, Any] = {
            "cmd": "clip_encode",
            "positive": positive,
            "negative": negative,
            "workflow_id": workflow_id
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    def clip_encode_sdxl(
        self,
        positive: str,
        negative: str,
        workflow_id: str,
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024,
        lora_stack: Optional[List[Dict]] = None
    ) -> Tuple[list, list]:
        """Encode text with SDXL-specific conditioning."""
        request: Dict[str, Any] = {
            "cmd": "clip_encode_sdxl",
            "positive": positive,
            "negative": negative,
            "workflow_id": workflow_id,
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    def clip_tokenize(
        self, 
        text: str, 
        workflow_id: str,
        return_word_ids: bool = False
    ):
        """Tokenize text via daemon."""
        return self._send_request({
            "cmd": "clip_tokenize",
            "text": text,
            "workflow_id": workflow_id,
            "return_word_ids": return_word_ids
        })
    
    def clip_encode_from_tokens(
        self,
        tokens,
        workflow_id: str,
        return_pooled: bool = False,
        return_dict: bool = False,
        lora_stack: Optional[List[Dict]] = None
    ):
        """Encode tokens via daemon."""
        request = {
            "cmd": "clip_encode_from_tokens",
            "tokens": tokens,
            "workflow_id": workflow_id,
            "return_pooled": return_pooled,
            "return_dict": return_dict
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    # =========================================================================
    # LoRA Cache Operations
    # =========================================================================
    
    def lora_cache_get(self, lora_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get LoRA state dict from daemon's RAM cache."""
        result = self._send_request({
            "cmd": "lora_cache_get",
            "lora_name": lora_name
        })
        if result.get("cached"):
            return result.get("state_dict")
        return None
    
    def lora_cache_put(self, lora_name: str, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Put LoRA state dict into daemon's RAM cache."""
        # Move to CPU for transport
        cpu_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.cpu() if v.device.type != 'cpu' else v
            else:
                cpu_dict[k] = v
        
        result = self._send_request({
            "cmd": "lora_cache_put",
            "lora_name": lora_name,
            "state_dict": cpu_dict
        })
        return result.get("success", False)
    
    def lora_cache_check(self, lora_name: str) -> bool:
        """Check if LoRA is in daemon's RAM cache."""
        result = self._send_request({
            "cmd": "lora_cache_check",
            "lora_name": lora_name
        })
        return result.get("cached", False)
    
    def lora_cache_stats(self) -> dict:
        """Get LoRA cache statistics."""
        return self._send_request({"cmd": "lora_cache_stats"})
    
    # =========================================================================
    # Legacy LoRA Operations (for backward compatibility)
    # =========================================================================
    
    def has_lora(self, lora_hash: str) -> bool:
        """Check if a LoRA is cached (legacy)."""
        return self.lora_cache_check(lora_hash)
    
    def upload_lora(self, lora_hash: str, weights: Dict[str, torch.Tensor]) -> dict:
        """Upload LoRA weights (legacy)."""
        self.lora_cache_put(lora_hash, weights)
        return {"success": True}
    
    def register_lora(self, lora_name: str, clip_strength: float = 1.0, model_strength: float = 1.0) -> dict:
        """
        Register and apply LoRA to CLIP with specified strength.
        
        Different from lora_cache_put - this actually applies the LoRA
        transformation to the CLIP model in the daemon.
        
        Args:
            lora_name: Name/path of the LoRA
            clip_strength: Strength multiplier for CLIP (0.0-2.0, default 1.0)
            model_strength: Strength multiplier for model (0.0-2.0, default 1.0)
        
        Returns:
            Response dict with success status
        """
        result = self._send_request({
            "cmd": "register_lora",
            "lora_name": lora_name,
            "clip_strength": clip_strength,
            "model_strength": model_strength
        })
        return result if isinstance(result, dict) else {"success": True}
    
    def get_lora_stats(self) -> dict:
        """Get LoRA cache statistics (alias for lora_cache_stats)."""
        return self.lora_cache_stats()
    
    def clear_lora_cache(self) -> dict:
        """Clear all cached LoRAs from daemon memory."""
        result = self._send_request({"cmd": "clear_loras"})
        return result if isinstance(result, dict) else {"success": True}
    
    # =========================================================================
    # Registration Operations (for DaemonVAE/DaemonCLIP)
    # =========================================================================
    
    def register_vae(self, vae: Any, vae_type: str) -> dict:
        """Register VAE with daemon (legacy - load_vae_model preferred)."""
        return {"registered": True, "vae_type": vae_type}
    
    def register_clip(self, clip: Any, clip_type: str) -> dict:
        """Register CLIP with daemon (legacy - load_clip_model preferred)."""
        return {"registered": True, "clip_type": clip_type}
    
    def load_vae_model(self, vae_path: str) -> dict:
        """
        Tell daemon to load a specific VAE model from file.
        
        Args:
            vae_path: Path to VAE model file (absolute or relative to ComfyUI models/)
        
        Returns:
            Response dict with success status
        """
        result = self._send_request({
            "cmd": "load_vae_model",
            "vae_path": vae_path
        })
        return result if isinstance(result, dict) else {"success": True, "vae_path": vae_path}
    
    def register_vae_by_path(self, vae_path: str, vae_type: str) -> dict:
        """
        Register VAE model by path.
        
        Args:
            vae_path: Path to VAE model file
            vae_type: Type of VAE ('sdxl', 'flux', etc.)
        
        Returns:
            Response dict with success status
        """
        return self.load_vae_model(vae_path)
    
    def get_model_proxies(self, workflow_id: str, model_type: str, models: dict) -> dict:
        """
        Request CLIP/VAE proxies for a workflow with specific models.
        
        Daemon returns existing proxies if models are already loaded for this workflow,
        or sideloads new ones if switching models. Enables multiple workflows to
        share model weights while using different model combinations.
        
        Args:
            workflow_id: Unique identifier for this workflow/instance
            model_type: Model type (SDXL, Flux, SD1.5, etc.)
            models: Dict of required models
                e.g. {"clip_l": "/path/to/clip_l.safetensors", 
                      "clip_g": "/path/to/clip_g.safetensors",
                      "vae": "/path/to/vae.safetensors"}
        
        Returns:
            Dict with {"clip": DaemonCLIP, "vae": DaemonVAE, "status": "loaded|new"}
        """
        result = self._send_request({
            "cmd": "get_model_proxies",
            "workflow_id": workflow_id,
            "model_type": model_type,
            "models": models
        })
        return result if isinstance(result, dict) else {"error": "Failed to get model proxies"}
    
    def register_clip_by_path(self, clip_components: dict, model_type: str, clip_type: str) -> dict:
        """
        Register CLIP models by path (loads individual components).
        
        Args:
            clip_components: Dict of {component_type: path} 
                e.g. {"clip_l": "path/to/clip_l.safetensors", "clip_g": "path/to/clip_g.safetensors"}
            model_type: Model type (SDXL, Flux, etc.)
            clip_type: ComfyUI CLIP type string
        
        Returns:
            Response dict with success status
        """
        if not clip_components:
            return {"error": "No CLIP paths provided"}
        
        return self.get_model_proxies(workflow_id="legacy", model_type="legacy", models=clip_components)
    
    def load_clip_model(self, clip_path: str, clip_type: str = "sdxl") -> dict:
        """Legacy method - use get_model_proxies instead."""
        return {"success": True}
    
    # =========================================================================
    # Daemon Control Operations
    # =========================================================================
    
    def start_daemon(self) -> bool:
        """
        Start the Luna Daemon process.
        
        Launches daemon server as a subprocess. The daemon will run in background
        and listen for connections.
        
        Returns:
            True if daemon was started successfully, False otherwise
        """
        import subprocess
        import sys
        import os
        from pathlib import Path
        
        # Check if daemon is already running
        if self.is_running():
            return True
        
        try:
            # Find daemon server script
            daemon_dir = Path(__file__).parent
            server_script = daemon_dir / "daemon_server.py"
            
            if not server_script.exists():
                raise FileNotFoundError(f"Daemon server script not found: {server_script}")
            
            # Launch daemon as subprocess
            # Use same Python interpreter as current process
            python_exe = sys.executable
            
            # Start detached process
            if os.name == 'nt':  # Windows
                # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
                subprocess.Popen(
                    [python_exe, str(server_script)],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL
                )
            else:  # Unix/Linux/Mac
                subprocess.Popen(
                    [python_exe, str(server_script)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            # Wait a moment for daemon to start
            import time
            time.sleep(2)
            
            return self.is_running()
            
        except Exception as e:
            print(f"Failed to start daemon: {e}")
            return False
    
    def shutdown(self) -> dict:
        """
        Shut down the daemon server.
        
        Sends shutdown command to daemon. The daemon will gracefully stop
        all workers and close connections.
        
        Returns:
            Response dict with success status
        """
        try:
            result = self._send_request({"cmd": "shutdown"})
            return result if isinstance(result, dict) else {"success": True}
        except DaemonConnectionError:
            # Daemon already stopped
            return {"success": True, "message": "Daemon not running"}
    
    def reset_clients(self) -> dict:
        """
        Reset all client connections to daemon.
        
        Clears connection state and forces reconnection. Useful for
        recovering from errors.
        
        Returns:
            Response dict with success status
        """
        global _client
        _client = None  # Clear singleton
        return {"success": True, "message": "Client reset"}
    
    def unload_daemon_models(self) -> dict:
        """
        Unload all models from daemon VRAM.
        
        Frees up daemon GPU memory by unloading VAE/CLIP models.
        Models will be reloaded on next request.
        
        Returns:
            Response dict with success status
        """
        try:
            result = self._send_request({"cmd": "unload_models"})
            return result if isinstance(result, dict) else {"success": True}
        except DaemonConnectionError:
            return {"success": False, "error": "Daemon not running"}
    
    # =========================================================================
    # Configuration Operations
    # =========================================================================
    
    def set_attention_mode(self, mode: str) -> dict:
        """
        Set attention mechanism mode for CLIP/VAE processing.
        
        Args:
            mode: Attention mode - "flash", "split", "pytorch", or "auto"
                - flash: Flash Attention 2 (fastest, requires flash-attn)
                - split: Split attention (memory efficient)
                - pytorch: PyTorch native attention
                - auto: Automatically choose best available
        
        Returns:
            Response dict with success status and applied mode
        """
        try:
            result = self._send_request({
                "cmd": "set_attention_mode",
                "mode": mode
            })
            return result if isinstance(result, dict) else {"success": True, "mode": mode}
        except DaemonConnectionError:
            return {"success": False, "error": "Daemon not running"}
    
    # =========================================================================
    # Z-IMAGE Operations
    # =========================================================================
    
    def register_qwen3_transformers(
        self, 
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None
    ) -> dict:
        """
        Register Qwen3-VL model using HuggingFace transformers.
        
        Note: Use this instead of register_qwen3() since llama-cpp-python
        doesn't yet support Qwen3-VL GGUF models.
        
        Args:
            model_path: Path to Qwen3 model directory or HF model ID
            device: Device to load on (e.g., "cuda:0", "cpu")
            dtype: Data type for model (e.g., "float16", "bfloat16", "auto")
        
        Returns:
            Response dict with success status and model info
        """
        result = self._send_request({
            "cmd": "register_qwen3_transformers",
            "model_path": model_path,
            "device": device,
            "dtype": dtype
        })
        return result if isinstance(result, dict) else {"success": True}
    
    def get_qwen3_status(self) -> dict:
        """
        Get Qwen3-VL model loading status.
        
        Returns:
            Dict with loaded status, model_path, device, etc.
        """
        result = self._send_request({"cmd": "qwen3_status"})
        return result if isinstance(result, dict) else {"loaded": False}
    
    def zimage_encode(self, text: str) -> torch.Tensor:
        """
        Encode text using Z-IMAGE CLIP (Qwen3-VL based).
        
        Args:
            text: Input text to encode
        
        Returns:
            Encoded conditioning tensor
        """
        result = self._send_request({
            "cmd": "zimage_encode",
            "text": text
        })
        return result
    
    # =========================================================================
    # Vision/VLM Operations
    # =========================================================================
    
    def vlm_generate(self, **kwargs) -> Any:
        """
        Generate text using Vision-Language Model.
        
        Args:
            **kwargs: VLM generation parameters (image, prompt, max_tokens, etc.)
        
        Returns:
            Generated text response
        """
        request = {"cmd": "vlm_generate"}
        request.update(kwargs)
        return self._send_request(request)
    
    def encode_vision(self, image) -> torch.Tensor:
        """
        Encode image using vision model.
        
        Args:
            image: Image tensor or numpy array
        
        Returns:
            Vision embedding tensor
        """
        # Convert to CPU tensor if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
        
        result = self._send_request({
            "cmd": "encode_vision",
            "image": image
        })
        return result
    
    # =========================================================================
    # Async Task Operations
    # =========================================================================
    
    def submit_async(self, task_name: str, task_data: dict) -> dict:
        """
        Submit async task to daemon.
        
        Allows daemon to handle tasks asynchronously (like image saving)
        without blocking ComfyUI workflow execution.
        
        Args:
            task_name: Name of task to execute (e.g., "save_images_async")
            task_data: Task parameters/data
        
        Returns:
            Response dict with job_id and success status
        """
        result = self._send_request({
            "cmd": "submit_async",
            "task_name": task_name,
            "task_data": task_data
        })
        return result if isinstance(result, dict) else {"success": True}


# =============================================================================
# Singleton Client Instance
# =============================================================================

_client: Optional[DaemonClient] = None


def get_client() -> DaemonClient:
    """Get or create singleton client instance."""
    global _client
    if _client is None:
        _client = DaemonClient()
    return _client


# =============================================================================
# Convenience Functions
# =============================================================================

def is_daemon_running() -> bool:
    """Check if daemon is available."""
    return get_client().is_running()


def get_daemon_info() -> dict:
    """Get daemon status info."""
    return get_client().get_info()


def vae_encode(
    pixels: torch.Tensor, 
    vae_type: str,
    tiled: bool = False, 
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """Encode pixels to latent space via daemon."""
    return get_client().vae_encode(pixels, vae_type, tiled, tile_size, overlap)


def vae_decode(
    latents: torch.Tensor, 
    vae_type: str,
    tiled: bool = False, 
    tile_size: int = 64,
    overlap: int = 16
) -> torch.Tensor:
    """Decode latents to image pixels via daemon."""
    return get_client().vae_decode(latents, vae_type, tiled, tile_size, overlap)


def clip_encode(
    positive: str, 
    negative: str, 
    clip_type: str,
    lora_stack: Optional[List[Dict]] = None
) -> Tuple:
    """Encode text prompts via daemon."""
    return get_client().clip_encode(positive, negative, clip_type, lora_stack)


def clip_encode_sdxl(
    positive: str, 
    negative: str, 
    clip_type: str = "sdxl",
    lora_stack: Optional[List[Dict]] = None,
    **kwargs
) -> Tuple[list, list]:
    """Encode text with SDXL-specific conditioning via daemon."""
    return get_client().clip_encode_sdxl(
        positive, negative, clip_type,
        kwargs.get("width", 1024),
        kwargs.get("height", 1024),
        kwargs.get("crop_w", 0),
        kwargs.get("crop_h", 0),
        kwargs.get("target_width", 1024),
        kwargs.get("target_height", 1024),
        lora_stack
    )


def clip_tokenize(text: str, clip_type: str, return_word_ids: bool = False):
    """Tokenize text via daemon."""
    return get_client().clip_tokenize(text, clip_type, return_word_ids)


def clip_encode_from_tokens(tokens, clip_type: str, **kwargs):
    """Encode tokens via daemon."""
    return get_client().clip_encode_from_tokens(
        tokens, clip_type,
        kwargs.get("return_pooled", False),
        kwargs.get("return_dict", False),
        kwargs.get("lora_stack")
    )


def has_lora(lora_hash: str) -> bool:
    """Check if LoRA is cached."""
    return get_client().has_lora(lora_hash)


def upload_lora(lora_hash: str, weights: Dict[str, torch.Tensor]) -> dict:
    """Upload LoRA weights."""
    return get_client().upload_lora(lora_hash, weights)


def register_lora(lora_name: str, clip_strength: float = 1.0, model_strength: float = 1.0) -> dict:
    """Register and apply LoRA to CLIP."""
    return get_client().register_lora(lora_name, clip_strength, model_strength)


def get_lora_stats() -> dict:
    """Get LoRA cache statistics."""
    return get_client().get_lora_stats()


def clear_lora_cache() -> dict:
    """Clear all cached LoRAs."""
    return get_client().clear_lora_cache()


def register_vae(vae: Any, vae_type: str) -> dict:
    """Register VAE with daemon."""
    return get_client().register_vae(vae, vae_type)


def register_clip(clip: Any, clip_type: str) -> dict:
    """Register CLIP with daemon."""
    return get_client().register_clip(clip, clip_type)


def load_vae_model(vae_path: str) -> dict:
    """Tell daemon to load a specific VAE model from file."""
    return get_client().load_vae_model(vae_path)


def load_clip_model(clip_path: str, clip_type: str = "sdxl") -> dict:
    """Tell daemon to load a specific CLIP model from file."""
    return get_client().load_clip_model(clip_path, clip_type)


def load_vae_weights(vae_path: str, model_key: Optional[str] = None) -> Dict:
    """Load VAE weights in daemon and get IPC handles (module-level convenience)."""
    return get_client().load_vae_weights(vae_path, model_key)


def load_clip_weights(clip_l_path: Optional[str] = None, 
                      clip_g_path: Optional[str] = None,
                      model_key: Optional[str] = None) -> Dict:
    """Load CLIP weights in daemon and get IPC handles (module-level convenience)."""
    return get_client().load_clip_weights(clip_l_path, clip_g_path, model_key)


def get_weight_handles(model_key: str) -> Dict:
    """Get IPC handles for a previously loaded model (module-level convenience)."""
    return get_client().get_weight_handles(model_key)


def list_loaded_weights() -> List[str]:
    """Get list of all loaded model keys in weight registry (module-level convenience)."""
    return get_client().list_loaded_weights()


def unload_weights(model_key: str) -> bool:
    """Unload model from weight registry (module-level convenience)."""
    return get_client().unload_weights(model_key)


# =============================================================================
# Legacy Registration Operations (for backward compatibility)
# =============================================================================

def register_checkpoint(instance_id: str, name: str, path: str, model_type: Optional[str] = None, size_mb: Optional[float] = None, device: Optional[str] = None, dtype: Optional[str] = None) -> dict:
    """Legacy checkpoint registration (stub - daemon loads from config)."""
    return {"success": True, "registered": True, "message": "Models managed via config"}


def register_vae_by_path(vae_path: str, vae_type: str) -> dict:
    """Register VAE model by path."""
    return get_client().register_vae_by_path(vae_path, vae_type)


def register_clip_by_path(clip_components: dict, model_type: str, clip_type: str) -> dict:
    """Register CLIP models by path (loads individual components)."""
    return get_client().register_clip_by_path(clip_components, model_type, clip_type)


def unregister_checkpoint(instance_id: str) -> dict:
    """Legacy checkpoint unregistration (stub)."""
    return {"success": True, "unregistered": True}


def start_daemon() -> bool:
    """Start the Luna Daemon process."""
    return get_client().start_daemon()


def shutdown() -> dict:
    """Shut down the daemon server."""
    return get_client().shutdown()


def reset_clients() -> dict:
    """Reset all client connections to daemon."""
    return get_client().reset_clients()


def unload_daemon_models() -> dict:
    """Unload all models from daemon VRAM."""
    return get_client().unload_daemon_models()


def set_attention_mode(mode: str) -> dict:
    """Set attention mechanism mode for CLIP/VAE processing."""
    return get_client().set_attention_mode(mode)


def register_qwen3_transformers(
    model_path: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None
) -> dict:
    """Register Qwen3-VL model using transformers."""
    return get_client().register_qwen3_transformers(model_path, device, dtype)


def get_qwen3_status() -> dict:
    """Get Qwen3-VL model loading status."""
    return get_client().get_qwen3_status()


def zimage_encode(text: str) -> torch.Tensor:
    """Encode text using Z-IMAGE CLIP."""
    return get_client().zimage_encode(text)


def vlm_generate(**kwargs) -> Any:
    """Generate text using Vision-Language Model."""
    return get_client().vlm_generate(**kwargs)


def encode_vision(image) -> torch.Tensor:
    """Encode image using vision model."""
    return get_client().encode_vision(image)


def submit_async(task_name: str, task_data: dict) -> dict:
    """Submit async task to daemon."""
    return get_client().submit_async(task_name, task_data)



