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
    # VAE Operations
    # =========================================================================
    
    def vae_encode(
        self, 
        pixels: torch.Tensor, 
        vae_type: str,
        tiled: bool = False, 
        tile_size: int = 512,
        overlap: int = 64
    ) -> torch.Tensor:
        """Encode pixels to latent space via daemon."""
        pixels_cpu = pixels.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_encode",
            "pixels": pixels_cpu,
            "vae_type": vae_type,
            "tiled": tiled,
            "tile_size": tile_size,
            "overlap": overlap
        })
        
        del pixels_cpu
        return result
    
    def vae_decode(
        self, 
        latents: torch.Tensor, 
        vae_type: str,
        tiled: bool = False, 
        tile_size: int = 64,
        overlap: int = 16
    ) -> torch.Tensor:
        """Decode latents to image pixels via daemon."""
        latents_cpu = latents.detach().cpu()
        
        result = self._send_request({
            "cmd": "vae_decode",
            "latents": latents_cpu,
            "vae_type": vae_type,
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
        clip_type: str,
        lora_stack: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text prompts to CLIP conditioning via daemon."""
        request: Dict[str, Any] = {
            "cmd": "clip_encode",
            "positive": positive,
            "negative": negative,
            "clip_type": clip_type
        }
        if lora_stack:
            request["lora_stack"] = lora_stack
        
        return self._send_request(request)
    
    def clip_encode_sdxl(
        self,
        positive: str,
        negative: str,
        clip_type: str = "sdxl",
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
            "clip_type": clip_type,
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
        clip_type: str,
        return_word_ids: bool = False
    ):
        """Tokenize text via daemon."""
        return self._send_request({
            "cmd": "clip_tokenize",
            "text": text,
            "clip_type": clip_type,
            "return_word_ids": return_word_ids
        })
    
    def clip_encode_from_tokens(
        self,
        tokens,
        clip_type: str,
        return_pooled: bool = False,
        return_dict: bool = False,
        lora_stack: Optional[List[Dict]] = None
    ):
        """Encode tokens via daemon."""
        request = {
            "cmd": "clip_encode_from_tokens",
            "tokens": tokens,
            "clip_type": clip_type,
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
    
    # =========================================================================
    # Registration Operations (for DaemonVAE/DaemonCLIP)
    # =========================================================================
    
    def register_vae(self, vae: Any, vae_type: str) -> dict:
        """Register VAE with daemon (placeholder - daemon loads from config)."""
        return {"registered": True, "vae_type": vae_type}
    
    def register_clip(self, clip: Any, clip_type: str) -> dict:
        """Register CLIP with daemon (placeholder - daemon loads from config)."""
        return {"registered": True, "clip_type": clip_type}


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


def register_vae(vae: Any, vae_type: str) -> dict:
    """Register VAE with daemon."""
    return get_client().register_vae(vae, vae_type)


def register_clip(clip: Any, clip_type: str) -> dict:
    """Register CLIP with daemon."""
    return get_client().register_clip(clip, clip_type)


# =============================================================================
# Legacy Registration Operations (for backward compatibility)
# =============================================================================

def register_checkpoint(instance_id: str, name: str, path: str, model_type: str = None, size_mb: float = None, device: str = None, dtype: str = None) -> dict:
    """Legacy checkpoint registration (stub - daemon loads from config)."""
    return {"success": True, "registered": True, "message": "Models managed via config"}


def register_vae_by_path(vae_path: str, vae_type: str) -> dict:
    """Legacy VAE registration by path (stub - daemon loads from config)."""
    return {"success": True, "registered": True, "vae_type": vae_type}


def register_clip_by_path(clip_components: dict, model_type: str, clip_type: str) -> dict:
    """Legacy CLIP registration by path (stub - daemon loads from config)."""
    return {"success": True, "registered": True, "clip_type": clip_type}


def unregister_checkpoint(instance_id: str) -> dict:
    """Legacy checkpoint unregistration (stub)."""
    return {"success": True, "unregistered": True}


def unload_daemon_models() -> dict:
    """Legacy daemon model unload (stub)."""
    return {"success": True, "unloaded": True}


