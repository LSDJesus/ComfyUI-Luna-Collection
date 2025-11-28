"""
Luna Daemon Client Library
Used by ComfyUI nodes to communicate with the shared VAE/CLIP daemon
"""

import socket
import pickle
import torch
from typing import Tuple, Optional, Any
from .config import DAEMON_HOST, DAEMON_PORT, CLIENT_TIMEOUT


class DaemonConnectionError(Exception):
    """Raised when daemon is not available"""
    pass


class DaemonClient:
    """Client for communicating with Luna VAE/CLIP Daemon"""
    
    def __init__(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        self.host = host
        self.port = port
        self.timeout = CLIENT_TIMEOUT
    
    def _send_request(self, request: dict) -> Any:
        """Send request to daemon and get response"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Serialize and send with end marker
            data = pickle.dumps(request) + b"<<END>>"
            sock.sendall(data)
            
            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(1048576)  # 1MB chunks
                if not chunk:
                    break
                response_data += chunk
            
            sock.close()
            
            result = pickle.loads(response_data)
            
            # Check for error response
            if isinstance(result, dict) and "error" in result:
                raise DaemonConnectionError(f"Daemon error: {result['error']}")
            
            return result
            
        except ConnectionRefusedError:
            raise DaemonConnectionError(
                "Luna VAE/CLIP Daemon is not running!\n"
                "Start it with: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server"
            )
        except socket.timeout:
            raise DaemonConnectionError(
                f"Daemon request timed out after {self.timeout}s"
            )
        except Exception as e:
            raise DaemonConnectionError(f"Daemon communication error: {e}")
    
    def is_running(self) -> bool:
        """Check if daemon is available"""
        try:
            result = self._send_request({"cmd": "health"})
            return result.get("status") == "ok"
        except:
            return False
    
    def get_info(self) -> dict:
        """Get daemon info (device, VRAM usage, loaded models)"""
        return self._send_request({"cmd": "info"})
    
    def vae_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode image pixels to latent space via daemon.
        
        Args:
            pixels: Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
        
        Returns:
            Latent tensor (B, 4, H//8, W//8)
        """
        return self._send_request({
            "cmd": "vae_encode",
            "pixels": pixels.cpu()
        })
    
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image pixels via daemon.
        
        Args:
            latents: Latent tensor (B, 4, H//8, W//8)
        
        Returns:
            Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
        """
        return self._send_request({
            "cmd": "vae_decode",
            "latents": latents.cpu()
        })
    
    def clip_encode(
        self, 
        positive: str, 
        negative: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to CLIP conditioning via daemon.
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
        
        Returns:
            Tuple of (cond, pooled, uncond, pooled_neg)
        """
        return self._send_request({
            "cmd": "clip_encode",
            "positive": positive,
            "negative": negative
        })
    
    def clip_encode_sdxl(
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
        """
        Encode text prompts with SDXL-specific conditioning (includes size embeddings).
        
        Returns:
            Tuple of (positive_conditioning, negative_conditioning) ready for KSampler
        """
        return self._send_request({
            "cmd": "clip_encode_sdxl",
            "positive": positive,
            "negative": negative,
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height
        })


# Singleton client instance
_client: Optional[DaemonClient] = None


def get_client() -> DaemonClient:
    """Get or create the singleton client instance"""
    global _client
    if _client is None:
        _client = DaemonClient()
    return _client


# Convenience functions
def is_daemon_running() -> bool:
    """Check if daemon is available"""
    return get_client().is_running()


def vae_encode(pixels: torch.Tensor) -> torch.Tensor:
    """Encode pixels via daemon"""
    return get_client().vae_encode(pixels)


def vae_decode(latents: torch.Tensor) -> torch.Tensor:
    """Decode latents via daemon"""
    return get_client().vae_decode(latents)


def clip_encode(positive: str, negative: str = "") -> Tuple:
    """Encode text via daemon"""
    return get_client().clip_encode(positive, negative)


def clip_encode_sdxl(positive: str, negative: str = "", **kwargs) -> Tuple[list, list]:
    """Encode text with SDXL conditioning via daemon"""
    return get_client().clip_encode_sdxl(positive, negative, **kwargs)
