"""
Luna Daemon Client Library
Used by ComfyUI nodes to communicate with the shared VAE/CLIP daemon.

The daemon loads models on-demand from the first workflow request,
then shares them across all ComfyUI instances.

Component-based architecture:
- CLIP components (clip_l, clip_g, t5xxl) can be shared across model families
- VAE components are family-specific (sdxl_vae, flux_vae, etc.)
"""

import socket
import pickle
import torch
from typing import Tuple, Optional, Any, List, Dict
from .config import DAEMON_HOST, DAEMON_PORT, CLIENT_TIMEOUT


class DaemonConnectionError(Exception):
    """Raised when daemon is not available"""
    pass


class ModelMismatchError(Exception):
    """Raised when workflow tries to use a different model than what's loaded in daemon"""
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
                # Check if it's a model mismatch error
                if result.get("type") == "model_mismatch":
                    raise ModelMismatchError(result["error"])
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
        """Get daemon info (device, VRAM usage, loaded models/components)"""
        return self._send_request({"cmd": "info"})
    
    def unload_models(self) -> dict:
        """Unload all models from daemon to allow loading different ones"""
        return self._send_request({"cmd": "unload"})
    
    def shutdown(self) -> dict:
        """Shutdown the daemon server"""
        return self._send_request({"cmd": "shutdown"})
    
    # =========================================================================
    # Model Registration (new component-based API)
    # =========================================================================
    
    def register_vae(self, vae: Any, vae_type: str) -> dict:
        """
        Register a VAE with the daemon.
        
        The daemon will extract and store the VAE state dict for future use.
        If a VAE of this type is already loaded, validates it matches.
        
        Args:
            vae: The VAE object from checkpoint loader
            vae_type: Type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Dict with registration status
        """
        # Extract state dict from VAE
        try:
            if hasattr(vae, 'first_stage_model'):
                state_dict = vae.first_stage_model.state_dict()
            else:
                state_dict = vae.state_dict()
        except Exception as e:
            raise DaemonConnectionError(f"Failed to extract VAE state dict: {e}")
        
        return self._send_request({
            "cmd": "register_vae",
            "vae_type": vae_type,
            "state_dict": state_dict
        })
    
    def register_clip(self, clip: Any, clip_type: str) -> dict:
        """
        Register CLIP with the daemon.
        
        The daemon will extract and store the CLIP components for future use.
        Components that are already loaded will be reused (shared).
        
        Args:
            clip: The CLIP object from checkpoint loader
            clip_type: Type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Dict with registration status and which components were loaded/shared
        """
        # Extract CLIP components based on type
        components = {}
        
        try:
            cond_model = getattr(clip, 'cond_stage_model', None)
            if cond_model is None and hasattr(clip, 'patcher'):
                cond_model = getattr(clip.patcher, 'model', None)
            
            if cond_model is not None:
                # Extract individual components
                if hasattr(cond_model, 'clip_l'):
                    components['clip_l'] = cond_model.clip_l.state_dict()
                if hasattr(cond_model, 'clip_g'):
                    components['clip_g'] = cond_model.clip_g.state_dict()
                if hasattr(cond_model, 't5xxl'):
                    components['t5xxl'] = cond_model.t5xxl.state_dict()
                
                # For SD1.5 which just has a single CLIP
                if not components and hasattr(cond_model, 'state_dict'):
                    components['clip_l'] = cond_model.state_dict()
        except Exception as e:
            raise DaemonConnectionError(f"Failed to extract CLIP components: {e}")
        
        return self._send_request({
            "cmd": "register_clip",
            "clip_type": clip_type,
            "components": components
        })
    
    # =========================================================================
    # VAE Operations
    # =========================================================================
    
    def vae_encode(self, pixels: torch.Tensor, vae_type: str) -> torch.Tensor:
        """
        Encode image pixels to latent space via daemon.
        
        Args:
            pixels: Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
            vae_type: VAE type string ('sdxl', 'flux', etc.)
        
        Returns:
            Latent tensor
        """
        return self._send_request({
            "cmd": "vae_encode",
            "pixels": pixels.cpu(),
            "vae_type": vae_type
        })
    
    def vae_decode(self, latents: torch.Tensor, vae_type: str) -> torch.Tensor:
        """
        Decode latents to image pixels via daemon.
        
        Args:
            latents: Latent tensor
            vae_type: VAE type string ('sdxl', 'flux', etc.)
        
        Returns:
            Image tensor in ComfyUI format (B, H, W, C), float32, 0-1 range
        """
        return self._send_request({
            "cmd": "vae_decode",
            "latents": latents.cpu(),
            "vae_type": vae_type
        })
    
    # =========================================================================
    # CLIP Operations
    # =========================================================================
    
    def clip_encode(
        self, 
        positive: str, 
        negative: str,
        clip_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to CLIP conditioning via daemon.
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
            clip_type: CLIP type string ('sdxl', 'flux', 'sd3', 'sd15')
        
        Returns:
            Tuple of (cond, pooled, uncond, pooled_neg)
        """
        return self._send_request({
            "cmd": "clip_encode",
            "positive": positive,
            "negative": negative,
            "clip_type": clip_type
        })
    
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
        target_height: int = 1024
    ) -> Tuple[list, list]:
        """
        Encode text prompts with SDXL-specific conditioning (includes size embeddings).
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
            clip_type: CLIP type string
            width, height: Original image dimensions
            crop_w, crop_h: Crop coordinates
            target_width, target_height: Target generation size
        
        Returns:
            Tuple of (positive_conditioning, negative_conditioning) ready for KSampler
        """
        return self._send_request({
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
        })


# =============================================================================
# Singleton client instance
# =============================================================================

_client: Optional[DaemonClient] = None


def get_client() -> DaemonClient:
    """Get or create the singleton client instance"""
    global _client
    if _client is None:
        _client = DaemonClient()
    return _client


# =============================================================================
# Convenience functions
# =============================================================================

def is_daemon_running() -> bool:
    """Check if daemon is available"""
    return get_client().is_running()


def get_daemon_info() -> dict:
    """Get daemon info including loaded models"""
    return get_client().get_info()


def unload_daemon_models() -> dict:
    """Unload all models from daemon"""
    return get_client().unload_models()


def register_vae(vae: Any, vae_type: str) -> dict:
    """Register VAE with daemon"""
    return get_client().register_vae(vae, vae_type)


def register_clip(clip: Any, clip_type: str) -> dict:
    """Register CLIP with daemon"""
    return get_client().register_clip(clip, clip_type)


def vae_encode(pixels: torch.Tensor, vae_type: str) -> torch.Tensor:
    """Encode pixels via daemon"""
    return get_client().vae_encode(pixels, vae_type)


def vae_decode(latents: torch.Tensor, vae_type: str) -> torch.Tensor:
    """Decode latents via daemon"""
    return get_client().vae_decode(latents, vae_type)


def clip_encode(positive: str, negative: str, clip_type: str) -> Tuple:
    """Encode text via daemon"""
    return get_client().clip_encode(positive, negative, clip_type)


def clip_encode_sdxl(positive: str, negative: str, clip_type: str = "sdxl", **kwargs) -> Tuple[list, list]:
    """Encode text with SDXL conditioning via daemon"""
    return get_client().clip_encode_sdxl(positive, negative, clip_type, **kwargs)
