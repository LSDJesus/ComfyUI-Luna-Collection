"""
Luna Shared VAE Nodes
ComfyUI nodes that use the shared VAE/CLIP daemon instead of loading models locally.

These nodes are drop-in replacements for standard VAE nodes, but they communicate
with the Luna daemon to use a single shared VAE instance across multiple ComfyUI processes.
"""

import torch
from typing import Tuple

# Import daemon client
try:
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonConnectionError = Exception


class LunaSharedVAEEncode:
    """
    Encode images to latent space using the shared VAE daemon.
    
    This node does NOT load a VAE model - it connects to the Luna daemon
    which holds a single VAE in VRAM shared across all ComfyUI instances.
    
    Use this instead of VAEEncode when running multiple ComfyUI workflows
    to save ~200MB VRAM per workflow.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
            }
        }
    
    def encode(self, pixels: torch.Tensor) -> Tuple[dict]:
        if not DAEMON_AVAILABLE:
            raise RuntimeError(
                "Luna daemon client not available. "
                "Make sure the luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna VAE/CLIP Daemon is not running!\n"
                "Start it with: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server\n"
                "Or use the startup script that launches it automatically."
            )
        
        try:
            # Send to daemon for encoding
            # ComfyUI format is (B, H, W, C) which the daemon expects
            latents = daemon_client.vae_encode(pixels)
            
            return ({"samples": latents},)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


class LunaSharedVAEDecode:
    """
    Decode latent space to images using the shared VAE daemon.
    
    This node does NOT load a VAE model - it connects to the Luna daemon
    which holds a single VAE in VRAM shared across all ComfyUI instances.
    
    Use this instead of VAEDecode when running multiple ComfyUI workflows
    to save ~200MB VRAM per workflow.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }
    
    def decode(self, samples: dict) -> Tuple[torch.Tensor]:
        if not DAEMON_AVAILABLE:
            raise RuntimeError(
                "Luna daemon client not available. "
                "Make sure the luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna VAE/CLIP Daemon is not running!\n"
                "Start it with: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server\n"
                "Or use the startup script that launches it automatically."
            )
        
        try:
            latents = samples["samples"]
            
            # Send to daemon for decoding
            pixels = daemon_client.vae_decode(latents)
            
            return (pixels,)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


class LunaSharedVAEEncodeTiled:
    """
    Tiled VAE encoding using the shared daemon.
    Useful for high-resolution images that would otherwise run out of VRAM.
    
    Note: Tiling is done client-side, only the tiles are sent to the daemon.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of tiles to encode. Smaller = less VRAM but slower."
                }),
                "overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams."
                }),
            }
        }
    
    def encode(self, pixels: torch.Tensor, tile_size: int, overlap: int) -> Tuple[dict]:
        if not DAEMON_AVAILABLE or not daemon_client.is_daemon_running():
            raise RuntimeError("Luna VAE/CLIP Daemon is not running!")
        
        # For now, just do regular encoding
        # TODO: Implement proper tiled encoding
        latents = daemon_client.vae_encode(pixels)
        
        return ({"samples": latents},)


class LunaSharedVAEDecodeTiled:
    """
    Tiled VAE decoding using the shared daemon.
    Useful for high-resolution latents that would otherwise run out of VRAM.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of tiles to decode. Smaller = less VRAM but slower."
                }),
                "overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams."
                }),
            }
        }
    
    def decode(self, samples: dict, tile_size: int, overlap: int) -> Tuple[torch.Tensor]:
        if not DAEMON_AVAILABLE or not daemon_client.is_daemon_running():
            raise RuntimeError("Luna VAE/CLIP Daemon is not running!")
        
        # For now, just do regular decoding
        # TODO: Implement proper tiled decoding
        latents = samples["samples"]
        pixels = daemon_client.vae_decode(latents)
        
        return (pixels,)


class LunaDaemonStatus:
    """
    Check the status of the Luna VAE/CLIP daemon.
    Outputs daemon info including VRAM usage and request count.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "is_running")
    FUNCTION = "check_status"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    def check_status(self) -> Tuple[str, bool]:
        if not DAEMON_AVAILABLE:
            return ("Daemon client not available", False)
        
        if not daemon_client.is_daemon_running():
            return ("Daemon not running", False)
        
        try:
            info = daemon_client.get_client().get_info()
            
            status_lines = [
                f"Status: {info.get('status', 'unknown')}",
                f"Device: {info.get('device', 'unknown')}",
                f"VRAM: {info.get('vram_used_gb', 0):.2f} / {info.get('vram_total_gb', 0):.2f} GB",
                f"Requests: {info.get('request_count', 0)}",
                f"Uptime: {info.get('uptime_seconds', 0):.0f}s",
                f"VAE loaded: {info.get('vae_loaded', False)}",
                f"CLIP loaded: {info.get('clip_loaded', False)}",
            ]
            
            return ("\n".join(status_lines), True)
            
        except Exception as e:
            return (f"Error: {e}", False)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSharedVAEEncode": LunaSharedVAEEncode,
    "LunaSharedVAEDecode": LunaSharedVAEDecode,
    "LunaSharedVAEEncodeTiled": LunaSharedVAEEncodeTiled,
    "LunaSharedVAEDecodeTiled": LunaSharedVAEDecodeTiled,
    "LunaDaemonStatus": LunaDaemonStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSharedVAEEncode": "Luna Shared VAE Encode",
    "LunaSharedVAEDecode": "Luna Shared VAE Decode",
    "LunaSharedVAEEncodeTiled": "Luna Shared VAE Encode (Tiled)",
    "LunaSharedVAEDecodeTiled": "Luna Shared VAE Decode (Tiled)",
    "LunaDaemonStatus": "Luna Daemon Status",
}
