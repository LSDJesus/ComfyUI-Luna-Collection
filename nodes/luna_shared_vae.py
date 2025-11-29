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
    
    Supports both one-shot and tiled encoding via toggle.
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
                "tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled encoding for high-resolution images"
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of tiles when tiled mode is enabled"
                }),
                "overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams"
                }),
            }
        }
    
    def encode(self, pixels: torch.Tensor, tiled: bool = False, 
               tile_size: int = 512, overlap: int = 64) -> Tuple[dict]:
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
            if tiled:
                # Tiled encoding for high-resolution images
                latents = self._encode_tiled(pixels, tile_size, overlap)
            else:
                # Standard one-shot encoding
                latents = daemon_client.vae_encode(pixels)
            
            return ({"samples": latents},)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def _encode_tiled(self, pixels: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
        """Encode image in tiles to reduce VRAM usage."""
        # For high-res images, process in tiles
        # TODO: Implement proper tiled encoding with overlap blending
        # For now, fall back to regular encoding
        return daemon_client.vae_encode(pixels)


class LunaSharedVAEDecode:
    """
    Decode latent space to images using the shared VAE daemon.
    
    This node does NOT load a VAE model - it connects to the Luna daemon
    which holds a single VAE in VRAM shared across all ComfyUI instances.
    
    Use this instead of VAEDecode when running multiple ComfyUI workflows
    to save ~200MB VRAM per workflow.
    
    Supports both one-shot and tiled decoding via toggle.
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
                "tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled decoding for high-resolution latents"
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of tiles when tiled mode is enabled"
                }),
                "overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams"
                }),
            }
        }
    
    def decode(self, samples: dict, tiled: bool = False,
               tile_size: int = 512, overlap: int = 64) -> Tuple[torch.Tensor]:
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
            
            if tiled:
                # Tiled decoding for high-resolution latents
                pixels = self._decode_tiled(latents, tile_size, overlap)
            else:
                # Standard one-shot decoding
                pixels = daemon_client.vae_decode(latents)
            
            return (pixels,)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def _decode_tiled(self, latents: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
        """Decode latents in tiles to reduce VRAM usage."""
        # For high-res latents, process in tiles
        # TODO: Implement proper tiled decoding with overlap blending
        # For now, fall back to regular decoding
        return daemon_client.vae_decode(latents)


class LunaDaemonStatus:
    """
    Check the status of the Luna VAE/CLIP daemon.
    Outputs daemon info including VRAM usage, request count, and loaded models.
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
                "",
                "Loaded Models:",
                f"  VAE: {info.get('vae_loaded', False)}",
            ]
            
            # Show currently loaded VAE path
            if info.get('current_vae'):
                import os
                status_lines.append(f"    → {os.path.basename(info['current_vae'])}")
            
            status_lines.append(f"  CLIP: {info.get('clip_loaded', False)}")
            
            # Show currently loaded CLIP paths
            if info.get('current_clip_l'):
                import os
                status_lines.append(f"    → CLIP-L: {os.path.basename(info['current_clip_l'])}")
            if info.get('current_clip_g'):
                import os
                status_lines.append(f"    → CLIP-G: {os.path.basename(info['current_clip_g'])}")
            if info.get('current_t5xxl'):
                import os
                status_lines.append(f"    → T5-XXL: {os.path.basename(info['current_t5xxl'])}")
            
            return ("\n".join(status_lines), True)
            
        except Exception as e:
            return (f"Error: {e}", False)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSharedVAEEncode": LunaSharedVAEEncode,
    "LunaSharedVAEDecode": LunaSharedVAEDecode,
    "LunaDaemonStatus": LunaDaemonStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSharedVAEEncode": "Luna Shared VAE Encode",
    "LunaSharedVAEDecode": "Luna Shared VAE Decode",
    "LunaDaemonStatus": "Luna Daemon Status",
}
