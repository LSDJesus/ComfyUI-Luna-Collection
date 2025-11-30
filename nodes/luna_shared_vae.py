"""
Luna Shared VAE Nodes
ComfyUI nodes that use the shared VAE/CLIP daemon instead of loading models locally.

These nodes are drop-in replacements for standard VAE nodes, but they communicate
with the Luna daemon to use a single shared VAE instance across multiple ComfyUI processes.
"""

import torch
import math
from typing import Tuple

# Import daemon client
try:
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonConnectionError = Exception


# =============================================================================
# TILED VAE UTILITIES
# =============================================================================

def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int) -> int:
    """Calculate number of tiles needed for a given image size."""
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


@torch.inference_mode()
def tiled_encode(pixels: torch.Tensor, encode_fn, tile_x: int = 512, tile_y: int = 512, 
                 overlap: int = 64, downscale_ratio: int = 8, latent_channels: int = 4) -> torch.Tensor:
    """
    Encode an image to latent space using tiles.
    
    Args:
        pixels: Input image tensor (B, H, W, C) in 0-1 range
        encode_fn: Function that encodes a single tile, returns latent
        tile_x, tile_y: Tile size in pixels
        overlap: Overlap between tiles in pixels
        downscale_ratio: VAE downscale factor (usually 8)
        latent_channels: Number of latent channels (usually 4)
    
    Returns:
        Latent tensor (B, C, H//8, W//8)
    """
    # Convert from ComfyUI format (B, H, W, C) to standard (B, C, H, W)
    if pixels.dim() == 4 and pixels.shape[-1] in [1, 3, 4]:
        pixels = pixels.permute(0, 3, 1, 2)
    
    batch_size, channels, height, width = pixels.shape
    
    # Calculate output size
    out_height = height // downscale_ratio
    out_width = width // downscale_ratio
    
    # Initialize output and weight tensors
    output = torch.zeros(batch_size, latent_channels, out_height, out_width, 
                         device=pixels.device, dtype=torch.float32)
    weight = torch.zeros(batch_size, 1, out_height, out_width, 
                         device=pixels.device, dtype=torch.float32)
    
    # Calculate tile positions
    tile_y_step = tile_y - overlap
    tile_x_step = tile_x - overlap
    
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / tile_y_step)
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / tile_x_step)
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile boundaries in pixel space
            y1 = row * tile_y_step
            x1 = col * tile_x_step
            y2 = min(y1 + tile_y, height)
            x2 = min(x1 + tile_x, width)
            
            # Adjust start if tile would extend beyond image
            if y2 - y1 < tile_y and row > 0:
                y1 = max(0, height - tile_y)
                y2 = height
            if x2 - x1 < tile_x and col > 0:
                x1 = max(0, width - tile_x)
                x2 = width
            
            # Extract tile and encode
            tile = pixels[:, :, y1:y2, x1:x2]
            # Convert back to ComfyUI format for encode_fn
            tile_hwc = tile.permute(0, 2, 3, 1)
            encoded_tile = encode_fn(tile_hwc)
            
            # Calculate output positions
            out_y1 = y1 // downscale_ratio
            out_x1 = x1 // downscale_ratio
            out_y2 = y2 // downscale_ratio
            out_x2 = x2 // downscale_ratio
            
            # Create blending weight (feather edges)
            tile_h = out_y2 - out_y1
            tile_w = out_x2 - out_x1
            feather = overlap // downscale_ratio // 2
            
            w = torch.ones(1, 1, tile_h, tile_w, device=pixels.device, dtype=torch.float32)
            if feather > 0:
                # Feather top edge
                if row > 0 and tile_h > feather:
                    for i in range(feather):
                        w[:, :, i, :] *= (i + 1) / (feather + 1)
                # Feather bottom edge
                if row < rows - 1 and tile_h > feather:
                    for i in range(feather):
                        w[:, :, -(i + 1), :] *= (i + 1) / (feather + 1)
                # Feather left edge
                if col > 0 and tile_w > feather:
                    for i in range(feather):
                        w[:, :, :, i] *= (i + 1) / (feather + 1)
                # Feather right edge
                if col < cols - 1 and tile_w > feather:
                    for i in range(feather):
                        w[:, :, :, -(i + 1)] *= (i + 1) / (feather + 1)
            
            # Accumulate weighted output
            output[:, :, out_y1:out_y2, out_x1:out_x2] += encoded_tile * w
            weight[:, :, out_y1:out_y2, out_x1:out_x2] += w
    
    # Normalize by weight
    output = output / weight.clamp(min=1e-8)
    
    return output


@torch.inference_mode()
def tiled_decode(samples: torch.Tensor, decode_fn, tile_x: int = 64, tile_y: int = 64,
                 overlap: int = 16, upscale_ratio: int = 8, out_channels: int = 3) -> torch.Tensor:
    """
    Decode latent space to image using tiles.
    
    Args:
        samples: Latent tensor (B, C, H, W)
        decode_fn: Function that decodes a single tile, returns pixels
        tile_x, tile_y: Tile size in latent space
        overlap: Overlap between tiles in latent space
        upscale_ratio: VAE upscale factor (usually 8)
        out_channels: Number of output channels (usually 3)
    
    Returns:
        Image tensor (B, H*8, W*8, C) in ComfyUI format
    """
    batch_size, channels, height, width = samples.shape
    
    # Calculate output size
    out_height = height * upscale_ratio
    out_width = width * upscale_ratio
    
    # Initialize output and weight tensors
    output = torch.zeros(batch_size, out_channels, out_height, out_width,
                         device=samples.device, dtype=torch.float32)
    weight = torch.zeros(batch_size, 1, out_height, out_width,
                         device=samples.device, dtype=torch.float32)
    
    # Calculate tile positions
    tile_y_step = tile_y - overlap
    tile_x_step = tile_x - overlap
    
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / tile_y_step)
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / tile_x_step)
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile boundaries in latent space
            y1 = row * tile_y_step
            x1 = col * tile_x_step
            y2 = min(y1 + tile_y, height)
            x2 = min(x1 + tile_x, width)
            
            # Adjust start if tile would extend beyond
            if y2 - y1 < tile_y and row > 0:
                y1 = max(0, height - tile_y)
                y2 = height
            if x2 - x1 < tile_x and col > 0:
                x1 = max(0, width - tile_x)
                x2 = width
            
            # Extract tile and decode
            tile = samples[:, :, y1:y2, x1:x2]
            decoded_tile = decode_fn(tile)
            
            # decoded_tile comes back as (B, H, W, C) from daemon
            # Convert to (B, C, H, W) for accumulation
            if decoded_tile.dim() == 4 and decoded_tile.shape[-1] in [1, 3, 4]:
                decoded_tile = decoded_tile.permute(0, 3, 1, 2)
            
            # Calculate output positions
            out_y1 = y1 * upscale_ratio
            out_x1 = x1 * upscale_ratio
            out_y2 = y2 * upscale_ratio
            out_x2 = x2 * upscale_ratio
            
            # Create blending weight (feather edges)
            tile_h = out_y2 - out_y1
            tile_w = out_x2 - out_x1
            feather = overlap * upscale_ratio // 2
            
            w = torch.ones(1, 1, tile_h, tile_w, device=samples.device, dtype=torch.float32)
            if feather > 0:
                # Feather edges
                if row > 0 and tile_h > feather:
                    for i in range(feather):
                        w[:, :, i, :] *= (i + 1) / (feather + 1)
                if row < rows - 1 and tile_h > feather:
                    for i in range(feather):
                        w[:, :, -(i + 1), :] *= (i + 1) / (feather + 1)
                if col > 0 and tile_w > feather:
                    for i in range(feather):
                        w[:, :, :, i] *= (i + 1) / (feather + 1)
                if col < cols - 1 and tile_w > feather:
                    for i in range(feather):
                        w[:, :, :, -(i + 1)] *= (i + 1) / (feather + 1)
            
            # Accumulate weighted output
            output[:, :, out_y1:out_y2, out_x1:out_x2] += decoded_tile * w
            weight[:, :, out_y1:out_y2, out_x1:out_x2] += w
    
    # Normalize by weight
    output = output / weight.clamp(min=1e-8)
    
    # Convert back to ComfyUI format (B, H, W, C)
    output = output.permute(0, 2, 3, 1)
    
    return output


class LunaSharedVAEEncode:
    """
    Encode images to latent space using the shared VAE daemon.
    
    This node does NOT load a VAE model - it connects to the Luna daemon
    which holds a single VAE in VRAM shared across all ComfyUI instances.
    
    Use this instead of VAEEncode when running multiple ComfyUI workflows
    to save ~200MB VRAM per workflow.
    
    Enable tiled mode for high-resolution images that would otherwise run out of VRAM.
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
                "use_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled encoding for high-resolution images to reduce VRAM usage."
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of tiles to encode (only used when tiled is enabled). Smaller = less VRAM but slower."
                }),
                "overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams (only used when tiled is enabled)."
                }),
            }
        }
    
    def encode(self, pixels: torch.Tensor, use_tiled: bool = False, 
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
            if use_tiled:
                # Check if tiling is actually needed
                _, h, w, _ = pixels.shape
                if h <= tile_size and w <= tile_size:
                    # Image fits in a single tile, use regular encode
                    latents = daemon_client.vae_encode(pixels)
                else:
                    # Use tiled encoding
                    def encode_fn(tile):
                        return daemon_client.vae_encode(tile)
                    
                    latents = tiled_encode(
                        pixels, 
                        encode_fn, 
                        tile_x=tile_size, 
                        tile_y=tile_size, 
                        overlap=overlap,
                        downscale_ratio=8,
                        latent_channels=4
                    )
            else:
                # Standard encoding
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
    
    Enable tiled mode for high-resolution latents that would otherwise run out of VRAM.
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
                "use_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled decoding for high-resolution latents to reduce VRAM usage."
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Size of tiles in latent space (8x in pixel space). Only used when tiled is enabled."
                }),
                "overlap": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 64,
                    "step": 4,
                    "tooltip": "Overlap between tiles in latent space. Only used when tiled is enabled."
                }),
            }
        }
    
    def decode(self, samples: dict, use_tiled: bool = False,
               tile_size: int = 64, overlap: int = 16) -> Tuple[torch.Tensor]:
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
            
            if use_tiled:
                _, _, h, w = latents.shape
                # Check if tiling is actually needed
                if h <= tile_size and w <= tile_size:
                    # Latents fit in a single tile, use regular decode
                    pixels = daemon_client.vae_decode(latents)
                else:
                    # Use tiled decoding
                    def decode_fn(tile):
                        return daemon_client.vae_decode(tile)
                    
                    pixels = tiled_decode(
                        latents,
                        decode_fn,
                        tile_x=tile_size,
                        tile_y=tile_size,
                        overlap=overlap,
                        upscale_ratio=8,
                        out_channels=3
                    )
            else:
                # Standard decoding
                pixels = daemon_client.vae_decode(latents)
            
            return (pixels,)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


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
    "LunaDaemonStatus": LunaDaemonStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSharedVAEEncode": "Luna Shared VAE Encode",
    "LunaSharedVAEDecode": "Luna Shared VAE Decode",
    "LunaDaemonStatus": "Luna Daemon Status",
}
