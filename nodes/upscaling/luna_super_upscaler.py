"""
Luna Super Upscaler

A unified super-resolution node that integrates SeedVR2's DiT upscaling
with Luna infrastructure for maximum efficiency.

Features:
- Automatic tiling for large images
- Batched tile processing (4 tiles at once)
- Optional daemon VAE for encode/decode (VRAM efficient)
- Integration with Luna Config Gateway
- Z-IMAGE and other model type support via Model Router

Based on SeedVR2 Video Upscaler (Apache 2.0 License)
"""

import os
import torch
from typing import Tuple, Any, Dict

try:
    import folder_paths
    import comfy.model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None
    comfy = None


def apply_film_grain(
    images: torch.Tensor,
    grain_intensity: float = 0.1,
    saturation: float = 0.5
) -> torch.Tensor:
    """
    Adds film grain to images as preprocessing for upscaling.
    
    This helps diffusion-based upscaling by providing texture to work with,
    preventing the "plastic" look on smooth gradients.
    
    Based on LTXVideo's film_grain implementation.
    
    Args:
        images: (B, H, W, C) tensor in [0, 1] range
        grain_intensity: Strength of grain effect (0.0 - 1.0)
        saturation: Color saturation of grain (0.0 = grayscale, 1.0 = full color)
    
    Returns:
        Images with film grain applied
    """
    if grain_intensity <= 0:
        return images
    
    device = images.device
    if HAS_COMFY:
        device = comfy.model_management.get_torch_device()
        images = images.to(device)
    
    grain = torch.zeros(images[0:1].shape, device=device)
    
    # Process images in-place for memory efficiency
    for i in range(images.shape[0]):
        # Generate colored grain - red and blue channels get more noise
        torch.randn(grain.shape, device=device, out=grain)
        grain[:, :, :, 0] *= 2  # Red channel
        grain[:, :, :, 2] *= 3  # Blue channel
        
        # Blend saturation with luminance
        grain = grain * saturation + grain[:, :, :, 1:2].expand(-1, -1, -1, 3) * (1 - saturation)
        
        # Apply grain to image
        images[i:i+1] = (images[i:i+1] + grain_intensity * grain).clamp_(0, 1)
    
    if HAS_COMFY:
        images = images.to(comfy.model_management.intermediate_device())
    
    return images

# SeedVR2 wrapper - try relative import first, then absolute
try:
    from .seedvr2_wrapper import (
        SEEDVR2_AVAILABLE,
        LunaSeedVR2Pipeline,
        UpscaleConfig,
        tile_image,
        untile_image,
        batch_tiles
    )
except ImportError:
    # Fallback for when loaded directly or from different context
    import sys
    import os
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    from seedvr2_wrapper import (
        SEEDVR2_AVAILABLE,
        LunaSeedVR2Pipeline,
        UpscaleConfig,
        tile_image,
        untile_image,
        batch_tiles
    )


class LunaSuperUpscaler:
    """
    Luna Super Upscaler - High-quality AI upscaling powered by SeedVR2.
    
    Takes an image and upscales it using diffusion-based super-resolution.
    Integrates with Luna Config Gateway for model management and daemon
    VAE for memory-efficient encode/decode operations.
    
    Flow:
    1. Image → tile into manageable chunks
    2. Tiles → VAE encode (optionally via daemon)
    3. Latents → DiT upscale
    4. Upscaled latents → VAE decode (optionally via daemon)
    5. Tiles → reassemble with blending
    """
    
    CATEGORY = "Luna/Upscaling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    
    # Available DiT models
    DIT_MODELS = [
        "seedvr2_ema_7b.safetensors",
        "seedvr2_ema_7b-Q8_0.gguf",
        "seedvr2_ema_7b-Q4_K_M.gguf",
        "seedvr2_ema_3b.safetensors",
    ]
    
    # Available VAE models
    VAE_MODELS = [
        "ema_vae_fp16.safetensors",
        "ema_vae_fp32.safetensors",
    ]
    
    COLOR_CORRECTIONS = ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available devices
        devices = ["cuda:0"]
        if torch.cuda.device_count() > 1:
            devices.extend([f"cuda:{i}" for i in range(1, torch.cuda.device_count())])
        devices.append("cpu")
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale"}),
                "dit_model": (cls.DIT_MODELS, {
                    "default": "seedvr2_ema_7b-Q4_K_M.gguf",
                    "tooltip": "DiT model for upscaling. Q4_K_M is fastest, fp32 is highest quality."
                }),
                "vae_model": (cls.VAE_MODELS, {
                    "default": "ema_vae_fp16.safetensors",
                    "tooltip": "VAE model for encode/decode. fp16 saves VRAM."
                }),
                "target_resolution": ("INT", {
                    "default": 2160,
                    "min": 512,
                    "max": 16384,
                    "step": 8,
                    "tooltip": "Target resolution for the shortest edge"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility"
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Size of processing tiles. Smaller = less VRAM, more tiles"
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Overlap between tiles for seamless blending"
                }),
                "tile_batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of tiles to process at once. Higher = faster but more VRAM"
                }),
                "color_correction": (cls.COLOR_CORRECTIONS, {
                    "default": "lab",
                    "tooltip": "Color correction method to match original colors"
                }),
                "dit_device": (["cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda:0",
                    "tooltip": "Device for DiT model (upscaling)"
                }),
                "vae_device": (["cuda:0", "cuda:1", "cpu", "daemon"], {
                    "default": "daemon",
                    "tooltip": "Device for VAE. 'daemon' uses Luna daemon for VRAM efficiency"
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed debug logging"
                }),
                "film_grain_intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Film grain preprocessing. Adds texture for better diffusion upscaling. 0 = disabled"
                }),
                "film_grain_saturation": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Color saturation of film grain. 0 = grayscale, 1 = full color"
                }),
            }
        }
    
    def upscale(
        self,
        image: torch.Tensor,
        dit_model: str,
        vae_model: str,
        target_resolution: int,
        seed: int,
        tile_size: int = 512,
        tile_overlap: int = 64,
        tile_batch_size: int = 4,
        color_correction: str = "lab",
        dit_device: str = "cuda:0",
        vae_device: str = "daemon",
        enable_debug: bool = False,
        film_grain_intensity: float = 0.1,
        film_grain_saturation: float = 0.5,
    ) -> Tuple[torch.Tensor]:
        """
        Upscale image using SeedVR2 DiT model.
        
        Args:
            image: Input image (B, H, W, C) in [0, 1] range
            dit_model: DiT model filename
            vae_model: VAE model filename
            target_resolution: Target resolution for shortest edge
            seed: Random seed
            tile_size: Size of processing tiles
            tile_overlap: Overlap between tiles
            tile_batch_size: Number of tiles to process together
            color_correction: Color correction method
            dit_device: Device for DiT inference
            vae_device: Device for VAE ('daemon' for Luna daemon)
            enable_debug: Enable debug logging
        
        Returns:
            Upscaled image (B, H', W', C)
        """
        if not SEEDVR2_AVAILABLE:
            raise RuntimeError(
                "SeedVR2 is required for Luna Super Upscaler.\n"
                "Please install via ComfyUI Manager: Search 'SeedVR2' → Install"
            )
        
        # Determine if using daemon VAE
        use_daemon_vae = vae_device == "daemon"
        actual_vae_device = dit_device if use_daemon_vae else vae_device
        
        print(f"[LunaSuperUpscaler] Starting upscale to {target_resolution}p")
        print(f"  DiT: {dit_model} on {dit_device}")
        print(f"  VAE: {vae_model} on {'daemon' if use_daemon_vae else actual_vae_device}")
        print(f"  Tiles: {tile_size}px with {tile_overlap}px overlap, batch {tile_batch_size}")
        
        # Apply film grain preprocessing
        if film_grain_intensity > 0:
            print(f"  Film grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}")
            image = apply_film_grain(image, film_grain_intensity, film_grain_saturation)
        
        # Process each image in batch
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            single_image = image[i]  # (H, W, C)
            
            # Calculate output size based on target resolution
            h, w = single_image.shape[:2]
            scale = target_resolution / min(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Tile the image
            tiles, positions = tile_image(
                single_image,
                tile_size=tile_size,
                overlap=tile_overlap
            )
            
            print(f"  Image {i+1}/{batch_size}: {h}x{w} → {new_h}x{new_w} ({len(tiles)} tiles)")
            
            # Batch tiles for efficient processing
            batched_tiles = batch_tiles(tiles, batch_size=tile_batch_size)
            
            # Create pipeline
            pipeline = LunaSeedVR2Pipeline(
                dit_model=dit_model,
                vae_model=vae_model,
                dit_device=dit_device,
                vae_device=actual_vae_device,
                use_daemon_vae=use_daemon_vae,
                debug_enabled=enable_debug
            )
            
            # Process each batch of tiles
            processed_tiles = []
            
            for batch_idx, tile_batch in enumerate(batched_tiles):
                print(f"    Processing tile batch {batch_idx + 1}/{len(batched_tiles)}...")
                
                config = UpscaleConfig(
                    resolution=int(tile_size * scale),  # Scale tiles proportionally
                    batch_size=tile_batch.shape[0],
                    color_correction=color_correction,
                    seed=seed + batch_idx,
                    tile_batch_size=tile_batch_size,
                    use_daemon_vae=use_daemon_vae
                )
                
                # Upscale the batch
                upscaled_batch = pipeline.upscale(tile_batch, config)
                
                # Split batch back into individual tiles
                for j in range(upscaled_batch.shape[0]):
                    processed_tiles.append(upscaled_batch[j])
            
            # Calculate scaled positions
            scaled_positions = [
                (int(x * scale), int(y * scale), int(tw * scale), int(th * scale))
                for (x, y, tw, th) in positions
            ]
            
            # Reassemble tiles
            result = untile_image(
                processed_tiles,
                scaled_positions,
                output_size=(new_h, new_w),
                overlap=int(tile_overlap * scale)
            )
            
            results.append(result)
        
        # Stack results
        output = torch.stack(results, dim=0)
        
        print(f"[LunaSuperUpscaler] Complete: {output.shape}")
        
        return (output,)


# For backwards compatibility and alternative interface
class LunaSuperUpscalerSimple:
    """
    Simplified version that takes CONFIG input from Luna Config Gateway.
    """
    
    CATEGORY = "Luna/Upscaling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "config": ("LUNA_CONFIG", {
                    "tooltip": "Configuration from Luna Config Gateway"
                }),
                "target_resolution": ("INT", {
                    "default": 2160,
                    "min": 512,
                    "max": 16384,
                    "step": 8
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "color_correction": (LunaSuperUpscaler.COLOR_CORRECTIONS, {"default": "lab"}),
            }
        }
    
    def upscale(
        self,
        image: torch.Tensor,
        config: Dict[str, Any],
        target_resolution: int,
        seed: int = 42,
        color_correction: str = "lab"
    ) -> Tuple[torch.Tensor]:
        """
        Upscale using config from Luna Config Gateway.
        
        The config should include model and VAE information.
        """
        # Extract model info from config
        # TODO: Integrate with actual Config Gateway format
        
        upscaler = LunaSuperUpscaler()
        return upscaler.upscale(
            image=image,
            dit_model="seedvr2_ema_7b-Q4_K_M.gguf",
            vae_model="ema_vae_fp16.safetensors",
            target_resolution=target_resolution,
            seed=seed,
            color_correction=color_correction,
            vae_device="daemon"
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSuperUpscaler": LunaSuperUpscaler,
    "LunaSuperUpscalerSimple": LunaSuperUpscalerSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSuperUpscaler": "Luna Super Upscaler ⚡",
    "LunaSuperUpscalerSimple": "Luna Super Upscaler (Simple)",
}
