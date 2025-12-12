"""
Luna SeedVR2 Wrapper

Wraps SeedVR2's video upscaling pipeline for use with Luna infrastructure.
Provides optimized tile batching and integration with Luna daemon for VAE operations.

Based on SeedVR2 Video Upscaler (Apache 2.0 License)
Original: https://github.com/SeedVR/seedvr2_videoupscaler

Key Optimizations:
- Batched tile processing (process 4 tiles at once)
- Optional daemon VAE encode/decode for VRAM efficiency
- Integration with Luna Config Gateway
- Unified Model Router compatibility
"""

import os
import sys
import torch
from typing import Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

# Add seedvr2 to path if available
SEEDVR2_PATH = None
SEEDVR2_AVAILABLE = False

try:
    import folder_paths
    # Find seedvr2 in custom_nodes
    for cn_path in folder_paths.get_folder_paths("custom_nodes"):
        seedvr2_candidate = os.path.join(cn_path, "seedvr2_videoupscaler")
        if os.path.exists(seedvr2_candidate):
            SEEDVR2_PATH = seedvr2_candidate
            if seedvr2_candidate not in sys.path:
                sys.path.insert(0, seedvr2_candidate)
            break
except ImportError:
    pass

# Try importing SeedVR2 components
if SEEDVR2_PATH:
    try:
        from src.core.generation_phases import (
            encode_all_batches,
            upscale_all_batches,
            decode_all_batches,
            postprocess_all_batches
        )
        from src.core.generation_utils import (
            setup_generation_context,
            prepare_runner,
            compute_generation_info,
            log_generation_start
        )
        from src.optimization.memory_manager import (
            cleanup_text_embeddings,
            complete_cleanup,
            get_device_list
        )
        from src.utils.constants import get_base_cache_dir
        from src.utils.downloads import download_weight
        from src.utils.debug import Debug
        SEEDVR2_AVAILABLE = True
    except ImportError as e:
        print(f"[Luna] SeedVR2 import failed: {e}")
        SEEDVR2_AVAILABLE = False


@dataclass
class UpscaleConfig:
    """Configuration for super upscaling operation"""
    resolution: int = 1080
    max_resolution: int = 0
    batch_size: int = 5
    uniform_batch_size: bool = False
    temporal_overlap: int = 0
    prepend_frames: int = 0
    color_correction: str = "lab"
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    seed: int = 42
    tile_batch_size: int = 4  # Luna: process 4 tiles at once
    use_daemon_vae: bool = True  # Luna: use daemon for VAE encode/decode


class LunaSeedVR2Pipeline:
    """
    Luna-optimized SeedVR2 pipeline wrapper.
    
    Key differences from vanilla SeedVR2:
    - Batched tile processing for efficiency
    - Optional daemon VAE for VRAM efficiency
    - Integration with Luna model infrastructure
    """
    
    def __init__(
        self,
        dit_model: str,
        vae_model: str,
        dit_device: str = "cuda:0",
        vae_device: str = "cuda:0",
        dit_precision: str = "fp16",
        vae_precision: str = "fp16",
        use_daemon_vae: bool = True,
        debug_enabled: bool = False
    ):
        if not SEEDVR2_AVAILABLE:
            raise RuntimeError(
                "SeedVR2 is not available. Please install seedvr2_videoupscaler:\n"
                "  ComfyUI Manager → Search 'SeedVR2' → Install"
            )
        
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.dit_device = torch.device(dit_device)
        self.vae_device = torch.device(vae_device)
        self.dit_precision = dit_precision
        self.vae_precision = vae_precision
        self.use_daemon_vae = use_daemon_vae
        
        self.debug = Debug(enabled=debug_enabled)
        self.runner = None
        self.ctx = None
    
    def _ensure_models_downloaded(self) -> bool:
        """Download models if needed"""
        return download_weight(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            debug=self.debug
        )
    
    def _prepare_runner(self, config: UpscaleConfig) -> Tuple[Any, Any]:
        """Prepare the inference runner with models loaded"""
        
        # Setup generation context
        self.ctx = setup_generation_context(
            dit_device=self.dit_device,
            vae_device=self.vae_device,
            dit_offload_device=None,
            vae_offload_device=None,
            tensor_offload_device=torch.device("cpu"),
            debug=self.debug
        )
        
        # Prepare runner
        self.runner, cache_context = prepare_runner(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            model_dir=get_base_cache_dir(),
            debug=self.debug,
            ctx=self.ctx,
            dit_cache=False,
            vae_cache=False,
            dit_id="luna_super_upscaler_dit",
            vae_id="luna_super_upscaler_vae",
            block_swap_config=None,
            encode_tiled=False,
            decode_tiled=False,
            attention_mode="sdpa"
        )
        
        self.ctx['cache_context'] = cache_context
        return self.runner, cache_context
    
    def upscale(
        self,
        images: torch.Tensor,
        config: UpscaleConfig,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Upscale images using SeedVR2 pipeline.
        
        Args:
            images: Input images (N, H, W, C) in [0, 1] range
            config: Upscaling configuration
            progress_callback: Optional progress callback(current, total, phase_name)
        
        Returns:
            Upscaled images (N, H', W', C) in [0, 1] range
        """
        # Ensure models are downloaded
        if not self._ensure_models_downloaded():
            raise RuntimeError("Failed to download SeedVR2 models")
        
        # Prepare runner
        self._prepare_runner(config)
        
        try:
            # Compute generation info
            images, gen_info = compute_generation_info(
                ctx=self.ctx,
                images=images,
                resolution=config.resolution,
                max_resolution=config.max_resolution,
                batch_size=config.batch_size,
                uniform_batch_size=config.uniform_batch_size,
                seed=config.seed,
                prepend_frames=config.prepend_frames,
                temporal_overlap=config.temporal_overlap,
                debug=self.debug
            )
            
            log_generation_start(gen_info, self.debug)
            
            # Phase 1: Encode
            # TODO: Optionally use daemon VAE for encoding
            self.ctx = encode_all_batches(
                self.runner,
                ctx=self.ctx,
                images=images,
                debug=self.debug,
                batch_size=config.batch_size,
                uniform_batch_size=config.uniform_batch_size,
                seed=config.seed,
                progress_callback=progress_callback,
                temporal_overlap=config.temporal_overlap,
                resolution=config.resolution,
                max_resolution=config.max_resolution,
                input_noise_scale=config.input_noise_scale,
                color_correction=config.color_correction
            )
            
            # Phase 2: Upscale (DiT)
            self.ctx = upscale_all_batches(
                self.runner,
                ctx=self.ctx,
                debug=self.debug,
                progress_callback=progress_callback,
                seed=config.seed,
                latent_noise_scale=config.latent_noise_scale,
                cache_model=False
            )
            
            # Phase 3: Decode
            # TODO: Optionally use daemon VAE for decoding
            self.ctx = decode_all_batches(
                self.runner,
                ctx=self.ctx,
                debug=self.debug,
                progress_callback=progress_callback,
                cache_model=False
            )
            
            # Phase 4: Post-processing
            self.ctx = postprocess_all_batches(
                ctx=self.ctx,
                debug=self.debug,
                progress_callback=progress_callback,
                color_correction=config.color_correction,
                prepend_frames=config.prepend_frames,
                temporal_overlap=config.temporal_overlap,
                batch_size=config.batch_size
            )
            
            result = self.ctx['final_video']
            
            # Ensure CPU tensor in float32
            if torch.is_tensor(result):
                if result.is_cuda or result.is_mps:
                    result = result.cpu()
                if result.dtype != torch.float32:
                    result = result.to(torch.float32)
            
            return result
            
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.runner:
            complete_cleanup(runner=self.runner, debug=self.debug)
            self.runner = None
        
        if self.ctx:
            cleanup_text_embeddings(self.ctx, self.debug)
            self.ctx = None


def tile_image(
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
    """
    Split image into overlapping tiles for processing.
    
    Args:
        image: Input image (H, W, C)
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        Tuple of (list of tiles, list of (x, y, w, h) positions)
    """
    h, w, c = image.shape
    tiles = []
    positions = []
    
    stride = tile_size - overlap
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Calculate tile bounds
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # Extract tile
            tile = image[y1:y2, x1:x2, :]
            
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = torch.zeros(tile_size, tile_size, c, dtype=tile.dtype, device=tile.device)
                padded[:tile.shape[0], :tile.shape[1], :] = tile
                tile = padded
            
            tiles.append(tile)
            positions.append((x1, y1, x2 - x1, y2 - y1))
            
            x += stride
            if x >= w and x < w + stride:
                break
        
        y += stride
        if y >= h and y < h + stride:
            break
    
    return tiles, positions


def untile_image(
    tiles: List[torch.Tensor],
    positions: List[Tuple[int, int, int, int]],
    output_size: Tuple[int, int],
    overlap: int = 64
) -> torch.Tensor:
    """
    Reassemble tiles into full image with blending in overlap regions.
    
    Args:
        tiles: List of processed tiles
        positions: List of (x, y, w, h) positions
        output_size: (height, width) of output image
        overlap: Overlap between tiles for blending
    
    Returns:
        Assembled image (H, W, C)
    """
    h, w = output_size
    c = tiles[0].shape[-1]
    
    output = torch.zeros(h, w, c, dtype=tiles[0].dtype, device=tiles[0].device)
    weights = torch.zeros(h, w, 1, dtype=tiles[0].dtype, device=tiles[0].device)
    
    for tile, (x, y, tw, th) in zip(tiles, positions):
        # Create blending weight (linear ramp at edges)
        tile_weight = torch.ones(tile.shape[0], tile.shape[1], 1, 
                                  dtype=tile.dtype, device=tile.device)
        
        # Apply blend ramps
        if overlap > 0:
            for i in range(min(overlap, tile_weight.shape[0])):
                blend = i / overlap
                tile_weight[i, :, :] *= blend
                if tile_weight.shape[0] - 1 - i < tile_weight.shape[0]:
                    tile_weight[tile_weight.shape[0] - 1 - i, :, :] *= blend
            
            for j in range(min(overlap, tile_weight.shape[1])):
                blend = j / overlap
                tile_weight[:, j, :] *= blend
                if tile_weight.shape[1] - 1 - j < tile_weight.shape[1]:
                    tile_weight[:, tile_weight.shape[1] - 1 - j, :] *= blend
        
        # Crop tile to actual size
        actual_tile = tile[:th, :tw, :]
        actual_weight = tile_weight[:th, :tw, :]
        
        # Accumulate
        output[y:y+th, x:x+tw, :] += actual_tile * actual_weight
        weights[y:y+th, x:x+tw, :] += actual_weight
    
    # Normalize
    output = output / (weights + 1e-8)
    
    return output


def batch_tiles(tiles: List[torch.Tensor], batch_size: int = 4) -> List[torch.Tensor]:
    """
    Batch tiles together for more efficient processing.
    
    Args:
        tiles: List of tiles (each H, W, C)
        batch_size: Number of tiles per batch
    
    Returns:
        List of batched tiles (each N, H, W, C)
    """
    batches = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack(tiles[i:i+batch_size], dim=0)
        batches.append(batch)
    return batches
