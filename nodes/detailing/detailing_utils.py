"""
Detailing Utilities - COMPATIBILITY SHIM

This module re-exports from the new split modules for backwards compatibility.
New code should import directly from:
- tile_ops
- image_utils  
- conditioning_utils
- refinement_engine

This file will be deprecated in a future version.
"""

# Re-export everything for backwards compatibility
from .tile_ops import (
    TileInfo,
    extract_tiles,
    composite_tiles,
    calculate_chess_grid,
    calculate_latent_chess_grid,
)

from .image_utils import (
    resize_image,
    lanczos_resize,
    normalize_mask,
    resize_mask,
    create_feathered_blend_mask,
    create_overlap_blend_mask,
    smoothstep,
    generate_scaled_noise,
    encode_vision_crops,
)

from .conditioning_utils import (
    encode_pixels,
    decode_latents,
    replicate_conditioning,
    apply_ip_adapter_batch,
    fuse_vision_conditioning,
    fuse_vision_conditioning_batch,
)

from .refinement_engine import (
    RefinementBatch,
    RefinementConfig,
    RefinementResult,
    RefinementEngine,
    create_refinement_batch,
    quick_refine,
)


__all__ = [
    # Tile Operations
    "TileInfo",
    "extract_tiles",
    "composite_tiles",
    "calculate_chess_grid",
    "calculate_latent_chess_grid",
    
    # Image Utilities
    "resize_image",
    "lanczos_resize",
    "normalize_mask",
    "resize_mask",
    "create_feathered_blend_mask",
    "create_overlap_blend_mask",
    "smoothstep",
    "generate_scaled_noise",
    "encode_vision_crops",
    
    # Conditioning
    "encode_pixels",
    "decode_latents",
    "replicate_conditioning",
    "apply_ip_adapter_batch",
    "fuse_vision_conditioning",
    "fuse_vision_conditioning_batch",
    
    # Refinement Engine
    "RefinementBatch",
    "RefinementConfig",
    "RefinementResult",
    "RefinementEngine",
    "create_refinement_batch",
    "quick_refine",
]
