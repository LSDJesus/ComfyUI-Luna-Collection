"""
Image Utilities - Image manipulation, masks, and blending

Handles:
- Image resizing (BHWC-aware)
- Mask normalization and resizing
- Blend mask creation (feathered, overlap)
- Noise generation
- Vision encoding helpers

Used by: tile_ops, RefinementEngine, LunaChessRefiner, LunaSemanticDetailer
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any


# =============================================================================
# Image Resizing
# =============================================================================

def resize_image(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear"
) -> torch.Tensor:
    """
    Resize image tensor.
    
    Args:
        image: [B, H, W, C] image tensor (BHWC format)
        target_size: (height, width) target
        mode: Interpolation mode
    
    Returns:
        Resized image [B, target_h, target_w, C]
    """
    # Convert BHWC → BCHW for interpolate
    img_bchw = image.permute(0, 3, 1, 2)
    
    resized = F.interpolate(
        img_bchw,
        size=target_size,
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None
    )
    
    # Convert back BCHW → BHWC
    return resized.permute(0, 2, 3, 1)


def lanczos_resize(
    image: torch.Tensor,
    target_h: int,
    target_w: int
) -> torch.Tensor:
    """
    High-quality Lanczos resize for images.
    
    Uses PIL for true Lanczos filtering (better than bicubic).
    
    Args:
        image: [B, H, W, C] image tensor
        target_h: Target height
        target_w: Target width
    
    Returns:
        Resized image [B, target_h, target_w, C]
    """
    try:
        from PIL import Image
        import numpy as np
        
        results = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            result_np = np.array(pil_resized).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np).to(image.device))
        
        return torch.stack(results, dim=0)
        
    except Exception:
        # Fallback to bicubic
        return resize_image(image, (target_h, target_w), mode="bicubic")


# =============================================================================
# Mask Helpers
# =============================================================================

def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Normalize mask to consistent format [H, W].
    
    Handles various input formats:
    - [H, W] - returned as-is
    - [1, H, W] - squeezed
    - [1, 1, H, W] - squeezed
    - [B, H, W] - first item
    - [B, C, H, W] - first item, first channel
    
    Args:
        mask: Input mask tensor
    
    Returns:
        2D mask tensor [H, W]
    """
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    
    return mask


def resize_mask(
    mask: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear"
) -> torch.Tensor:
    """
    Resize mask to target size.
    
    Args:
        mask: Input mask (any dimensionality)
        target_size: (height, width) tuple
        mode: Interpolation mode
    
    Returns:
        Resized mask [H, W]
    """
    # Normalize to 2D first
    mask = normalize_mask(mask)
    
    # Add batch and channel dims for interpolate
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    
    resized = F.interpolate(
        mask_4d,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None
    )
    
    return resized[0, 0]


# =============================================================================
# Blend Mask Helpers
# =============================================================================

def create_feathered_blend_mask(
    height: int,
    width: int,
    feather_pixels: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create a blend mask with feathered edges.
    
    Args:
        height: Mask height
        width: Mask width
        feather_pixels: Number of pixels to feather
        device: Target device
    
    Returns:
        Blend mask [1, 1, H, W] with values 0-1
    """
    mask = torch.ones(height, width, device=device)
    
    if feather_pixels > 0:
        # Create linear ramps for each edge
        for i in range(feather_pixels):
            alpha = (i + 1) / feather_pixels
            
            # Top edge
            mask[i, :] *= alpha
            # Bottom edge
            mask[-(i+1), :] *= alpha
            # Left edge
            mask[:, i] *= alpha
            # Right edge
            mask[:, -(i+1)] *= alpha
    
    return mask.unsqueeze(0).unsqueeze(0)


def create_overlap_blend_mask(
    height: int,
    width: int,
    overlap_h: int,
    overlap_w: int,
    feather: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Create a blend mask optimized for overlapping tiles.
    
    The mask has stronger feathering in overlap regions.
    
    Args:
        height: Tile height
        width: Tile width
        overlap_h: Vertical overlap in pixels
        overlap_w: Horizontal overlap in pixels
        feather: Feathering amount
        device: Target device
    
    Returns:
        Blend mask [1, 1, H, W]
    """
    mask = torch.ones(height, width, device=device)
    
    # Apply feathering to overlap regions
    eff_feather_h = min(feather, overlap_h) if overlap_h > 0 else feather
    eff_feather_w = min(feather, overlap_w) if overlap_w > 0 else feather
    
    # Horizontal ramps
    if eff_feather_w > 0:
        for i in range(eff_feather_w):
            alpha = (i + 1) / eff_feather_w
            mask[:, i] *= alpha
            mask[:, -(i+1)] *= alpha
    
    # Vertical ramps
    if eff_feather_h > 0:
        for i in range(eff_feather_h):
            alpha = (i + 1) / eff_feather_h
            mask[i, :] *= alpha
            mask[-(i+1), :] *= alpha
    
    return mask.unsqueeze(0).unsqueeze(0)


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    """
    Smoothstep function for smooth blending transitions.
    
    f(t) = 3t² - 2t³
    
    Args:
        t: Input values in [0, 1] range
    
    Returns:
        Smoothed values
    """
    return t * t * (3 - 2 * t)


# =============================================================================
# Noise Helpers
# =============================================================================

def generate_scaled_noise(
    shape: Tuple[int, ...],
    seed: int,
    scale_factor: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Generate noise with variance correction for different scales.
    
    When downscaling during tiled refinement, the effective noise
    needs to be scaled to maintain consistent denoising behavior.
    
    Args:
        shape: Output tensor shape
        seed: Random seed for reproducibility
        scale_factor: Variance scaling factor
        device: Target device
    
    Returns:
        Noise tensor
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    noise = torch.randn(shape, generator=generator, device=device)
    
    if scale_factor != 1.0:
        noise = noise * scale_factor
    
    return noise


# =============================================================================
# Vision Router Helpers
# =============================================================================

def encode_vision_crops(
    vision_router: Any,
    image: torch.Tensor,
    crop_coords: List[Tuple[int, int, int, int]],
    tile_size: int = 1024
) -> List[torch.Tensor]:
    """
    Encode image crops with CLIP-ViT for IP-Adapter.
    
    Args:
        vision_router: VisionRouter instance (daemon or local)
        image: Full image tensor [1, H, W, C]
        crop_coords: List of (x1, y1, x2, y2) coordinates
        tile_size: Target size for vision encoding
    
    Returns:
        List of vision embeddings, one per crop
    """
    if vision_router is None or not vision_router.available:
        return []
    
    return vision_router.encode_crops(image, crop_coords, tile_size)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Resizing
    "resize_image",
    "lanczos_resize",
    
    # Masks
    "normalize_mask",
    "resize_mask",
    
    # Blend Masks
    "create_feathered_blend_mask",
    "create_overlap_blend_mask",
    "smoothstep",
    
    # Noise
    "generate_scaled_noise",
    
    # Vision
    "encode_vision_crops",
]
