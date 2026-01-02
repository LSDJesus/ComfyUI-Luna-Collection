"""
Tile Operations - Geometry and manipulation for tiled refinement

Handles:
- Tile extraction from images
- Tile compositing back to canvas
- Chess grid pattern calculation
- Coordinate manipulation

Used by: LunaChessRefiner, LunaSemanticDetailer
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TileInfo:
    """Information about an extracted tile."""
    index: int                    # Tile index in batch
    x: int                        # Left coordinate in source
    y: int                        # Top coordinate in source
    width: int                    # Tile width
    height: int                   # Tile height
    original_width: int           # Original width before resize
    original_height: int          # Original height before resize
    
    @property
    def coords(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) coordinates."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


# =============================================================================
# Tile Extraction
# =============================================================================

def extract_tiles(
    image: torch.Tensor,
    tile_coords: List[Tuple[int, int, int, int]],
    target_size: Optional[Tuple[int, int]] = None,
    padding: int = 0
) -> Tuple[torch.Tensor, List[TileInfo]]:
    """
    Extract tiles from an image at specified coordinates.
    
    Args:
        image: Source image [1, H, W, C] or [H, W, C]
        tile_coords: List of (x1, y1, x2, y2) coordinates
        target_size: Optional (height, width) to resize tiles to
        padding: Extra padding around each tile
    
    Returns:
        Tuple of:
        - Stacked tiles [N, H, W, C]
        - List of TileInfo for compositing
    """
    from .image_utils import resize_image
    
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    tiles = []
    tile_infos = []
    
    _, img_h, img_w, _ = image.shape
    
    for idx, (x1, y1, x2, y2) in enumerate(tile_coords):
        # Apply padding
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(img_w, x2 + padding)
        y2_pad = min(img_h, y2 + padding)
        
        # Extract tile
        tile = image[:, y1_pad:y2_pad, x1_pad:x2_pad, :]
        
        orig_h, orig_w = tile.shape[1], tile.shape[2]
        
        # Resize if target size specified
        if target_size is not None:
            tile = resize_image(tile, target_size)
        
        tiles.append(tile)
        tile_infos.append(TileInfo(
            index=idx,
            x=x1_pad,
            y=y1_pad,
            width=x2_pad - x1_pad,
            height=y2_pad - y1_pad,
            original_width=orig_w,
            original_height=orig_h
        ))
    
    # Stack into batch
    batch = torch.cat(tiles, dim=0)
    
    return batch, tile_infos


def composite_tiles(
    canvas: torch.Tensor,
    tiles: torch.Tensor,
    tile_infos: List[TileInfo],
    blend_masks: Optional[torch.Tensor] = None,
    feather: int = 0
) -> torch.Tensor:
    """
    Composite refined tiles back onto canvas.
    
    Args:
        canvas: Target canvas [1, H, W, C]
        tiles: Refined tiles [N, H, W, C]
        tile_infos: List of TileInfo from extraction
        blend_masks: Optional [N, H, W] blend weights
        feather: Feathering pixels for edge blending
    
    Returns:
        Canvas with tiles composited
    """
    from .image_utils import resize_image, create_feathered_blend_mask
    
    result = canvas.clone()
    
    for i, info in enumerate(tile_infos):
        tile = tiles[i:i+1]
        
        # Resize tile back to original size if needed
        if tile.shape[1] != info.height or tile.shape[2] != info.width:
            tile = resize_image(tile, (info.height, info.width))
        
        # Get blend mask
        if blend_masks is not None:
            mask = blend_masks[i]
        elif feather > 0:
            mask = create_feathered_blend_mask(
                info.height, info.width, feather, 
                device=tile.device
            ).squeeze(0).squeeze(0)
        else:
            mask = None
        
        # Composite
        if mask is not None:
            # Expand mask to match tile channels
            mask = mask.unsqueeze(-1)  # [H, W, 1]
            result[:, info.y:info.y+info.height, info.x:info.x+info.width, :] = (
                result[:, info.y:info.y+info.height, info.x:info.x+info.width, :] * (1 - mask) +
                tile * mask
            )
        else:
            # Direct paste
            result[:, info.y:info.y+info.height, info.x:info.x+info.width, :] = tile
    
    return result


# =============================================================================
# Chess Grid Calculation
# =============================================================================

def calculate_chess_grid(
    height: int,
    width: int,
    tile_size: int,
    overlap_ratio: float = 0.25
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], int, int]:
    """
    Calculate chess pattern tile grid with overlap.
    
    The chess pattern processes tiles in two passes:
    - Pass 1 (even parity): Tiles at even row+col positions
    - Pass 2 (odd parity): Tiles at odd row+col positions
    
    This ensures each pass has fresh context from already-refined neighbors,
    eliminating seams without post-processing.
    
    Args:
        height: Image height
        width: Image width
        tile_size: Tile size in pixels
        overlap_ratio: Overlap as fraction of tile size (0.0-0.5)
    
    Returns:
        Tuple of:
        - even_tiles: List of (x, y, w, h) for even parity tiles
        - odd_tiles: List of (x, y, w, h) for odd parity tiles
        - overlap_h: Vertical overlap in pixels
        - overlap_w: Horizontal overlap in pixels
    """
    # Calculate overlap
    overlap_h = int(tile_size * overlap_ratio)
    overlap_w = int(tile_size * overlap_ratio)
    
    # Calculate step size (tile size minus overlap)
    step_h = tile_size - overlap_h
    step_w = tile_size - overlap_w
    
    # Calculate grid dimensions
    rows = max(1, (height - overlap_h + step_h - 1) // step_h)
    cols = max(1, (width - overlap_w + step_w - 1) // step_w)
    
    even_tiles = []
    odd_tiles = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile position
            y = row * step_h
            x = col * step_w
            
            # Clamp to image bounds
            y = min(y, max(0, height - tile_size))
            x = min(x, max(0, width - tile_size))
            
            # Determine parity
            parity = (row + col) % 2
            
            tile_info = (x, y, tile_size, tile_size)
            
            if parity == 0:
                even_tiles.append(tile_info)
            else:
                odd_tiles.append(tile_info)
    
    return even_tiles, odd_tiles, overlap_h, overlap_w


def calculate_latent_chess_grid(
    latent_h: int,
    latent_w: int,
    tile_size: int = 128,
    overlap_ratio: float = 0.25
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], int, int]:
    """
    Calculate chess pattern for latent space dimensions.
    
    Convenience wrapper that works directly with latent dimensions.
    Tile size is in latent pixels (typically 1/8 of pixel space).
    
    Args:
        latent_h: Latent height
        latent_w: Latent width
        tile_size: Tile size in latent pixels (128 = 1024px)
        overlap_ratio: Overlap as fraction
    
    Returns:
        Same as calculate_chess_grid
    """
    return calculate_chess_grid(latent_h, latent_w, tile_size, overlap_ratio)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TileInfo",
    "extract_tiles",
    "composite_tiles",
    "calculate_chess_grid",
    "calculate_latent_chess_grid",
]
