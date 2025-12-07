# utils/tiling.py

## Purpose
Tiling orchestration utility for large image upscaling providing chessboard and sequential processing strategies to handle memory constraints.

## Exports
- `luna_tiling_orchestrator(image_tensor, model, tile_x, tile_y, overlap, strategy)`: Main tiling function

## Key Imports
- `torch`: Tensor operations for image processing
- `math`: Mathematical calculations for tile calculations

## ComfyUI Node Configuration
N/A - Utility function, not a node.

## Input Schema
N/A - Single utility function.

## Key Methods
- `luna_tiling_orchestrator(...)`: Orchestrate tiled upscaling with overlap handling and strategy selection

## Dependencies
- `torch`: Tensor operations for image manipulation
- `math`: Ceiling calculations for tile grid determination

## Integration Points
- Used by upscaling nodes for large image processing
- Supports 'chess' strategy for alternating tile processing
- Handles overlap regions for seamless tile stitching
- Compatible with various upscaling models with scale attribute

## Notes
- Implements two-pass tiling: primary tiles then secondary (chess strategy)
- Automatic overlap calculation for seamless boundaries
- Memory-efficient processing for large images
- Supports both sequential and chessboard processing patterns
- Used by Luna upscaling nodes for high-resolution image processing