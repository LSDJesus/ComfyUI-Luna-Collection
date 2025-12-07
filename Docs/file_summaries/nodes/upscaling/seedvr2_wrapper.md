# upscaling/seedvr2_wrapper.py

## Purpose
Wrapper for SeedVR2 video upscaling pipeline optimized for Luna infrastructure with batched tile processing and daemon VAE integration.

## Exports
- `LunaSeedVR2Pipeline`: Main pipeline wrapper class
- `UpscaleConfig`: Configuration dataclass for upscaling
- `tile_image()`: Function to split images into overlapping tiles
- `untile_image()`: Function to reassemble tiles with blending
- `batch_tiles()`: Function to batch tiles for efficient processing

## Key Imports
- `os`, `sys`, `torch`, `typing`, `dataclasses`
- SeedVR2 components (if available): generation_phases, generation_utils, memory_manager, constants, downloads, debug

## ComfyUI Node Configuration
N/A - Utility module

## Input Schema
N/A

## Key Methods
- `LunaSeedVR2Pipeline.upscale(images, config, progress_callback)` - Main upscaling method
- `LunaSeedVR2Pipeline._prepare_runner(config)` - Prepare inference runner
- `tile_image(image, tile_size, overlap)` - Split image into tiles
- `untile_image(tiles, positions, output_size, overlap)` - Reassemble tiles with blending
- `batch_tiles(tiles, batch_size)` - Batch tiles for processing

## Dependencies
- SeedVR2 custom node (seedvr2_videoupscaler)
- folder_paths for custom node discovery

## Integration Points
- Used by `luna_super_upscaler.py` for high-quality upscaling
- Integrates with Luna daemon for VAE encode/decode operations
- Compatible with Luna Config Gateway

## Notes
Luna-optimized wrapper for SeedVR2 providing batched tile processing (4 tiles at once), optional daemon VAE for VRAM efficiency, and integration with Luna model infrastructure.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\seedvr2_wrapper.py.md