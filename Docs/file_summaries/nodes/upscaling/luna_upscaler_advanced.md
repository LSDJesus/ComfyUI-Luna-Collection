# upscaling/luna_upscaler_advanced.py

## Purpose
Advanced upscaler with tiling strategies, TensorRT acceleration, and adaptive resizing for high-quality image upscaling.

## Exports
- `Luna_Advanced_Upscaler`: Main upscaler class
- `NODE_CLASS_MAPPINGS`: Node class mappings for ComfyUI
- `NODE_DISPLAY_NAME_MAPPINGS`: Node display name mappings

## Key Imports
- `torch`, `comfy.utils`, `comfy.model_management`, `folder_paths`
- `PIL`, `numpy`, `spandrel`
- `trt_engine` (optional)

## ComfyUI Node Configuration
- `CATEGORY = "Luna/Upscaling"`
- `RETURN_TYPES = ("IMAGE",)`
- `RETURN_NAMES = ("upscaled_image",)`
- `FUNCTION = "upscale"`
- `OUTPUT_NODE = True`

## Input Schema
**Required:**
- `image: ("IMAGE",)`
- `scale_by: ("FLOAT",)` - Scale factor
- `resampling: (["bicubic", "bilinear", "lanczos", "nearest-exact", "area"],)` - Resampling method
- `supersample: ("BOOLEAN",)` - Enable supersampling
- `rescale_after_model: ("BOOLEAN",)` - Rescale after model processing
- `tile_strategy: (["linear", "chess", "none"],)` - Tiling strategy
- `tile_mode: (["default", "auto"],)` - Tile sizing mode
- `tile_resolution: ("INT",)` - Tile resolution
- `tile_overlap: ("INT",)` - Tile overlap
- `show_preview: ("BOOLEAN",)` - Show preview output
- `upscale_model: ("UPSCALE_MODEL",)` - ComfyUI upscaler model

**Optional:**
- `tensorrt_engine_path: ("STRING",)` - TensorRT engine path (if available)

## Key Methods
- `upscale(image, scale_by, resampling, supersample, rescale_after_model, tile_strategy, tile_mode, tile_resolution, tile_overlap, show_preview, upscale_model, tensorrt_engine_path)` - Main upscaling method
- `_basic_tiling_upscale(in_img, upscale_model, tile_x, tile_y, tile_overlap)` - Basic tiling implementation
- `_adaptive_resize(tensor, target_height, target_width, resampling, use_antialias)` - Adaptive resizing with fallback
- `_calculate_auto_tile_size(image_width, image_height, target_resolution)` - Auto tile size calculation

## Dependencies
- ComfyUI core (torch, comfy.utils, comfy.model_management, folder_paths)
- spandrel for model loading
- trt_engine (optional for TensorRT acceleration)

## Integration Points
- luna_tiling_orchestrator for advanced tiling (if available)
- TensorRT engines for accelerated inference
- ComfyUI upscale models

## Notes
Advanced upscaling with configurable tiling strategies (linear/chess/none), auto tile sizing, adaptive resizing with CPU fallback for memory issues, and optional TensorRT acceleration.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\luna_upscaler_advanced.py.md