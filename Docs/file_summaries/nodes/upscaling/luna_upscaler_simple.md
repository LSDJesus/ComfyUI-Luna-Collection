# upscaling/luna_upscaler_simple.py

## Purpose
Simple upscaler with basic functionality, optional TensorRT acceleration, and Luna validation integration.

## Exports
- `Luna_SimpleUpscaler`: Main upscaler class
- `NODE_CLASS_MAPPINGS`: Node class mappings for ComfyUI
- `NODE_DISPLAY_NAME_MAPPINGS`: Node display name mappings

## Key Imports
- `torch`, `comfy.utils`, `comfy.model_management`, `folder_paths`
- `PIL`, `numpy`, `spandrel`
- `trt_engine` (optional)
- `validation` (optional)

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
- `resampling: (["bicubic", "bilinear"],)` - Resampling method
- `show_preview: ("BOOLEAN",)` - Show preview output
- `upscale_model: ("UPSCALE_MODEL",)` - ComfyUI upscaler model

**Optional:**
- `tensorrt_engine_path: ("STRING",)` - TensorRT engine path (if available)

## Key Methods
- `upscale(image, scale_by, resampling, show_preview, upscale_model, tensorrt_engine_path)` - Main upscaling method

## Dependencies
- ComfyUI core (torch, comfy.utils, comfy.model_management, folder_paths)
- spandrel for model loading
- trt_engine (optional for TensorRT acceleration)
- validation (optional for Luna validation system)

## Integration Points
- Uses `comfy.utils.tiled_scale` for basic tiling
- TensorRT engines for accelerated inference
- Luna validation system for input validation

## Notes
Simplified upscaler with basic tiling using ComfyUI's tiled_scale, GPU-native resizing, and optional TensorRT acceleration. Includes conditional validation decorator.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\luna_upscaler_simple.py.md