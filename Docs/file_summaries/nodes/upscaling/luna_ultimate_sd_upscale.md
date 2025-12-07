# upscaling/luna_ultimate_sd_upscale.py

## Purpose
Advanced tiled upscaling with diffusion enhancement and seam fixing, supporting TensorRT acceleration and Luna pipe integration.

## Exports
- `Luna_UltimateSDUpscale`: Main upscaler class with tiling and enhancement
- `NODE_CLASS_MAPPINGS`: Node class mappings for ComfyUI
- `NODE_DISPLAY_NAME_MAPPINGS`: Node display name mappings

## Key Imports
- `torch`, `comfy.utils`, `folder_paths`, `PIL`, `numpy`
- `luna_performance_monitor` (optional)
- `trt_engine` (optional)

## ComfyUI Node Configuration
- `CATEGORY = "Luna/Meta"`
- `RETURN_TYPES = ("IMAGE", "PERFORMANCE_STATS")`
- `RETURN_NAMES = ("upscaled_image", "performance_stats")`
- `FUNCTION = "upscale"`
- `OUTPUT_NODE = True`

## Input Schema
**Required:**
- `image: ("IMAGE",)`

**Optional:**
- `luna_pipe: ("LUNA_PIPE",)` - Luna pipe with model/conditioning
- `model_opt: ("MODEL",)` - Diffusion model
- `positive_opt: ("CONDITIONING",)` - Positive conditioning
- `negative_opt: ("CONDITIONING",)` - Negative conditioning
- `vae_opt: ("VAE",)` - VAE model
- `upscale_model: ("UPSCALE_MODEL",)` - ComfyUI upscaler
- `upscaler_trt_model: ("UPSCALER_TRT_MODEL",)` - TensorRT upscaler
- `upscale_by: ("FLOAT",)` - Scale factor
- `tile_width: ("INT",)` - Tile width
- `tile_height: ("INT",)` - Tile height
- `mask_blur: ("INT",)` - Mask blur radius
- `tile_padding: ("INT",)` - Tile padding
- `redraw_mode: (["Linear", "Chess", "None"],)` - Diffusion enhancement mode
- `seam_fix_mode: (["None", "Band Pass", "Half Tile", "Half Tile + Intersections"],)` - Seam fixing method
- `seam_fix_denoise: ("FLOAT",)` - Seam fix denoising strength
- `seam_fix_width: ("INT",)` - Seam fix width
- `seam_fix_mask_blur: ("INT",)` - Seam fix mask blur
- `seam_fix_padding: ("INT",)` - Seam fix padding
- `force_uniform_tiles: ("BOOLEAN",)` - Force uniform tile sizes
- `tiled_decode: ("BOOLEAN",)` - Use tiled VAE decoding
- `seed_opt: ("INT",)` - Random seed
- `steps_opt: ("INT",)` - Sampling steps
- `cfg_opt: ("FLOAT",)` - CFG scale
- `sampler_name_opt: ("STRING",)` - Sampler name
- `scheduler_opt: ("STRING",)` - Scheduler name
- `denoise_opt: ("FLOAT",)` - Denoising strength

## Key Methods
- `upscale(image, luna_pipe=None, ...)` - Main upscaling method
- `basic_upscale(pil_img, target_width, target_height, upscale_model)` - Basic upscaling
- `apply_diffusion_enhancement(pil_img, model, positive, negative, vae, ...)` - Diffusion enhancement
- `apply_seam_fixing(pil_img, model, positive, negative, vae, ...)` - Seam fixing
- `linear_diffusion_enhancement(...)` - Linear tile processing
- `chess_diffusion_enhancement(...)` - Chess pattern tile processing
- `band_pass_seam_fix(...)` - Band pass seam fixing
- `half_tile_seam_fix(...)` - Half tile seam fixing

## Dependencies
- ComfyUI core (torch, comfy.utils, folder_paths)
- PIL, numpy
- luna_performance_monitor (optional)
- trt_engine (optional for TensorRT acceleration)

## Integration Points
- Luna pipe format for model/conditioning bundling
- TensorRT engines for accelerated inference
- ComfyUI samplers and VAE decoders
- Performance monitoring integration

## Notes
Complex tiled upscaling system with multiple diffusion enhancement modes (Linear/Chess), seam fixing techniques (Band Pass/Half Tile), and TensorRT acceleration support for high-performance upscaling.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\luna_ultimate_sd_upscale.md