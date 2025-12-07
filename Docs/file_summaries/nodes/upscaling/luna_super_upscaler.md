# upscaling/luna_super_upscaler.py

## Purpose
Unified super-resolution node using SeedVR2 DiT upscaling with Luna infrastructure for efficient high-quality image upscaling.

## Exports
- `LunaSuperUpscaler`: Main upscaler with full configuration options
- `LunaSuperUpscalerSimple`: Simplified version using Luna Config Gateway
- `NODE_CLASS_MAPPINGS`: Node class mappings for ComfyUI
- `NODE_DISPLAY_NAME_MAPPINGS`: Node display name mappings

## Key Imports
- `os`, `torch`, `typing`
- `folder_paths` (optional)
- `comfy.model_management` (optional)
- `seedvr2_wrapper` (relative import)

## ComfyUI Node Configuration
- `CATEGORY = "Luna/Upscaling"`
- `RETURN_TYPES = ("IMAGE",)`
- `RETURN_NAMES = ("upscaled_image",)`
- `FUNCTION = "upscale"`

## Input Schema
**Required:**
- `image: ("IMAGE",)`
- `dit_model: (DIT_MODELS,)` - DiT model selection
- `vae_model: (VAE_MODELS,)` - VAE model selection
- `target_resolution: ("INT",)` - Target resolution for shortest edge
- `seed: ("INT",)` - Random seed

**Optional:**
- `tile_size: ("INT",)` - Processing tile size
- `tile_overlap: ("INT",)` - Tile overlap for blending
- `tile_batch_size: ("INT",)` - Number of tiles to process together
- `color_correction: (COLOR_CORRECTIONS,)` - Color correction method
- `dit_device: (["cuda:0", "cuda:1", "cpu"],)` - Device for DiT
- `vae_device: (["cuda:0", "cuda:1", "cpu", "daemon"],)` - Device for VAE
- `enable_debug: ("BOOLEAN",)` - Debug logging
- `film_grain_intensity: ("FLOAT",)` - Film grain preprocessing
- `film_grain_saturation: ("FLOAT",)` - Film grain color saturation

## Key Methods
- `upscale(image, dit_model, vae_model, target_resolution, seed, ...) -> Tuple[torch.Tensor]`

## Dependencies
- SeedVR2 (via seedvr2_wrapper)
- folder_paths (optional)
- comfy.model_management (optional)

## Integration Points
- Uses `seedvr2_wrapper` for SeedVR2 pipeline
- Integrates with Luna daemon for VAE operations
- Compatible with Luna Config Gateway

## Notes
High-quality diffusion-based upscaling with automatic tiling, batch processing, film grain preprocessing, and multiple color correction methods.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\luna_super_upscaler.md