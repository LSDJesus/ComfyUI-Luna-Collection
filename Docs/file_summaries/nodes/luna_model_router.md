# luna_model_router.py

## Purpose
Primary model loader node with explicit architecture type selection. Supports FLUX, SD1.5, SDXL, SD3, Z-IMAGE with preset configurations and GGUF loading.

## Exports
**Classes:**
- `LunaModelRouter` - ComfyUI node for explicit model loading

**Functions:**
- None

**Constants:**
- `CLIP_TYPE_MAP` - Maps model types to ComfyUI CLIPType strings
- `CLIP_REQUIREMENTS` - CLIP configuration requirements by model type
- `MODEL_SOURCES` - Available model source folders
- `MODEL_TYPES` - Supported model architectures
- `PRECISION_OPTIONS` - Dynamic precision conversion options
- `DAEMON_MODES` - Daemon routing modes

## Key Imports
- `folder_paths` - ComfyUI model path resolution
- `comfy.sd` - Model and CLIP loading functions
- `comfy.utils` - Utility functions
- `nodes` - CLIPVisionLoader, VAELoader
- `aiohttp` - Web API endpoints
- `torch` - PyTorch operations
- `pathlib`, `os` - File system operations
- Optional: `luna_daemon` modules, ComfyUI-GGUF modules

## ComfyUI Node Configuration
- **Category:** `Luna/Core`
- **Display Name:** `Luna Model Router ⚡`
- **Return Types:** `(MODEL, CLIP, VAE, LLM, CLIP_VISION, STRING, STRING)`
- **Return Names:** `(model, clip, vae, llm, clip_vision, model_name, status)`
- **Function:** `load`

## Input Schema
**Required:**
- `model_source` (MODEL_SOURCES): Folder to load model from. Changes which models appear in the dropdown.
- `model_name` (all_models): Model file to load (filtered by model_source)
- `model_type` (MODEL_TYPES): Model architecture - determines CLIP requirements
- `dynamic_precision` (PRECISION_OPTIONS): Enable to auto-convert UNet to optimized precision. 'None' = use source precision.
- `clip_1` (clip_list): Primary CLIP encoder (CLIP-L for most, Qwen3 for Z-IMAGE)
- `clip_2` (clip_list): Secondary CLIP encoder (CLIP-G for SDXL/SD3)
- `clip_3` (clip_list): Tertiary CLIP encoder (T5-XXL for Flux/SD3)
- `clip_4` (clip_list): Vision encoder (SigLIP/CLIP-H for vision models)
- `vae_name` (vae_list): VAE for encoding/decoding. 'None' uses VAE from checkpoint.
- `daemon_mode` (DAEMON_MODES): auto: use daemon if running | force_daemon: require daemon | force_local: never use daemon

**Optional:**
- `local_weights_dir` (STRING): Directory for converted UNet cache (default: models/unet/optimized)

## Key Methods/Functions
- `load(model_source, model_name, model_type, dynamic_precision, clip_1, clip_2, clip_3, clip_4, vae_name, daemon_mode, **kwargs) -> (MODEL, CLIP, VAE, LLM, CLIP_VISION, str, str)`
  - Main entry point, routes to preset-specific loaders
  - Validates CLIP configuration, loads model with optional precision conversion
  - Handles daemon routing for CLIP/VAE, supports Z-IMAGE Qwen3 loading
  - Returns model components and status string
- `_validate_clip_config(model_type, clip_config) -> None`
  - Validates CLIP configuration for selected model type
  - Raises RuntimeError if required CLIPs are missing
- `_load_model(source, name, precision, local_weights_dir) -> (Any, str)`
  - Loads model from specified source with optional precision conversion
  - Supports GGUF loading, checkpoint extraction, dynamic precision conversion
- `_load_standard_clip(model_type, clip_config, daemon_running, use_daemon) -> Any`
  - Loads and combines standard CLIP encoders for non-Z-IMAGE models
  - Routes through daemon if available, falls back to local loading
- `_load_zimage_clip_and_llm(clip_config, daemon_running, use_daemon) -> (Any, Any)`
  - Loads Z-IMAGE CLIP and LLM from full Qwen3 model
  - Uses Lumina2 CLIP type for hidden state extraction
  - Auto-detects mmproj for vision features
- `_load_clip_vision(model_type, clip_config, daemon_running, use_daemon) -> Any`
  - Loads vision encoder for image→embedding conversion
  - Supports CLIP-H/SigLIP for standard models, mmproj for Z-IMAGE
- `_load_vae(vae_name, daemon_running, use_daemon) -> Any`
  - Loads VAE with optional daemon routing
  - Detects VAE type from filename for daemon registration

## Dependencies
**Internal:**
- None (standalone node)

**External:**
- Required: `comfy`, `folder_paths`, `torch`, `aiohttp`
- Optional: `luna_daemon` modules (for VRAM sharing), ComfyUI-GGUF (for .gguf files)

## Integration Points
**Input:** Model filenames from ComfyUI model directories (checkpoints, diffusion_models, unet, clip, vae)
**Output:** (MODEL, CLIP, VAE, LLM, CLIP_VISION, model_name, status) tuple for samplers and downstream nodes
**Side Effects:** Model loading into VRAM, file I/O for cached precision conversions, web API registration

## Notes
- Z-IMAGE uses 'lumina2' CLIPType for Qwen3 hidden state extraction
- GGUF loading auto-detects mmproj files with naming pattern `*mmproj*.gguf`
- Attaches `model_path` and `mmproj_path` to CLIP object for downstream nodes
- Supports dynamic precision conversion with caching (bf16, fp16, fp8, GGUF)
- Web API endpoints for dynamic model list filtering by source folder