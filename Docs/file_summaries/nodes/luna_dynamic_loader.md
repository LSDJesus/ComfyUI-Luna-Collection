# luna_dynamic_loader.py

## Purpose
Smart precision loader for multi-GPU setups. Extracts UNet from checkpoints, converts to optimized precision, and stores locally while loading CLIP/VAE from source on-demand.

## Exports
**Classes:**
- `LunaDynamicModelLoader` - Smart precision loader with lazy evaluation
- `LunaOptimizedWeightsManager` - Utility for managing local optimized weights

**Functions:**
- `get_unet_keys(state_dict) -> Dict[str, torch.Tensor]` - Extracts UNet-related tensors from checkpoint
- `convert_to_precision(src_path, dst_path, precision, strip_components) -> (float, float)` - Converts checkpoint to target precision safetensors
- `convert_to_gguf(src_path, dst_path, quant_type) -> (float, float)` - Converts checkpoint to GGUF format

**Constants:**
- `PRECISION_OPTIONS` - Available precision options with hardware notes

## Key Imports
- `folder_paths` - ComfyUI model path resolution
- `comfy.sd`, `comfy.utils` - ComfyUI model loading utilities
- `safetensors.torch` - SafeTensor file operations (load_file, save_file)
- `torch` - PyTorch tensor operations
- `gguf` - GGUF format support (optional)
- ComfyUI-GGUF nodes (optional import)

## ComfyUI Node Configuration
- **LunaDynamicModelLoader**
  - Category: `Luna/Loaders`
  - Display Name: `Luna Dynamic Model Loader`
  - Return Types: `(MODEL, CLIP, VAE, STRING)`
  - Return Names: `(model, clip, vae, unet_path)`
  - Function: `load_smart`
- **LunaOptimizedWeightsManager**
  - Category: `Luna/Utilities`
  - Display Name: `Luna Optimized Weights Manager`
  - Return Types: `(STRING,)`
  - Return Names: `(report,)`
  - Function: `manage_weights`
  - OUTPUT_NODE: `True`

## Input Schema
**LunaDynamicModelLoader:**
- `ckpt_name` (checkpoints list): Source checkpoint (FP16/FP32 on HDD)
- `precision` (PRECISION_OPTIONS): Target UNet precision (bf16, fp16, fp8, GGUF variants)
- `local_weights_dir` (STRING, optional): Local NVMe directory for optimized UNets

**LunaOptimizedWeightsManager:**
- `action` (["list", "stats", "clear_old", "purge_all"]): Management action
- `weights_directory` (STRING, optional): Override weights location
- `days_old` (INT, default=30): Age threshold for clear_old action

## Key Methods/Functions
- `LunaDynamicModelLoader.load_smart(ckpt_name, precision, local_weights_dir="", **kwargs) -> (Any, Any, Any, str)`
  - Smart loading with lazy evaluation based on connected outputs
  - Converts and caches optimized UNet on first use, loads CLIP/VAE only when needed
  - Returns optimized model, source CLIP/VAE, and unet path
- `LunaDynamicModelLoader.check_lazy_status(ckpt_name, precision, local_weights_dir, dynprompt, unique_id) -> List[int]`
  - Determines which outputs need computation based on graph connections
  - Uses DynamicPrompt API to check if CLIP/VAE outputs are connected
- `convert_to_precision(src_path, dst_path, precision, strip_components) -> (original_size_mb, converted_size_mb)`
  - Loads checkpoint, extracts UNet tensors, converts to target precision
  - Saves as safetensors with size reporting
- `convert_to_gguf(src_path, dst_path, quant_type) -> (original_size_mb, converted_size_mb)`
  - Converts checkpoint to GGUF format using Luna or ComfyUI-GGUF converter
  - Supports Q8_0, Q4_K_M quantization types
- `get_unet_keys(state_dict) -> Dict[str, torch.Tensor]`
  - Filters checkpoint state dict to include only UNet-related tensors
  - Excludes VAE, CLIP, and other components
- `LunaOptimizedWeightsManager.manage_weights(action, weights_directory="", days_old=30) -> (str,)`
  - Manages local optimized weights: list files, show stats, clear old, purge all
  - Provides disk usage and file management utilities

## Dependencies
**Internal:**
- Optional: `luna_gguf_converter` (for internal GGUF conversion)

**External:**
- Required: `comfy`, `folder_paths`, `safetensors`, `torch`
- Optional: `gguf` (for GGUF format), ComfyUI-GGUF (for .gguf loading)

## Integration Points
**Input:** Checkpoint filenames from ComfyUI checkpoints directory, precision options
**Output:** Optimized MODEL from local cache, source CLIP/VAE when needed, unet path string
**Side Effects:** Creates optimized UNet files in local weights directory, file I/O for conversions and caching

## Notes
- Hybrid loading: Optimized UNet from NVMe + CLIP/VAE from HDD source
- Lazy evaluation: Only loads CLIP/VAE if outputs are connected in workflow
- Precision options: bf16 (recommended), fp16, fp8_e4m3fn, gguf_Q8_0, gguf_Q4_K_M
- Local weights management: Utilities for cleanup and monitoring optimized files
- First use: Extracts UNet, converts precision, caches locally (~2-4GB vs 6.5GB full checkpoint)
- Multi-GPU optimization: Different precision per PC while sharing HDD checkpoints