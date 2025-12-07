# luna_gguf_converter.py

## Purpose
Convert safetensors checkpoints to GGUF format with various quantization levels. Extracts UNet weights for use with Luna Daemon.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, json, struct, threading, pathlib (Path), typing (Dict, List, Optional, Tuple, Any), enum (Enum), numpy, torch, safetensors.torch (load_file, save_file), folder_paths (optional)

## ComfyUI Node Configuration
- LunaGGUFConverter: CATEGORY="Luna/Utilities", RETURN_TYPES=("STRING", "INT", "FLOAT"), FUNCTION="convert", OUTPUT_NODE=True
- LunaGGUFBatchConverter: CATEGORY="Luna/Utilities", RETURN_TYPES=("STRING", "INT", "INT"), FUNCTION="batch_convert", OUTPUT_NODE=True

## Input Schema
- LunaGGUFConverter: source_checkpoint (STRING), output_directory (STRING), quantization (from quant_options), output_filename (STRING, optional), extract_unet_only (BOOLEAN)
- LunaGGUFBatchConverter: source_directory (STRING), output_directory (STRING), quantization (from quant_options), max_files (INT), skip_existing (BOOLEAN, optional)

## Key Methods
- LunaGGUFConverter.convert(source_checkpoint, output_directory, quantization, output_filename, extract_unet_only) -> Tuple[str, int, float]
- LunaGGUFBatchConverter.batch_convert(source_directory, output_directory, quantization, max_files, skip_existing) -> Tuple[str, int, int]
- get_unet_keys(state_dict) -> Dict[str, torch.Tensor]
- quantize_tensor_q8_0(tensor) -> bytes
- quantize_tensor_q4_0(tensor) -> bytes

## Dependencies
numpy, torch, safetensors, folder_paths (optional)

## Integration Points
GGUF format for llama-cpp-python, Luna Daemon for shared VAE/CLIP, ComfyUI-GGUF loader

## Notes
GPU-specific quantization recommendations, vectorized quantization for GPU acceleration, extracts UNet-only for daemon workflow, supports batch conversion</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_gguf_converter.md