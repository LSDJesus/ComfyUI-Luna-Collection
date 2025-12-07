# luna_batch_prompt_extractor.py

## Purpose
Batch extraction and loading of prompts/LoRAs from image metadata. Supports ComfyUI, A1111/Forge, and EXIF formats. Includes dimension scaling for model compatibility.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, json, re, PIL (Image, PngImagePlugin, ExifTags), typing (Tuple, Dict, List, Any, Optional), folder_paths (optional)

## ComfyUI Node Configuration
- LunaBatchPromptExtractor: CATEGORY="Luna/Utils", RETURN_TYPES=("STRING", "INT", "INT"), FUNCTION="extract_metadata", OUTPUT_NODE=True
- LunaBatchPromptLoader: CATEGORY="Luna/Utils", RETURN_TYPES=("STRING", "STRING", "LORA_STACK", "INT", "INT", "INT", "BOOLEAN", "INT", "INT"), FUNCTION="load_metadata"
- LunaDimensionScaler: CATEGORY="Luna/Utils", RETURN_TYPES=("INT", "INT"), FUNCTION="scale_dimensions"

## Input Schema
- LunaBatchPromptExtractor: image_directory (STRING), output_file (STRING), save_to_input_dir (BOOLEAN), output_directory (STRING), overwrite (BOOLEAN), include_path (BOOLEAN)
- LunaBatchPromptLoader: json_file (from _get_json_files()), index (INT), lora_output (["stack_only", "inline_only", "both"]), lora_validation (["include_all", "only_existing"])
- LunaDimensionScaler: width (INT), height (INT), model_type (from MODEL_NATIVE_SIZES.keys()), custom_max_size (INT), round_to (INT)

## Key Methods
- LunaBatchPromptExtractor.extract_metadata(image_directory, output_file, save_to_input_dir, output_directory, overwrite, include_path) -> Tuple[str, int, int]
- LunaBatchPromptExtractor.parse_loras_from_prompt(prompt) -> Tuple[str, List[Dict[str, Any]]]
- LunaBatchPromptExtractor.extract_comfyui_metadata(image_path) -> Optional[Dict[str, Any]]
- LunaBatchPromptExtractor.extract_a1111_metadata(image_path) -> Optional[Dict[str, Any]]
- LunaBatchPromptExtractor.extract_exif_metadata(image_path) -> Optional[Dict[str, Any]]
- LunaBatchPromptLoader.load_metadata(json_file, index, lora_output, lora_validation) -> Tuple[str, str, List[Tuple[str, float, float]], int, int, int, bool, int, int]
- LunaBatchPromptLoader.resolve_lora_path(lora_name) -> Optional[str]
- LunaDimensionScaler.scale_dimensions(width, height, model_type, custom_max_size, round_to) -> Tuple[int, int]

## Dependencies
PIL, folder_paths (optional)

## Integration Points
ComfyUI input directory for JSON files, folder_paths for LoRA/embedding resolution, LORA_STACK format compatibility

## Notes
Supports multiple metadata formats with fallback extraction, caches LoRA/embedding paths for validation, handles inline <lora:> tags, scales dimensions to model native resolutions with 8-pixel rounding</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_batch_prompt_extractor.md