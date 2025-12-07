# luna_lora_validator.py

## Purpose
Validate LoRAs in prompt JSON files against local installation. Optionally search CivitAI for missing LoRAs with download links.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, json, urllib.request/error/parse, ssl, time, typing (TYPE_CHECKING, Dict, List, Optional, Tuple, Any), folder_paths (optional)

## ComfyUI Node Configuration
- LunaLoRAValidator: CATEGORY="Luna/Utilities", RETURN_TYPES=("STRING", "STRING", "STRING", "INT", "INT"), FUNCTION="validate_loras", OUTPUT_NODE=True

## Input Schema
- json_file (from _get_json_files()), search_civitai (BOOLEAN), civitai_api_key (STRING)

## Key Methods
- LunaLoRAValidator.validate_loras(json_file, search_civitai, civitai_api_key) -> Tuple[str, str, str, int, int]
- LunaLoRAValidator.lora_exists(lora_name) -> Tuple[bool, Optional[str]]
- LunaLoRAValidator.search_civitai(lora_name, api_key) -> Optional[Dict[str, Any]]

## Dependencies
folder_paths (optional), urllib for CivitAI API

## Integration Points
ComfyUI input directory for JSON files, folder_paths for LoRA path resolution, CivitAI API v1 for model search

## Notes
Caches LoRA filenames for validation, supports partial name matching, provides formatted ASCII report, rate-limited CivitAI searches with API key support</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_lora_validator.md