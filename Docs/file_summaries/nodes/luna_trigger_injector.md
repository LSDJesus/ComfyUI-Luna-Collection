# luna_trigger_injector.py

## Purpose
Extract trigger words from LoRAs and inject into prompts. Supports LORA_STACK input, inline LoRA parsing, CivitAI metadata, and embedded safetensors metadata.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, re, json, struct, typing (TYPE_CHECKING, Dict, List, Optional, Tuple, Any), folder_paths (optional)

## ComfyUI Node Configuration
- LunaLoRATriggerInjector: CATEGORY="Luna/LoRA", RETURN_TYPES=("STRING", "STRING", "LORA_STACK", "STRING"), FUNCTION="process"

## Input Schema
- injection_mode (["prepend", "append", "none"]), max_triggers_per_lora (INT), separator (STRING), deduplicate (BOOLEAN), prompt (STRING, optional), lora_stack (LORA_STACK, optional)

## Key Methods
- LunaLoRATriggerInjector.process(injection_mode, max_triggers_per_lora, separator, deduplicate, prompt, lora_stack) -> Tuple[str, str, List[Tuple[str, float, float]], str]
- LunaLoRATriggerInjector._extract_inline_loras(prompt) -> Tuple[str, List[Tuple[str, float, float]]]
- LunaLoRATriggerInjector._get_triggers_for_lora(lora_name) -> List[str]
- LunaLoRATriggerInjector._read_safetensors_metadata(filepath) -> Dict[str, Any]

## Dependencies
folder_paths (optional), utils.luna_metadata_db (optional)

## Integration Points
LORA_STACK format, folder_paths for LoRA path resolution, CivitAI metadata cache, safetensors embedded metadata

## Notes
Caches trigger lookups, supports multiple metadata sources (CivitAI, embedded, sidecar files), configurable trigger limits and injection modes, removes inline LoRA tags from prompts</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_trigger_injector.md