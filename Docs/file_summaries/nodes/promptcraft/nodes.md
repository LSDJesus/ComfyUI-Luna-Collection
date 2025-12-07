# promptcraft/nodes.py

## Purpose
Smart wildcard resolution with constraints, modifiers, expanders, and LoRA linking. Main node for Luna PromptCraft system.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, json, random, typing (Dict, List, Tuple, Any, Optional), folder_paths (optional), engine (LunaPromptEngine, create_engine)

## ComfyUI Node Configuration
- LunaPromptCraft: CATEGORY="Luna/PromptCraft", RETURN_TYPES=("STRING", "INT", "LORA_STACK", "STRING", "STRING"), FUNCTION="process"
- LunaPromptCraftDebug: CATEGORY="Luna/PromptCraft", RETURN_TYPES=("STRING",), FUNCTION="format_debug", OUTPUT_NODE=True

## Input Schema
- LunaPromptCraft: template (STRING), seed (INT), wildcards_path (STRING, optional), enable_constraints/modifiers/expanders/lora_links/add_trigger_words (BOOLEAN, optional)
- LunaPromptCraftDebug: debug_json (STRING), show_paths/tags/loras (BOOLEAN, optional)

## Key Methods
- LunaPromptCraft.process(template, seed, wildcards_path, enable_constraints, enable_modifiers, enable_expanders, enable_lora_links, add_trigger_words) -> Tuple[str, int, List, str, str]
- LunaPromptCraftDebug.format_debug(debug_json, show_paths, show_tags, show_loras) -> Tuple[str]

## Dependencies
folder_paths (optional), engine module

## Integration Points
Luna PromptCraft engine, LORA_STACK format, JS Connection Manager panel for configuration

## Notes
Shared engine instance, template syntax with {wildcards}, supports constraints/modifiers/expanders/LoRA linking, outputs debug JSON for visualization</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\promptcraft\nodes.md