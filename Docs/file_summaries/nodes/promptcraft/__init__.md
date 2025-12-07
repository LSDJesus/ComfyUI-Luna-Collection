# promptcraft/__init__.py

## Purpose
Module initialization for Luna PromptCraft, exposing ComfyUI node mappings.

## Exports
- `NODE_CLASS_MAPPINGS`: Dict mapping node names to classes
- `NODE_DISPLAY_NAME_MAPPINGS`: Dict mapping node names to display names

## Key Imports
- From `.nodes`: NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## ComfyUI Node Configuration
N/A - Module init file

## Input Schema
N/A

## Key Methods
N/A

## Dependencies
- `nodes/promptcraft/nodes.py`

## Integration Points
- Imported by ComfyUI's auto-discovery mechanism
- Exposes LunaPromptCraft and LunaPromptCraftDebug nodes

## Notes
Simple module init that re-exports node mappings from nodes.py for ComfyUI integration.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\promptcraft\__init__.md