# upscaling/__init__.py

## Purpose
Module initialization for Luna upscaling nodes, exposing all upscaler node mappings.

## Exports
- `NODE_CLASS_MAPPINGS`: Dict mapping all upscaler node names to classes
- `NODE_DISPLAY_NAME_MAPPINGS`: Dict mapping all upscaler node names to display names

## Key Imports
- From `.luna_ultimate_sd_upscale`: Luna_UltimateSDUpscale
- From `.luna_upscaler_advanced`: Luna_Advanced_Upscaler
- From `.luna_upscaler_simple`: Luna_SimpleUpscaler
- From `.luna_super_upscaler`: LunaSuperUpscaler, LunaSuperUpscalerSimple

## ComfyUI Node Configuration
N/A - Module init file

## Input Schema
N/A

## Key Methods
N/A

## Dependencies
- All upscaling node modules in the upscaling/ directory

## Integration Points
- Imported by ComfyUI's auto-discovery mechanism
- Exposes 5 upscaling nodes: Ultimate SD Upscale, Advanced Upscaler, Simple Upscaler, Super Upscaler, and Super Upscaler Simple

## Notes
Simple module init that aggregates all Luna upscaling nodes for ComfyUI integration, providing comprehensive upscaling capabilities from basic to advanced diffusion-based methods.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\upscaling\__init__.md