# luna_multi_saver.py

## Purpose
Advanced multi-image saver with custom paths, quality filtering, parallel saving, and comprehensive metadata embedding.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, json, threading, time, datetime, typing (Dict, Any), PIL (Image, PngImagePlugin), numpy, folder_paths, nodes, comfy.utils, comfy.cli_args, piexif (optional)

## ComfyUI Node Configuration
- LunaMultiSaver: CATEGORY="Luna/Image", RETURN_TYPES=(), FUNCTION="save_images", OUTPUT_NODE=True

## Input Schema
- Required: save_path (STRING), filename (STRING), save_mode (["parallel", "sequential"]), quality_gate (["disabled", "variance", "edge_density", "both"]), min_quality_threshold (FLOAT)
- Optional: model_name (STRING), image_1-5 (IMAGE), affix_1-5 (STRING), format_1-5 (["png", "webp", "jpeg"]), subdir_1-5 (BOOLEAN), png_compression (INT), lossy_quality (INT), lossless_webp (BOOLEAN), embed_workflow (BOOLEAN), filename_index (INT), custom_metadata (STRING), metadata (METADATA)

## Key Methods
- LunaMultiSaver.save_images(save_path, filename, save_mode, quality_gate, min_quality_threshold, ...) -> dict
- LunaMultiSaver.save_single_image(image, affix_name, use_subdir, model_name_raw, custom_path, custom_metadata, metadata, prompt, extra_pnginfo, batch_timestamp, batch_counter, filename_template, filename_index, extension, lossless_webp, lossy_quality, png_compression, embed_workflow) -> list
- LunaMultiSaver.quality_check_image(image, threshold, mode) -> tuple
- LunaMultiSaver._process_template(template_str, model_path, model_name, model_dir, timestamp, index) -> str

## Dependencies
PIL, numpy, folder_paths, nodes, comfy.utils, piexif (optional)

## Integration Points
ComfyUI output directory, folder_paths for output management, metadata from Luna Config Gateway, supports PNG/WebP/JPEG with different metadata embedding

## Notes
Supports template variables (%model_name%, %time:FORMAT%, etc.), quality filtering with variance/edge detection, parallel thread-based saving, comprehensive metadata embedding in EXIF/PNG chunks</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_multi_saver.md