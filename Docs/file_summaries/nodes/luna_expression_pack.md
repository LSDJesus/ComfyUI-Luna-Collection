# luna_expression_pack.py

## Purpose
Generate character expression packs for Luna Narrates and SillyTavern, supporting SDXL, Illustrious, Pony, and Flux models.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, pathlib (Path), typing (Tuple), torch, numpy, PIL (Image), folder_paths (optional)

## ComfyUI Node Configuration
- LunaExpressionPromptBuilder: CATEGORY="Luna", RETURN_TYPES=("STRING", "STRING"), FUNCTION="build_prompt"
- LunaExpressionSlicerSaver: CATEGORY="Luna", RETURN_TYPES=("IMAGE", "STRING", "STRING"), OUTPUT_IS_LIST=(True, True, False), FUNCTION="slice_and_save", OUTPUT_NODE=True

## Input Schema
- LunaExpressionPromptBuilder: character_description (STRING), expression (STRING), model_type (["illustrious", "flux", "sdxl"]), character_name (STRING, optional), additional_prompt (STRING, optional), negative_additions (STRING, optional)
- LunaExpressionSlicerSaver: sheet_image (IMAGE), character_name (STRING), columns (INT), rows (INT), expression_names (STRING, optional), output_folder (STRING, optional), format (["png", "webp"]), padding (INT), save_to_disk (BOOLEAN)

## Key Methods
- LunaExpressionPromptBuilder.build_prompt(character_description, expression, model_type, character_name, additional_prompt, negative_additions) -> Tuple[str, str]
- LunaExpressionSlicerSaver.slice_and_save(sheet_image, character_name, columns, rows, expression_names, expression_names, output_folder, format, padding, save_to_disk) -> Tuple[list, list, str]

## Dependencies
torch, numpy, PIL, folder_paths (optional)

## Integration Points
ComfyUI output directory, SillyTavern folder structure, supports SillyTavern expression names

## Notes
Includes preset prompt templates for different models, supports SillyTavern compatible expression sets, slices grid-based expression sheets into individual images, saves to organized folder structure</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_expression_pack.py