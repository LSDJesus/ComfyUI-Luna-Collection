"""
Luna Expression Pack - Generate character expression packs for Luna Narrates and SillyTavern
Supports SDXL, Illustrious, Pony, and Flux models
"""

import os
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from PIL import Image

# ComfyUI imports
try:
    import folder_paths
except ImportError:
    folder_paths = None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION PRESETS (for reference - expressions are passed via workflow)
# ═══════════════════════════════════════════════════════════════════════════════

# SillyTavern compatible expression set (29 expressions)
SILLYTAVERN_EXPRESSIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise", "wonder"
]

# Natural language prompt templates for different models
PROMPT_TEMPLATES = {
    "illustrious": {
        "expression": (
            "{character_desc}, portrait, {expression} expression, "
            "head shot, looking at viewer, simple background, "
            "masterpiece, best quality, highly detailed"
        ),
    },
    "flux": {
        "expression": (
            "Portrait of {character_desc} with {expression} expression, "
            "head and shoulders, neutral background"
        ),
    },
    "sdxl": {
        "expression": (
            "{character_desc}, portrait, {expression} expression, "
            "head shot, looking at viewer, simple background, "
            "masterpiece, best quality"
        ),
    }
}


class LunaExpressionPromptBuilder:
    """
    Builds natural language prompts for expression generation.
    Optimized for different model types.
    
    Use with a loop/batch to generate multiple expressions.
    Pass expression names from your workflow (e.g., from a text file or list).
    """
    
    CATEGORY = "Luna/Character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "filename")
    FUNCTION = "build_prompt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "a young woman with long brown hair and green eyes",
                    "tooltip": "Description of the character's appearance"
                }),
                "expression": ("STRING", {
                    "default": "neutral",
                    "tooltip": "The expression to generate (e.g., joy, anger, surprise)"
                }),
                "model_type": (["illustrious", "flux", "sdxl"], {
                    "default": "illustrious",
                    "tooltip": "Target model type for prompt optimization"
                }),
            },
            "optional": {
                "character_name": ("STRING", {
                    "default": "",
                    "tooltip": "Character name for filename"
                }),
                "additional_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional prompt elements to append"
                }),
                "negative_additions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional negative prompt elements"
                }),
            }
        }
    
    def build_prompt(
        self,
        character_description: str,
        expression: str,
        model_type: str,
        character_name: str = "",
        additional_prompt: str = "",
        negative_additions: str = ""
    ) -> Tuple[str, str]:
        
        templates = PROMPT_TEMPLATES.get(model_type, PROMPT_TEMPLATES["illustrious"])
        template = templates.get("expression")
        
        # Build the prompt
        prompt = template.format(
            character_desc=character_description,
            expression=expression
        )
        
        if additional_prompt.strip():
            prompt = f"{prompt}, {additional_prompt.strip()}"
        
        # Build filename
        safe_name = character_name.replace(" ", "_").lower() if character_name else "character"
        safe_expr = expression.replace(" ", "_").lower()
        filename = f"{safe_name}_{safe_expr}"
        
        return (prompt, filename)


class LunaExpressionSlicerSaver:
    """
    Slices an expression sheet into individual images AND saves them.
    Combined node for efficient expression pack creation.
    
    Input: A grid-based expression sheet image
    Output: Individual expression images saved to disk
    
    Works with SillyTavern folder structure for easy character import.
    """
    
    CATEGORY = "Luna/Character"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "expression_names", "output_folder")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "slice_and_save"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet_image": ("IMAGE", {
                    "tooltip": "The expression sheet image to slice"
                }),
                "character_name": ("STRING", {
                    "default": "character",
                    "tooltip": "Character name for folder/filenames"
                }),
                "columns": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of columns in the grid"
                }),
                "rows": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of rows in the grid"
                }),
            },
            "optional": {
                "expression_names": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Expression names, one per line (in grid order left-to-right, top-to-bottom)"
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output folder (default: ComfyUI output/expressions)"
                }),
                "format": (["png", "webp"], {
                    "default": "png",
                    "tooltip": "Output image format"
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Padding between cells to crop out"
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save images to disk (disable to just slice)"
                }),
            }
        }
    
    def slice_and_save(
        self,
        sheet_image,
        character_name: str,
        columns: int,
        rows: int,
        expression_names: str = "",
        output_folder: str = "",
        format: str = "png",
        padding: int = 0,
        save_to_disk: bool = True
    ) -> Tuple[list, list, str]:
        
        # Get image dimensions (B, H, W, C)
        batch, height, width, channels = sheet_image.shape
        
        # Calculate cell dimensions
        cell_height = height // rows
        cell_width = width // columns
        
        # Parse expression names or use defaults
        if expression_names.strip():
            names = [n.strip() for n in expression_names.strip().split("\n") if n.strip()]
        else:
            # Use SillyTavern default names if not specified
            names = SILLYTAVERN_EXPRESSIONS[:rows * columns]
        
        # Ensure we have enough names
        while len(names) < rows * columns:
            names.append(f"expression_{len(names) + 1}")
        
        # Determine output path
        if output_folder:
            base_path = Path(output_folder)
        elif folder_paths:
            base_path = Path(folder_paths.get_output_directory())
        else:
            base_path = Path("./output")
        
        # Create character folder (SillyTavern structure)
        safe_char_name = character_name.lower().replace(" ", "_")
        char_folder = base_path / "expressions" / safe_char_name
        
        if save_to_disk:
            char_folder.mkdir(parents=True, exist_ok=True)
        
        # Slice the sheet and optionally save
        sliced_images = []
        sliced_names = []
        
        for row in range(rows):
            for col in range(columns):
                idx = row * columns + col
                if idx >= len(names):
                    break
                
                expr_name = names[idx]
                
                # Calculate crop coordinates
                y1 = row * cell_height + padding
                y2 = (row + 1) * cell_height - padding
                x1 = col * cell_width + padding
                x2 = (col + 1) * cell_width - padding
                
                # Crop the cell
                cell = sheet_image[:, y1:y2, x1:x2, :]
                sliced_images.append(cell)
                sliced_names.append(expr_name)
                
                # Save to disk if enabled
                if save_to_disk:
                    # Convert tensor to PIL
                    img_tensor = cell[0] if len(cell.shape) == 4 else cell
                    img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    
                    # Build filename (SillyTavern expects exact expression names)
                    safe_expr = expr_name.lower().replace(" ", "_")
                    filename = f"{safe_expr}.{format}"
                    filepath = char_folder / filename
                    
                    # Save with appropriate settings
                    if format == "webp":
                        img.save(filepath, format="WEBP", quality=95, lossless=False)
                    else:
                        img.save(filepath, format="PNG", compress_level=6)
                    
                    print(f"[LunaExpression] Saved: {filepath}")
        
        print(f"[LunaExpression] Saved {len(sliced_images)} expressions to {char_folder}")
        
        return (sliced_images, sliced_names, str(char_folder))


# ═══════════════════════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "LunaExpressionPromptBuilder": LunaExpressionPromptBuilder,
    "LunaExpressionSlicerSaver": LunaExpressionSlicerSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaExpressionPromptBuilder": "🌙 Luna Expression Prompt Builder",
    "LunaExpressionSlicerSaver": "🌙 Luna Expression Slicer & Saver",
}
