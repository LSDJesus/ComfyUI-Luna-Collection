"""
Luna Expression Pack - Generate character expression sheets for SillyTavern and other uses
Supports SDXL, Illustrious, Pony, Z-Image, and Flux models
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

# ComfyUI imports
try:
    import folder_paths
except ImportError:
    folder_paths = None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# SillyTavern compatible expression set (29 expressions)
SILLYTAVERN_EXPRESSIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise", "wonder"
]

# Extended expression set with more variety
EXTENDED_EXPRESSIONS = SILLYTAVERN_EXPRESSIONS + [
    "seductive", "flirty", "shy", "determined", "confident", "playful",
    "mischievous", "smug", "bored", "tired", "sleepy", "crying",
    "laughing", "smiling", "frowning", "pouting", "winking", "thinking"
]

# Common full-body poses for character sheets
COMMON_POSES = [
    "standing front view", "standing side view", "standing back view",
    "standing three-quarter view", "walking pose", "running pose",
    "sitting pose", "kneeling pose", "action pose", "fighting stance",
    "relaxed pose", "confident pose", "shy pose", "waving",
    "arms crossed", "hands on hips", "pointing", "thinking pose",
    "dancing pose", "jumping pose"
]

# Natural language prompt templates for different models
PROMPT_TEMPLATES = {
    "z-image": {
        "expression": (
            "A detailed portrait photograph of {character_desc}, "
            "showing a clear {expression} expression on their face. "
            "Head and shoulders shot, facing the camera. "
            "The subject displays {expression} emotion with natural facial features. "
            "Clean neutral background, studio lighting, high quality photograph."
        ),
        "pose": (
            "A full body photograph of {character_desc}, "
            "in a {pose} position. "
            "The subject is clearly visible from head to toe. "
            "Clean neutral background, studio lighting, high quality photograph."
        ),
        "sheet": (
            "A character expression sheet showing {character_desc} "
            "with multiple facial expressions arranged in a {grid} grid. "
            "Each expression is clearly distinct: {expressions_list}. "
            "Clean white background, consistent lighting across all expressions."
        )
    },
    "sdxl": {
        "expression": (
            "{character_desc}, portrait, {expression} expression, "
            "head shot, looking at viewer, simple background, "
            "masterpiece, best quality, highly detailed"
        ),
        "pose": (
            "{character_desc}, full body, {pose}, "
            "simple background, masterpiece, best quality, highly detailed"
        ),
        "sheet": (
            "character sheet, {character_desc}, multiple expressions, "
            "{grid} grid layout, {expressions_list}, "
            "white background, reference sheet, masterpiece, best quality"
        )
    },
    "flux": {
        "expression": (
            "Portrait of {character_desc} with {expression} expression, "
            "head and shoulders, neutral background"
        ),
        "pose": (
            "Full body shot of {character_desc} in {pose}, "
            "neutral background, studio lighting"
        ),
        "sheet": (
            "Character expression sheet of {character_desc}, "
            "{grid} grid, expressions: {expressions_list}, white background"
        )
    }
}


class LunaExpressionList:
    """
    Provides expression and pose lists for character sheet generation.
    Outputs individual items or formatted lists for batch processing.
    """
    
    CATEGORY = "Luna/Character"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("expression_list", "current_expression", "count")
    FUNCTION = "get_expressions"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["sillytavern", "extended", "poses", "custom"], {
                    "default": "sillytavern",
                    "tooltip": "Expression/pose preset to use"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Current expression index (for looping)"
                }),
            },
            "optional": {
                "custom_list": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom expressions, one per line"
                }),
            }
        }
    
    def get_expressions(self, preset: str, index: int, custom_list: str = "") -> Tuple[str, str, int]:
        if preset == "sillytavern":
            expressions = SILLYTAVERN_EXPRESSIONS
        elif preset == "extended":
            expressions = EXTENDED_EXPRESSIONS
        elif preset == "poses":
            expressions = COMMON_POSES
        elif preset == "custom" and custom_list.strip():
            expressions = [e.strip() for e in custom_list.strip().split("\n") if e.strip()]
        else:
            expressions = SILLYTAVERN_EXPRESSIONS
        
        count = len(expressions)
        current = expressions[index % count] if expressions else ""
        full_list = "\n".join(expressions)
        
        return (full_list, current, count)


class LunaExpressionPromptBuilder:
    """
    Builds natural language prompts for expression/pose generation.
    Optimized for different model types (Z-Image, SDXL, Flux).
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
                "expression_or_pose": ("STRING", {
                    "default": "neutral",
                    "tooltip": "The expression or pose to generate"
                }),
                "mode": (["expression", "pose", "sheet"], {
                    "default": "expression",
                    "tooltip": "Generation mode"
                }),
                "model_type": (["z-image", "sdxl", "flux"], {
                    "default": "z-image",
                    "tooltip": "Target model type for prompt optimization"
                }),
            },
            "optional": {
                "character_name": ("STRING", {
                    "default": "",
                    "tooltip": "Character name for filename"
                }),
                "grid_size": ("STRING", {
                    "default": "4x4",
                    "tooltip": "Grid size for sheet mode (e.g., 4x4, 5x4)"
                }),
                "additional_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional prompt elements to append"
                }),
            }
        }
    
    def build_prompt(
        self,
        character_description: str,
        expression_or_pose: str,
        mode: str,
        model_type: str,
        character_name: str = "",
        grid_size: str = "4x4",
        additional_prompt: str = ""
    ) -> Tuple[str, str]:
        
        templates = PROMPT_TEMPLATES.get(model_type, PROMPT_TEMPLATES["sdxl"])
        template = templates.get(mode, templates["expression"])
        
        # Build the prompt
        prompt = template.format(
            character_desc=character_description,
            expression=expression_or_pose,
            pose=expression_or_pose,
            grid=grid_size,
            expressions_list=expression_or_pose
        )
        
        if additional_prompt.strip():
            prompt = f"{prompt} {additional_prompt.strip()}"
        
        # Build filename
        safe_name = character_name.replace(" ", "_").lower() if character_name else "character"
        safe_expr = expression_or_pose.replace(" ", "_").lower()
        filename = f"{safe_name}_{safe_expr}"
        
        return (prompt, filename)


class LunaExpressionSheetSlicer:
    """
    Slices a generated expression sheet into individual images.
    Handles grid layouts and outputs individual expression images.
    """
    
    CATEGORY = "Luna/Character"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "expression_names")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "slice_sheet"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet_image": ("IMAGE", {
                    "tooltip": "The expression sheet image to slice"
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
                    "tooltip": "Expression names, one per line (in grid order)"
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Padding between cells to remove"
                }),
            }
        }
    
    def slice_sheet(
        self,
        sheet_image,
        columns: int,
        rows: int,
        expression_names: str = "",
        padding: int = 0
    ) -> Tuple[list, list]:
        import torch
        
        # Get image dimensions (B, H, W, C)
        batch, height, width, channels = sheet_image.shape
        
        # Calculate cell dimensions
        cell_height = height // rows
        cell_width = width // columns
        
        # Parse expression names
        if expression_names.strip():
            names = [n.strip() for n in expression_names.strip().split("\n") if n.strip()]
        else:
            names = [f"expr_{i}" for i in range(rows * columns)]
        
        # Slice the sheet
        sliced_images = []
        sliced_names = []
        
        for row in range(rows):
            for col in range(columns):
                idx = row * columns + col
                if idx >= len(names):
                    break
                
                # Calculate crop coordinates
                y1 = row * cell_height + padding
                y2 = (row + 1) * cell_height - padding
                x1 = col * cell_width + padding
                x2 = (col + 1) * cell_width - padding
                
                # Crop the cell
                cell = sheet_image[:, y1:y2, x1:x2, :]
                sliced_images.append(cell)
                sliced_names.append(names[idx])
        
        return (sliced_images, sliced_names)


class LunaExpressionBatchSaver:
    """
    Saves a batch of expression images with proper naming for SillyTavern.
    Creates the correct folder structure and filenames.
    """
    
    CATEGORY = "Luna/Character"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save_expressions"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Images to save"
                }),
                "character_name": ("STRING", {
                    "default": "character",
                    "tooltip": "Character name for folder/filenames"
                }),
                "expression_name": ("STRING", {
                    "default": "neutral",
                    "tooltip": "Expression name for filename"
                }),
            },
            "optional": {
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output folder (default: ComfyUI output)"
                }),
                "format": (["png", "webp", "jpg"], {
                    "default": "png"
                }),
                "create_sillytavern_structure": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create SillyTavern-compatible folder structure"
                }),
            }
        }
    
    def save_expressions(
        self,
        images,
        character_name: str,
        expression_name: str,
        output_folder: str = "",
        format: str = "png",
        create_sillytavern_structure: bool = True
    ) -> Tuple[str]:
        from PIL import Image
        import numpy as np
        
        # Determine output path
        if output_folder:
            base_path = Path(output_folder)
        elif folder_paths:
            base_path = Path(folder_paths.get_output_directory())
        else:
            base_path = Path("./output")
        
        # Create SillyTavern structure if requested
        if create_sillytavern_structure:
            char_folder = base_path / "expressions" / character_name.lower().replace(" ", "_")
        else:
            char_folder = base_path / character_name.lower().replace(" ", "_")
        
        char_folder.mkdir(parents=True, exist_ok=True)
        
        # Save images
        saved_paths = []
        for i, img_tensor in enumerate(images):
            # Convert tensor to PIL
            if len(img_tensor.shape) == 4:
                img_tensor = img_tensor[0]
            
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            # Build filename
            safe_expr = expression_name.lower().replace(" ", "_")
            if images.shape[0] > 1:
                filename = f"{safe_expr}_{i+1}.{format}"
            else:
                filename = f"{safe_expr}.{format}"
            
            filepath = char_folder / filename
            img.save(filepath)
            saved_paths.append(str(filepath))
        
        return (str(char_folder),)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "LunaExpressionList": LunaExpressionList,
    "LunaExpressionPromptBuilder": LunaExpressionPromptBuilder,
    "LunaExpressionSheetSlicer": LunaExpressionSheetSlicer,
    "LunaExpressionBatchSaver": LunaExpressionBatchSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaExpressionList": "🌙 Luna Expression List",
    "LunaExpressionPromptBuilder": "🌙 Luna Expression Prompt Builder",
    "LunaExpressionSheetSlicer": "🌙 Luna Expression Sheet Slicer",
    "LunaExpressionBatchSaver": "🌙 Luna Expression Batch Saver",
}
