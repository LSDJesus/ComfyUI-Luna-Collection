"""
Luna GGUF Converter V2 - Proper Q4/Q8 quantization using llama-cpp-python

Node wrapper for the gguf_converter utility module.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gguf_converter import convert_checkpoint_to_gguf


class LunaGGUFConverter:
    """
    Convert checkpoints to GGUF with proper Q4/Q8 quantization.
    
    Uses llama-cpp-python for real quantization (~70% compression for Q4).
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "tensor_count", "size_mb")
    FUNCTION = "convert"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Path to source .safetensors checkpoint"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save converted GGUF (UNet only, VAE/CLIP stripped)"
                }),
                "quantization": (["F16", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q8_0"], {
                    "default": "Q4_K_M",
                    "tooltip": "Q4_K_M recommended (best quality ~2GB), Q4_0 for compatibility"
                }),
            },
            "optional": {
                "output_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename (without .gguf)"
                }),
            }
        }
    
    def convert(self, source_checkpoint: str, output_directory: str, quantization: str,
                output_filename: str = "") -> Tuple[str, int, float]:
        """Convert checkpoint to GGUF with proper quantization (UNet only)."""
        
        return convert_checkpoint_to_gguf(
            source_checkpoint=source_checkpoint,
            output_directory=output_directory,
            quantization=quantization,
            output_filename=output_filename
        )


NODE_CLASS_MAPPINGS = {
    "LunaGGUFConverter": LunaGGUFConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaGGUFConverter": "Luna GGUF Converter",
}