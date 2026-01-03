"""
Luna GGUF Converter V2 - Proper Q4/Q8 quantization using llama-cpp-python

Node wrapper for the gguf_converter utility module.
"""

from pathlib import Path
from typing import Tuple
import importlib.util

# Import centralized path constants
try:
    from __init__ import LUNA_PATH
except (ImportError, ModuleNotFoundError, AttributeError):
    LUNA_PATH = None

# Load gguf_converter using centralized LUNA_PATH
if LUNA_PATH:
    converter_path = Path(LUNA_PATH) / "utils" / "gguf_converter.py"
else:
    # Fallback: luna_gguf_converter.py is at nodes/utilities/, go up 3 levels
    converter_path = Path(__file__).parent.parent.parent / "utils" / "gguf_converter.py"

try:
    spec = importlib.util.spec_from_file_location("gguf_converter", converter_path)
    if not spec or not spec.loader:
        raise ImportError("Could not create spec for gguf_converter")
    gguf_converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gguf_converter)
    convert_checkpoint_to_gguf = gguf_converter.convert_checkpoint_to_gguf
except Exception as e:
    raise ImportError(f"Failed to load gguf_converter module from {converter_path}: {e}")


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
                "quantization": (["F16", "Q4_0", "Q4_K", "Q4_K_S", "Q5_0", "Q5_K_M", "Q8_0"], {
                    "default": "Q4_K",
                    "tooltip": "Q4_K recommended (best quality ~2GB), Q4_0 for compatibility"
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