"""
Luna SAM3 - Vendored SAM3 image segmentation for ComfyUI

Vendored from comfyui-sam3 to provide:
- Local-first model loading (no HuggingFace dependency)
- Safetensors support
- ComfyUI model_management integration
- Text-based image segmentation

Video tracking and interactive features are excluded.
"""

from .model_builder import build_sam3_video_model, _load_checkpoint_file
from .sam3_video_predictor import Sam3VideoPredictor

__all__ = [
    "build_sam3_video_model",
    "Sam3VideoPredictor",
    "_load_checkpoint_file",
]
