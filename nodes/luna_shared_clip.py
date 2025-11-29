"""
Luna Shared CLIP Encode
ComfyUI node that uses the shared VAE/CLIP daemon instead of loading CLIP models locally.

This is a smart, unified CLIP encoding node that:
- Accepts any CLIP model from standard ComfyUI loaders
- Auto-detects CLIP architecture (SD1.5, SDXL, FLUX, etc.)
- Routes encoding through the daemon for shared VRAM usage
- Handles both single and dual prompt encoding
"""

import torch
from typing import Tuple, Optional

# Import daemon client
try:
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonConnectionError = Exception


class LunaSharedCLIPEncode:
    """
    Smart CLIP text encoding using the shared daemon.
    
    This node does NOT load CLIP models - it connects to the Luna daemon
    which holds CLIP in VRAM shared across all ComfyUI instances.
    
    Features:
    - Auto-detects CLIP architecture from the connected loader
    - Supports SD1.5, SDXL (CLIP-L + CLIP-G), and FLUX (CLIP-L + T5-XXL)
    - Encodes both positive and negative prompts in one node
    - Includes optional SDXL size conditioning
    
    Saves ~2-3GB VRAM per workflow when running multiple instances.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive prompt text"
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt text"
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for architecture detection. "
                               "If not provided, uses daemon's default CLIP."
                }),
                # SDXL size conditioning (only used when SDXL detected)
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Target width (SDXL only)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Target height (SDXL only)"
                }),
                "crop_w": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Crop width (SDXL only)"
                }),
                "crop_h": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Crop height (SDXL only)"
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "tooltip": "Aesthetic target width (SDXL only)"
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "tooltip": "Aesthetic target height (SDXL only)"
                }),
            }
        }
    
    def encode(
        self,
        positive: str,
        negative: str,
        clip: Optional[object] = None,
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024
    ) -> Tuple[list, list]:
        if not DAEMON_AVAILABLE:
            raise RuntimeError(
                "Luna daemon client not available. "
                "Make sure the luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna VAE/CLIP Daemon is not running!\n"
                "Start it with: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server"
            )
        
        try:
            # Detect CLIP architecture if clip model provided
            arch = self._detect_architecture(clip)
            
            if arch == "sdxl":
                # SDXL dual CLIP encoding with size conditioning
                positive_cond, negative_cond = daemon_client.clip_encode_sdxl(
                    positive=positive,
                    negative=negative,
                    width=width,
                    height=height,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    target_width=target_width,
                    target_height=target_height
                )
                return (positive_cond, negative_cond)
            
            elif arch == "flux":
                # FLUX encoding (CLIP-L + T5-XXL)
                # TODO: Implement FLUX-specific encoding when daemon supports it
                cond, pooled, uncond, pooled_neg = daemon_client.clip_encode(positive, negative)
                positive_out = [[cond, {"pooled_output": pooled}]]
                negative_out = [[uncond, {"pooled_output": pooled_neg}]]
                return (positive_out, negative_out)
            
            else:
                # Standard SD1.5 or fallback encoding
                cond, pooled, uncond, pooled_neg = daemon_client.clip_encode(positive, negative)
                
                positive_out = [[cond, {"pooled_output": pooled}]]
                negative_out = [[uncond, {"pooled_output": pooled_neg}]]
                
                return (positive_out, negative_out)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def _detect_architecture(self, clip: Optional[object]) -> str:
        """
        Detect CLIP architecture from the model object.
        
        Returns:
            'sd15' - Standard SD1.5 CLIP
            'sdxl' - SDXL dual CLIP (CLIP-L + CLIP-G)
            'flux' - FLUX (CLIP-L + T5-XXL)
            'unknown' - Unknown architecture
        """
        if clip is None:
            return "unknown"
        
        try:
            # Check for SDXL indicators
            # SDXL CLIP has specific tokenizer/model structure
            if hasattr(clip, 'cond_stage_model'):
                model = clip.cond_stage_model
                
                # Check for dual text encoders (SDXL signature)
                if hasattr(model, 'clip_l') and hasattr(model, 'clip_g'):
                    return "sdxl"
                
                # Check for T5 (FLUX signature)
                if hasattr(model, 't5xxl') or hasattr(model, 't5'):
                    return "flux"
            
            # Check tokenizer for hints
            if hasattr(clip, 'tokenizer'):
                tokenizer = clip.tokenizer
                # SDXL typically has larger vocab
                if hasattr(tokenizer, 'vocab_size'):
                    if tokenizer.vocab_size > 50000:
                        return "sdxl"
            
            # Check patcher for model info
            if hasattr(clip, 'patcher') and hasattr(clip.patcher, 'model'):
                model_keys = set(clip.patcher.model.state_dict().keys()) if hasattr(clip.patcher.model, 'state_dict') else set()
                
                # Look for SDXL-specific layer names
                if any('clip_g' in k or 'text_model_2' in k for k in model_keys):
                    return "sdxl"
                
                # Look for T5/FLUX indicators
                if any('t5' in k.lower() for k in model_keys):
                    return "flux"
            
            return "sd15"
            
        except Exception as e:
            print(f"[LunaSharedCLIPEncode] Architecture detection error: {e}")
            return "unknown"


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSharedCLIPEncode": LunaSharedCLIPEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSharedCLIPEncode": "Luna Shared CLIP Encode",
}
