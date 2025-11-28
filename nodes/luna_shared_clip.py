"""
Luna Shared CLIP Nodes
ComfyUI nodes that use the shared VAE/CLIP daemon instead of loading CLIP models locally.

These nodes are drop-in replacements for standard CLIP encoding nodes, but they communicate
with the Luna daemon to use a single shared CLIP instance across multiple ComfyUI processes.
"""

import torch
from typing import Tuple

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
    Encode text prompts using the shared CLIP daemon.
    
    This node does NOT load CLIP models - it connects to the Luna daemon
    which holds CLIP in VRAM shared across all ComfyUI instances.
    
    Use this instead of CLIPTextEncode when running multiple ComfyUI workflows
    to save ~2-3GB VRAM per workflow.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    def encode(self, text: str) -> Tuple[list]:
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
            # Get conditioning from daemon
            cond, pooled, _, _ = daemon_client.clip_encode(text, "")
            
            # Format as ComfyUI conditioning
            conditioning = [[cond, {"pooled_output": pooled}]]
            
            return (conditioning,)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


class LunaSharedCLIPEncodeSDXL:
    """
    SDXL-style dual text encoding using the shared CLIP daemon.
    Encodes both positive and negative prompts with SDXL size conditioning.
    
    This node does NOT load CLIP models - it connects to the Luna daemon
    which holds CLIP-L and CLIP-G in VRAM shared across all ComfyUI instances.
    
    Saves ~2.5GB VRAM per workflow when running multiple instances.
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
                    "default": ""
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 8
                }),
            },
            "optional": {
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "target_width": ("INT", {"default": 1024, "min": 512, "max": 4096}),
                "target_height": ("INT", {"default": 1024, "min": 512, "max": 4096}),
            }
        }
    
    def encode(
        self,
        positive: str,
        negative: str,
        width: int,
        height: int,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024
    ) -> Tuple[list, list]:
        if not DAEMON_AVAILABLE:
            raise RuntimeError("Luna daemon client not available.")
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError("Luna VAE/CLIP Daemon is not running!")
        
        try:
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
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


class LunaSharedCLIPEncodeDual:
    """
    Encode both positive and negative prompts in one node using the shared daemon.
    More efficient than two separate CLIPTextEncode nodes.
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
                    "default": ""
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    def encode(self, positive: str, negative: str) -> Tuple[list, list]:
        if not DAEMON_AVAILABLE or not daemon_client.is_daemon_running():
            raise RuntimeError("Luna VAE/CLIP Daemon is not running!")
        
        try:
            cond, pooled, uncond, pooled_neg = daemon_client.clip_encode(positive, negative)
            
            positive_out = [[cond, {"pooled_output": pooled}]]
            negative_out = [[uncond, {"pooled_output": pooled_neg}]]
            
            return (positive_out, negative_out)
            
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSharedCLIPEncode": LunaSharedCLIPEncode,
    "LunaSharedCLIPEncodeSDXL": LunaSharedCLIPEncodeSDXL,
    "LunaSharedCLIPEncodeDual": LunaSharedCLIPEncodeDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSharedCLIPEncode": "Luna Shared CLIP Encode",
    "LunaSharedCLIPEncodeSDXL": "Luna Shared CLIP Encode (SDXL)",
    "LunaSharedCLIPEncodeDual": "Luna Shared CLIP Encode (Dual)",
}
