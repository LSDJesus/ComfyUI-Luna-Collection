"""
Luna Vision Node - Image to Vision Embedding Converter

This node takes an image and a vision model (CLIP-H/SigLIP/Qwen3 mmproj)
and produces a vision embedding that can be combined with text conditioning.

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Luna Vision Node                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUTS:                                                                    │
│    clip_vision: CLIP_VISION (from Model Router)                            │
│    image: IMAGE (from Load Image or other source)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS:                                                                   │
│    vision_embed: LUNA_VISION_EMBED                                         │
│                  (embedding tensor for Config Gateway)                      │
└─────────────────────────────────────────────────────────────────────────────┘

Usage Flow:
===========
  1. LunaModelRouter outputs clip_vision (CLIP-H/SigLIP or Qwen3 mmproj)
  2. This node takes that + an image → produces vision_embed
  3. LunaConfigGateway accepts optional vision_embed and combines with text
  4. Combined conditioning goes to KSampler
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict

import torch
import numpy as np

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None

# Daemon support
try:
    from ..luna_daemon import client as daemon_client
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    daemon_client = None


class LunaVisionNode:
    """
    Convert an image to a vision embedding using CLIP-H/SigLIP or Qwen3 mmproj.
    
    The output embedding can be passed to LunaConfigGateway for vision-conditioned
    image generation (IP-Adapter style or native vision model conditioning).
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("LUNA_VISION_EMBED",)
    RETURN_NAMES = ("vision_embed",)
    FUNCTION = "encode"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION", {
                    "tooltip": "Vision encoder from Luna Model Router (clip_vision output)"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image to encode into vision embedding"
                }),
            },
            "optional": {
                "crop_mode": (["center", "none"], {
                    "default": "center",
                    "tooltip": "How to crop image to match vision encoder's expected size"
                }),
            }
        }
    
    def encode(
        self,
        clip_vision: Any,
        image: torch.Tensor,
        crop_mode: str = "center"
    ) -> Tuple[Dict[str, Any]]:
        """
        Encode image using vision encoder.
        
        Args:
            clip_vision: Vision model from Model Router
            image: Image tensor (B, H, W, C) in 0-1 range
            crop_mode: "center" for center crop, "none" for resize only
            
        Returns:
            vision_embed: Dict containing embedding tensor and metadata
        """
        
        # Handle different vision encoder types
        if isinstance(clip_vision, dict):
            # This is a reference dict (Qwen3 mmproj or deferred loading)
            if clip_vision.get("type") == "qwen3_mmproj":
                return self._encode_qwen3_vision(clip_vision, image, crop_mode)
            else:
                raise ValueError(f"Unknown vision encoder type: {clip_vision.get('type')}")
        else:
            # Standard CLIP Vision model
            return self._encode_clip_vision(clip_vision, image, crop_mode)
    
    def _encode_clip_vision(
        self,
        clip_vision: Any,
        image: torch.Tensor,
        crop_mode: str
    ) -> Tuple[Dict[str, Any]]:
        """Encode using standard CLIP Vision (CLIP-H/SigLIP)."""
        
        # Prepare image for CLIP Vision
        # ComfyUI's CLIP Vision expects (B, H, W, C) tensor
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Encode using CLIP Vision
        try:
            # ComfyUI CLIPVision.encode_image returns encoded dict
            encoded = clip_vision.encode_image(image)
            
            # Extract the actual embedding
            if hasattr(encoded, 'image_embeds'):
                embed = encoded.image_embeds
            elif isinstance(encoded, dict):
                embed = encoded.get('image_embeds', encoded.get('last_hidden_state'))
            else:
                embed = encoded
            
            return ({
                "type": "clip_vision",
                "embedding": embed,
                "shape": list(embed.shape) if hasattr(embed, 'shape') else None,
                "encoder": "clip_h_or_siglip",
            },)
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode image with CLIP Vision: {e}")
    
    def _encode_qwen3_vision(
        self,
        vision_config: Dict[str, Any],
        image: torch.Tensor,
        crop_mode: str
    ) -> Tuple[Dict[str, Any]]:
        """Encode using Qwen3-VL mmproj (multimodal projector)."""
        
        mmproj_path = vision_config.get("mmproj_path")
        use_daemon = vision_config.get("use_daemon", False)
        
        if not mmproj_path or not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")
        
        # Convert image tensor to format expected by Qwen3-VL
        # Image should be (B, H, W, C) in 0-1 range
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # If daemon is available and running, use it
        if use_daemon and DAEMON_AVAILABLE and daemon_client is not None:
            if daemon_client.is_daemon_running():
                try:
                    # Convert to numpy for transfer
                    img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
                    
                    # Call daemon's vision encoding
                    embed = daemon_client.encode_vision(img_np)
                    
                    return ({
                        "type": "qwen3_vision",
                        "embedding": torch.tensor(embed),
                        "shape": list(embed.shape),
                        "encoder": "qwen3_mmproj_daemon",
                    },)
                except Exception as e:
                    print(f"[LunaVisionNode] Daemon vision encoding failed: {e}")
                    # Fall through to local loading
        
        # Local mmproj loading
        try:
            # Load mmproj weights
            embed = self._local_qwen3_vision_encode(mmproj_path, image, vision_config.get("model_path"))
            
            return ({
                "type": "qwen3_vision",
                "embedding": embed,
                "shape": list(embed.shape) if hasattr(embed, 'shape') else None,
                "encoder": "qwen3_mmproj_local",
            },)
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode with Qwen3 mmproj: {e}")
    
    def _local_qwen3_vision_encode(
        self,
        mmproj_path: str,
        image: torch.Tensor,
        model_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Load mmproj and encode image locally.
        
        This is a simplified implementation - full Qwen3-VL vision encoding
        may require the transformers library for complete compatibility.
        """
        
        # Try to use llama-cpp-python for GGUF mmproj
        if mmproj_path.endswith('.gguf'):
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                
                # For GGUF, we need the main model too
                if not model_path:
                    raise RuntimeError("Qwen3 model path required for GGUF mmproj")
                
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                
                # Convert tensor to PIL
                from PIL import Image
                img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Get image embedding through chat handler
                # This extracts the vision embedding without generating text
                embed = chat_handler.get_image_embedding(pil_img)
                
                return torch.tensor(embed)
                
            except ImportError:
                print("[LunaVisionNode] llama-cpp-python not available for GGUF mmproj")
        
        # Fallback: Try transformers for safetensors
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from PIL import Image
            
            # Load processor for image preprocessing
            processor = AutoProcessor.from_pretrained(
                os.path.dirname(model_path) if model_path else os.path.dirname(mmproj_path),
                trust_remote_code=True
            )
            
            # Load just the vision part
            # This is a simplified approach - production would be more optimized
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                os.path.dirname(model_path) if model_path else os.path.dirname(mmproj_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Convert tensor to PIL
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Process image
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get vision embeddings
            with torch.no_grad():
                vision_outputs = model.visual(inputs["pixel_values"])
            
            return vision_outputs.last_hidden_state.mean(dim=1)  # Pool to get embedding
            
        except ImportError as e:
            raise RuntimeError(
                f"Neither llama-cpp-python nor transformers available for vision encoding: {e}\n"
                "Install with: pip install llama-cpp-python transformers"
            )


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaVisionNode": LunaVisionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaVisionNode": "Luna Vision Encoder 👁️",
}
