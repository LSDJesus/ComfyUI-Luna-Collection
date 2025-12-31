"""
IP-Adapter Wrapper for Luna Scaffold Detailing

Wraps the IPAdapterPlus implementation to provide:
- Per-tile IP-Adapter injection for structural anchoring
- Dynamic model patching for batched tile processing
- Integration with Luna's VisionRouter for daemon offload

This is the CORRECT way to fuse vision embeddings with diffusion models.
Direct embedding concatenation/padding is architecturally wrong.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict

import comfy.model_management as model_management

# Try to import IPAdapterPlus components
try:
    from custom_nodes.comfyui_ipadapter_plus.IPAdapterPlus import (
        IPAdapter,
        ipadapter_execute,
    )
    from custom_nodes.comfyui_ipadapter_plus.CrossAttentionPatch import (
        Attn2Replace,
        ipadapter_attention,
    )
    from custom_nodes.comfyui_ipadapter_plus.utils import (
        encode_image_masked,
        ipadapter_model_loader,
    )
    HAS_IPADAPTER = True
except ImportError:
    HAS_IPADAPTER = False
    IPAdapter = None
    ipadapter_execute = None


def check_ipadapter_available() -> bool:
    """Check if IPAdapterPlus is installed and available."""
    return HAS_IPADAPTER


class LunaIPAdapterWrapper:
    """
    Wrapper for IP-Adapter that handles per-tile structural anchoring.
    
    Key Insight: IP-Adapter works by:
    1. Projecting CLIP-ViT embeddings through learned projection layers
    2. Injecting projected embeddings into cross-attention layers
    3. Modifying the attention computation to blend image and text guidance
    
    For LSD, we need to:
    1. Encode each tile/crop with CLIP-ViT
    2. Project the embeddings through IP-Adapter
    3. Apply to model during sampling for that tile
    """
    
    def __init__(
        self,
        ipadapter_model: Dict[str, Any],
        clip_vision: Any,
        is_sdxl: bool = True,
        is_plus: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the IP-Adapter wrapper.
        
        Args:
            ipadapter_model: Loaded IP-Adapter state dict
            clip_vision: CLIP Vision model for encoding
            is_sdxl: Whether this is for SDXL (vs SD1.5)
            is_plus: Whether this is IP-Adapter-Plus (higher quality)
            device: Device to run on
        """
        if not HAS_IPADAPTER:
            raise RuntimeError(
                "IPAdapterPlus is not installed. "
                "Please install from: https://github.com/cubiq/ComfyUI_IPAdapter_plus"
            )
        
        self.ipadapter_model = ipadapter_model
        self.clip_vision = clip_vision
        self.is_sdxl = is_sdxl
        self.is_plus = is_plus
        self.device = device or model_management.get_torch_device()
        self.dtype = model_management.unet_dtype()
        
        # Detect model configuration from weights
        self._detect_model_config()
        
        # Create IPAdapter instance
        self.ipa = IPAdapter(
            ipadapter_model,
            cross_attention_dim=self.cross_attention_dim,
            output_cross_attention_dim=self.output_cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
            is_sdxl=self.is_sdxl,
            is_plus=self.is_plus,
            is_full=self.is_full,
            is_faceid=False,
            is_portrait_unnorm=False,
        ).to(self.device, dtype=self.dtype)
    
    def _detect_model_config(self):
        """Detect IP-Adapter configuration from weights."""
        ip_adapter = self.ipadapter_model
        
        # Detect model type from weight keys
        self.is_full = "proj.3.weight" in ip_adapter.get("image_proj", {})
        self.is_portrait = "proj.2.weight" in ip_adapter.get("image_proj", {}) and not self.is_full
        
        # Get output dimension from weights
        if "ip_adapter" in ip_adapter and "1.to_k_ip.weight" in ip_adapter["ip_adapter"]:
            self.output_cross_attention_dim = ip_adapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        else:
            self.output_cross_attention_dim = 2048 if self.is_sdxl else 768
        
        # Set cross attention dimension
        if (self.is_plus and self.is_sdxl) or self.is_portrait:
            self.cross_attention_dim = 1280
        else:
            self.cross_attention_dim = self.output_cross_attention_dim
        
        # Set clip embeddings dimension based on model type
        if self.is_plus:
            self.clip_embeddings_dim = 1280  # ViT-H hidden states
            self.clip_extra_context_tokens = 16
        else:
            self.clip_embeddings_dim = 1024  # ViT-L/G image embeds
            self.clip_extra_context_tokens = 4
    
    def encode_images(
        self,
        images: torch.Tensor,
        batch_size: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images with CLIP Vision.
        
        Args:
            images: Image tensor [N, H, W, 3] in BHWC format
            batch_size: Batch size for encoding (0 = all at once)
        
        Returns:
            (cond_embeds, uncond_embeds) tuple
        """
        # Encode with CLIP Vision
        clip_output = encode_image_masked(self.clip_vision, images, batch_size=batch_size)
        
        if self.is_plus:
            # Use penultimate hidden states for Plus models
            img_cond_embeds = clip_output.penultimate_hidden_states
        else:
            # Use final image embeds for standard models
            img_cond_embeds = clip_output.image_embeds
        
        # Create zero unconditional embeddings
        if self.is_plus:
            zero_image = torch.zeros([1, 224, 224, 3])
            zero_output = encode_image_masked(self.clip_vision, zero_image, batch_size=batch_size)
            img_uncond_embeds = zero_output.penultimate_hidden_states
        else:
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        
        return img_cond_embeds, img_uncond_embeds
    
    def get_projected_embeds(
        self,
        img_cond_embeds: torch.Tensor,
        img_uncond_embeds: torch.Tensor,
        batch_size: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project CLIP embeddings through IP-Adapter.
        
        Args:
            img_cond_embeds: Conditional image embeddings
            img_uncond_embeds: Unconditional image embeddings
            batch_size: Batch size for projection
        
        Returns:
            (image_prompt_embeds, uncond_image_prompt_embeds) tuple
        """
        img_cond_embeds = img_cond_embeds.to(self.device, dtype=self.dtype)
        img_uncond_embeds = img_uncond_embeds.to(self.device, dtype=self.dtype)
        
        return self.ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, batch_size)
    
    def apply_to_model(
        self,
        model: Any,
        image_prompt_embeds: torch.Tensor,
        uncond_image_prompt_embeds: torch.Tensor,
        weight: float = 1.0,
        start_at: float = 0.0,
        end_at: float = 1.0,
        weight_type: str = "linear",
        attn_mask: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Apply IP-Adapter to model via cross-attention patching.
        
        Args:
            model: ComfyUI model to patch
            image_prompt_embeds: Projected conditional embeddings
            uncond_image_prompt_embeds: Projected unconditional embeddings
            weight: IP-Adapter strength (0.0-1.0)
            start_at: Start timestep for application (0.0-1.0)
            end_at: End timestep for application (0.0-1.0)
            weight_type: Weight schedule type
            attn_mask: Optional attention mask
        
        Returns:
            Patched model clone
        """
        # Clone model to avoid modifying original
        work_model = model.clone()
        
        # Get the diffusion model
        if hasattr(work_model, 'model'):
            diffusion_model = work_model.model.diffusion_model
        else:
            diffusion_model = work_model.diffusion_model
        
        # Apply cross-attention patches
        patch_kwargs = {
            "ipadapter": self.ipa,
            "weight": weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": start_at,
            "sigma_end": end_at,
            "unfold_batch": False,
            "embeds_scaling": "V only",
        }
        
        # Patch cross-attention layers
        work_model.set_model_attn2_replace(
            Attn2Replace(ipadapter_attention, **patch_kwargs),
            patch_kwargs
        )
        
        return work_model


def load_ipadapter_model(model_path: str) -> Dict[str, Any]:
    """
    Load IP-Adapter model from file.
    
    Args:
        model_path: Path to IP-Adapter safetensors/pt file
    
    Returns:
        Loaded model state dict
    """
    if not HAS_IPADAPTER:
        raise RuntimeError("IPAdapterPlus is not installed")
    
    return ipadapter_model_loader(model_path)


def create_ipadapter_wrapper(
    ipadapter_path: str,
    clip_vision: Any,
    is_sdxl: bool = True
) -> LunaIPAdapterWrapper:
    """
    Factory function to create an IP-Adapter wrapper.
    
    Args:
        ipadapter_path: Path to IP-Adapter model
        clip_vision: CLIP Vision model
        is_sdxl: Whether this is for SDXL
    
    Returns:
        Configured LunaIPAdapterWrapper
    """
    ipadapter_model = load_ipadapter_model(ipadapter_path)
    
    # Detect if it's a Plus model
    is_plus = (
        "proj.3.weight" in ipadapter_model.get("image_proj", {}) or
        "latents" in ipadapter_model.get("image_proj", {}) or
        "perceiver_resampler.proj_in.weight" in ipadapter_model.get("image_proj", {})
    )
    
    return LunaIPAdapterWrapper(
        ipadapter_model=ipadapter_model,
        clip_vision=clip_vision,
        is_sdxl=is_sdxl,
        is_plus=is_plus
    )
