"""
Conditioning Utilities - Model conditioning and IP-Adapter helpers

Handles:
- Conditioning replication for batched inference
- IP-Adapter application (batched)
- Vision embedding fusion
- VAE encode/decode wrappers

Used by: RefinementEngine, LunaChessRefiner, LunaSemanticDetailer
"""

import torch
from typing import List, Any, Optional


# =============================================================================
# VAE Helpers
# =============================================================================

def encode_pixels(vae: Any, pixels: torch.Tensor) -> torch.Tensor:
    """
    Encode pixels to latent space using VAE.
    
    ComfyUI's vae.encode() expects BHWC format and handles conversion internally.
    
    Args:
        vae: ComfyUI VAE wrapper
        pixels: Tensor in BHWC format [batch, height, width, channels]
    
    Returns:
        Latent tensor [batch, channels, latent_h, latent_w]
    """
    return vae.encode(pixels)


def decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode latents to pixel space using VAE.
    
    Args:
        vae: ComfyUI VAE wrapper
        latents: Latent tensor [batch, channels, latent_h, latent_w]
    
    Returns:
        Pixels tensor in BHWC format [batch, height, width, channels]
    """
    return vae.decode(latents)


# =============================================================================
# Conditioning Helpers
# =============================================================================

def replicate_conditioning(cond: List, count: int) -> List:
    """
    Replicate conditioning for batch processing.
    
    Args:
        cond: ComfyUI conditioning format - list of [tensor, dict]
        count: Number of times to replicate
    
    Returns:
        Replicated conditioning with batched tensors
    """
    if count == 1:
        return cond
    
    replicated = []
    for c in cond:
        tensor, opts = c
        # Repeat tensor along batch dimension
        batched_tensor = tensor.repeat(count, 1, 1)
        replicated.append([batched_tensor, opts.copy()])
    
    return replicated


# =============================================================================
# IP-Adapter Helpers
# =============================================================================

def apply_ip_adapter_batch(
    model: Any,
    ip_adapter: Any,
    vision_batch: torch.Tensor,
    uncond_batch: torch.Tensor,
    weight: float = 0.5
) -> Any:
    """
    Apply IP-Adapter patch to model with batched vision embeddings.
    
    KEY INSIGHT: PyTorch attention maps Latent[i] → Embed[i] when batch dims match.
    This enables TRUE BATCHING - N tiles get N distinct vision anchors in ONE pass.
    
    Args:
        model: ComfyUI model wrapper (will be cloned)
        ip_adapter: IP-Adapter model
        vision_batch: [N, seq_len, dim] vision embeddings
        uncond_batch: [N, seq_len, dim] unconditional embeddings
        weight: IP-Adapter weight (0.0-1.0)
    
    Returns:
        Patched model clone with IP-Adapter conditioning
    """
    if ip_adapter is None:
        return model
    
    try:
        # IPAdapterUnifiedLoader format
        if isinstance(ip_adapter, dict) and "ipadapter" in ip_adapter:
            ipadapter_model = ip_adapter["ipadapter"]["model"]
        else:
            ipadapter_model = ip_adapter
        
        # Clone model to avoid modifying original
        work_model = model.clone()
        
        # Get the IPAdapter application function
        from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterApply
        
        # Create conditioning dict
        cond = {
            "c_crossattn": vision_batch,
            "uncond": uncond_batch,
        }
        
        # Apply IP-Adapter patch
        ipadapter_model.set_model_ipadapter_patch(
            work_model.model,
            cond,
            weight=weight
        )
        
        return work_model
        
    except Exception as e:
        print(f"[ConditioningUtils] IP-Adapter batch application failed: {e}")
        return model


def fuse_vision_conditioning(
    text_cond: List,
    vision_embeds: torch.Tensor,
    method: str = "concat"
) -> List:
    """
    Fuse vision embeddings with text conditioning.
    
    Args:
        text_cond: ComfyUI conditioning format - list of [tensor, dict]
        vision_embeds: Vision embeddings tensor
        method: Fusion method - "concat", "add", "replace"
    
    Returns:
        Fused conditioning
    """
    fused = []
    for c in text_cond:
        tensor, opts = c
        
        if method == "concat":
            # Concatenate vision embeddings to text embeddings
            # Expand vision to match text batch size if needed
            if vision_embeds.shape[0] == 1 and tensor.shape[0] > 1:
                vision_embeds = vision_embeds.expand(tensor.shape[0], -1, -1)
            
            combined = torch.cat([tensor, vision_embeds.to(tensor.device)], dim=1)
            fused.append([combined, opts.copy()])
            
        elif method == "add":
            # Add vision to text (requires same dimensions)
            combined = tensor + vision_embeds.to(tensor.device)
            fused.append([combined, opts.copy()])
            
        elif method == "replace":
            fused.append([vision_embeds.to(tensor.device), opts.copy()])
            
        else:
            fused.append([tensor, opts.copy()])
    
    return fused


def fuse_vision_conditioning_batch(
    text_cond: List,
    vision_embeds: torch.Tensor,
    batch_size: int
) -> List:
    """
    Fuse batched vision embeddings with text conditioning.
    
    For tiled refinement, each tile gets its own vision anchor.
    
    Args:
        text_cond: ComfyUI conditioning format
        vision_embeds: [batch_size, seq_len, dim] vision embeddings
        batch_size: Number of tiles in batch
    
    Returns:
        Fused conditioning with per-tile vision
    """
    fused = []
    for c in text_cond:
        tensor, opts = c
        
        # Expand text to batch size
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(batch_size, 1, 1)
        
        # Concatenate vision per-tile
        combined = torch.cat([tensor, vision_embeds.to(tensor.device)], dim=1)
        fused.append([combined, opts.copy()])
    
    return fused


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # VAE
    "encode_pixels",
    "decode_latents",
    
    # Conditioning
    "replicate_conditioning",
    
    # IP-Adapter
    "apply_ip_adapter_batch",
    "fuse_vision_conditioning",
    "fuse_vision_conditioning_batch",
]
