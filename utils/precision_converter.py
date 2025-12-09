"""
Luna Precision Converter Utility - Convert checkpoints to bf16/fp16/fp8

Supports:
- bf16: BFloat16 (Ampere+)
- fp16: Float16 (universal)
- fp8_e4m3fn: Native FP8 (Ada/Blackwell, 75% smaller)
"""

import os
from typing import Tuple, Dict

import torch
from safetensors.torch import load_file, save_file


def get_unet_keys(state_dict: dict) -> dict:
    """Extract UNet weights from a state dict."""
    unet_keys = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            unet_keys[key] = value
    return unet_keys


def convert_checkpoint_precision(
    source_checkpoint: str,
    output_path: str,
    precision: str = "bf16",
    unet_only: bool = True
) -> Tuple[float, float]:
    """
    Convert checkpoint to target precision.
    
    Always extracts UNet only by default for Luna Daemon workflow.
    VAE/CLIP are handled by the daemon for multi-instance sharing.
    
    Args:
        source_checkpoint: Path to source .safetensors
        output_path: Path to save converted checkpoint
        precision: Target precision (bf16, fp16, fp8_e4m3fn)
        unet_only: Extract only UNet weights (default True, always recommended)
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
    }
    
    target_dtype = dtype_map.get(precision)
    if target_dtype is None:
        raise ValueError(f"Unsupported precision: {precision}. Choose from {list(dtype_map.keys())}")
    
    print(f"[LunaPrecision] Loading {source_checkpoint}...")
    state_dict = load_file(source_checkpoint)
    original_size = os.path.getsize(source_checkpoint) / (1024 * 1024)
    print(f"[LunaPrecision] Original: {original_size:.1f} MB")
    
    # Extract UNet if requested
    if unet_only:
        print("[LunaPrecision] Extracting UNet weights...")
        state_dict = get_unet_keys(state_dict)
        if not state_dict:
            raise ValueError("No UNet keys found. Is this a valid model?")
        print(f"[LunaPrecision] Extracted {len(state_dict)} UNet tensors")
    
    # Convert precision
    print(f"[LunaPrecision] Converting to {precision}...")
    new_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            new_dict[key] = tensor.to(target_dtype)
        else:
            # Keep non-float tensors as-is (e.g., int, bool)
            new_dict[key] = tensor
    
    # Save
    print(f"[LunaPrecision] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(new_dict, output_path)
    
    converted_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - converted_size / original_size) * 100
    print(f"[LunaPrecision] ✓ {original_size:.1f}MB → {converted_size:.1f}MB ({reduction:.1f}% reduction)")
    
    return (original_size, converted_size)
