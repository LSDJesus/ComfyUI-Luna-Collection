"""
BitsAndBytes Quantization Converter for ComfyUI Models

Converts safetensors checkpoints to 4-bit NF4 or 8-bit INT8 using BitsAndBytes.
Optimized for QLoRA workflows with minimal quality loss.

Supports:
- NF4: 4-bit NormalFloat quantization (QLoRA standard, ~75% VRAM reduction)
- INT8: 8-bit integer quantization (~50% VRAM reduction)

Architecture:
- Extracts UNet only (VAE/CLIP loaded via Luna Daemon)
- Quantizes weights in-place using bitsandbytes library
- Saves as safetensors with quantized tensors
"""

import os
from typing import Tuple, Dict, Any
import torch
from safetensors.torch import load_file, save_file

# Check for bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def get_unet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract only UNet-related keys from a checkpoint state dict.
    Filters out VAE, CLIP, and other non-UNet components.
    """
    unet_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "unet.",
        "model.model.",
    ]
    
    exclude_prefixes = [
        "first_stage_model.",  # VAE
        "cond_stage_model.",   # CLIP
        "conditioner.",        # CLIP (SDXL format)
        "vae.",
        "text_encoder.",
        "text_model.",
        "clip.",
    ]
    
    unet_tensors = {}
    
    for key, tensor in state_dict.items():
        if any(key.startswith(exc) for exc in exclude_prefixes):
            continue
        
        if any(key.startswith(pre) for pre in unet_prefixes):
            unet_tensors[key] = tensor
        elif "diffusion" in key.lower() or "unet" in key.lower():
            unet_tensors[key] = tensor
    
    return unet_tensors


def quantize_tensor_nf4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Quantize a single tensor to 4-bit NormalFloat format.
    
    NF4 is optimized for normally-distributed weights (neural networks).
    Uses optimal quantization bins for Gaussian distributions.
    """
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for NF4 quantization.\n"
            "Install with: pip install bitsandbytes"
        )
    
    # Only quantize float tensors
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        return tensor
    
    # Determine device for quantization (bnb requires CUDA for packing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to float32 and move to device
    tensor_fp32 = tensor.to(device=device, dtype=torch.float32)
    
    # Quantize using bitsandbytes NF4
    # Note: This creates a Params4bit object, we need to extract the quantized data
    quantized = bnb.nn.Params4bit(  # type: ignore
        tensor_fp32,
        requires_grad=False,
        compress_statistics=True,  # type: ignore
        quant_type="nf4"  # type: ignore
    )
    
    # Ensure it is on the correct device (Params4bit might be lazy)
    if device == "cuda" and quantized.data.device.type == "cpu":
         quantized.cuda(device)
    
    # Return the quantized data (move back to CPU for saving)
    # BitsAndBytes stores this as uint8 with metadata
    return quantized.data.cpu()


def quantize_tensor_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Quantize a single tensor to 8-bit integer format.
    
    Uses symmetric quantization with per-tensor scaling.
    """
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for INT8 quantization.\n"
            "Install with: pip install bitsandbytes"
        )
    
    # Only quantize float tensors
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        return tensor
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to float32 and move to device
    tensor_fp32 = tensor.to(device=device, dtype=torch.float32)
    
    # Quantize using bitsandbytes INT8
    quantized = bnb.nn.Int8Params(  # type: ignore
        tensor_fp32,
        requires_grad=False,
        has_fp16_weights=False  # type: ignore
    ).cuda(device)
    
    # Return the quantized data
    return quantized.data.cpu()


def convert_checkpoint_to_bnb(
    source_checkpoint: str,
    output_path: str,
    quantization: str = "nf4"
) -> Tuple[float, float]:
    """
    Convert a checkpoint to BitsAndBytes quantized format.
    
    Args:
        source_checkpoint: Path to source safetensors checkpoint
        output_path: Path to save quantized checkpoint
        quantization: "nf4" or "int8"
    
    Returns:
        Tuple of (original_size_mb, quantized_size_mb)
    """
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for quantization.\n"
            "Install with: pip install bitsandbytes"
        )
    
    if quantization not in ["nf4", "int8"]:
        raise ValueError(f"Unsupported quantization type: {quantization}. Use 'nf4' or 'int8'")
    
    print(f"[BnB Converter] Loading checkpoint: {os.path.basename(source_checkpoint)}")
    state_dict = load_file(source_checkpoint)
    original_size = os.path.getsize(source_checkpoint) / (1024 * 1024)
    
    # Extract UNet only
    print(f"[BnB Converter] Extracting UNet weights...")
    unet_dict = get_unet_keys(state_dict)
    if not unet_dict:
        raise ValueError("No UNet keys found in checkpoint. Is this a valid model?")
    print(f"[BnB Converter] Extracted {len(unet_dict)} UNet tensors")
    
    # Quantize tensors
    print(f"[BnB Converter] Quantizing to {quantization.upper()}...")
    quantized_dict = {}
    quantize_fn = quantize_tensor_nf4 if quantization == "nf4" else quantize_tensor_int8
    
    total_tensors = len(unet_dict)
    for idx, (key, tensor) in enumerate(unet_dict.items(), 1):
        if idx % 100 == 0:
            print(f"[BnB Converter] Progress: {idx}/{total_tensors} tensors")
        
        try:
            quantized_dict[key] = quantize_fn(tensor)
        except Exception as e:
            print(f"[BnB Converter] Warning: Failed to quantize {key}, keeping original: {e}")
            quantized_dict[key] = tensor
    
    # Save quantized checkpoint
    print(f"[BnB Converter] Saving to {output_path}...")
    print(f"[BnB Converter] WARNING: This file contains raw packed {quantization.upper()} data without quantization state.")
    print(f"[BnB Converter] It requires a specialized loader to be used.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(quantized_dict, output_path)
    
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = (1 - quantized_size / original_size) * 100
    
    print(f"[BnB Converter] Conversion complete!")
    print(f"[BnB Converter] Original: {original_size:.1f}MB")
    print(f"[BnB Converter] Quantized: {quantized_size:.1f}MB")
    print(f"[BnB Converter] Compression: {compression_ratio:.1f}% reduction")
    
    return (original_size, quantized_size)
