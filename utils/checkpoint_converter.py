"""
Checkpoint Conversion Wrappers

High-level conversion API that wraps the specific converter utilities.
Used by multiple nodes (Luna Dynamic Loader, Luna Model Router, etc.)
to provide consistent conversion interface.

All conversions:
- Extract UNet only (VAE/CLIP loaded separately via Luna Daemon)
- Return (original_size_mb, converted_size_mb) tuple
"""

import os
from typing import Tuple


def convert_to_precision(
    src_path: str, 
    dst_path: str, 
    precision: str, 
    strip_components: bool = True
) -> Tuple[float, float]:
    """
    Convert checkpoint to target precision safetensors.
    
    Wrapper around utils.precision_converter for bf16, fp16, fp8_e4m3fn.
    Always strips VAE/CLIP, keeping only UNet weights for Luna Daemon workflow.
    
    Args:
        src_path: Path to source checkpoint
        dst_path: Path to save converted checkpoint
        precision: Target precision (bf16, fp16, fp8_e4m3fn)
        strip_components: Deprecated, kept for compatibility. Always strips to UNet only.
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    import importlib.util
    from pathlib import Path
    
    precision_path = Path(__file__).parent / "precision_converter.py"
    spec = importlib.util.spec_from_file_location("precision_converter", precision_path)
    precision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precision_module)
    convert_checkpoint_precision = precision_module.convert_checkpoint_precision
    
    print(f"[Checkpoint Converter] Converting to {precision} (UNet only)")
    original_size, converted_size = convert_checkpoint_precision(
        source_checkpoint=src_path,
        output_path=dst_path,
        precision=precision
    )
    
    return (original_size, converted_size)


def convert_to_gguf(
    src_path: str,
    dst_path: str,
    quant_type: str
) -> Tuple[float, float]:
    """
    Convert checkpoint to GGUF format with quantization.
    
    Wrapper around utils.gguf_converter for Q4/Q8 GGUF quantization.
    Always strips VAE/CLIP, keeping only UNet weights for Luna Daemon workflow.
    
    Args:
        src_path: Path to source checkpoint
        dst_path: Path to save converted GGUF file
        quant_type: GGUF quantization type (Q4_K_M, Q8_0, etc.)
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    import importlib.util
    from pathlib import Path
    
    gguf_path = Path(__file__).parent / "gguf_converter.py"
    spec = importlib.util.spec_from_file_location("gguf_converter", gguf_path)
    gguf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gguf_module)
    convert_checkpoint_to_gguf = gguf_module.convert_checkpoint_to_gguf
    
    output_dir = os.path.dirname(dst_path)
    output_name = os.path.splitext(os.path.basename(dst_path))[0]
    
    print(f"[Checkpoint Converter] Converting to GGUF: {quant_type} (UNet only)")
    output_path, tensor_count, final_size = convert_checkpoint_to_gguf(
        source_checkpoint=src_path,
        output_directory=output_dir,
        quantization=quant_type,
        output_filename=output_name
    )
    
    original_size = os.path.getsize(src_path) / (1024 * 1024)
    return (original_size, final_size)


def convert_to_bnb(
    src_path: str,
    dst_path: str,
    quant_type: str
) -> Tuple[float, float]:
    """
    Convert checkpoint to BitsAndBytes quantized format.
    
    Wrapper around utils.bitsandbytes_converter for NF4/INT8 quantization.
    Always strips VAE/CLIP, keeping only UNet weights for Luna Daemon workflow.
    
    Args:
        src_path: Path to source checkpoint
        dst_path: Path to save quantized checkpoint
        quant_type: BitsAndBytes quantization type (nf4, int8)
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    import importlib.util
    from pathlib import Path
    
    bnb_path = Path(__file__).parent / "bitsandbytes_converter.py"
    spec = importlib.util.spec_from_file_location("bitsandbytes_converter", bnb_path)
    bnb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnb_module)
    convert_checkpoint_to_bnb = bnb_module.convert_checkpoint_to_bnb
    
    print(f"[Checkpoint Converter] Converting to BitsAndBytes: {quant_type} (UNet only)")
    original_size, converted_size = convert_checkpoint_to_bnb(
        source_checkpoint=src_path,
        output_path=dst_path,
        quantization=quant_type
    )
    
    return (original_size, converted_size)
