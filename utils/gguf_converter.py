"""
Luna GGUF Converter Utility - Proper Q4/Q8 quantization using llama-cpp-python

Two-step process:
1. Convert safetensors → F16 GGUF
2. Quantize F16 GGUF → Q4/Q8 GGUF using llama.cpp

This ensures proper ~70% compression for Q4 (6.6GB → ~2GB)
"""

import os
import ctypes
from pathlib import Path
from typing import Tuple, Dict

from safetensors.torch import load_file

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

try:
    import llama_cpp
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False


def get_unet_keys(state_dict: dict) -> dict:
    """Extract UNet weights from a state dict."""
    unet_keys = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            unet_keys[key] = value
    return unet_keys


def detect_model_architecture(state_dict: dict) -> str:
    """
    Detect model architecture from state dict.
    Returns architecture string compatible with ComfyUI-GGUF.
    
    Supported: flux, sd1, sdxl, sd3, aura, hidream, cosmos, ltxv, hyvid, wan, lumina2, qwen_image
    """
    # Check for Flux architecture
    if any("double_blocks" in key for key in state_dict.keys()):
        return "flux"
    
    # Check for SD3 architecture  
    if any("joint_blocks" in key for key in state_dict.keys()):
        return "sd3"
    
    # Check for SDXL (has add_embedding and larger transformer blocks)
    if any("add_embedding" in key for key in state_dict.keys()):
        return "sdxl"
    
    # Default to SD1.5
    return "sd1"


def write_f16_gguf(source_checkpoint: str, tensors: Dict, output_path: str, unet_only: bool = True, target_quant: str = "F16"):
    """Write tensors to F16 GGUF file."""
    if not HAS_GGUF:
        raise RuntimeError("gguf library required: pip install gguf")
    
    # Detect architecture from tensor keys
    arch = detect_model_architecture(tensors)
    print(f"[GGUF Converter] Detected architecture: {arch}")
    
    metadata = {
        "general.architecture": arch,  # Use detected architecture
        "general.name": Path(source_checkpoint).stem,
        "general.quantization": "F16",
        "luna.source_file": source_checkpoint,
        "luna.unet_only": "true" if unet_only else "false",
        "luna.dtype": target_quant,  # What the final dtype will be after quantization
    }
    
    writer = gguf.GGUFWriter(output_path, arch=arch)  # Pass detected arch to writer
    
    # Add metadata
    for key, value in metadata.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, (int, float)):
            writer.add_int32(key, int(value))
    
    # Add tensors as F16
    for i, (name, tensor) in enumerate(tensors.items()):
        if (i + 1) % 100 == 0:
            print(f"[LunaGGUF]   {i + 1}/{len(tensors)}...")
        
        tensor_np = tensor.half().cpu().numpy()
        writer.add_tensor(name, tensor_np, raw_dtype=gguf.GGMLQuantizationType.F16)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def quantize_gguf(f16_path: str, output_path: str, quant_type: str) -> bool:
    """Quantize F16 GGUF to Q4/Q8 using llama-cpp-python."""
    if not HAS_LLAMA:
        print("[LunaGGUF] llama-cpp-python not available")
        print("[LunaGGUF] Install: pip install llama-cpp-python")
        return False
    
    # Map to llama.cpp ftype codes
    ftype_map = {
        "Q4_0": 2,     # LLAMA_FTYPE_MOSTLY_Q4_0
        "Q4_K_S": 12,  # LLAMA_FTYPE_MOSTLY_Q4_K_S
        "Q4_K_M": 13,  # LLAMA_FTYPE_MOSTLY_Q4_K_M
        "Q5_0": 7,     # LLAMA_FTYPE_MOSTLY_Q5_0
        "Q5_K_M": 15,  # LLAMA_FTYPE_MOSTLY_Q5_K_M
        "Q8_0": 7,     # LLAMA_FTYPE_MOSTLY_Q8_0
    }
    
    ftype = ftype_map.get(quant_type, 13)  # Default Q4_K_M
    
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = ftype
    params.nthread = os.cpu_count() or 4
    
    input_bytes = f16_path.encode('utf-8')
    output_bytes = output_path.encode('utf-8')
    
    print(f"[LunaGGUF]   Quantizing (ftype={ftype}, threads={params.nthread})...")
    
    try:
        result = llama_cpp.llama_model_quantize(
            input_bytes,
            output_bytes,
            ctypes.pointer(params)
        )
        
        if result == 0:
            return True
        else:
            print(f"[LunaGGUF]   Error code: {result}")
            return False
            
    except Exception as e:
        print(f"[LunaGGUF]   Failed: {e}")
        return False


def convert_checkpoint_to_gguf(
    source_checkpoint: str,
    output_directory: str,
    quantization: str = "Q4_K_M",
    output_filename: str = "",
    unet_only: bool = True
) -> Tuple[str, int, float]:
    """
    Convert a safetensors checkpoint to GGUF format.
    
    Always extracts UNet only by default for Luna Daemon workflow.
    VAE/CLIP are handled by the daemon for multi-instance sharing.
    
    Args:
        source_checkpoint: Path to source .safetensors file
        output_directory: Directory to save output GGUF
        quantization: Quantization type (F16, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q8_0)
        output_filename: Custom output filename (without .gguf extension)
        unet_only: Extract only UNet weights (default True, always recommended)
    
    Returns:
        Tuple of (output_path, tensor_count, size_mb)
    """
    
    if not HAS_GGUF:
        raise RuntimeError("gguf library required: pip install gguf")
    
    # Validate
    if not os.path.exists(source_checkpoint):
        raise ValueError(f"Source not found: {source_checkpoint}")
    
    if not source_checkpoint.endswith(".safetensors"):
        raise ValueError("Source must be a .safetensors file")
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate filename
    if not output_filename:
        base = Path(source_checkpoint).stem
        output_filename = f"{base}_{quantization.lower()}"
    
    output_path = os.path.join(output_directory, f"{output_filename}.gguf")
    
    # Load checkpoint
    print(f"[LunaGGUF] Loading {source_checkpoint}...")
    state_dict = load_file(source_checkpoint)
    original_mb = os.path.getsize(source_checkpoint) / (1024 * 1024)
    print(f"[LunaGGUF] Original: {original_mb:.1f} MB")
    
    # Extract UNet
    if unet_only:
        tensors = get_unet_keys(state_dict)
        print(f"[LunaGGUF] Extracted {len(tensors)} UNet tensors")
    else:
        tensors = state_dict
    
    # Filter scalars
    valid_tensors = {k: v for k, v in tensors.items() if len(v.shape) > 0}
    print(f"[LunaGGUF] Valid tensors: {len(valid_tensors)}")
    
    # STEP 1: Write F16 GGUF
    if quantization == "F16":
        f16_path = output_path
    else:
        f16_path = output_path.replace(".gguf", "_f16_temp.gguf")
    
    print(f"[LunaGGUF] Step 1/2: Writing F16 GGUF...")
    write_f16_gguf(source_checkpoint, valid_tensors, f16_path, unet_only, quantization)
    
    f16_mb = os.path.getsize(f16_path) / (1024 * 1024)
    print(f"[LunaGGUF] F16: {f16_mb:.1f} MB")
    
    # STEP 2: Quantize
    if quantization != "F16":
        if not HAS_LLAMA:
            print("[LunaGGUF] llama-cpp-python not available, returning F16 version")
            if f16_path != output_path:
                os.rename(f16_path, output_path)
            return (output_path, len(valid_tensors), f16_mb)
        
        print(f"[LunaGGUF] Step 2/2: Quantizing to {quantization}...")
        success = quantize_gguf(f16_path, output_path, quantization)
        
        if success:
            # Cleanup temp file
            if os.path.exists(f16_path) and f16_path != output_path:
                os.remove(f16_path)
            
            final_mb = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (1 - final_mb / original_mb) * 100
            print(f"[LunaGGUF] ✓ {original_mb:.1f}MB → {final_mb:.1f}MB ({reduction:.1f}% reduction)")
            return (output_path, len(valid_tensors), final_mb)
        else:
            print("[LunaGGUF] Quantization failed, returning F16 version")
            if f16_path != output_path:
                os.rename(f16_path, output_path)
            return (output_path, len(valid_tensors), f16_mb)
    else:
        return (f16_path, len(valid_tensors), f16_mb)
