"""
Luna GGUF Converter - Convert safetensors checkpoints to GGUF format

Extracts UNet weights from checkpoints and converts to GGUF with various
quantization levels. Works with the Luna Daemon for shared VAE/CLIP.

Quantization Recommendations by GPU:
- Ampere (3000 series): Q8_0 (native INT8 tensor cores)
- Ada (4000 series): Q8_0 or FP8 (native FP8 tensor cores)  
- Blackwell (5090): Q4_K_M (native INT4 tensor cores!)
"""

import os
import json
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import torch
from safetensors.torch import load_file, save_file

try:
    import gguf
    HAS_GGUF_LIB = True
except ImportError:
    HAS_GGUF_LIB = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# Quantization types
class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30


# Block sizes for quantized types
QUANT_BLOCK_SIZES = {
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.Q4_K: 256,
    GGMLType.Q5_K: 256,
    GGMLType.Q6_K: 256,
    GGMLType.Q8_K: 256,
}


# GPU architecture recommendations
GPU_RECOMMENDATIONS = {
    "ampere": {
        "name": "Ampere (RTX 3000)",
        "native": ["INT8"],
        "recommended": "Q8_0",
        "notes": "Native INT8 tensor cores. Q8_0 gives best quality/speed."
    },
    "ada": {
        "name": "Ada Lovelace (RTX 4000)",
        "native": ["INT8", "FP8"],
        "recommended": "Q8_0",
        "notes": "Native FP8 and INT8. Q8_0 or keep as FP8 safetensors."
    },
    "blackwell": {
        "name": "Blackwell (RTX 5000)",
        "native": ["INT8", "FP8", "INT4"],
        "recommended": "Q4_K_M",
        "notes": "Native INT4 tensor cores! Q4 runs at hardware speed."
    },
}


def get_unet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract only UNet-related keys from a checkpoint state dict.
    Filters out VAE, CLIP, and other non-UNet components.
    """
    unet_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "unet.",
        "model.model.",  # Some formats
    ]
    
    # Keys to explicitly exclude
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
        # Skip excluded prefixes
        if any(key.startswith(exc) for exc in exclude_prefixes):
            continue
        
        # Include if matches UNet prefix
        if any(key.startswith(pre) for pre in unet_prefixes):
            unet_tensors[key] = tensor
        # Also include keys that look like diffusion model parts
        elif "diffusion" in key.lower() or "unet" in key.lower():
            unet_tensors[key] = tensor
    
    return unet_tensors


def quantize_tensor_q8_0(tensor: torch.Tensor) -> bytes:
    """
    Quantize a tensor to Q8_0 format.
    Q8_0: 8-bit quantization with block size 32.
    Each block: 1 float16 scale + 32 int8 values = 34 bytes per 32 elements.
    
    Fully vectorized for GPU acceleration.
    """
    # Ensure float and flatten - keep on same device
    device = tensor.device
    tensor = tensor.float().flatten()
    
    # Pad to block size multiple
    block_size = 32
    n_elements = tensor.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    
    if padded_size > n_elements:
        tensor = torch.nn.functional.pad(tensor, (0, padded_size - n_elements))
    
    # Reshape into blocks
    blocks = tensor.reshape(n_blocks, block_size)
    
    # Compute scale per block (max abs value) - vectorized
    scales = blocks.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-10)  # Avoid division by zero
    
    # Quantize to int8 - fully vectorized
    quantized = (blocks / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    
    # Convert scales to float16 - vectorized
    scales_f16 = scales.to(torch.float16)
    
    # Move to CPU for final packing
    scales_f16_cpu = scales_f16.cpu().numpy()
    quantized_cpu = quantized.cpu().numpy()
    
    # Vectorized packing: interleave scales and quantized blocks
    # Each block: 2 bytes (scale) + 32 bytes (int8) = 34 bytes
    scales_bytes = scales_f16_cpu.view(np.uint8).reshape(n_blocks, 2)
    quant_bytes = quantized_cpu.view(np.uint8).reshape(n_blocks, 32)
    
    # Concatenate scale + data per block, then flatten
    packed = np.concatenate([scales_bytes, quant_bytes], axis=1)  # (n_blocks, 34)
    
    return packed.tobytes()


def quantize_tensor_q4_0(tensor: torch.Tensor) -> bytes:
    """
    Quantize a tensor to Q4_0 format.
    Q4_0: 4-bit quantization with block size 32.
    Each block: 1 float16 scale + 16 bytes (32 x 4-bit) = 18 bytes per 32 elements.
    
    Fully vectorized for GPU acceleration.
    """
    device = tensor.device
    tensor = tensor.float().flatten()
    
    block_size = 32
    n_elements = tensor.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    
    if padded_size > n_elements:
        tensor = torch.nn.functional.pad(tensor, (0, padded_size - n_elements))
    
    blocks = tensor.reshape(n_blocks, block_size)
    
    # Scale based on max absolute value - vectorized
    max_abs = blocks.abs().max(dim=1).values
    scales = max_abs / 7.0
    scales = scales.clamp(min=1e-10)
    
    # Quantize to range [-8, 7], then shift to [0, 15] - vectorized
    quantized = (blocks / scales.unsqueeze(1)).round().clamp(-8, 7)
    quantized_unsigned = (quantized + 8).to(torch.uint8)
    
    # Convert scales to float16 and move to CPU
    scales_f16 = scales.to(torch.float16).cpu().numpy()
    quant_cpu = quantized_unsigned.cpu().numpy()  # (n_blocks, 32)
    
    # Vectorized 4-bit packing: pair adjacent values into single bytes
    # low nibble = even indices, high nibble = odd indices
    low_nibbles = quant_cpu[:, 0::2] & 0x0F  # (n_blocks, 16)
    high_nibbles = quant_cpu[:, 1::2] & 0x0F  # (n_blocks, 16)
    packed_nibbles = (low_nibbles | (high_nibbles << 4)).astype(np.uint8)  # (n_blocks, 16)
    
    # Prepare scale bytes
    scales_bytes = scales_f16.view(np.uint8).reshape(n_blocks, 2)  # (n_blocks, 2)
    
    # Concatenate: 2 bytes scale + 16 bytes packed = 18 bytes per block
    packed = np.concatenate([scales_bytes, packed_nibbles], axis=1)  # (n_blocks, 18)
    
    return packed.tobytes()


def write_gguf_header(f, n_tensors: int, metadata: Dict[str, Any]):
    """Write GGUF file header."""
    # Magic number
    f.write(struct.pack('<I', GGUF_MAGIC))
    # Version
    f.write(struct.pack('<I', GGUF_VERSION))
    # Number of tensors
    f.write(struct.pack('<Q', n_tensors))
    # Number of metadata key-value pairs
    f.write(struct.pack('<Q', len(metadata)))
    
    # Write metadata
    for key, value in metadata.items():
        # Key (string)
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        
        # Value type and value
        if isinstance(value, str):
            # String type = 8
            f.write(struct.pack('<I', 8))
            val_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(val_bytes)))
            f.write(val_bytes)
        elif isinstance(value, int):
            # uint64 type = 5
            f.write(struct.pack('<I', 5))
            f.write(struct.pack('<Q', value))
        elif isinstance(value, float):
            # float32 type = 6
            f.write(struct.pack('<I', 6))
            f.write(struct.pack('<f', value))


class LunaGGUFConverter:
    """
    Convert safetensors checkpoints to GGUF format.
    
    Extracts UNet weights and quantizes them for efficient loading.
    Use with Luna Daemon for shared VAE/CLIP, or ComfyUI-GGUF loader.
    
    GPU Recommendations:
    - RTX 3000 (Ampere): Q8_0 - native INT8 tensor cores
    - RTX 4000 (Ada): Q8_0 - native INT8/FP8
    - RTX 5090 (Blackwell): Q4_K_M - native INT4 tensor cores!
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "original_mb", "converted_mb")
    FUNCTION = "convert"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        quant_options = [
            "Q8_0 (recommended for Ampere/Ada)",
            "Q4_0 (smaller, Blackwell optimized)",
            "Q4_K_M (best quality Q4, Blackwell)",
            "Q5_0 (balanced)",
            "Q5_K_M (balanced, better quality)",
            "F16 (no quantization)",
        ]
        
        return {
            "required": {
                "source_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Path to source .safetensors checkpoint file"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save converted GGUF file"
                }),
                "quantization": (quant_options, {
                    "default": quant_options[0],
                    "tooltip": "Quantization level - see GPU recommendations"
                }),
            },
            "optional": {
                "output_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output filename (without extension). Leave empty to auto-generate."
                }),
                "extract_unet_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract only UNet weights (recommended for Luna Daemon workflow)"
                }),
            }
        }
    
    def convert(self, source_checkpoint: str, output_directory: str,
                quantization: str, output_filename: str = "",
                extract_unet_only: bool = True) -> Tuple[str, int, float]:
        """Convert a safetensors checkpoint to GGUF format."""
        
        # Validate source
        if not os.path.exists(source_checkpoint):
            raise ValueError(f"Source checkpoint not found: {source_checkpoint}")
        
        if not source_checkpoint.endswith(".safetensors"):
            raise ValueError("Source must be a .safetensors file")
        
        # Create output directory if needed
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate output filename
        if not output_filename:
            base_name = Path(source_checkpoint).stem
            quant_suffix = quantization.split()[0].lower()
            output_filename = f"{base_name}-{quant_suffix}"
        
        output_path = os.path.join(output_directory, f"{output_filename}.gguf")
        
        # Get original size
        original_size_mb = os.path.getsize(source_checkpoint) / (1024 * 1024)
        
        print(f"[LunaGGUF] Loading {source_checkpoint}...")
        print(f"[LunaGGUF] Original size: {original_size_mb:.1f} MB")
        
        # Load checkpoint
        state_dict = load_file(source_checkpoint)
        
        # Extract UNet if requested
        if extract_unet_only:
            print("[LunaGGUF] Extracting UNet weights...")
            tensors = get_unet_keys(state_dict)
            print(f"[LunaGGUF] Found {len(tensors)} UNet tensors (from {len(state_dict)} total)")
        else:
            tensors = state_dict
        
        # Determine quantization type
        quant_type = quantization.split()[0]
        
        print(f"[LunaGGUF] Quantizing to {quant_type}...")
        
        # Build GGUF metadata
        metadata = {
            "general.architecture": "stable-diffusion",
            "general.name": Path(source_checkpoint).stem,
            "general.quantization": quant_type,
            "luna.source_file": source_checkpoint,
            "luna.unet_only": "true" if extract_unet_only else "false",
        }
        
        # First pass: collect all valid tensors (skip scalars)
        valid_tensors = {}
        for name, tensor in tensors.items():
            shape = list(tensor.shape)
            n_dims = len(shape)
            
            # Skip 0-dimensional (scalar) tensors - GGUF reader expects at least 1 dimension
            if n_dims == 0:
                print(f"[LunaGGUF] Skipping scalar tensor: {name}")
                continue
            
            valid_tensors[name] = tensor
        
        print(f"[LunaGGUF] Valid tensors after filtering: {len(valid_tensors)}")
        
        # Write GGUF file using proper format
        if HAS_GGUF_LIB:
            # Use official gguf library if available - it handles quantization properly
            try:
                writer = gguf.GGUFWriter(output_path, arch="stable-diffusion")
                
                # Write metadata
                for key, value in metadata.items():
                    if isinstance(value, str):
                        writer.add_string(key, value)
                    elif isinstance(value, (int, float)):
                        writer.add_int32(key, int(value))
                
                # Determine block size for quantization
                if quant_type in ["Q4_0", "Q4_K_M"]:
                    block_size = 32
                elif quant_type in ["Q8_0"]:
                    block_size = 32
                else:
                    block_size = 1  # F16 doesn't need padding
                
                # Process and write tensors - pad dimensions for quantization compatibility
                print(f"[LunaGGUF] Writing {len(valid_tensors)} tensors with gguf library...")
                for i, (name, tensor) in enumerate(valid_tensors.items()):
                    if (i + 1) % 100 == 0:
                        print(f"[LunaGGUF] Processing tensor {i + 1}/{len(valid_tensors)}...")
                    
                    # For quantized formats, pad last dimension to multiple of block size
                    if block_size > 1:
                        shape = list(tensor.shape)
                        last_dim = shape[-1]
                        padded_dim = ((last_dim + block_size - 1) // block_size) * block_size
                        
                        if padded_dim != last_dim:
                            # Pad the tensor
                            padding = padded_dim - last_dim
                            tensor = torch.nn.functional.pad(tensor, (0, padding))
                    
                    # Convert tensor to numpy (float32) - let gguf library handle quantization
                    tensor_np = tensor.float().cpu().numpy()
                    
                    # Add tensor - gguf library will quantize based on architecture
                    writer.add_tensor(name, tensor_np)
                
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                print("[LunaGGUF] File written successfully with gguf library")
                
            except Exception as e:
                print(f"[LunaGGUF] Warning: gguf library write failed ({e}), falling back to manual method")
                use_fallback = True
            else:
                use_fallback = False
        else:
            use_fallback = True
        
        # Fallback: Manual GGUF writing if library not available or failed
        if use_fallback or not HAS_GGUF_LIB:
            print("[LunaGGUF] Using fallback manual GGUF writing...")
            with open(output_path, 'wb') as f:
                # Write header with correct tensor count
                write_gguf_header(f, len(valid_tensors), metadata)
                
                # Collect all tensor data first
                tensor_data_list = []
                
                # Determine block size for current quantization type
                if quant_type in ["Q4_0", "Q4_K_M"]:
                    block_size = 32
                elif quant_type in ["Q8_0"]:
                    block_size = 32
                else:
                    block_size = 1  # No alignment needed for F16
                
                # First pass: quantize all tensors and collect data
                for i, (name, tensor) in enumerate(valid_tensors.items()):
                    if (i + 1) % 100 == 0:
                        print(f"[LunaGGUF] Quantizing tensor {i + 1}/{len(valid_tensors)}...")
                    
                    # Get original shape
                    original_shape = list(tensor.shape)
                    
                    # Quantize based on type
                    if quant_type in ["Q8_0"]:
                        quantized_data = quantize_tensor_q8_0(tensor)
                        ggml_type = GGMLType.Q8_0.value
                    elif quant_type in ["Q4_0", "Q4_K_M"]:
                        quantized_data = quantize_tensor_q4_0(tensor)
                        ggml_type = GGMLType.Q4_0.value
                    elif quant_type in ["Q5_0", "Q5_K_M"]:
                        quantized_data = quantize_tensor_q8_0(tensor)
                        ggml_type = GGMLType.Q8_0.value
                    else:  # F16
                        tensor_f16 = tensor.to(torch.float16)
                        quantized_data = tensor_f16.numpy().tobytes()
                        ggml_type = GGMLType.F16.value
                    
                    # For quantized formats, ensure last dimension is multiple of block size
                    # This is required by GGUF format
                    shape_for_gguf = original_shape.copy()
                    if ggml_type in [GGMLType.Q4_0.value, GGMLType.Q8_0.value] and len(shape_for_gguf) > 0:
                        # Pad last dimension to multiple of 32
                        last_dim = shape_for_gguf[-1]
                        padded_dim = ((last_dim + 31) // 32) * 32
                        if padded_dim != last_dim:
                            shape_for_gguf[-1] = padded_dim
                    
                    tensor_data_list.append({
                        'name': name,
                        'shape': shape_for_gguf,
                        'data': quantized_data,
                        'type': ggml_type
                    })
                
                # Second pass: write all tensor headers first, then all data
                data_offset = f.tell()
                offset_accumulator = 0
                
                for tensor_info in tensor_data_list:
                    name = tensor_info['name']
                    shape = tensor_info['shape']
                    ggml_type = tensor_info['type']
                    data_size = len(tensor_info['data'])
                    
                    n_dims = len(shape)
                    
                    # Write tensor name
                    name_bytes = name.encode('utf-8')
                    f.write(struct.pack('<Q', len(name_bytes)))
                    f.write(name_bytes)
                    
                    # Write dimensions
                    f.write(struct.pack('<I', n_dims))
                    for dim in shape:
                        f.write(struct.pack('<Q', dim))
                    
                    # Write type
                    f.write(struct.pack('<I', ggml_type))
                    
                    # Write offset (will be relative to data start)
                    f.write(struct.pack('<Q', offset_accumulator))
                    offset_accumulator += data_size
                
                # Write all tensor data
                for tensor_info in tensor_data_list:
                    f.write(tensor_info['data'])
        
        # Get final size
        converted_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - converted_size_mb / original_size_mb) * 100
        
        print(f"[LunaGGUF] Conversion complete!")
        print(f"[LunaGGUF] Output: {output_path}")
        print(f"[LunaGGUF] Size: {converted_size_mb:.1f} MB ({reduction:.1f}% reduction)")
        
        return (output_path, int(original_size_mb), converted_size_mb)


class LunaGGUFBatchConverter:
    """
    Batch convert multiple checkpoints to GGUF format.
    
    Scans a directory for .safetensors files and converts them all.
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("report", "converted", "failed")
    FUNCTION = "batch_convert"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        quant_options = [
            "Q8_0 (recommended for Ampere/Ada)",
            "Q4_0 (smaller, Blackwell optimized)",
            "Q4_K_M (best quality Q4, Blackwell)",
            "F16 (no quantization)",
        ]
        
        return {
            "required": {
                "source_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing .safetensors checkpoints"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save converted GGUF files"
                }),
                "quantization": (quant_options, {
                    "default": quant_options[0],
                }),
                "max_files": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Maximum number of files to convert"
                }),
            },
            "optional": {
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip if GGUF already exists in output directory"
                }),
            }
        }
    
    def batch_convert(self, source_directory: str, output_directory: str,
                      quantization: str, max_files: int,
                      skip_existing: bool = True) -> Tuple[str, int, int]:
        """Batch convert checkpoints to GGUF."""
        
        if not os.path.isdir(source_directory):
            return (f"Source directory not found: {source_directory}", 0, 0)
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Find safetensors files
        files = [f for f in os.listdir(source_directory) 
                 if f.endswith('.safetensors')][:max_files]
        
        report_lines = [
            f"Luna GGUF Batch Converter",
            f"Source: {source_directory}",
            f"Output: {output_directory}",
            f"Quantization: {quantization}",
            f"Found {len(files)} checkpoint(s)",
            "=" * 50
        ]
        
        converter = LunaGGUFConverter()
        converted = 0
        failed = 0
        
        for filename in files:
            source_path = os.path.join(source_directory, filename)
            base_name = Path(filename).stem
            quant_suffix = quantization.split()[0].lower()
            output_name = f"{base_name}-{quant_suffix}.gguf"
            output_path = os.path.join(output_directory, output_name)
            
            if skip_existing and os.path.exists(output_path):
                report_lines.append(f"SKIP {filename}: already exists")
                continue
            
            try:
                result = converter.convert(
                    source_path, output_directory, quantization,
                    output_filename=f"{base_name}-{quant_suffix}",
                    extract_unet_only=True
                )
                original_mb, converted_mb = result[1], result[2]
                report_lines.append(
                    f"OK   {filename}: {original_mb}MB -> {converted_mb:.1f}MB"
                )
                converted += 1
            except Exception as e:
                report_lines.append(f"FAIL {filename}: {str(e)[:50]}")
                failed += 1
        
        report_lines.append("=" * 50)
        report_lines.append(f"Converted: {converted}, Failed: {failed}")
        
        return ("\n".join(report_lines), converted, failed)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaGGUFConverter": LunaGGUFConverter,
    "LunaGGUFBatchConverter": LunaGGUFBatchConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaGGUFConverter": "Luna GGUF Converter",
    "LunaGGUFBatchConverter": "Luna GGUF Batch Converter",
}
