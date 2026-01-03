"""
Luna Conversion Cache Manager

Manages standardized conversion output paths and searches for existing
converted models to avoid redundant conversions.

Naming Convention:
- Precision (fp16/bf16/fp8): {basename}_{precision}.safetensors
- BitsAndBytes (nf4/int8): {basename}_{quant}.safetensors
- GGUF (Q4/Q8): {basename}_{quant}.gguf

Output Paths:
- Precision conversions: models/diffusion_models/converted/
- BitsAndBytes: models/diffusion_models/converted/
- GGUF: models/diffusion_models/gguf/
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import folder_paths


# =============================================================================
# Model Inspection
# =============================================================================

def detect_model_precision(model_path: str) -> Optional[str]:
    """
    Detect actual precision of a model by inspecting metadata or weights.
    
    First checks for 'luna_dtype' metadata (set by Luna converters).
    Falls back to inspecting tensor dtypes if metadata not found.
    
    This prevents converting fp16 → fp16 or creating duplicate files.
    
    Args:
        model_path: Path to model file (.safetensors or .gguf)
    
    Returns:
        Precision string ('fp32', 'fp16', 'bf16', 'fp8_e4m3fn', 'int8', 'nf4', 'unknown')
        or None if file doesn't exist/can't be read
    """
    if not os.path.exists(model_path):
        return None
    
    try:
        if model_path.endswith('.gguf'):
            # GGUF files store Luna metadata in the header
            import gguf
            reader = gguf.GGUFReader(model_path)
            if hasattr(reader, 'get_field'):
                field_value = reader.get_field('luna.dtype')
                if field_value and hasattr(field_value, 'parts') and field_value.parts:
                    value = field_value.parts[0]
                    # Handle both bytes and string types
                    if isinstance(value, bytes):
                        return value.decode()
                    else:
                        return str(value)
            
            # Fallback to filename if Luna metadata not present
            basename = Path(model_path).stem.lower()
            if 'q4' in basename:
                return 'Q4'
            elif 'q8' in basename:
                return 'Q8'
            elif 'q5' in basename:
                return 'Q5'
            else:
                return 'GGUF'
        
        # Safetensors - check metadata first, then inspect tensors
        try:
            with open(model_path, 'rb') as f:
                # SAFETENSORS header: first 8 bytes are length of JSON header
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    return None
                
                header_len = int.from_bytes(header_len_bytes, 'little')
                header_json = f.read(header_len).decode('utf-8')
                
                import json
                header = json.loads(header_json)
                
                # Check for Luna metadata first (most reliable)
                metadata = header.get('__metadata__', {})
                if isinstance(metadata, dict) and 'luna_dtype' in metadata:
                    return metadata['luna_dtype']
                
                # Fallback: Get first tensor's dtype
                for tensor_name, tensor_info in header.items():
                    if tensor_name == '__metadata__':
                        continue
                    
                    dtype_str = tensor_info.get('dtype', 'unknown')
                    
                    # Map dtype string to precision name
                    dtype_map = {
                        'F32': 'fp32',
                        'F16': 'fp16',
                        'BF16': 'bf16',
                        'F8': 'fp8_e4m3fn',
                        'U8': 'uint8',
                    }
                    
                    precision = dtype_map.get(dtype_str.upper(), dtype_str.lower())
                    return precision
                
                return 'unknown'
        
        except Exception as e:
            print(f"[PrecisionDetect] Warning: Could not read safetensors header: {e}")
            return None
    
    except Exception as e:
        print(f"[PrecisionDetect] Error detecting precision of {model_path}: {e}")
        return None


def should_skip_conversion(source_path: str, target_precision: str) -> Tuple[bool, Optional[str]]:
    """
    Check if conversion should be skipped because source is already target precision.
    
    Returns:
        Tuple of (should_skip, reason_or_none)
        - (True, "Already fp16"): Source is already target precision
        - (False, None): Conversion is needed
    """
    source_precision = detect_model_precision(source_path)
    
    if source_precision is None:
        return (False, None)
    
    # Normalize precision names for comparison
    source_norm = source_precision.lower()
    target_norm = target_precision.lower()
    
    # Direct match
    if source_norm == target_norm:
        return (True, f"Source is already {source_precision}")
    
    # Check for common aliases
    precision_aliases = {
        'fp8': 'fp8_e4m3fn',
        'fp8_e4m3fn': 'fp8_e4m3fn',
        'float8': 'fp8_e4m3fn',
        'nf4': 'nf4',
        'int8': 'int8',
        'uint8': 'int8',
    }
    
    source_alias = precision_aliases.get(source_norm, source_norm)
    target_alias = precision_aliases.get(target_norm, target_norm)
    
    if source_alias == target_alias:
        return (True, f"Source is already {source_precision} (alias match)")
    
    # GGUF types (Q4, Q8, Q5)
    if source_norm.startswith('q') and target_norm.startswith('q'):
        if source_norm == target_norm:
            return (True, f"Source is already {source_precision}")
    
    return (False, None)


# =============================================================================
# Path Management
# =============================================================================

def get_conversion_output_dir(conversion_type: str) -> str:
    """
    Get standardized output directory for conversion type.
    
    ComfyUI directory structure:
    - models/diffusion_models/converted/ ← fp16, bf16, fp8, int8, nf4 (precision/BNB)
    - models/unet/converted/ ← GGUF Q8_0, Q4_K, etc.
    
    Args:
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        Absolute path to output directory
    """
    if conversion_type in ['precision', 'bnb']:
        # Precision (fp16/bf16/fp8) and BitsAndBytes (nf4/int8) go to diffusion_models/converted
        try:
            diffusion_models_paths = folder_paths.get_folder_paths("diffusion_models")
            if diffusion_models_paths:
                # folder_paths returns [models/unet, models/diffusion_models]
                # We want models/diffusion_models, so find the one that doesn't contain 'unet' 
                # or default to the last path
                output_dir = None
                for path in diffusion_models_paths:
                    if 'diffusion_models' in path:
                        output_dir = os.path.join(path, "converted")
                        break
                if output_dir is None:
                    output_dir = os.path.join(diffusion_models_paths[-1], "converted")
                print(f"[ConversionCache] Using diffusion_models path: {output_dir}")
            else:
                raise ValueError("No diffusion_models paths found")
        except Exception as e:
            print(f"[ConversionCache] Warning: Could not determine conversion path: {e}")
            # Absolute fallback - create in models folder
            output_dir = os.path.join(os.getcwd(), "models", "diffusion_models", "converted")
            print(f"[ConversionCache] Using fallback path: {output_dir}")
    
    elif conversion_type == 'gguf':
        # GGUF models go to unet/converted (ComfyUI convention for quantized models)
        try:
            unet_paths = folder_paths.get_folder_paths("unet")
            if unet_paths:
                output_dir = os.path.join(unet_paths[0], "converted")
                print(f"[ConversionCache] Using unet path: {output_dir}")
            else:
                raise ValueError("No unet paths found")
        except Exception as e:
            print(f"[ConversionCache] Warning: Could not determine GGUF path: {e}")
            output_dir = os.path.join(os.getcwd(), "models", "unet", "converted")
            print(f"[ConversionCache] Using fallback path: {output_dir}")

    
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def generate_converted_filename(
    source_path: str,
    target_precision: str,
    conversion_type: str = 'precision'
) -> str:
    """
    Generate standardized filename for converted model.
    
    Args:
        source_path: Path to source model
        target_precision: Target precision/quantization (fp16, bf16, fp8, nf4, int8, Q4_K, etc.)
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        Just the filename (not full path)
    """
    basename = Path(source_path).stem  # Remove extension
    
    if conversion_type == 'gguf':
        ext = '.gguf'
    else:
        ext = '.safetensors'
    
    return f"{basename}_{target_precision}{ext}"


def get_converted_model_path(
    source_path: str,
    target_precision: str,
    conversion_type: str = 'precision'
) -> str:
    """
    Get full path for converted model (checking if it exists).
    
    Args:
        source_path: Path to source model
        target_precision: Target precision/quantization
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        Full path to converted model
    """
    output_dir = get_conversion_output_dir(conversion_type)
    filename = generate_converted_filename(source_path, target_precision, conversion_type)
    return os.path.join(output_dir, filename)


def find_existing_conversion(
    source_path: str,
    target_precision: str,
    conversion_type: str = 'precision'
) -> Optional[str]:
    """
    Check if a converted model already exists.
    
    Args:
        source_path: Path to source model
        target_precision: Target precision/quantization
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        Path to existing converted model, or None if not found
    """
    expected_path = get_converted_model_path(source_path, target_precision, conversion_type)
    
    if os.path.exists(expected_path):
        return expected_path
    
    return None


def should_convert(
    source_path: str,
    target_precision: str,
    conversion_type: str = 'precision'
) -> bool:
    """
    Determine if conversion is needed.
    
    Returns True if:
    1. Converted model doesn't exist yet
    2. Source is different from converted (e.g., different file)
    
    Args:
        source_path: Path to source model
        target_precision: Target precision/quantization
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        True if conversion needed, False if already exists
    """
    existing = find_existing_conversion(source_path, target_precision, conversion_type)
    return existing is None


def list_converted_models(conversion_type: str = 'precision') -> list:
    """
    List all converted models in the output directory.
    
    Args:
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        List of tuples (basename, precision, full_path)
    """
    output_dir = get_conversion_output_dir(conversion_type)
    models = []
    
    if conversion_type == 'gguf':
        pattern = '*.gguf'
    else:
        pattern = '*.safetensors'
    
    for filepath in Path(output_dir).glob(pattern):
        filename = filepath.name
        # Parse: {basename}_{precision}.{ext}
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            basename = parts[0]
            precision_and_ext = parts[1]
            precision = precision_and_ext.rsplit('.', 1)[0]
            models.append((basename, precision, str(filepath)))
    
    return models


def cleanup_old_conversions(
    source_path: str,
    conversion_type: str = 'precision',
    keep_count: int = 3
) -> list:
    """
    Clean up old converted models for same source, keeping most recent N.
    
    Args:
        source_path: Path to source model
        conversion_type: 'precision', 'bnb', or 'gguf'
        keep_count: Number of most recent to keep
    
    Returns:
        List of deleted paths
    """
    basename = Path(source_path).stem
    all_models = list_converted_models(conversion_type)
    
    # Filter to models from same source
    same_source = [m for m in all_models if m[0] == basename]
    
    if len(same_source) <= keep_count:
        return []
    
    # Sort by modification time (most recent first)
    same_source.sort(
        key=lambda m: os.path.getmtime(m[2]),
        reverse=True
    )
    
    # Delete older ones
    deleted = []
    for _, _, filepath in same_source[keep_count:]:
        try:
            os.remove(filepath)
            deleted.append(filepath)
        except Exception as e:
            print(f"[ConversionCache] Failed to delete {filepath}: {e}")
    
    return deleted


# =============================================================================
# Conversion Report
# =============================================================================

def get_conversion_stats(conversion_type: str = 'precision') -> dict:
    """
    Get statistics about converted models.
    
    Args:
        conversion_type: 'precision', 'bnb', or 'gguf'
    
    Returns:
        Dict with stats (count, total_size_gb, models_by_precision)
    """
    models = list_converted_models(conversion_type)
    
    total_size = 0
    by_precision = {}
    
    for basename, precision, filepath in models:
        size = os.path.getsize(filepath)
        total_size += size
        
        if precision not in by_precision:
            by_precision[precision] = []
        by_precision[precision].append({
            'basename': basename,
            'path': filepath,
            'size_mb': size / (1024 * 1024)
        })
    
    return {
        'count': len(models),
        'total_size_gb': total_size / (1024**3),
        'models_by_precision': by_precision,
        'output_dir': get_conversion_output_dir(conversion_type)
    }


# =============================================================================
# Conversion Suggestion
# =============================================================================

def suggest_conversion(source_path: str, target_precision: str) -> Tuple[str, Optional[str], str]:
    """
    Suggest best conversion approach.
    
    Determines whether to use:
    - Precision converter (fp16, bf16, fp8_e4m3fn)
    - GGUF (Q4_K, Q4_K_S, Q8_0, etc.)
    
    Note: BitsAndBytes (nf4) removed - cannot be properly serialized to safetensors.
    Use fp8_e4m3fn_scaled or GGUF instead.
    
    Args:
        source_path: Path to source model
        target_precision: Target precision/quantization (e.g., 'fp8_e4m3fn', 'gguf_Q8_0')
    
    Returns:
        Tuple of (conversion_type, existing_model_path or None, normalized_precision)
    """
    precision_types = ['fp16', 'bf16', 'fp8', 'fp8_e4m3fn', 'fp8_e4m3fn_scaled', 'fp8_e5m2']
    # BnB types removed - nf4 doesn't work with safetensors serialization
    # gguf_types - accept both with and without 'gguf_' prefix
    gguf_types = [
        'Q4_0', 'Q4_K_S', 'Q4_K', 'Q5_0', 'Q5_K_M', 'Q8_0',
        'gguf_Q4_0', 'gguf_Q4_K_S', 'gguf_Q4_K', 'gguf_Q5_0', 'gguf_Q5_K_M', 'gguf_Q8_0'
    ]
    
    # Normalize GGUF precision - strip 'gguf_' prefix if present
    normalized_precision = target_precision
    if target_precision.startswith('gguf_'):
        normalized_precision = target_precision[5:]  # Remove 'gguf_' prefix
    
    if target_precision in precision_types:
        conv_type = 'precision'
    elif target_precision in gguf_types or normalized_precision in gguf_types:
        conv_type = 'gguf'
        # Use normalized precision for GGUF operations
        target_precision = normalized_precision
    else:
        raise ValueError(f"Unknown precision type: {target_precision}")
    
    # Check if already converted (use normalized precision)
    existing = find_existing_conversion(source_path, target_precision, conv_type)
    
    return (conv_type, existing, target_precision)
