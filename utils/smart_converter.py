"""
Luna Smart Conversion Wrapper

High-level API that combines conversion_cache + checkpoint_converter.

Handles:
1. Checking source precision to skip redundant conversions
2. Checking for existing converted models in cache
3. Converting if needed
4. Returning path to converted model (existing or newly created)

Used by model_router to simplify conversion logic.
"""

import os
import sys
from typing import Tuple, Optional
from pathlib import Path

# Add utils to path for dynamic imports
_utils_path = str(Path(__file__).parent)
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)

from conversion_cache import (
    find_existing_conversion,
    should_convert,
    get_converted_model_path,
    suggest_conversion,
    detect_model_precision,
    should_skip_conversion
)
from checkpoint_converter import (
    convert_to_precision,
    convert_to_gguf,
    convert_to_bnb
)


def smart_convert(
    source_path: str,
    target_precision: str
) -> Tuple[str, bool]:
    """
    Smart conversion: check source precision, cache, convert if needed, return path.
    
    Three-stage check:
    1. If source is already target precision → return source (no conversion)
    2. If converted version exists in cache → return cached (no reconversion)
    3. Otherwise → convert and return new file
    
    Args:
        source_path: Path to source model
        target_precision: Target precision (fp16, bf16, fp8, nf4, int8, Q4_K_M, etc.)
    
    Returns:
        Tuple of (model_path, was_newly_converted)
        - model_path: Full path to model (source if no conversion, converted if done)
        - was_newly_converted: True if newly converted, False if reused/source
    """
    # Step 1: Check if source is already target precision
    skip, reason = should_skip_conversion(source_path, target_precision)
    if skip:
        print(f"[SmartConvert] ✓ Skipping conversion - {reason}")
        return (source_path, False)
    
    # Step 2: Determine conversion type and check for existing
    try:
        conv_type, existing, normalized_precision = suggest_conversion(source_path, target_precision)
        # Use normalized precision for the rest of the conversion process
        target_precision = normalized_precision
    except ValueError as e:
        print(f"[SmartConvert] Error: {e}")
        raise
    
    if existing:
        print(f"[SmartConvert] ✓ Using cached conversion: {existing}")
        return (existing, False)
    
    # Step 3: Need to convert
    output_path = get_converted_model_path(source_path, target_precision, conv_type)
    
    print(f"[SmartConvert] Converting {os.path.basename(source_path)} → {target_precision}")
    print(f"[SmartConvert] Output: {output_path}")
    
    try:
        if conv_type == 'precision':
            convert_to_precision(source_path, output_path, target_precision)
        elif conv_type == 'bnb':
            convert_to_bnb(source_path, output_path, target_precision)
        elif conv_type == 'gguf':
            convert_to_gguf(source_path, output_path, target_precision)
        
        if os.path.exists(output_path):
            # Verify the output is actually the right precision
            output_precision = detect_model_precision(output_path)
            print(f"[SmartConvert] ✓ Conversion complete: {output_path}")
            print(f"[SmartConvert]   Output precision verified: {output_precision}")
            return (output_path, True)
        else:
            raise RuntimeError(f"Conversion output not found: {output_path}")
    
    except Exception as e:
        print(f"[SmartConvert] ✗ Conversion failed: {e}")
        raise
    
    # Need to convert
    output_path = get_converted_model_path(source_path, target_precision, conv_type)
    
    print(f"[SmartConvert] Converting {source_path} → {target_precision}")
    print(f"[SmartConvert] Output: {output_path}")
    
    try:
        if conv_type == 'precision':
            convert_to_precision(source_path, output_path, target_precision)
        elif conv_type == 'bnb':
            convert_to_bnb(source_path, output_path, target_precision)
        elif conv_type == 'gguf':
            convert_to_gguf(source_path, output_path, target_precision)
        
        if os.path.exists(output_path):
            print(f"[SmartConvert] ✓ Conversion complete: {output_path}")
            return (output_path, True)
        else:
            raise RuntimeError(f"Conversion output not found: {output_path}")
    
    except Exception as e:
        print(f"[SmartConvert] ✗ Conversion failed: {e}")
        raise


def batch_check_conversions(source_path: str, precisions: list) -> dict:
    """
    Check status of multiple conversions for same source.
    
    Args:
        source_path: Path to source model
        precisions: List of target precisions to check
    
    Returns:
        Dict mapping precision → (exists, path)
    """
    results = {}
    
    for precision in precisions:
        try:
            conv_type, existing = suggest_conversion(source_path, precision)
            if existing:
                results[precision] = (True, existing)
            else:
                path = get_converted_model_path(source_path, precision, conv_type)
                results[precision] = (False, path)
        except Exception as e:
            print(f"[SmartConvert] Error checking {precision}: {e}")
            results[precision] = (False, None)
    
    return results
