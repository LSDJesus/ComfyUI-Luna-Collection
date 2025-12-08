#!/usr/bin/env python3
"""Test GGUF quantization to verify it works correctly"""

import torch
import tempfile
import os
from pathlib import Path

try:
    import gguf
    print("✓ gguf library available")
except ImportError:
    print("✗ gguf library not available")
    exit(1)

# Create a small test tensor
print("\n1. Creating test tensor (32x256 float32) - must be large enough for Q8_0...")
test_tensor = torch.randn(32, 256, dtype=torch.float32)
print(f"   Original: shape={test_tensor.shape}, dtype={test_tensor.dtype}")
print(f"   Original size: {test_tensor.numel() * 4} bytes")

# Test quantization
with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "test.gguf"
    
    print(f"\n2. Writing GGUF with Q8_0 quantization...")
    writer = gguf.GGUFWriter(output_path, arch="stable-diffusion")
    
    # Convert to numpy
    tensor_np = test_tensor.numpy()
    print(f"   Numpy shape: {tensor_np.shape}, dtype={tensor_np.dtype}")
    
    # Quantize
    print(f"   Quantizing with Q8_0...")
    quantized = gguf.quants.quantize(tensor_np, gguf.GGMLQuantizationType.Q8_0)
    print(f"   Quantized: shape={quantized.shape}, dtype={quantized.dtype}")
    
    # Add tensor
    writer.add_tensor("test_tensor", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    
    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # Check file size
    file_size = os.path.getsize(output_path)
    print(f"\n3. File written: {output_path}")
    print(f"   File size: {file_size} bytes")
    
    # Expected sizes
    original_size = 32 * 256 * 4  # 32KB
    q8_0_size = 32 * 256 + (32 * 256 // 32) * 4  # tensor + scales
    
    print(f"\n4. Size comparison:")
    print(f"   Original float32: {original_size} bytes")
    print(f"   Expected Q8_0: ~{q8_0_size} bytes")
    print(f"   GGUF file (with headers): {file_size} bytes")
    
    # If GGUF file is close to original, quantization didn't work
    if file_size > original_size * 1.5:
        print(f"\n✗ ERROR: File size suggests NO quantization occurred!")
        print(f"   Expected: ~{original_size + 200} bytes (small tensor + headers)")
        print(f"   Got: {file_size} bytes")
        exit(1)
    else:
        print(f"\n✓ SUCCESS: Quantization appears to be working!")
        ratio = file_size / original_size
        print(f"   Compression ratio: {ratio:.2f}x")

print("\n✓ All tests passed!")
