# GGUF Quantization Fix - Investigation & Solution

## Problem Identified

The GGUF converter was producing **23GB files instead of expected compressed files** because:

1. **Not calling `gguf.quants.quantize()`** - The converter was using `writer.add_tensor()` without specifying a quantization type, which defaults to uncompressed storage
2. **No explicit quantization type mapping** - The gguf library needs explicit quantization type specification
3. **Wrong architecture string** - Using `arch="stable-diffusion"` instead of a proper architecture

## Root Cause Analysis

From the official `llama.cpp` GGUF conversion code (`convert_hf_to_gguf.py`), the proper flow is:

```python
# 1. Convert tensor to numpy (float32)
data = tensor_torch.numpy()

# 2. Explicitly quantize using gguf.quants.quantize()
try:
    data = gguf.quants.quantize(data, data_qtype)  # <-- THIS WAS MISSING
except gguf.QuantError as e:
    logger.warning(f"Quantization failed: {e}, falling back to F16")
    data = gguf.quants.quantize(data, gguf.GGMLQuantizationType.F16)

# 3. Add with explicit raw_dtype
writer.add_tensor(name, data, raw_dtype=data_qtype)  # <-- NEED raw_dtype
```

Our code was doing:
```python
writer.add_tensor(name, tensor_np)  # ❌ No quantization, no raw_dtype
```

## Test Results

Created `test_gguf_quant.py` to verify:

- ✓ Tensor (32x256 float32): 32KB → 8.8KB after Q8_0 quantization
- ✓ Compression ratio: **0.27x** (3.7:1 reduction)
- ✓ File size is as expected

## Fix Applied

Updated `nodes/luna_gguf_converter.py`:

```python
# Map quantization types to gguf library enums
quant_map = {
    "F16": gguf.GGMLQuantizationType.F16,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q4_0": gguf.GGMLQuantizationType.Q4_0,
    "Q4_K_M": gguf.GGMLQuantizationType.Q4_K_M,
    "Q5_0": gguf.GGMLQuantizationType.Q5_0,
    "Q5_K_M": gguf.GGMLQuantizationType.Q5_K_M,
}

quant_type_gguf = quant_map.get(quant_type, gguf.GGMLQuantizationType.F16)

# Convert and quantize each tensor
tensor_np = tensor.float().cpu().numpy()
quantized_data = gguf.quants.quantize(tensor_np, quant_type_gguf)

# Add with explicit quantization type
writer.add_tensor(name, quantized_data, raw_dtype=quant_type_gguf)
```

## Architecture Issue

Note: The `arch="stable-diffusion"` parameter is still being used. This appears to be a fallback that works but isn't ideal. The proper architecture string for UNet weights would depend on the actual model architecture. For now it works, but this could be improved in future to properly detect and use the correct architecture.

## Files Affected

- `nodes/luna_gguf_converter.py` - Fixed quantization call
- `test_gguf_quant.py` - New test file to verify quantization

## Next Steps

1. Test the actual GGUF converter with a real model
2. Verify file sizes match expected compression ratios
3. Test that generated GGUF files are readable by llama.cpp
4. Consider improving architecture detection for better compatibility
