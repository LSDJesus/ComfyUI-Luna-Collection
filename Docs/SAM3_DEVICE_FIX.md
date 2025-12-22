# SAM3 Device Mismatch Fix

## Problem
When using SAM3 on a non-default CUDA device (e.g., `cuda:1`), the following error occurred:

```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cuda:1)
```

**Root Cause:** In SAM3's `_encode_prompt` method (sam3_image.py:182), text token indices (`txt_ids`) were created on CPU via the tokenizer, but the `language_features` tensor was on the target CUDA device. The indexing operation `backbone_out["language_features"][:, txt_ids]` failed due to this device mismatch.

## Solution
Monkey-patch the SAM3 model's `text_tokenizer.encode` method to automatically move all tokenizer outputs to the model's target device.

### Implementation
In [daemon_server.py](../luna_daemon/daemon_server.py) after SAM3 model initialization:

```python
# Patch the text_tokenizer.encode to return tensors on the correct device
if hasattr(sam3_model, 'text_tokenizer') and hasattr(sam3_model.text_tokenizer, 'encode'):
    _original_tokenize = sam3_model.text_tokenizer.encode
    
    def encode_to_device(*args, **kwargs):
        """Ensure tokenizer output is on the model's device"""
        result = _original_tokenize(*args, **kwargs)
        
        # Move result to device if it's a tensor
        if isinstance(result, torch.Tensor):
            return result.to(device)
        # Handle dict/list/tuple of tensors
        elif isinstance(result, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in result.items()}
        elif isinstance(result, (list, tuple)):
            moved = [item.to(device) if isinstance(item, torch.Tensor) else item 
                    for item in result]
            return type(result)(moved)
        
        return result
    
    sam3_model.text_tokenizer.encode = encode_to_device
```

### Why This Works
- **Non-invasive:** Doesn't require modifying SAM3 source code
- **Comprehensive:** Handles all tensor types (single, dict, list, tuple)
- **Early intervention:** Fixes the issue at tokenization time, before indexing
- **Preserves structure:** Returns the same data structure types as the original

## Testing
Restart the Luna Daemon and test with SAM3 grounding on non-default CUDA device:

```python
# In daemon client or node
result = daemon_client.request({
    "type": "sam3_grounding_batch",
    "image": image_data,
    "prompts": ["cat", "dog"],
    "threshold": 0.25
})
```

Should now complete without device mismatch errors.

## Related Files
- [luna_daemon/daemon_server.py](../luna_daemon/daemon_server.py) - Fix implementation
- [nodes/detailing/luna_chess_refiner.py](../nodes/detailing/luna_chess_refiner.py) - SAM3 consumer node

## Date
2025-12-21
