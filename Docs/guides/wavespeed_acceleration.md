# WaveSpeed Acceleration Guide

## Overview

[Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed) provides significant inference acceleration for diffusion models **without any of TensorRT's constraints**. In our testing, WaveSpeed consistently outperforms TensorRT while maintaining full model flexibility.

### Performance Comparison

| Method | Inference Time | Notes |
|--------|---------------|-------|
| **Standard (no acceleration)** | ~9.5s | Baseline |
| **TensorRT Engine** | ~5.9s | Requires engine building, fixed resolutions, LoRA complications |
| **WaveSpeed First Block Cache** | **~5.6s** | No conversion needed, works with all LoRAs, dynamic resolutions |

**WaveSpeed is faster than TRT with none of the downsides.**

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/chengzeyi/Comfy-WaveSpeed.git
# Restart ComfyUI
```

No additional dependencies required - it uses native PyTorch optimizations.

---

## Key Nodes

### Apply First Block Cache (Recommended)

This is the primary acceleration node. It caches computations from the first transformer block, dramatically reducing redundant calculations during denoising.

**Usage:**
1. Add `Apply First Block Cache` node to your workflow
2. Connect your model through it before the sampler
3. That's it - no configuration needed

```
[Load Checkpoint] → [Apply First Block Cache] → [KSampler]
```

### Other WaveSpeed Nodes

| Node | Purpose | Use Case |
|------|---------|----------|
| `Apply First Block Cache` | Cache first transformer block | **Primary - use this** |
| `Apply FBCNN Acceleration` | Alternative caching strategy | Experimental |
| `Apply Stable Fast` | torch.compile optimizations | Requires compilation time |

---

## Why WaveSpeed Over TensorRT?

### TensorRT Limitations (Avoided with WaveSpeed)

| TensorRT Problem | WaveSpeed Solution |
|------------------|-------------------|
| Must build engine per model | Works instantly with any model |
| Fixed resolution ranges | Dynamic resolutions supported |
| LoRA merging complications | LoRAs work normally |
| Long rebuild time when model changes | No rebuild needed |
| Engine files are large (GB+) | No extra files |
| CUDA version dependencies | Standard PyTorch only |

### WaveSpeed Advantages

- ✅ **Zero configuration** - just add the node
- ✅ **Works with all LoRAs** - no merging or rebuilding
- ✅ **Dynamic resolutions** - any size works
- ✅ **Model hot-swapping** - change models freely
- ✅ **Lower VRAM overhead** - no engine loading
- ✅ **Faster than TRT** - yes, really

---

## Integration with Luna Collection

### Recommended Workflow Setup

```
[Luna Checkpoint Loader]
        ↓
[Apply First Block Cache]  ← WaveSpeed node
        ↓
[Luna LoRA Stacker]       ← LoRAs work normally!
        ↓
[Luna Shared CLIP Encode]  ← Uses daemon
        ↓
[KSampler]
        ↓
[Luna Shared VAE Decode]   ← Uses daemon
        ↓
[Luna Simple Upscaler]
```

### Multi-Instance Setup with Luna Daemon

With the Luna VAE/CLIP Daemon offloading encoding/decoding to a secondary GPU, your primary GPU focuses entirely on UNet inference. Adding WaveSpeed to this setup provides:

- **~40% faster inference** (WaveSpeed)
- **~30% VRAM savings** (Daemon offload)
- **Near-instant model swapping** (no TRT rebuild)

---

## Performance Tips

### 1. Combine with Daemon

Use Luna Shared VAE/CLIP nodes to offload encoding to `cuda:1`:
```
ComfyUI (cuda:0): UNet only + WaveSpeed
Luna Daemon (cuda:1): VAE + CLIP
```

### 2. Batch Processing

WaveSpeed caching benefits compound with batch size:
- Batch 1: ~40% speedup
- Batch 4: ~50% speedup (cache reuse across batch)

### 3. Consistent Seeds

The first block cache is most effective when prompts/conditioning are similar. For batch generation with different prompts, the speedup is still significant but slightly less.

---

## Troubleshooting

### WaveSpeed Not Speeding Up

1. Ensure node is connected **before** the sampler
2. Check that you're using `Apply First Block Cache`, not other experimental nodes
3. Verify WaveSpeed is installed: look for nodes under `WaveSpeed/` category

### Compatibility Issues

WaveSpeed is compatible with:
- ✅ SDXL, SD1.5, Pony, Illustrious
- ✅ LoRAs, textual inversions
- ✅ ControlNet, IPAdapter
- ✅ Luna Collection nodes

### Memory Errors

If you see OOM errors after adding WaveSpeed:
- The cache uses minimal additional VRAM (~100MB)
- Check for other memory-intensive nodes
- Consider using Luna Daemon to offload VAE/CLIP

---

## Legacy TensorRT Support

Luna Collection still includes `trt_engine.py` and TensorRT-capable nodes (`Luna Detailer` with engine paths) for backwards compatibility. However, **we recommend WaveSpeed for all new workflows**.

If you have existing TRT engines, they'll continue to work, but consider migrating to WaveSpeed for:
- Easier maintenance
- Better LoRA support
- Faster iteration on model changes

---

## Resources

- [Comfy-WaveSpeed GitHub](https://github.com/chengzeyi/Comfy-WaveSpeed)
- [Luna Daemon Setup](../../luna_daemon/README.md)
- [Multi-Instance Guide](../../Docs/luna_narrates_integration.md)
