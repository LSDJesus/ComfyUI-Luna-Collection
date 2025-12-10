# Luna Daemon Image Saver - Quick Start

## In 3 Steps:

### 1. Start the Daemon
```powershell
cd D:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection
python -m luna_daemon.server
```

You'll see:
```
Socket server: 127.0.0.1:19283
Image save pool configured on CPU (up to 4 parallel workers)
Ready to accept connections!
```

### 2. Add Node to Workflow
- Right-click in ComfyUI graph
- Search: `Luna Daemon Image Saver`
- Configure:
  - **save_path**: `my_outputs/%model_name%`
  - **filename**: `%time:YYYY-mm-dd_%index%`
  - **image_1**: Connect your image
  - **affix_1**: `"generation"`
  - **format_1**: `"png"`

### 3. Run Workflow
- Click **Queue Prompt**
- Node returns immediately with job ID
- Images save in background while workflow continues

## Example Setup

```
[KSampler] → [VAE Decode] → [Luna Daemon Image Saver]
                                  ↓
                            (Job submitted, returns)
                                  ↓
                            [Next Generation]  ← starts immediately!
```

## What You Get

- ✅ Workflow doesn't block on disk I/O
- ✅ Multiple images save in parallel (2-4 workers)
- ✅ Same features as Luna Multi Saver (quality gates, metadata, etc.)
- ✅ Real-time job tracking via job ID

## Comparison: Before & After

**Before (Luna Multi Saver):**
```
Generate (2s) → Save (3s) → Generate (2s) → Save (3s) → ...
Total per generation: 5 seconds
```

**After (Daemon Image Saver):**
```
Generate (2s) → Queue Save, Return (0.01s) → Generate (2s) → ...
Total per generation: 2 seconds (saves happen in parallel)
```

For 4K images or batches, savings are even more dramatic.

## Key Differences from Luna Multi Saver

| Feature | Multi Saver | Daemon Saver |
|---------|-------------|--------------|
| Blocks workflow | Yes | **No** |
| Return value | Filenames | **Job ID** |
| Best for | Quick saves | **Production/batch** |
| Requires daemon | No | **Yes** |

## If Daemon Is Not Running

The node will return an error. Either:

1. **Start the daemon** (see step 1 above), or
2. **Switch to Luna Multi Saver** (same parameters, blocks locally)

## Parameters Quick Reference

### Required
```
save_path: "subfolder/%model_name%"     # Where to save
filename: "%time:YYYY-mm-dd_%index%"    # File template
image_1: [image tensor]                  # Your image
affix_1: "generation"                    # Label (e.g., RAW, UPSCALED)
format_1: "png"                          # Format (png, webp, jpeg)
```

### Optional
```
quality_gate: "disabled"                 # Or: variance, edge_density, both
min_quality_threshold: 0.3               # Filter bad images (0.0-1.0)
png_compression: 4                       # 0-9 (higher = slower)
embed_workflow: True                     # Include metadata
```

## Template Variables

| Variable | Example | Result |
|----------|---------|--------|
| `%model_name%` | `hsUltrahdCG_v60` | Checkpoint name |
| `%index%` | `0` | Batch index |
| `%time:YYYY-mm-dd%` | `2025-12-09` | Date |
| `%time:HH.MM.SS%` | `23.45.30` | Time |

## Monitor Daemon

In another terminal, check daemon status:

```powershell
# Check if running
curl http://localhost:19283/health

# WebSocket monitor (real-time updates)
wscat -c ws://localhost:19284
```

## Troubleshooting

**"Connection refused"?**
- Is daemon running? Check step 1
- Wrong port? Default is 19283

**Images not appearing?**
- Check `save_path` directory exists
- Check quality_gate threshold (might be filtering)
- Check daemon logs for errors

**Slow saves?**
- Upgrade to SSD if on HDD
- Images too large? Reduce resolution
- Daemon workers overloaded? Reduce parallel saves

---

**That's it!** Your workflow now saves asynchronously. Generate images faster. 🚀
