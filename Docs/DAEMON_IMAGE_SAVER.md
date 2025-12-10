# Luna Daemon Image Saver - Async Image Saving

## Overview

The **Luna Daemon Image Saver** adds asynchronous image saving to your workflows by offloading disk I/O to daemon worker threads. Instead of blocking the workflow during image saves, jobs are submitted to the daemon and executed in the background.

```
Traditional:  Generate → Save (blocked) → Generate → Save (blocked) → ...
With Daemon:  Generate → Save (async) → Generate → Save (async) → ...
              (immediate)              (happening in background)
```

## Key Benefits

1. **No Workflow Blocking** - Saves happen in parallel background workers
2. **Higher Throughput** - Generate next image while previous saves to disk
3. **Job Tracking** - Each save gets a job ID for monitoring
4. **Parallel Saves** - Multiple images saved simultaneously via daemon workers
5. **Integrates with Luna Multi Saver** - Uses same templating and quality gates

## Architecture

```
ComfyUI Workflow
       ↓
[Luna Daemon Image Saver Node]
       ↓ (immediate return)
Luna Daemon (localhost:19283)
       ↓
[Async Image Save Worker Pool]
       ↓ (non-blocking)
Disk I/O (parallel)
```

## Setup

### 1. Start the Luna Daemon

```powershell
# In ComfyUI directory
python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
```

The daemon starts with a **CPU-only image save pool** (2-4 worker threads) alongside VAE/CLIP workers.

### 2. Add Node to Workflow

1. Right-click in ComfyUI graph
2. Search for **"Luna Daemon Image Saver"**
3. Connect your images and configure parameters
4. Run workflow - images submit immediately, saves happen in background

## Node Parameters

### Required
- **save_path**: Directory relative to output/ (supports %model_name%, %index%, %time:FORMAT%)
- **filename**: File template (supports %model_name%, %index%, %time:YYYY-mm-dd.HH.MM.SS%)
- **quality_gate**: Quality filtering (disabled, variance, edge_density, both)
- **min_quality_threshold**: Quality score threshold (0.0-1.0)

### Images & Formats
- **image_1 through image_5**: Input image tensors
- **affix_1 through affix_5**: Label for each image (e.g., "RAW", "UPSCALED")
- **format_1 through format_5**: Save format (png, webp, jpeg)
- **subdir_1 through subdir_5**: Save to subdirectory or root

### Compression & Metadata
- **png_compression**: Level 0-9 (higher = slower but smaller)
- **lossy_quality**: JPEG/WebP quality 70-100
- **lossless_webp**: Use lossless WebP compression
- **embed_workflow**: Include workflow metadata in image
- **custom_metadata**: Additional metadata as JSON
- **metadata**: Metadata from Luna Load Parameters node

## Usage Example

### Simple Batch Save
```
[Generate Image]
    ↓
[Luna Daemon Image Saver]
    - save_path: "outputs/%model_name%"
    - filename: "%time:YYYY-mm-dd_%index%"
    - image_1: [image tensor]
    - affix_1: "generation"
    - format_1: "png"
    ↓ (returns immediately)
[Next Node - continues without waiting]
```

### Multi-Image Pipeline
```
[Generate Base Image]
    ↓
[Upscale]
    ↓
[Enhance Details]
    ↓
[Luna Daemon Image Saver]
    - image_1: [base image] → affix: "RAW"
    - image_2: [upscaled] → affix: "UPSCALED"
    - image_3: [enhanced] → affix: "DETAILED"
    ↓ (all three save in parallel in daemon)
[Queue Next Generation]  ← happens while saves are running!
```

## Return Value

The node returns a status string indicating successful submission:

```
"[Luna Daemon Image Saver] Submitted 3 images to daemon. Job ID: a1b2c3d4"
```

If the daemon is not running or an error occurs:

```
"[Luna Daemon Image Saver] ERROR: Connection refused. Is daemon running?"
```

## Comparison with Luna Multi Saver

| Feature | Multi Saver | Daemon Saver |
|---------|-------------|--------------|
| Save location | Local threads | Daemon workers |
| Blocks workflow | Yes (during save) | No (async) |
| Multiple images | Up to 5 | Up to 5 |
| Parallel saves | Yes (local) | Yes (daemon) |
| Quality gates | ✓ | ✓ |
| Metadata embedding | ✓ | ✓ |
| Best for | Small batches | Production pipelines |

## Performance Considerations

### When to Use Daemon Saver
- **High-speed generation** (multiple images/minute)
- **Large images** (4K+) where disk I/O matters
- **Multi-GPU pipelines** where save time blocks next generation
- **Batch generation** where throughput matters

### When to Use Multi Saver
- **Simple workflows** with few images
- **GPU with limited VRAM** (no daemon overhead)
- **Immediate feedback** on save completion
- **Debugging** (easier to see errors immediately)

## Daemon Worker Pool

The daemon allocates **2-4 CPU worker threads** for image saving:

- **Min workers**: 2 (active from startup)
- **Max workers**: 4 (scales up if queue backs up)
- **Idle timeout**: 5 minutes (threads shut down if unused)
- **Device**: CPU only (no GPU overhead)

This allows up to 4 images to save simultaneously while generation continues on GPU.

## Error Handling

If daemon is not running, the node will return an error message. To fix:

1. **Start the daemon:**
   ```powershell
   python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
   ```

2. **Or fall back to Luna Multi Saver** - same parameters, blocks locally

3. **Check daemon logs** for detailed error info

## Configuration

### Daemon Settings (in `luna_daemon/config.py`)
```python
# Image save worker pool configuration
SAVE_POOL_MIN_WORKERS = 2      # Minimum save threads
SAVE_POOL_MAX_WORKERS = 4      # Maximum save threads
SAVE_POOL_IDLE_TIMEOUT = 300   # Seconds before idle shutdown
```

### Workflow Integration

The node auto-detects daemon availability:
- If daemon is running → use async saving
- If daemon is not available → returns error (use Multi Saver instead)

## Monitoring

### WebSocket Status
The daemon provides real-time status via WebSocket at `ws://localhost:19284`:

```json
{
  "event": "save_job_queued",
  "job_id": "a1b2c3d4",
  "num_images": 3,
  "timestamp": "2025-12-09T23:45:00"
}
```

### Logs
Check daemon console output:
```
[Image Save] Job a1b2c3d4: 3 images submitted
[Image Save] Job a1b2c3d4: Starting save operation
[Image Save] Job a1b2c3d4: Saved to outputs/model_name/...
```

## Advanced: Custom Metadata

Pass metadata from **Luna Load Parameters** node:

```
[Luna Load Parameters]
    ↓
[Luna Daemon Image Saver]
    - metadata: [from above]
    ↓
Images saved with workflow + generation params embedded
```

## Troubleshooting

### "Connection refused" error
- Daemon not running
- Wrong host/port in config
- Daemon crashed (check logs)

### Images not saving
- Check output directory permissions
- Verify save_path template is valid
- Check quality_gate threshold (filtering too aggressively?)

### Slow saves despite async
- Disk I/O bottleneck (upgrade to SSD)
- Too many concurrent images (reduce number per batch)
- Quality gate checking is expensive (disable if not needed)

## Future Enhancements

Potential additions:
- [ ] Automatic quality assessment without blocking
- [ ] Batched save operations (group multiple jobs)
- [ ] Network remote saving (S3, FTP, etc.)
- [ ] Archive creation (ZIP multiple images)
- [ ] Thumbnail generation (automatic previews)

---

**Version**: 1.0  
**Status**: Stable  
**Dependencies**: Luna Daemon must be running
