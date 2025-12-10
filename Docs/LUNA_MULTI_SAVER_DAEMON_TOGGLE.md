# Luna Multi Saver - Daemon Async Saving Integration

## Overview

The Luna Multi Saver node now includes a **daemon save toggle** that switches between traditional blocking saves and asynchronous daemon-based saves without needing a separate node.

## How It Works

### Before (Traditional Blocking Save)
```
[Generate] → [Luna Multi Saver: daemon_save=OFF]
                      ↓
                   (blocked for 3 seconds while saving)
                      ↓
[Queue Next] ← waits 3 seconds for save to complete
```

### After (Daemon Async Save)
```
[Generate] → [Luna Multi Saver: daemon_save=ON]
                      ↓
                 (returns immediately)
                      ↓
[Queue Next] ← proceeds immediately!
                      ↓
              (daemon saves in background)
```

## UI Changes

The node now has **one new toggle** right above `save_mode`:

```
save_path: "outputs/%model_name%"
filename: "%time:YYYY-mm-dd_%index%"
┌─────────────────────────────────────────┐
│ daemon_save: [OFF] ←→ [ON]             │  ← NEW TOGGLE
│ (Blocking)     (Async via daemon)       │
└─────────────────────────────────────────┘
save_mode: parallel ↔ sequential
quality_gate: disabled | variance | edge_density | both
min_quality_threshold: 0.3
[... rest of parameters ...]
```

### Toggle Behavior

| Setting | Behavior | When to Use |
|---------|----------|------------|
| **OFF (Blocking)** | Traditional local saves (original behavior) | Default, or when daemon not running |
| **ON (Daemon Async)** | Submits to daemon, returns immediately | High-speed pipelines, multi-GPU |

## Setup

### No Extra Node Needed!

Just use the existing **Luna Multi Saver** node with the toggle:

1. **Enable daemon toggle** (if daemon is running)
2. Configure save parameters normally
3. Run workflow - saves happen in background

### Requirements

- **Daemon running** (if `daemon_save=ON`): 
  ```powershell
  python -m luna_daemon.server
  ```
- **Falls back automatically** if daemon unavailable

## Behavior

### When daemon_save = ON

```python
# Node returns immediately with status message:
"[LunaMultiSaver] Submitted 3 images to daemon (Job ID: a1b2c3d4)"

# Actual saving happens in daemon worker pool (background)
```

### If Daemon Not Running

```python
# Automatically falls back to local blocking save
"[LunaMultiSaver] Daemon save selected but daemon not available..."
# Then saves locally using original behavior
```

### All Parameters Still Work

- ✅ Quality gates (variance, edge_density, both)
- ✅ Custom metadata embedding
- ✅ Template variables (%model_name%, %time:FORMAT%, %index%)
- ✅ Format options (PNG, WebP, JPEG)
- ✅ Compression control
- ✅ Workflow embedding
- ✅ Multiple images (1-5)

## Performance

### Without Daemon
```
Generate (2s) → Save (3s) → Generate (2s) → Save (3s) = 10s per 2 images
```

### With Daemon
```
Generate (2s) → Queue Save (0.01s) → Generate (2s) → [saves in background]
= 4s per 2 images (saves don't block!)
```

**Speedup**: Up to 2.5x for save-heavy workflows

## Implementation Details

### No Breaking Changes

- All existing workflows continue to work unchanged
- `daemon_save` defaults to `OFF` (original behavior)
- Toggle to `ON` only when daemon is available

### Automatic Fallback

If daemon unavailable:
1. Logs warning message
2. Falls back to local parallel save (same as before)
3. Workflow continues normally

### Image Transmission

- Images converted to numpy arrays (not raw tensors)
- Sent via socket to daemon
- Daemon uses same save logic as local saver
- Quality gates work identically

## Advanced: Using in Batch Workflows

```
[Batch Loop] → [Generate] → [Luna Multi Saver: daemon_save=ON]
                                      ↓
                              (returns immediately)
                                      ↓
                          [Next batch item]  ← can start immediately!
                                      ↓
                              (previous saves in background)
```

This enables true pipeline parallelism:
- CPU/ComfyUI: Generating next image
- Daemon (CPU workers): Saving previous images
- GPU: Not blocked by I/O

## Configuration

### Daemon Settings (in `luna_daemon/config.py`)

```python
# Image save worker pool
SAVE_POOL_MIN_WORKERS = 2      # Always active
SAVE_POOL_MAX_WORKERS = 4      # Scales up if needed
SAVE_POOL_IDLE_TIMEOUT = 300   # Seconds before idle shutdown
```

## Troubleshooting

### "Daemon save selected but daemon not available"
- Is daemon running? → `python -m luna_daemon.server`
- Wrong port? → Check `DAEMON_PORT` in config
- Daemon crashed? → Check logs

### Images saving locally even with toggle ON
- Daemon not responding
- Check daemon logs for errors
- Node auto-fallbacks to local save

### Slow saves despite async
- Disk I/O bottleneck (upgrade SSD)
- Too many images at once
- Quality gates too expensive (disable if not needed)

## Comparison: Local vs Daemon

| Feature | Blocking (OFF) | Daemon (ON) |
|---------|---|---|
| Blocks workflow | Yes | No |
| Return value | N/A | Job ID |
| VRAM overhead | None | None (CPU) |
| GPU impact | None | None |
| Saves to disk | Immediately | Queued |
| Parallel workers | Local threads | Daemon pool |
| Fallback needed | No | Yes (if daemon down) |

## Migration from Separate Node

If you were using the standalone `Luna Daemon Image Saver` node:

**Old workflow:**
```
[Generate] → [Luna Daemon Image Saver] (separate node)
```

**New workflow:**
```
[Generate] → [Luna Multi Saver] (with daemon_save=ON toggle)
```

Exactly the same functionality, one less node to manage!

## Future Enhancements

Potential additions:
- [ ] Progress tracking via job ID
- [ ] Batch save operations (group multiple jobs)
- [ ] Remote storage backends (S3, etc.)
- [ ] Thumbnail generation (automatic previews)
- [ ] Archive creation (ZIP multiple images)

---

**Summary**: The Luna Multi Saver now has a simple toggle for daemon async saving. Use it when daemon is running for maximum throughput, or leave it OFF for traditional blocking saves. Zero breaking changes.
