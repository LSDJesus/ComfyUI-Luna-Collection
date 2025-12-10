# Luna Daemon Image Saver - Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Workflow                         │
│  [Generate] → [Luna Daemon Image Saver] → [Continue]       │
│                         ↓                    (immediately)    │
│                  (returns job ID)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ 127.0.0.1:19283
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   Luna Daemon Server                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Main Socket Server                        │   │
│  │  (Accepts incoming requests from ComfyUI)           │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │ cmd="save_images_async"                 │
│                   ↓                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Async Image Save Worker Pool (CPU)             │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ Worker 1   │  │ Worker 2   │  │ Worker 3   │... │   │
│  │  │ Save Job   │  │ Save Job   │  │ (idle)     │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  └────────┬──────────────┬──────────────────────────────┘   │
└───────────┼──────────────┼─────────────────────────────────┘
            ↓              ↓
         Disk I/O (parallel writes)
      image_1.png   image_2.png ...
```

## Request Flow

### 1. Workflow Submission
```python
# Luna Daemon Image Saver Node
client = DaemonClient()
result = client.submit_async(
    "save_images_async",
    {
        "images": [image1, image2, image3],
        "save_path": "outputs/...",
        "filename": "...",
        "png_compression": 4,
        # ... other params
    }
)
# Returns immediately with job_id
print(result)  # {"success": True, "job_id": "abc123", ...}
```

### 2. Daemon Reception
```python
# luna_daemon/server.py → handle_request()
cmd = request.get("cmd")  # "save_images_async"

if cmd == "save_images_async":
    # Submit to worker pool (non-blocking)
    self.save_pool.submit("save_images", save_request)
    
    # Return job ID immediately
    result = {
        "success": True,
        "job_id": job_id,
        "num_images": len(images)
    }
    # Send response back to client immediately
```

### 3. Worker Processing
```python
# Worker thread (in save_pool)
# This happens AFTER response sent, in background
def save_images_worker(save_request):
    for image_data in save_request["images"]:
        # Convert tensor to PIL
        # Apply quality gates if needed
        # Write to disk
        # Embed metadata
    
    # All done - workflow is long finished by now
```

## Protocol Details

### Length-Prefix Protocol
All daemon communication uses efficient binary protocol:

```
[4-byte length][pickled request data]
     ↓
Unpickle
     ↓
Process
     ↓
[4-byte length][pickled response data]
```

Benefits:
- Efficient binary serialization (pickle)
- No accumulator slowdowns (know exact payload size)
- Works with any data type (tensors, dicts, etc.)

### Request Structure
```python
{
    "cmd": "save_images_async",
    "save_path": "outputs/model_name",
    "filename": "gen_%index%",
    "model_name": "model.safetensors",
    "quality_gate": "disabled",
    "min_quality_threshold": 0.3,
    "png_compression": 4,
    "lossy_quality": 90,
    "lossless_webp": False,
    "embed_workflow": True,
    "filename_index": 0,
    "custom_metadata": "",
    "metadata": {},
    "prompt": "...",  # Workflow metadata
    "extra_pnginfo": {},
    "images": [
        {
            "image": numpy_array,
            "affix": "RAW",
            "format": "png",
            "subdir": True
        },
        # ... more images
    ],
    "output_dir": "/path/to/ComfyUI/output",
    "timestamp": "2025-12-09T23:45:00"
}
```

### Response Structure
```python
# Immediate response
{
    "success": True,
    "job_id": "a1b2c3d4",
    "message": "Image save job submitted (ID: a1b2c3d4)",
    "num_images": 3
}

# Or error
{
    "error": "Image save submission failed: ...",
    "job_id": "a1b2c3d4"
}
```

## Worker Pool Architecture

The daemon uses a **dynamic scaling worker pool** for image saves:

```
┌─────────────────────────────────────────┐
│        Async Image Save Pool            │
├─────────────────────────────────────────┤
│                                         │
│  Config:                                │
│  - min_workers: 2                       │
│  - max_workers: 4                       │
│  - idle_timeout: 300 seconds            │
│  - device: CPU (no GPU overhead)        │
│                                         │
│  Job Queue:                             │
│  ┌────────────────────────────────┐    │
│  │ save_job_1  (processing)       │    │
│  │ save_job_2  (processing)       │    │
│  │ save_job_3  (waiting in queue) │    │
│  └────────────────────────────────┘    │
│                                         │
│  Workers:                               │
│  - Worker 1: saving image_1.png         │
│  - Worker 2: saving image_2.webp       │
│  - Worker 3: (idle, will shutdown      │
│               after 5 minutes)         │
│  - Worker 4: (not created yet)         │
│                                         │
└─────────────────────────────────────────┘
```

### Scaling Logic
```python
# In WorkerPool.submit()
if queue_length > threshold and num_workers < max_workers:
    # Demand exceeded, spin up new worker
    self.scale_up()

# In Worker idle check (every 5 minutes)
if time_idle > idle_timeout and num_workers > min_workers:
    # Nothing to do, shut down this worker
    self.shutdown()
```

## Memory Considerations

### Why CPU-Only Workers?

**Before (GPU saving):**
```
GPU: [UNet (2GB)] + [VAE (512MB)] + [Images being saved (500MB)]
     = 3GB+ needed
```

**After (CPU daemon):**
```
GPU: [UNet (2GB)] + [VAE (512MB)]        = 2.5GB
CPU: [Image tensors being saved (500MB)] = 500MB
```

By offloading saves to CPU daemon:
- Frees GPU VRAM for larger models
- Parallel saves don't fight GPU memory
- Can scale to 4 workers without GPU impact

### Image Tensor Handling
```python
# Client converts tensor → numpy for transmission
image_tensor: torch.Tensor  # [B, H, W, C] on GPU
    ↓ (in save node)
image_numpy: np.uint8      # [H, W, C] on CPU memory
    ↓ (sent via socket)
daemon receives numpy array
    ↓ (save worker)
PIL.Image.fromarray()
    ↓
Write to disk
```

## Integration with Luna Multi Saver

Both nodes share the same **image processing pipeline**:

```python
LunaMultiSaver.save_images()
    ↓
├─ _process_template()         # Template processing
├─ check_variance()            # Quality gate: variance
├─ check_edge_density()        # Quality gate: edge
├─ _save_single_image()        # Actual PNG/WEBP/JPEG write
└─ embed_metadata()            # Workflow embedding

LunaDaemonImageSaver.save_async()
    ↓
submit_async() to daemon
    ↓
daemon worker runs:
    └─ Same pipeline as above!
```

The daemon saver just adds the **async submission layer** on top of the proven Multi Saver logic.

## Daemon Startup Sequence

```
1. python -m luna_daemon.server

2. Load config.py
   ├─ DAEMON_HOST: "127.0.0.1"
   ├─ DAEMON_PORT: 19283
   ├─ MODEL_PRECISION: "fp16"
   └─ SAVE_POOL settings

3. Initialize DynamicDaemon
   ├─ Create LoRA registry (for CLIP)
   ├─ Create Model registry (for dynamic loading)
   └─ Initialize worker pools:
       ├─ VAE pool (2-4 workers, GPU)
       ├─ CLIP pool (2-4 workers, GPU)
       └─ Save pool (2-4 workers, CPU) ← NEW

4. Start pools
   ├─ VAE pool starts (lazy init workers)
   ├─ CLIP pool starts (lazy init workers)
   └─ Save pool starts with 2 workers

5. Start socket server
   └─ Listen on 127.0.0.1:19283

6. Ready to accept connections
   ├─ VAE encode/decode commands
   ├─ CLIP encode commands
   ├─ Model registration
   └─ save_images_async commands ← NEW
```

## Error Handling

### Daemon Unavailable
```python
# In Luna Daemon Image Saver node
try:
    client = DaemonClient()
    result = client.submit_async("save_images_async", request)
except Exception as e:
    return (f"ERROR: {e}",)
    # User sees error, can switch to Multi Saver
```

### Save Failure (in daemon)
```python
# Worker catches exception
try:
    save_image(image, path, format)
except Exception as e:
    logger.error(f"[Image Save] Job {job_id}: {e}")
    # Job fails silently (no way to report back)
    # Solution: Check daemon logs for errors
```

### Network Issues
```python
# Client timeout handling
sock.settimeout(30)  # 30 second timeout
try:
    result = sock.recv(response_len)
except socket.timeout:
    raise DaemonConnectionError("Daemon not responding")
```

## Performance Metrics

### Typical Throughput
```
Single Image (1024×1024, PNG):
- Async submit: 1ms
- Background save: 50ms
- Network latency: 2ms
- Total workflow overhead: ~3ms

Batch (5 images, parallel save):
- Async submit: 1ms
- Background saves: 50ms × 5 = 250ms total (parallel)
- Network: 10ms
- Total overhead: ~11ms (vs 250ms if blocking!)
- Speedup: ~23x for this specific batch
```

### Worker Scaling
```
Light load (1-2 images):
  2 workers (baseline) = ~50ms per image

Medium load (5+ images/minute):
  Scale to 3-4 workers = ~50-100ms per image (parallel)

Heavy load (10+ images/minute):
  4 workers + queue backlog = monitor needed
```

## Future Optimizations

Potential improvements:
1. **Binary image format** (skip PIL conversion, direct numpy → disk)
2. **Compression presets** (cache optimal settings per format)
3. **Batch metadata** (single metadata write for multiple images)
4. **Remote storage** (S3, FTP backends)
5. **Image streaming** (start save before full tensor received)

---

**Version**: 1.0  
**Protocol**: Length-prefix binary (socket.AF_INET, SOCK_STREAM)  
**Default Port**: 19283  
**Worker Count**: 2-4 CPU threads
