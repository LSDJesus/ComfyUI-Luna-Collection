# Luna VAE/CLIP Daemon

A shared model server that allows multiple ComfyUI instances to share a single VAE and CLIP in VRAM.

## The Problem

When running multiple ComfyUI workflows on the same machine:
- Each instance loads its own VAE (~200MB VRAM)
- Each instance loads its own CLIP models (~2-3GB VRAM)
- With 2 workflows: ~6GB wasted on duplicate models
- Limits you to fewer concurrent workflows

## The Solution

The Luna Daemon:
1. Loads VAE and CLIP **once** on a dedicated GPU (e.g., your 3080)
2. Serves encode/decode requests to all ComfyUI instances via local socket
3. Frees up VRAM on your main GPU for more UNet instances

## VRAM Savings

| Setup | VRAM per Workflow | 3090 Capacity |
|-------|-------------------|---------------|
| Standard | ~9GB (UNet + CLIP + VAE) | 2 workflows |
| With Daemon | ~6GB (UNet only) | **4 workflows** |

## Quick Start

### 1. Configure Paths

Edit `luna_daemon/config.py` to match your setup:

```python
SHARED_DEVICE = "cuda:1"  # GPU for shared models
VAE_PATH = "D:/AI/SD Models/vae/sdxl_vae.safetensors"
CLIP_L_PATH = "D:/AI/SD Models/clip/clip_l.safetensors"
CLIP_G_PATH = "D:/AI/SD Models/clip/clip_g.safetensors"
```

### 2. Start the Daemon

```powershell
# From ComfyUI directory
.\custom_nodes\ComfyUI-Luna-Collection\scripts\start_daemon.ps1
```

Or manually:
```powershell
cd D:\AI\ComfyUI
.\venv\Scripts\Activate.ps1
python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
```

### 3. Use Shared Nodes in Workflows

Replace standard nodes with Luna Shared nodes:

| Standard Node | Luna Shared Node |
|---------------|------------------|
| VAE Encode | Luna Shared VAE Encode |
| VAE Decode | Luna Shared VAE Decode |
| CLIP Text Encode | Luna Shared CLIP Encode |
| CLIP Text Encode (SDXL) | Luna Shared CLIP Encode (SDXL) |

### 4. Start Multiple ComfyUI Instances

```powershell
# Terminal 1
.\scripts\start_server_workflow.ps1 -Port 8188

# Terminal 2
.\scripts\start_server_workflow.ps1 -Port 8189

# Terminal 3
.\scripts\start_server_workflow.ps1 -Port 8190
```

## Available Nodes

### Luna Shared VAE Encode
Drop-in replacement for VAE Encode. Sends image to daemon for encoding.

### Luna Shared VAE Decode
Drop-in replacement for VAE Decode. Sends latents to daemon for decoding.

### Luna Shared VAE Encode (Tiled)
Tiled encoding for high-resolution images.

### Luna Shared VAE Decode (Tiled)
Tiled decoding for high-resolution latents.

### Luna Shared CLIP Encode
Basic text encoding. Returns conditioning.

### Luna Shared CLIP Encode (SDXL)
SDXL-style encoding with size embeddings. Returns both positive and negative conditioning.

### Luna Shared CLIP Encode (Dual)
Encodes both positive and negative prompts in one node.

### Luna Daemon Status
Debug node that shows daemon status, VRAM usage, and request count.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   3080 (cuda:1)                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Luna VAE/CLIP Daemon                   │   │
│  │  • SDXL VAE loaded once                         │   │
│  │  • CLIP-L + CLIP-G loaded once                  │   │
│  │  • Listening on 127.0.0.1:19283                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
└─────────────────────────┼───────────────────────────────┘
                          │ Local Socket
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ComfyUI :8188 │ │ ComfyUI :8189 │ │ ComfyUI :8190 │
│ UNet only     │ │ UNet only     │ │ UNet only     │
│ ~6GB VRAM     │ │ ~6GB VRAM     │ │ ~6GB VRAM     │
└───────────────┘ └───────────────┘ └───────────────┘
              3090 (cuda:0) - 24GB total
```

## Configuration

### `luna_daemon/config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `DAEMON_HOST` | `127.0.0.1` | Bind address |
| `DAEMON_PORT` | `19283` | Listen port |
| `SHARED_DEVICE` | `cuda:1` | GPU for models |
| `VAE_PATH` | - | Path to SDXL VAE |
| `CLIP_L_PATH` | - | Path to CLIP-L |
| `CLIP_G_PATH` | - | Path to CLIP-G |
| `EMBEDDINGS_DIR` | - | Textual inversions |

## Troubleshooting

### "Daemon not running" error

1. Make sure the daemon is started before ComfyUI
2. Check the daemon terminal for errors
3. Verify paths in `config.py` exist

### Slow encode/decode

- Latency should be <10ms for local socket
- If slow, check if daemon GPU is being used by something else
- Monitor with `Luna Daemon Status` node

### Models not loading

- Check that model paths in `config.py` are correct
- Ensure you have enough VRAM on the daemon GPU
- Check daemon terminal for error messages

## Performance Notes

- Socket communication adds ~1-5ms per encode/decode
- At 20 steps with 1 encode + 1 decode, that's ~40-100ms overhead
- Totally negligible compared to the 30-60 seconds of generation time
- Net effect: **2x more concurrent workflows**

## Future Improvements

- [ ] Automatic model detection from ComfyUI model paths
- [ ] Multiple VAE support (SD1.5, SDXL, Flux)
- [ ] Web UI for daemon monitoring
- [ ] Prometheus metrics export
