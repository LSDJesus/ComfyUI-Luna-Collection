# ComfyUI-Luna-Collection Integration Guide for Luna Narrates

## Overview

This document describes how Luna Narrates integrates with ComfyUI for image generation. The architecture follows a **clean separation of concerns**:

- **Luna Narrates** = Orchestration layer (calls ComfyUI API with workflows)
- **ComfyUI** = Rendering engine (executes workflows, returns images)
- **Luna Daemon** = Shared VAE/CLIP service (offloads models to secondary GPU)

**Key Principle:** ComfyUI nodes should be generic and reusable. All Narrates-specific orchestration logic lives in the Narrates server, not in custom ComfyUI nodes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LUNA-Narrates Server                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Preprocessor│→ │ Strategist  │→ │   Writer    │→ │   Dreamer   │    │
│  │   Agent     │  │   Agent     │  │   Agent     │  │   Agent     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────┬──────┘    │
│                                                             │           │
│                    Monitors daemon health ←─────────────────┼───┐       │
│                                                             │   │       │
└─────────────────────────────────────────────────────────────┼───┼───────┘
                                                              │   │
                              ┌────────────────────────────────   │
                              │ HTTP POST /prompt                 │ WebSocket
                              │ (workflow + parameters)           │ ws://127.0.0.1:19284
                              ▼                                   │
┌─────────────────────────────────────────────────────────────────┼───────┐
│                    ComfyUI Instance(s)                          │       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │       │
│  │ Port 8188   │  │ Port 8189   │  │ Port 8190 ...           │  │       │
│  │ (Anime)     │  │ (Realistic) │  │ (Other styles)          │  │       │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │       │
│         │                │                     │                │       │
│         └────────────────┴─────────────────────┘                │       │
│                          │                                      │       │
│              Uses Luna Shared VAE/CLIP nodes                    │       │
│                          │                                      │       │
└──────────────────────────┼──────────────────────────────────────┘       │
                           │ Socket (127.0.0.1:19283)                     │
                           ▼                                              │
┌─────────────────────────────────────────────────────────────────────────┤
│                    Luna VAE/CLIP Daemon (cuda:1)                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Dynamic Worker Scaling                                          │   │
│  │  ┌─────────────┐  ┌───────────────────────────────────────────┐ │   │
│  │  │  CLIP Pool  │  │              VAE Pool                     │ │   │
│  │  │  1-2 workers│  │  1-4 workers (scales with demand)         │ │◄──┘
│  │  └─────────────┘  └───────────────────────────────────────────┘ │
│  └─────────────────────────────────────────────────────────────────┘
│  WebSocket: ws://127.0.0.1:19284 (status monitoring)
│  Socket: 127.0.0.1:19283 (VAE/CLIP operations)
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Luna Daemon WebSocket Monitoring

Luna Narrates can monitor the daemon's health and scaling status via WebSocket.

### Connection

```
ws://127.0.0.1:19284
```

### Message Types

#### 1. Status Updates (automatic, every 1 second)

```json
{
  "type": "status",
  "data": {
    "status": "ok",
    "version": "2.0-dynamic",
    "device": "cuda:1",
    "precision": "bf16",
    "uptime_seconds": 3600.5,
    "total_requests": 1234,
    "vram_used_gb": 3.2,
    "vram_total_gb": 12.0,
    "vae_pool": {
      "type": "vae",
      "active_workers": 2,
      "queue_depth": 0,
      "total_requests": 800,
      "worker_ids": [0, 1]
    },
    "clip_pool": {
      "type": "clip",
      "active_workers": 1,
      "queue_depth": 0,
      "total_requests": 434,
      "worker_ids": [0]
    }
  }
}
```

#### 2. Scaling Events (real-time)

```json
{
  "type": "scaling",
  "data": {
    "event": "scale_up",
    "pool": "vae",
    "worker_id": 2,
    "active_workers": 3,
    "vram_available_gb": 4.5
  }
}
```

```json
{
  "type": "scaling",
  "data": {
    "event": "scale_down",
    "pool": "vae",
    "worker_id": 2,
    "active_workers": 2,
    "vram_available_gb": 5.1
  }
}
```

#### 3. On-Demand Status Request

**Send:**
```json
{"type": "get_status"}
```

**Receive:** Full status message immediately.

### Example: Python Client

```python
import asyncio
import websockets
import json

async def monitor_daemon():
    uri = "ws://127.0.0.1:19284"
    
    async with websockets.connect(uri) as ws:
        # Receive messages
        async for message in ws:
            data = json.loads(message)
            
            if data["type"] == "status":
                status = data["data"]
                print(f"Daemon: {status['status']}, VRAM: {status['vram_used_gb']:.1f}GB")
                print(f"  VAE workers: {status['vae_pool']['active_workers']}")
                print(f"  CLIP workers: {status['clip_pool']['active_workers']}")
                
            elif data["type"] == "scaling":
                event = data["data"]
                print(f"Scaling: {event['pool']} {event['event']} → {event['active_workers']} workers")

asyncio.run(monitor_daemon())
```

---

## ComfyUI API Integration

Luna Narrates calls ComfyUI's standard HTTP API - no custom bridge nodes needed.

### Queue a Workflow

```http
POST http://127.0.0.1:8188/prompt
Content-Type: application/json

{
  "prompt": { /* workflow JSON */ },
  "client_id": "luna-narrates-session-123"
}
```

### Monitor Execution (WebSocket)

```
ws://127.0.0.1:8188/ws?clientId=luna-narrates-session-123
```

Messages received:
- `{"type": "status", "data": {"status": {"exec_info": {...}}}}`
- `{"type": "executing", "data": {"node": "3"}}`
- `{"type": "progress", "data": {"value": 5, "max": 20}}`
- `{"type": "executed", "data": {"node": "9", "output": {...}}}`

### Retrieve Output

```http
GET http://127.0.0.1:8188/history/{prompt_id}
GET http://127.0.0.1:8188/view?filename={filename}&subfolder={subfolder}&type=output
```

---

## Workflow Templates

Luna Narrates should store workflow templates as JSON and inject parameters at runtime.

### Parameter Injection Points

Workflows use node IDs. Common injection points:

| Parameter | Node Type | Field |
|-----------|-----------|-------|
| Positive prompt | CLIPTextEncode / Luna Shared CLIP Encode | `text` |
| Negative prompt | CLIPTextEncode | `text` |
| Seed | KSampler | `seed` |
| Image dimensions | EmptyLatentImage | `width`, `height` |
| LoRA selection | LoraLoader / Luna LoRA Stacker | `lora_name`, `strength_model` |

### Example: Injecting Prompt

```python
import copy

def inject_params(workflow_template, params):
    workflow = copy.deepcopy(workflow_template)
    
    # Find CLIP text encode node and update prompt
    for node_id, node in workflow.items():
        if node["class_type"] == "CLIPTextEncode":
            if "positive" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["text"] = params["positive_prompt"]
            elif "negative" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["text"] = params["negative_prompt"]
        
        # Update seed
        if node["class_type"] == "KSampler":
            node["inputs"]["seed"] = params.get("seed", random.randint(0, 2**32))
    
    return workflow
```

---

## Multi-Instance Load Balancing

With multiple ComfyUI instances (each with WaveSpeed acceleration), Luna Narrates can route requests by style:

| Port | Instance | Model | Use Case |
|------|----------|-------|----------|
| 8188 | ComfyUI-Anime | Illustrious | Anime/manga style |
| 8189 | ComfyUI-Real | RealVisXL | Photorealistic |
| 8190 | ComfyUI-Pony | Pony | Semi-realistic |
| 8191 | ComfyUI-Art | Artium | Artistic/stylized |

> **Performance Note:** Using [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed) "Apply First Block Cache" provides ~40% speedup (faster than TensorRT) with no model conversion or resolution constraints.

### Style Router Example

```python
STYLE_PORTS = {
    "anime": 8188,
    "realistic": 8189,
    "semi_realistic": 8190,
    "artistic": 8191,
}

def get_comfyui_endpoint(style: str) -> str:
    port = STYLE_PORTS.get(style, 8188)
    return f"http://127.0.0.1:{port}"
```

---

## Luna Collection Nodes for Narrates Workflows

These ComfyUI-Luna-Collection nodes are useful in Narrates workflows:

### Shared VAE/CLIP (Daemon Nodes)

| Node | Purpose |
|------|---------|
| `Luna Shared CLIP Encode` | Text encoding via daemon (frees main GPU) |
| `Luna Shared VAE Encode` | Image → latent via daemon |
| `Luna Shared VAE Decode` | Latent → image via daemon |
| `Luna Daemon Status` | Check daemon health in workflow |

### LoRA Management

| Node | Purpose |
|------|---------|
| `Luna LoRA Stacker` | Stack multiple LoRAs with strengths |
| `Luna LoRA Randomizer` | Random LoRA selection from YAML |

### Prompt Processing

| Node | Purpose |
|------|---------|
| `Luna YAML Wildcard` | Template-based prompt generation |
| `Luna Prompt Combiner` | Combine prompt fragments |

### Upscaling & Detailing

| Node | Purpose |
|------|---------|
| `Luna Simple Upscaler` | Model-based upscaling |
| `Luna Face Detailer` | MediaPipe-based face enhancement |

---

## Daemon Configuration

### Config File: `luna_daemon/config.py`

```python
# Network
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283      # Socket for VAE/CLIP ops
DAEMON_WS_PORT = 19284   # WebSocket for monitoring

# Device
SHARED_DEVICE = "cuda:1"  # Secondary GPU

# Model paths
CLIP_L_PATH = r"D:\AI\SD Models\clip\Clip-L\clip-L_noMERGE_Universal_CLIP_FLUX_illustrious_Base-fp32.safetensors"
CLIP_G_PATH = r"D:\AI\SD Models\clip\Clip-G\clip-G_noMERGE_Universal_CLIP_FLUX_illustrious_Base-fp32.safetensors"
VAE_PATH = "path/to/vae.safetensors"

# Precision (fp32 models converted on load)
MODEL_PRECISION = "bf16"

# Dynamic scaling
MAX_VAE_WORKERS = 4
MAX_CLIP_WORKERS = 2
MIN_VAE_WORKERS = 1
MIN_CLIP_WORKERS = 1
IDLE_TIMEOUT_SEC = 60.0
```

### Starting the Daemon

```powershell
# Static mode (single worker each)
.\scripts\start_daemon.ps1

# Dynamic scaling mode (recommended for Narrates)
.\scripts\start_daemon.ps1 -Dynamic
```

---

## Health Checks

Luna Narrates should verify services before dispatching work:

### 1. Daemon Health (Socket)

```python
import socket

def check_daemon_health(host="127.0.0.1", port=19283) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((host, port))
        sock.close()
        return True
    except:
        return False
```

### 2. Daemon Health (WebSocket - Recommended)

```python
import asyncio
import websockets
import json

async def check_daemon_health_ws(host="127.0.0.1", port=19284) -> dict:
    try:
        uri = f"ws://{host}:{port}"
        async with websockets.connect(uri, close_timeout=2.0) as ws:
            # Request status
            await ws.send(json.dumps({"type": "get_status"}))
            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
            return json.loads(response)["data"]
    except:
        return {"status": "offline"}
```

### 3. ComfyUI Health

```python
import httpx

async def check_comfyui_health(port=8188) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://127.0.0.1:{port}/system_stats", timeout=2.0)
            return r.status_code == 200
    except:
        return False
```

---

## Removed Components

The following were removed as they belong in Luna Narrates, not ComfyUI:

| Removed | Reason |
|---------|--------|
| `luna_narrates_bridge.py` | HTTP bridge to Narrates - orchestration belongs in Narrates server |

**Principle:** ComfyUI receives workflows via API. It doesn't need to know about Narrates sessions, characters, or story context.

---

## Files Reference

| File | Purpose |
|------|---------|
| `luna_daemon/server.py` | Static daemon (v1) |
| `luna_daemon/server_v2.py` | Dynamic scaling daemon with WebSocket |
| `luna_daemon/client.py` | Client library for nodes |
| `luna_daemon/config.py` | Configuration |
| `nodes/loaders/luna_shared_*.py` | Daemon-connected nodes |
| `scripts/start_daemon.ps1` | Daemon startup script |
| `scripts/start_server_workflow.ps1` | ComfyUI + daemon launcher |
