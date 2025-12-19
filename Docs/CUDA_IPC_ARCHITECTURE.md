# CUDA IPC Weight Sharing Architecture

## Problem
Current daemon sends data tensors over sockets (pickle overhead = 15s slower per workflow).

## Solution: Share Model Weights, Not Data

### Setup Phase (One-time per instance)

```
ComfyUI Instance A          Luna Daemon              ComfyUI Instance B
──────────────────          ───────────              ──────────────────

1. ModelRouter starts
   ↓
2. Request VAE weights ─→  Load VAE to GPU
                           Create IPC handle
                           ↓
3. ←─ Receive handle      Store in registry
   ↓
4. Open IPC handle
   Create local VAE
   using shared weights
   ↓                                                 5. Request same VAE ─→
5. Ready to use                                         ↓
                                                    6. ←─ Same handle
                                                        ↓
                                                    7. Open same handle
                                                       Share weights!
```

### Runtime (Zero Socket Overhead)

```
ComfyUI Process A                    Shared GPU Memory                ComfyUI Process B
─────────────────                    ─────────────────                ─────────────────

vae.encode(pixels)                   ┌──────────────┐                vae.encode(pixels)
  ↓                                  │ VAE Weights  │                  ↓
Access shared weights ◄──────────────┤ (loaded once)├────────────────► Access shared weights
  ↓                                  └──────────────┘                  ↓
Run encode LOCALLY                                                    Run encode LOCALLY
  ↓                                                                     ↓
Return result (no socket!)                                           Return result (no socket!)
```

## Key Components

### Daemon Side
- **Weight Registry**: Map of `model_path → IPC handle`
- **Load on demand**: When first instance requests a model
- **Handle distribution**: Return IPC handle metadata to clients

### Client Side
- **Handle cache**: Store IPC handles for loaded models
- **Local execution**: DaemonVAE/CLIP wraps shared weights
- **Zero latency**: No socket calls after initial setup

## Benefits

1. **No socket overhead**: Run locally using shared memory
2. **VRAM deduplication**: N instances share 1 copy of weights
3. **True multi-instance**: Each process independent, shared resources
4. **Simple**: Just share pointers, not data

## Implementation Status

### VAE (IPC Weight Sharing)
- ✅ Daemon: Weight registry with IPC handle generation
- ✅ Client: Handle retrieval and local model wrapping  
- ✅ DaemonVAE: Local encode/decode with shared weights
- ⏳ Testing: Multi-instance validation needed
- ⏳ Memory lifecycle: Proper cleanup on instance shutdown

### CLIP (Socket API - By Design)
- ✅ Keep socket-based architecture
- ✅ Daemon handles LoRA loading/caching/transient application
- ✅ Treat as API: send text → receive embeddings
- ✅ Acceptable overhead (1-2s once per workflow vs VAE's repeated overhead)

**Why different approaches?**
- **VAE**: Repeated operations (20+ per workflow) → IPC eliminates cumulative 15s overhead
- **CLIP**: Single operation per workflow + dynamic LoRA application → Socket mode is simpler and fast enough
