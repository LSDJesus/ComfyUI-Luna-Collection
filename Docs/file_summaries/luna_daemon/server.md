# luna_daemon/server.py

## Purpose
Main daemon server implementation providing shared VAE/CLIP model serving with dynamic worker scaling, LoRA support, and split architecture for multi-GPU setups.

## Exports
- `ScalingConfig`: Configuration class for worker scaling parameters
- `LoRARegistry`: LRU cache for LoRA weights with F-150 architecture
- `ModelRegistry`: Dynamic model loading from client registration
- `TransientLoRAContext`: Context manager for temporary LoRA application
- `ModelWorker`: Individual worker holding models and processing requests
- `WorkerPool`: Dynamic scaling worker pool manager
- `WebSocketServer`: Status monitoring WebSocket server
- `DynamicDaemon`: Main daemon server with split service types
- `main()`: CLI entry point with argument parsing

## Key Imports
- `socket`, `threading`, `queue`: Network and concurrency primitives
- `torch`: Tensor operations and CUDA management
- `comfy.sd`: ComfyUI model loading utilities
- `folder_paths`: ComfyUI model path resolution
- `pickle`, `struct`: Binary serialization for socket protocol

## ComfyUI Node Configuration
N/A - This is a standalone daemon server, not a ComfyUI node.

## Input Schema
N/A - Socket-based server accepting pickled request dictionaries.

## Key Methods
- `DynamicDaemon.__init__(device, precision, service_type, port)`: Initialize daemon with configuration
- `DynamicDaemon.start_pools()`: Initialize VAE/CLIP worker pools based on service type
- `DynamicDaemon.handle_request(conn, addr)`: Process incoming socket requests with length-prefix protocol
- `DynamicDaemon.run()`: Main server loop accepting connections
- `WorkerPool.submit(cmd, data)`: Submit request to worker pool and wait for result
- `ModelWorker.process_vae_encode(pixels, tiled, tile_size, overlap)`: VAE encoding with OOM fallback to tiled mode
- `ModelWorker.process_vae_decode(latents, tiled, tile_size, overlap)`: VAE decoding with tiled support
- `ModelWorker.process_clip_encode(positive, negative, lora_stack)`: CLIP encoding with transient LoRA application
- `ModelWorker.process_clip_encode_sdxl(...)`: SDXL-specific CLIP encoding with size conditioning
- `LoRARegistry.put(hash, weights)`: Cache LoRA weights with size limits
- `ModelRegistry.register_vae/clip*()`: Dynamic model registration from clients

## Dependencies
- `torch`: Core tensor operations and CUDA IPC
- `comfy.sd`: Model loading and CLIP/VAE classes
- `folder_paths`: ComfyUI model directory resolution
- `socket`: Network communication
- `threading`: Worker thread management
- `queue`: Request/result queueing
- `pickle`: Request serialization

## Integration Points
- ComfyUI `folder_paths` for model path resolution
- Socket server on configurable port with length-prefix protocol
- WebSocket monitoring server compatible with LUNA-Narrates
- CUDA IPC for zero-copy tensor sharing between processes
- Dynamic model registration allowing clients to load custom models
- LoRA registry (F-150) for transient weight injection in CLIP workers
- Split architecture supporting FULL/CLIP_ONLY/VAE_ONLY service types

## Notes
- Implements v2.1 split daemon architecture for multi-GPU VRAM sharing
- Dynamic scaling based on queue depth and VRAM availability
- Lazy loading: workers start only when first request arrives
- F-150 LoRA architecture: lock-based transient weight application
- WebSocket broadcasting for real-time status monitoring
- CUDA IPC negotiation for same-GPU zero-copy operations
- CLI interface supporting service type, device, precision, and port configuration