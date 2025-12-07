# luna_daemon/client.py

## Purpose
Client library for communicating with Luna VAE/CLIP daemon, enabling shared model operations for multi-instance ComfyUI setups with efficient serialization and CUDA IPC support.

## Exports
- `DaemonClient`: Main client class for daemon communication
- Convenience functions: `vae_encode()`, `vae_decode()`, `clip_encode()`, `clip_encode_sdxl()`, `zimage_encode()`, etc.
- Singleton clients: `get_client()`, `get_vae_client()`, `get_clip_client()`
- Utility functions: `is_daemon_running()`, `get_daemon_info()`, `reset_clients()`

## Key Imports
- `socket`, `pickle`, `struct`, `torch`, `os`, `typing`

## ComfyUI Node Configuration
N/A - Utility library

## Input Schema
N/A

## Key Methods
- `DaemonClient._send_request(request)` - Send request with length-prefix protocol
- `DaemonClient.register_vae(vae, vae_type)` - Register VAE with daemon
- `DaemonClient.register_clip(clip, clip_type)` - Register CLIP components with daemon
- `DaemonClient.vae_encode(pixels, vae_type, ...)` - Encode images to latents via daemon
- `DaemonClient.vae_decode(latents, vae_type, ...)` - Decode latents to images via daemon
- `DaemonClient.clip_encode(positive, negative, clip_type, ...)` - Encode text prompts via daemon
- `DaemonClient.clip_encode_sdxl(...)` - SDXL-specific text encoding with size embeddings
- `DaemonClient.register_lora(lora_name, clip_strength)` - Register LoRA for daemon disk loading
- `DaemonClient.zimage_encode(text)` - Encode text using Qwen3 encoder for Z-IMAGE
- `DaemonClient.describe_image(image, prompt)` - Generate image descriptions using VLM

## Dependencies
- `luna_daemon.config` for daemon connection settings

## Integration Points
- Used by ComfyUI nodes for VRAM-efficient model operations
- Supports CUDA IPC for zero-copy tensor transfer on same GPU
- Component-based architecture allowing CLIP sharing across model families
- Split daemon support (separate CLIP and VAE daemons)

## Notes
Advanced client with length-prefix protocol for efficient serialization, CUDA IPC for same-GPU zero-copy, component-based model registration, and support for Z-IMAGE/Qwen3-VL operations.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\luna_daemon\client.md