# luna_daemon/config.py

## Purpose
Configuration file for Luna Daemon with split daemon architecture supporting full, CLIP-only, and VAE-only service modes for optimal multi-GPU utilization.

## Exports
- `ServiceType`: Enum for daemon service modes (FULL, CLIP_ONLY, VAE_ONLY)
- Network settings: `DAEMON_HOST`, `DAEMON_PORT`, `DAEMON_VAE_PORT`, `DAEMON_WS_PORT`
- Device settings: `CLIP_DEVICE`, `VAE_DEVICE`, `LLM_DEVICE`, `SHARED_DEVICE`
- Model paths: `VAE_PATH`, `CLIP_L_PATH`, `CLIP_G_PATH`, `EMBEDDINGS_DIR`
- Client settings: `CLIENT_TIMEOUT`
- Worker pool: `MAX_WORKERS`, `MAX_VAE_WORKERS`, `MAX_CLIP_WORKERS`, etc.
- Precision settings: `CLIP_PRECISION`, `VAE_PRECISION`, `MODEL_PRECISION`
- VRAM management: `VRAM_LIMIT_GB`, `VRAM_SAFETY_MARGIN_GB`, `LORA_CACHE_SIZE_MB`
- Qwen3-VL settings: `QWEN3_VL_MODEL`, `QWEN3_VL_AUTO_LOAD`, etc.
- Logging: `LOG_LEVEL`
- CUDA IPC: `ENABLE_CUDA_IPC`, `IPC_SHM_PREFIX`

## Key Imports
- `os`, `enum`

## ComfyUI Node Configuration
N/A - Configuration file

## Input Schema
N/A

## Key Methods
N/A

## Dependencies
None

## Integration Points
- Used by `luna_daemon/server.py` for daemon configuration
- Used by `luna_daemon/client.py` for connection settings
- Supports split daemon architecture for multi-GPU setups

## Notes
Comprehensive configuration supporting dynamic worker scaling, VRAM management, Qwen3-VL integration, CUDA IPC for zero-copy tensor transfer, and lazy/auto model loading.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\luna_daemon\config.md