"""
Luna Daemon Configuration - v1.3 Split Daemon Architecture

Supports running as:
- "full" (default): Both CLIP and VAE services
- "clip_only": Just CLIP encoding (for secondary GPU)
- "vae_only": Just VAE encode/decode (for primary GPU with CUDA IPC)
"""

import os
from enum import Enum


class ServiceType(Enum):
    """Daemon service mode"""
    FULL = "full"         # Both CLIP and VAE
    CLIP_ONLY = "clip"    # Just CLIP (text encoder)
    VAE_ONLY = "vae"      # Just VAE (encode/decode)


# =============================================================================
# Network Configuration
# =============================================================================

DAEMON_HOST = "127.0.0.1"

# Port assignments for split daemon architecture
DAEMON_PORT = 19283           # Primary daemon port (or CLIP in split mode)
DAEMON_VAE_PORT = 19284       # VAE-only daemon port (when split)
DAEMON_WS_PORT = 19285        # WebSocket monitoring port

# Service type - controls which models are loaded
# Override with --service-type cli arg or SERVICE_TYPE env var
SERVICE_TYPE = ServiceType.FULL

# =============================================================================
# Device Configuration
# =============================================================================

# Device assignment - which GPU to load shared models on
SHARED_DEVICE = "cuda:1"  # Secondary GPU for CLIP in split mode

# For VAE daemon, typically same GPU as ComfyUI workers
VAE_DEVICE = "cuda:0"     # Primary GPU for VAE (enables CUDA IPC)

# =============================================================================
# Model Paths
# =============================================================================

# Default model paths - override per-installation
VAE_PATH = "D:/AI/SD Models/vae/sdxl_vae.safetensors"
CLIP_L_PATH = "D:/AI/SD Models/clip/clip_l.safetensors"
CLIP_G_PATH = "D:/AI/SD Models/clip/clip_g.safetensors"

# Embeddings directory for textual inversions
EMBEDDINGS_DIR = "D:/AI/SD Models/embeddings"

# =============================================================================
# Client Settings
# =============================================================================

# Timeout for client connections (seconds)
CLIENT_TIMEOUT = 120

# =============================================================================
# Worker Pool Configuration
# =============================================================================

# Maximum concurrent requests (legacy, for thread pool)
MAX_WORKERS = 8

# Model precision settings
# Options: "bf16", "fp16", "fp32"
#
# bf16: Recommended - same range as fp32, native on Ampere+, no overflow risk
# fp16: Legacy - higher precision but limited range, NaN risk in VAE
# fp32: Maximum precision, 2x VRAM usage, rarely needed
#
# Note: VAE is more sensitive to precision than CLIP due to exp/sigmoid ops.
# bf16 eliminates fp16 NaN issues while using same VRAM. No visible quality loss.

CLIP_PRECISION = "bf16"   # CLIP text encoder precision
VAE_PRECISION = "bf16"    # VAE encoder/decoder precision

# Legacy setting (used if CLIP/VAE precision not set)
MODEL_PRECISION = "bf16"

# Dynamic scaling configuration
MAX_VAE_WORKERS = 4
MAX_CLIP_WORKERS = 2
MIN_VAE_WORKERS = 1
MIN_CLIP_WORKERS = 1

# Queue management
QUEUE_THRESHOLD = 3          # Queue depth before considering scale-up
SCALE_UP_DELAY_SEC = 0.5     # Delay before scaling up
IDLE_TIMEOUT_SEC = 30.0      # Idle time before scaling down

# =============================================================================
# VRAM Management
# =============================================================================

# VRAM budget for dynamic scaling
VRAM_LIMIT_GB = 20.0         # Maximum VRAM to use
VRAM_SAFETY_MARGIN_GB = 2.0  # Buffer to prevent OOM

# LoRA registry cache size
LORA_CACHE_SIZE_MB = 2048.0  # 2GB for cached LoRAs

# =============================================================================
# Qwen3-VL / Z-IMAGE Configuration
# =============================================================================

# Qwen3-VL model for unified text encoding + vision-language
# Provides Z-IMAGE compatible embeddings AND VLM capabilities
# Set to HuggingFace model ID or local path
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # Default: official 4B instruct

# Alternative: local GGUF path (for llama.cpp backend)
# QWEN3_VL_MODEL = "F:/LLM/Models/huihui/Qwen3-VL-4B-instruct-abliterated/Huihui-Qwen3-VL-4B-Instruct-abliterated.i1-Q4_K_M.gguf"

# Whether to auto-load Qwen3-VL on daemon startup
# If False, loads on first Z-IMAGE request
QWEN3_VL_AUTO_LOAD = False

# Qwen3-VL inference settings
QWEN3_VL_MAX_TEXT_LENGTH = 256   # Max tokens for CLIP-style encoding
QWEN3_VL_MAX_NEW_TOKENS = 512    # Max tokens for VLM generation
QWEN3_VL_TEMPERATURE = 0.7       # VLM generation temperature

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# =============================================================================
# CUDA IPC Configuration (v1.3)
# =============================================================================

# Enable CUDA IPC when client and server share same GPU
# Eliminates pickle serialization overhead for VAE tensors
ENABLE_CUDA_IPC = True

# IPC shared memory name prefix
IPC_SHM_PREFIX = "luna_daemon_"
