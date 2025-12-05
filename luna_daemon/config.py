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

# Model precision: "bf16", "fp16", or "fp32"
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
