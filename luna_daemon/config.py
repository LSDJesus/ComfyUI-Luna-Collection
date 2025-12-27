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
CLIP_DEVICE = "cuda:1"    # Secondary GPU for CLIP/Text Encoders
VAE_DEVICE = "cuda:0"     # Primary GPU for VAE (enables CUDA IPC)
LLM_DEVICE = "cuda:1"     # GPU for Large Language Models (Qwen3-VL, etc.)

# Legacy alias for backward compatibility
SHARED_DEVICE = CLIP_DEVICE

# =============================================================================
# Attention Mechanism Configuration
# =============================================================================

def _detect_comfyui_attention_mode():
    """
    Auto-detect the attention mode ComfyUI is currently using.
    This allows the daemon to automatically match ComfyUI's settings.
    
    Checks CLI args for specific attention flags in priority order.
    PyTorch is the default fallback - only report it if no specific mode is set.
    
    Priority: sage > flash > xformers > split > pytorch (default)
    
    Returns:
        str: The detected attention mode ("sage", "xformers", "flash", "pytorch", "split", or "auto")
    """
    try:
        # First try to get CLI args (most reliable when daemon starts after ComfyUI)
        try:
            import comfy.cli_args  # type: ignore
            args = comfy.cli_args.args
            
            # Check CLI flags in priority order (skip pytorch check - it's always enabled as fallback)
            if getattr(args, 'use_sage_attention', False):
                return "sage"
            elif getattr(args, 'use_flash_attention', False):
                return "flash"
            elif getattr(args, 'use_quad_cross_attention', False):
                return "split"
            elif getattr(args, 'use_split_cross_attention', False):
                return "split"
            # Don't check disable_xformers - just see if xformers is actually enabled
            # Fall through to check xformers availability below
        except (ImportError, AttributeError):
            pass  # CLI args not available, try runtime detection
        
        # Check if xformers is available and enabled (before falling back to pytorch)
        import comfy.model_management as mm  # type: ignore
        if hasattr(mm, 'xformers_enabled') and mm.xformers_enabled():
            return "xformers"
        
        # Fallback to runtime detection functions (less reliable but better than nothing)
        if hasattr(mm, 'sage_attention_enabled') and mm.sage_attention_enabled():  # type: ignore
            return "sage"
        elif hasattr(mm, 'flash_attention_enabled') and mm.flash_attention_enabled():  # type: ignore
            return "flash"
        
        # Default fallback - pytorch is always available
        return "pytorch"
        
    except ImportError:
        # ComfyUI not available yet (daemon starting before ComfyUI)
        return "auto"
    except Exception as e:
        print(f"[Luna.Config] Warning: Failed to detect ComfyUI attention mode: {e}")
        return "auto"


# Attention implementation to use: "auto", "xformers", "flash", "sage", "pytorch", "split"
# - "auto": Auto-detect from ComfyUI (default, recommended)
# - "xformers": Use xformers (best for RTX 3000/4000)
# - "flash": Flash Attention 2 (requires flash-attn package)
# - "sage": Sage Attention (memory efficient, requires sage-attention package)
# - "pytorch": PyTorch native attention (slowest, most compatible)
# - "split": Split attention (for older GPUs)
#
# The daemon will automatically detect and match ComfyUI's attention mode when set to "auto".
# This means if you start ComfyUI with --use-sage-attention, the daemon will use sage too.
#
# Override via environment variable: 
#   LUNA_ATTENTION_MODE=sage .\scripts\start_daemon.ps1
#   or use PowerShell parameter: .\scripts\start_daemon.ps1 -AttentionMode sage
#
# Or set directly here to force a specific mode:
#   ATTENTION_MODE = "sage"
#
ATTENTION_MODE = os.environ.get("LUNA_ATTENTION_MODE", "auto")

# Note: Auto-detection happens at daemon startup in configure_attention_mode(),
# not at config import time, to ensure ComfyUI is fully initialized.

# =============================================================================
# Model Paths
# =============================================================================

# Auto-load Configuration
# The daemon can automatically load models on startup to be ready for requests.
# You can provide:
# 1. A filename (e.g. "sdxl_vae.safetensors") - will be searched in ComfyUI model paths
# 2. An absolute path (e.g. "C:/Models/vae.safetensors")
# 3. None or "" - Daemon starts empty and loads models on first request (Lazy Loading)

VAE_PATH = None
CLIP_L_PATH = None
CLIP_G_PATH = None

# Embeddings directory for textual inversions
# Leave None to automatically use ComfyUI's embeddings directory
EMBEDDINGS_DIR = None

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

# Model type - determines CLIP type for daemon loading
# Options: "SD1.5", "SDXL", "SDXL + Vision", "Flux", "Flux + Vision", "SD3", "Z-IMAGE"
# This maps to the correct ComfyUI CLIPType enum internally
MODEL_TYPE = "SDXL"

# Dynamic scaling configuration
MAX_VAE_WORKERS = 4
MAX_CLIP_WORKERS = 2

# Minimum workers to keep alive
# Set to 0 to allow full unloading (True Lazy Loading)
# Set to 1 to keep at least one worker ready (Hot Cache)
MIN_VAE_WORKERS = 1
MIN_CLIP_WORKERS = 1

# Queue management
QUEUE_THRESHOLD = 3          # Queue depth before considering scale-up
SCALE_UP_DELAY_SEC = 0.5     # Delay before scaling up
IDLE_TIMEOUT_SEC = 60.0      # Idle time before scaling down

# =============================================================================
# VRAM Management
# =============================================================================

# VRAM budget for dynamic scaling
VRAM_LIMIT_GB = 10.0         # Maximum VRAM to use
VRAM_SAFETY_MARGIN_GB = 2.0  # Buffer to prevent OOM

# LoRA registry cache size
LORA_CACHE_SIZE_MB = 2048.0  # 2GB for cached LoRAs

# =============================================================================
# Qwen3-VL / Z-IMAGE Configuration
# =============================================================================

# Qwen3-VL model for unified text encoding + vision-language
# Provides Z-IMAGE compatible embeddings AND VLM capabilities
# Set to HuggingFace model ID or local path
QWEN3_VL_MODEL = "Qwen3-VL-4B-instruct-abliterated"  # Default: official 4B instruct

# Alternative: local GGUF path (for llama.cpp backend)
# QWEN3_VL_MODEL = "F:/LLM/Models/huihui/Qwen3-VL-4B-instruct-abliterated/Huihui-Qwen3-VL-4B-Instruct-abliterated.i1-Q4_K_M.gguf"

# Whether to auto-load Qwen3-VL on daemon startup
# If False, loads on first Z-IMAGE request
QWEN3_VL_AUTO_LOAD = False

# Qwen3-VL inference settings
QWEN3_VL_MAX_TEXT_LENGTH = 256   # Max tokens for CLIP-style encoding
QWEN3_VL_MAX_NEW_TOKENS = 512    # Max tokens for VLM generation
QWEN3_VL_TEMPERATURE = 0.5       # VLM generation temperature

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
