"""
Luna Daemon Configuration
Edit these paths to match your setup, or leave empty to use ComfyUI defaults.
"""

import os
import sys

# Try to import folder_paths from ComfyUI
try:
    # Add ComfyUI root to path if running standalone
    comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    folder_paths = None
    HAS_FOLDER_PATHS = False


def get_model_path(model_type: str, filename: str = "") -> str:
    """Get path to a model file using folder_paths or fallback to default."""
    if HAS_FOLDER_PATHS and filename:
        try:
            full_path = folder_paths.get_full_path(model_type, filename)
            if full_path:
                return full_path
        except:
            pass
    
    if HAS_FOLDER_PATHS:
        try:
            paths = folder_paths.get_folder_paths(model_type)
            if paths:
                if filename:
                    return os.path.join(paths[0], filename)
                return paths[0]
        except:
            pass
    
    # Fallback to models directory relative to ComfyUI
    models_dir = os.path.join(comfyui_root, "models", model_type)
    if filename:
        return os.path.join(models_dir, filename)
    return models_dir


# Network config
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283
DAEMON_WS_PORT = 19284  # WebSocket port for status monitoring (LUNA-Narrates compatible)

# Device assignment - which GPU to load shared models on
SHARED_DEVICE = "cuda:1"  # Your secondary GPU for shared models

# Model paths - uses folder_paths if available, otherwise set manually
# Leave filename empty to auto-detect, or specify exact filename

# SDXL VAE - extracted from checkpoint or standalone VAE file
# Your illustriousXL VAE is 164MB
VAE_PATH = get_model_path("vae", "illustriousXL_v20_vae.safetensors")

# SDXL CLIP models (both needed for SDXL/Illustrious)
# Your Universal CLIP files (fp32, will be converted to bf16 on load)
CLIP_L_PATH = r"D:\AI\SD Models\clip\Clip-L\clip-L_noMERGE_Universal_CLIP_FLUX_illustrious_Base-fp32.safetensors"
CLIP_G_PATH = r"D:\AI\SD Models\clip\Clip-G\clip-G_noMERGE_Universal_CLIP_FLUX_illustrious_Base-fp32.safetensors"

# Embeddings directory for textual inversions
EMBEDDINGS_DIR = get_model_path("embeddings")

# Timeout for client connections (seconds)
CLIENT_TIMEOUT = 60

# Maximum concurrent requests (threads)
MAX_WORKERS = 4

# Model precision - convert fp32 models to this format on load
# Options: "fp32" (no conversion), "bf16" (recommended), "fp16" (may cause black images with VAE)
MODEL_PRECISION = "bf16"

# ============================================================================
# Dynamic Scaling Configuration (for server_v2.py)
# ============================================================================

# VRAM limits (GB) - adjust for your GPU
VRAM_LIMIT_GB = 9.5  # 3080 Ti usable VRAM after OS overhead
VRAM_SAFETY_MARGIN_GB = 1.5  # Keep this much free for VAE encode/decode operations

# Worker limits
MAX_VAE_WORKERS = 4   # VAE decode is slower, may need more workers
MAX_CLIP_WORKERS = 2  # CLIP is fast, rarely needs more than 1-2
MIN_VAE_WORKERS = 1   # Always keep at least 1
MIN_CLIP_WORKERS = 1  # Always keep at least 1

# Scaling behavior
QUEUE_THRESHOLD = 2        # Scale up when queue depth exceeds this
SCALE_UP_DELAY_SEC = 0.5   # Wait this long before scaling up (prevents thrashing)
IDLE_TIMEOUT_SEC = 60.0    # Spin down idle workers after this many seconds

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
