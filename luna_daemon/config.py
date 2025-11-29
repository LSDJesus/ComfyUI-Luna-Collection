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

# Device assignment - which GPU to load shared models on
SHARED_DEVICE = "cuda:1"  # Your secondary GPU for shared models

# Model paths - uses folder_paths if available, otherwise set manually
# Leave filename empty to auto-detect, or specify exact filename

# SDXL VAE - set to your VAE filename or leave empty
VAE_PATH = get_model_path("vae", "sdxl_vae.safetensors")

# SDXL CLIP models (both needed for SDXL)
CLIP_L_PATH = get_model_path("clip", "clip_l.safetensors")
CLIP_G_PATH = get_model_path("clip", "clip_g.safetensors")

# Embeddings directory for textual inversions
EMBEDDINGS_DIR = get_model_path("embeddings")

# Timeout for client connections (seconds)
CLIENT_TIMEOUT = 60

# Maximum concurrent requests (threads)
MAX_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
