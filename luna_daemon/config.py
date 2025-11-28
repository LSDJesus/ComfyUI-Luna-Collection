"""
Luna Daemon Configuration
Edit these paths to match your setup
"""

import os

# Network config
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283

# Device assignment - which GPU to load shared models on
SHARED_DEVICE = "cuda:1"  # Your 3080 on the server

# Model paths - adjust to your setup
# These are the models that will be loaded ONCE and shared across all ComfyUI instances

# SDXL VAE
VAE_PATH = "D:/AI/SD Models/vae/sdxl_vae.safetensors"

# SDXL CLIP models (both needed for SDXL)
# If you use a single combined clip, adjust accordingly
CLIP_L_PATH = "D:/AI/SD Models/clip/clip_l.safetensors"
CLIP_G_PATH = "D:/AI/SD Models/clip/clip_g.safetensors"

# Embeddings directory for textual inversions
EMBEDDINGS_DIR = "D:/AI/SD Models/embeddings"

# Timeout for client connections (seconds)
CLIENT_TIMEOUT = 60

# Maximum concurrent requests (threads)
MAX_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
