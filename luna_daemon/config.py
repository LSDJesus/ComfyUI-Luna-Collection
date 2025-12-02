"""
Luna Daemon Configuration
"""

import os

# Network config
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283

# Device assignment - which GPU to load shared models on
SHARED_DEVICE = "cuda:1"  # Your 3080 on the server

# Embeddings directory for textual inversions (used when loading CLIP)
EMBEDDINGS_DIR = "D:/AI/SD Models/embeddings"

# Timeout for client connections (seconds)
CLIENT_TIMEOUT = 60

# Maximum concurrent requests (threads)
MAX_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
