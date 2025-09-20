"""
Luna Collection Constants

Centralized constants for the Luna Collection to ensure consistency
across all nodes and utilities.
"""

# Category constants
LUNA_BASE_CATEGORY = "Luna"
CATEGORY_PREPROCESSING = f"{LUNA_BASE_CATEGORY}/Preprocessing"
CATEGORY_PERFORMANCE = f"{LUNA_BASE_CATEGORY}/Utils"
CATEGORY_UPSCALING = f"{LUNA_BASE_CATEGORY}/Upscaling"
CATEGORY_DETAILING = f"{LUNA_BASE_CATEGORY}/Detailing"
CATEGORY_LOADERS = f"{LUNA_BASE_CATEGORY}/Loaders"
CATEGORY_SAMPLING = f"{LUNA_BASE_CATEGORY}/Sampling"
CATEGORY_TEXT = f"{LUNA_BASE_CATEGORY}/Text"
CATEGORY_META = f"{LUNA_BASE_CATEGORY}/Meta"

# Default values
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RESOLUTION = 4096
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 32

# File extensions
EMBEDDING_EXTENSIONS = [".safetensors", ".pt", ".pth"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
MODEL_EXTENSIONS = [".safetensors", ".ckpt", ".pth"]

# Performance monitoring
PERFORMANCE_LOG_PREFIX = "[LunaPerformance]"
LOG_FORMAT = "[{node_name}] {message}"

# Error messages
ERROR_MESSAGES = {
    "trt_engine_unavailable": "TensorRT Engine not available. Please ensure trt_engine.py is properly installed.",
    "mediapipe_unavailable": "MediaPipe engine not available",
    "performance_monitor_unavailable": "Performance monitoring not available",
    "segs_unavailable": "Segmentation utilities not available",
}

# Success messages
SUCCESS_MESSAGES = {
    "processing_complete": "Processing completed successfully",
    "cache_hit": "Cache hit for {key} ({load_time:.3f}s)",
    "file_loaded": "Loaded {filename} successfully",
}