"""
Luna Daemon Core - Shared types, enums, and base utilities.

This module contains the fundamental building blocks used across the daemon.
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("LunaDaemon")


# =============================================================================
# Service Types
# =============================================================================

class ServiceType(Enum):
    """
    Defines what services the daemon provides.
    
    FULL: Complete daemon with VAE, CLIP workers
    VAE_ONLY: Only VAE encode/decode 
    CLIP_ONLY: Only CLIP text encoding
    """
    FULL = auto()
    VAE_ONLY = auto()
    CLIP_ONLY = auto()


# =============================================================================
# Model Types
# =============================================================================

class ModelFamily(Enum):
    """Supported model families."""
    SDXL = "sdxl"
    SD15 = "sd15"
    FLUX = "flux"
    SD3 = "sd3"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_type: str
    path: Optional[str] = None
    size_mb: float = 0.0
    device: str = "cuda:0"
    loaded: bool = False


@dataclass
class LoRACacheEntry:
    """Entry in the LoRA RAM cache."""
    name: str
    state_dict: Dict[str, Any]
    size_mb: float
    hash: str
    access_count: int = 0


@dataclass 
class CLIPInfo:
    """Information about registered CLIP models."""
    clip_type: str  # 'sdxl', 'flux', etc.
    components: list  # ['clip_l', 'clip_g'] etc.
    size_mb: float = 0.0
    loaded: bool = False


@dataclass
class VAEInfo:
    """Information about registered VAE."""
    vae_type: str  # 'sdxl', 'flux', etc.
    path: Optional[str] = None
    size_mb: float = 0.0
    loaded: bool = False


# =============================================================================
# Exceptions
# =============================================================================

class DaemonError(Exception):
    """Base exception for daemon errors."""
    pass


class ModelNotLoadedError(DaemonError):
    """Raised when trying to use a model that isn't loaded."""
    pass


class CLIPNotLoadedError(DaemonError):
    """Raised when CLIP isn't loaded."""
    pass


class VAENotLoadedError(DaemonError):
    """Raised when VAE isn't loaded."""
    pass


class LoRANotFoundError(DaemonError):
    """Raised when a LoRA isn't in cache or on disk."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def detect_model_family(model_type: str) -> ModelFamily:
    """Map model type string to ModelFamily enum."""
    model_type_lower = model_type.lower()
    if 'flux' in model_type_lower:
        return ModelFamily.FLUX
    elif 'sdxl' in model_type_lower or 'xl' in model_type_lower:
        return ModelFamily.SDXL
    elif 'sd3' in model_type_lower:
        return ModelFamily.SD3
    elif 'sd15' in model_type_lower or 'sd1' in model_type_lower:
        return ModelFamily.SD15
    return ModelFamily.UNKNOWN
