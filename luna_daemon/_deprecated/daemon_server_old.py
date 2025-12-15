"""
Luna VAE/CLIP Daemon Server - Dynamic Worker Scaling
Intelligently scales VAE and CLIP workers based on demand and available VRAM.

Features:
- Starts with 1 CLIP worker and 1 VAE worker
- Automatically spins up additional workers when queue backs up
- Spins down idle workers after configurable timeout
- VRAM-aware: won't scale beyond available memory
- Separate scaling for CLIP (fast) vs VAE (slower, needs more workers)

Usage:
    python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon.server
"""

import os
import sys
import socket
import pickle
import threading
import time
import queue
import logging
import json
import hashlib
import base64
import struct
from typing import Any, Dict, Tuple, Optional, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

# NOTE: sys.path is configured centrally in __init__.py
# All necessary paths (nodes, daemon, utils, luna_root) are already added at import time
# This file can be run standalone, but when imported as part of Luna Collection,
# the centralized path setup takes precedence.

# Import folder_paths for LoRA disk loading
try:
    import folder_paths  # type: ignore
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False
    folder_paths = None  # type: ignore
    
# Import safetensors for LoRA loading
try:
    from safetensors.torch import load_file as load_safetensors  # type: ignore
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    load_safetensors = None  # type: ignore

# Wavespeed FB cache integration
try:
    from .wavespeed_utils import apply_fb_cache_transient, FBCacheConfig  # type: ignore
    HAS_FB_CACHE = True
except ImportError:
    # Fallback if wavespeed utils not available
    import contextlib
    HAS_FB_CACHE = False
    FBCacheConfig = None  # type: ignore
    @contextlib.contextmanager
    def apply_fb_cache_transient(model, fb_config):  # type: ignore
        yield None
    load_safetensors = None

# Try relative import first (when used as module), fall back to direct import
try:
    from .config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_VAE_PORT, DAEMON_WS_PORT, 
        CLIP_DEVICE, VAE_DEVICE, LLM_DEVICE, SHARED_DEVICE,
        VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
        MAX_WORKERS, LOG_LEVEL, MODEL_PRECISION,
        VRAM_LIMIT_GB, VRAM_SAFETY_MARGIN_GB,
        MAX_VAE_WORKERS, MAX_CLIP_WORKERS, MIN_VAE_WORKERS, MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD, SCALE_UP_DELAY_SEC, IDLE_TIMEOUT_SEC,
        ServiceType, SERVICE_TYPE, ENABLE_CUDA_IPC
    )
    # Import new precision settings with fallback to MODEL_PRECISION
    try:
        from .config import CLIP_PRECISION, VAE_PRECISION
    except ImportError:
        CLIP_PRECISION = MODEL_PRECISION
        VAE_PRECISION = MODEL_PRECISION

    # Import MODEL_TYPE setting (ties to CLIP type via CLIP_TYPE_MAP)
    try:
        from .config import MODEL_TYPE
    except ImportError:
        MODEL_TYPE = "SDXL"
    
    # Import attention mode configuration
    try:
        from .config import ATTENTION_MODE
    except ImportError:
        ATTENTION_MODE = "auto"
except ImportError:
    from config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_VAE_PORT, DAEMON_WS_PORT, 
        CLIP_DEVICE, VAE_DEVICE, LLM_DEVICE, SHARED_DEVICE,
        VAE_PATH, CLIP_L_PATH, CLIP_G_PATH, EMBEDDINGS_DIR,
        MAX_WORKERS, LOG_LEVEL, MODEL_PRECISION,
        VRAM_LIMIT_GB, VRAM_SAFETY_MARGIN_GB,
        MAX_VAE_WORKERS, MAX_CLIP_WORKERS, MIN_VAE_WORKERS, MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD, SCALE_UP_DELAY_SEC, IDLE_TIMEOUT_SEC,
        ServiceType, SERVICE_TYPE, ENABLE_CUDA_IPC
    )
    # Import new precision settings with fallback to MODEL_PRECISION
    try:
        from config import CLIP_PRECISION, VAE_PRECISION
    except ImportError:
        CLIP_PRECISION = MODEL_PRECISION
        VAE_PRECISION = MODEL_PRECISION
    
    # Import MODEL_TYPE setting (ties to CLIP type via CLIP_TYPE_MAP)
    try:
        from config import MODEL_TYPE
    except ImportError:
        MODEL_TYPE = "SDXL"
    
    # Import attention mode configuration
    try:
        from config import ATTENTION_MODE
    except ImportError:
        ATTENTION_MODE = "auto"

# Import CLIP_TYPE_MAP from model router for model_type → clip_type mapping
try:
    from luna_model_router import CLIP_TYPE_MAP  # type: ignore
except ImportError:
    # Fallback mapping if import fails
    CLIP_TYPE_MAP = {
        "SD1.5": "stable_diffusion",
        "SDXL": "stable_diffusion",
        "SDXL + Vision": "stable_diffusion",
        "Flux": "flux",
        "Flux + Vision": "flux",
        "SD3": "sd3",
        "Z-IMAGE": "lumina2",
    }

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [Daemon] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# VRAM Monitoring Utilities
# ============================================================================

def log_all_gpu_vram(prefix: str = ""):
    """
    Log VRAM usage for all available GPUs.
    Useful for tracking memory allocation during operations.
    """
    if not torch.cuda.is_available():
        return
    
    gpu_count = torch.cuda.device_count()
    vram_info = []
    
    for gpu_id in range(gpu_count):
        try:
            # Get system-level memory info (actual usage across all processes)
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            used_gb = (total_mem - free_mem) / 1024**3
            total_gb = total_mem / 1024**3
            percent = round((used_gb / total_gb) * 100, 1)
            
            vram_info.append(f"GPU{gpu_id}: {used_gb:.2f}/{total_gb:.1f}GB ({percent}%)")
        except Exception as e:
            vram_info.append(f"GPU{gpu_id}: ERROR")
    
    logger.info(f"[VRAM] {prefix}{' | '.join(vram_info)}")


# ============================================================================
# Multi-GPU Device Patching (ComfyUI-MultiGPU pattern)
# ============================================================================
# CRITICAL: Patch ComfyUI's device management to enable multi-GPU routing
# This must happen BEFORE any model loading occurs

import comfy.model_management as mm  # type: ignore
import threading

# Thread-local context for per-operation device override
# Allows get_clip() and get_vae() to specify which GPU to use
_daemon_device_context = threading.local()
_original_get_torch_device = mm.get_torch_device
_original_text_encoder_device = mm.text_encoder_device

def _daemon_get_torch_device_patched():
    """
    Context-aware device patch for general model loading.
    """
    if hasattr(_daemon_device_context, 'device') and _daemon_device_context.device is not None:
        device = torch.device(_daemon_device_context.device)
        logger.debug(f"[PATCH] get_torch_device() -> {device} (context override)")
        return device
    result = _original_get_torch_device()
    logger.debug(f"[PATCH] get_torch_device() -> {result} (default)")
    return result

def _daemon_text_encoder_device_patched():
    """
    Context-aware device patch specifically for CLIP/text encoders.
    This is the CRITICAL patch for CLIP placement!
    """
    if hasattr(_daemon_device_context, 'device') and _daemon_device_context.device is not None:
        device = torch.device(_daemon_device_context.device)
        logger.info(f"[PATCH] text_encoder_device() -> {device} (CLIP context override)")
        return device
    result = _original_text_encoder_device()
    logger.debug(f"[PATCH] text_encoder_device() -> {result} (default)")
    return result

# Apply the patches permanently
mm.get_torch_device = _daemon_get_torch_device_patched
mm.text_encoder_device = _daemon_text_encoder_device_patched
logger.info(f"[PATCH] Installed context-aware multi-GPU patches (get_torch_device + text_encoder_device)")


# ============================================================================
# Configuration for Dynamic Scaling
# ============================================================================

@dataclass
class ScalingConfig:
    """Configuration for dynamic worker scaling"""
    # VRAM limits (GB)
    vram_limit_gb: float = VRAM_LIMIT_GB
    vram_safety_margin_gb: float = VRAM_SAFETY_MARGIN_GB
    
    # Model sizes in VRAM (bf16) - will be adjusted at runtime
    vae_size_gb: float = 0.082  # ~82 MB for VAE
    clip_size_gb: float = 1.6   # ~1.6 GB for CLIP-L + CLIP-G combined
    
    # Worker limits
    max_vae_workers: int = MAX_VAE_WORKERS
    max_clip_workers: int = MAX_CLIP_WORKERS
    min_vae_workers: int = MIN_VAE_WORKERS
    min_clip_workers: int = MIN_CLIP_WORKERS
    
    # Scaling triggers
    queue_threshold: int = QUEUE_THRESHOLD
    scale_up_delay_sec: float = SCALE_UP_DELAY_SEC
    
    # Idle timeout (seconds)
    idle_timeout_sec: float = IDLE_TIMEOUT_SEC
    
    # How often to check for scaling decisions
    scaling_check_interval_sec: float = 0.25


class WorkerType(Enum):
    VAE = "vae"
    CLIP = "clip"
    IMAGE_SAVE = "image_save"


# ============================================================================
# LoRA Registry - F-150 Architecture (Reliable, Locking-Based)
# ============================================================================

@dataclass
class LoRAEntry:
    """A cached LoRA in the registry"""
    lora_hash: str
    weights: Dict[str, torch.Tensor]  # layer_key -> delta tensor
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0


class LoRARegistry:
    """
    LRU cache for LoRA weights on the daemon.
    
    Stores LoRA CLIP weights (deltas) keyed by content hash.
    Implements LRU eviction when VRAM budget exceeded.
    
    F-150 Philosophy: Simple, reliable, thread-safe.
    """
    
    def __init__(self, max_size_mb: float = 2048.0, device: str = "cuda"):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.device = device
        self.entries: Dict[str, LoRAEntry] = {}
        self.lock = threading.Lock()
        self.total_size_bytes = 0
        
        logger.info(f"[LoRARegistry] Initialized with {max_size_mb:.0f}MB cache limit")
    
    def _compute_size(self, weights: Dict[str, torch.Tensor]) -> int:
        """Compute total size of weight tensors in bytes"""
        return sum(t.numel() * t.element_size() for t in weights.values())
    
    def has(self, lora_hash: str) -> bool:
        """Check if LoRA is in registry"""
        with self.lock:
            return lora_hash in self.entries
    
    def get(self, lora_hash: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get LoRA weights by hash, updating LRU timestamp"""
        with self.lock:
            if lora_hash not in self.entries:
                return None
            
            entry = self.entries[lora_hash]
            entry.last_used = time.time()
            entry.use_count += 1
            return entry.weights
    
    def put(self, lora_hash: str, weights: Dict[str, torch.Tensor]) -> bool:
        """
        Store LoRA weights in registry.
        
        Moves tensors to device and evicts old entries if needed.
        Returns True if stored successfully.
        """
        with self.lock:
            # Already exists
            if lora_hash in self.entries:
                self.entries[lora_hash].last_used = time.time()
                return True
            
            # Move weights to device
            device_weights = {}
            for key, tensor in weights.items():
                device_weights[key] = tensor.to(self.device)
            
            size = self._compute_size(device_weights)
            
            # Evict if needed
            while self.total_size_bytes + size > self.max_size_bytes and self.entries:
                self._evict_lru()
            
            # Check if single LoRA is too large
            if size > self.max_size_bytes:
                logger.warning(f"[LoRARegistry] LoRA {lora_hash[:12]}... too large ({size/1024/1024:.1f}MB)")
                return False
            
            # Store entry
            entry = LoRAEntry(
                lora_hash=lora_hash,
                weights=device_weights,
                size_bytes=size
            )
            self.entries[lora_hash] = entry
            self.total_size_bytes += size
            
            logger.info(f"[LoRARegistry] Cached {lora_hash[:12]}... ({size/1024/1024:.1f}MB, "
                       f"total: {self.total_size_bytes/1024/1024:.1f}MB)")
            return True
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.entries:
            return
        
        # Find oldest entry
        oldest_hash = min(self.entries.keys(), 
                         key=lambda h: self.entries[h].last_used)
        entry = self.entries.pop(oldest_hash)
        
        # Free tensors
        for tensor in entry.weights.values():
            del tensor
        
        self.total_size_bytes -= entry.size_bytes
        logger.info(f"[LoRARegistry] Evicted {oldest_hash[:12]}... (freed {entry.size_bytes/1024/1024:.1f}MB)")
    
    def clear(self):
        """Clear all cached LoRAs"""
        with self.lock:
            for entry in self.entries.values():
                for tensor in entry.weights.values():
                    del tensor
            self.entries.clear()
            self.total_size_bytes = 0
            torch.cuda.empty_cache()
            logger.info("[LoRARegistry] Cleared all cached LoRAs")
    
    def register_from_file(self, lora_name: str) -> Optional[str]:
        """
        Load LoRA from disk and cache CLIP weights.
        
        This is the preferred method for Config Gateway - avoids socket serialization
        by having the daemon load directly from disk.
        
        Args:
            lora_name: LoRA filename (e.g., "my_lora.safetensors" or path/my_lora.safetensors)
        
        Returns:
            Content hash if successful, None if failed
        """
        if not HAS_FOLDER_PATHS or not HAS_SAFETENSORS or folder_paths is None:
            logger.error("[LoRARegistry] folder_paths or safetensors not available")
            return None
        
        # Find the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            # folder_paths.get_full_path() doesn't always handle subdirs well
            # Try building path manually from loras base directory
            loras_dirs = folder_paths.get_folder_paths("loras") if folder_paths else []
            for base_dir in loras_dirs:
                test_path = os.path.join(base_dir, lora_name)
                if os.path.exists(test_path):
                    lora_path = test_path
                    break
            else:
                # Still not found - try with/without extensions
                base_name = lora_name
                for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin'):
                    if base_name.lower().endswith(ext):
                        base_name = base_name[:-len(ext)]
                        break
                
                # Try common extensions
                for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin'):
                    test_name = base_name + ext
                    lora_path = folder_paths.get_full_path("loras", test_name)  # type: ignore
                    if lora_path and os.path.exists(lora_path):
                        break
                else:
                    logger.error(f"[LoRARegistry] LoRA not found: {lora_name}")
                    logger.error(f"[LoRARegistry] Searched in: {loras_dirs}")
                    return None
        
        logger.info(f"[LoRARegistry] Loading LoRA from disk: {os.path.basename(lora_path)}")
        
        try:
            # Load the full LoRA file using ComfyUI's loader (supports BF16)
            import comfy.utils  # type: ignore
            all_weights = comfy.utils.load_torch_file(lora_path)  # type: ignore
            
            # Extract CLIP-specific weights
            clip_weights = {}
            for key, tensor in all_weights.items():
                is_clip = any(pattern in key.lower() for pattern in [
                    'clip_l', 'clip_g', 'te1', 'te2', 'text_encoder',
                    'text_model', 'lora_te', 'clip.'
                ])
                if is_clip:
                    clip_weights[key] = tensor
            
            if not clip_weights:
                logger.info(f"[LoRARegistry] No CLIP weights in LoRA: {lora_name} (UNet-only LoRA)")
                # Return a placeholder hash for UNet-only LoRAs
                return hashlib.sha256(lora_path.encode()).hexdigest()
            
            # Compute content hash
            hasher = hashlib.sha256()
            for key in sorted(clip_weights.keys()):
                tensor = clip_weights[key]
                hasher.update(key.encode('utf-8'))
                hasher.update(str(tensor.shape).encode('utf-8'))
                if tensor.numel() > 10000:
                    step = tensor.numel() // 10000
                    sample = tensor.flatten()[::step]
                    hasher.update(sample.numpy().tobytes())
                else:
                    hasher.update(tensor.numpy().tobytes())
            
            lora_hash = hasher.hexdigest()
            
            # Check if already cached
            if self.has(lora_hash):
                logger.info(f"[LoRARegistry] LoRA already cached: {lora_hash[:12]}...")
                return lora_hash
            
            # Store in registry
            if self.put(lora_hash, clip_weights):
                logger.info(f"[LoRARegistry] Cached LoRA from disk: {lora_hash[:12]}... "
                           f"({len(clip_weights)} CLIP tensors)")
                return lora_hash
            else:
                logger.warning(f"[LoRARegistry] Failed to cache LoRA: {lora_name}")
                return None
                
        except Exception as e:
            logger.error(f"[LoRARegistry] Error loading LoRA {lora_name}: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get registry statistics"""
        with self.lock:
            return {
                "cached_loras": len(self.entries),
                "total_size_mb": self.total_size_bytes / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "entries": [
                    {
                        "hash": h[:12] + "...",
                        "size_mb": e.size_bytes / 1024 / 1024,
                        "use_count": e.use_count,
                        "age_seconds": time.time() - e.created_at
                    }
                    for h, e in sorted(self.entries.items(), 
                                      key=lambda x: x[1].last_used, reverse=True)
                ]
            }


# ============================================================================
# Model Registry - Dynamic Model Loading from Client Registration
# ============================================================================

@dataclass
class RegisteredModel:
    """Stores a registered model's state dict and metadata"""
    model_type: str  # e.g., "SDXL", "Flux", "Z-IMAGE"
    clip_type: str   # e.g., "stable_diffusion", "flux", "lumina2"
    state_dict: Optional[Dict[str, Any]] = None  # VAE state dict
    components: Optional[Dict[str, Dict[str, Any]]] = None  # CLIP components
    path: Optional[str] = None  # Single file path (for VAE)
    paths: Optional[List[str]] = None  # Multiple file paths (for CLIP)
    registered_at: float = field(default_factory=time.time)
    loaded: bool = False


class ModelRegistry:
    """
    Registry for dynamically registered models from client loader nodes.
    
    When a loader node (Model Router, etc.) creates a DaemonCLIP or DaemonVAE,
    it sends registration data here. Workers then load from this registry
    instead of static config paths.
    
    Thread-safe for concurrent registration and access.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lock = threading.Lock()
        
        # Current registered models (only one VAE and one CLIP at a time)
        self.vae: Optional[RegisteredModel] = None
        self.clip: Optional[RegisteredModel] = None
        
        # Loaded model objects (shared across workers of same type)
        self._loaded_vae = None
        self._loaded_clip = None
        
        logger.info(f"[ModelRegistry] Initialized on device {device} - awaiting model registration from clients")
    
    def register_vae(self, vae_type: str, state_dict: Dict[str, Any]) -> dict:
        """
        Register a VAE model from client.
        
        Args:
            vae_type: Type string (e.g., "sdxl", "flux")
            state_dict: The VAE state dict
        
        Returns:
            Registration status dict
        """
        with self.lock:
            # Clear any previously loaded model
            if self._loaded_vae is not None:
                del self._loaded_vae
                self._loaded_vae = None
                torch.cuda.empty_cache()
            
            self.vae = RegisteredModel(
                model_type=vae_type,
                clip_type="",  # Not applicable for VAE
                state_dict=state_dict,
                loaded=False
            )
            
            size_mb = sum(t.numel() * t.element_size() for t in state_dict.values() 
                         if isinstance(t, torch.Tensor)) / 1024 / 1024
            
            logger.info(f"[ModelRegistry] Registered VAE: {vae_type} ({size_mb:.1f} MB)")
            
            return {
                "success": True,
                "vae_type": vae_type,
                "size_mb": size_mb,
                "message": f"VAE registered, ready for loading"
            }
    
    def register_clip(self, clip_type: str, model_type: str, 
                      components: Dict[str, Dict[str, Any]]) -> dict:
        """
        Register CLIP components from client.
        
        Args:
            clip_type: Type string for ComfyUI CLIPType (e.g., "stable_diffusion", "flux")
            model_type: Luna model type string (e.g., "SDXL", "Flux")
            components: Dict of CLIP components (clip_l, clip_g, t5xxl)
        
        Returns:
            Registration status dict
        """
        with self.lock:
            # Clear any previously loaded model
            if self._loaded_clip is not None:
                del self._loaded_clip
                self._loaded_clip = None
                torch.cuda.empty_cache()
            
            self.clip = RegisteredModel(
                model_type=model_type,
                clip_type=clip_type,
                components=components,
                loaded=False
            )
            
            total_size_mb = 0
            component_list = []
            for name, sd in components.items():
                size_mb = sum(t.numel() * t.element_size() for t in sd.values() 
                             if isinstance(t, torch.Tensor)) / 1024 / 1024
                total_size_mb += size_mb
                component_list.append(name)
            
            logger.info(f"[ModelRegistry] Registered CLIP: {model_type} -> {clip_type} "
                       f"({', '.join(component_list)}, {total_size_mb:.1f} MB total)")
            
            return {
                "success": True,
                "model_type": model_type,
                "clip_type": clip_type,
                "components": component_list,
                "size_mb": total_size_mb,
                "message": f"CLIP registered with {len(components)} components"
            }
    
    def get_vae(self, precision: str = "bf16") -> Any:
        """
        Get or load the registered VAE model.
        
        Loads on first access, returns cached model on subsequent calls.
        Sets device context to VAE_DEVICE before loading.
        """
        with self.lock:
            if self.vae is None:
                raise RuntimeError("No VAE registered. Call register_vae first.")
            
            if self._loaded_vae is not None:
                return self._loaded_vae
            
            # Load the VAE
            import comfy.sd  # type: ignore
            
            # Set device context for VAE loading on GPU 0
            target_device = torch.device(VAE_DEVICE)
            logger.info(f"[ModelRegistry] ========== VAE LOADING START ==========")
            logger.info(f"[ModelRegistry] Target device: {target_device} ({VAE_DEVICE})")
            log_all_gpu_vram("BEFORE VAE load: ")
            
            state_dict = self.vae.state_dict
            if precision != "fp32":
                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                state_dict = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v 
                             for k, v in state_dict.items()}  # type: ignore
            
            # Set thread-local device context for VAE
            _daemon_device_context.device = VAE_DEVICE
            try:
                self._loaded_vae = comfy.sd.VAE(sd=state_dict)  # type: ignore
                self.vae.loaded = True
            finally:
                # Clear context after loading
                _daemon_device_context.device = None
            
            log_all_gpu_vram("AFTER VAE load: ")
            
            # Free the state_dict from registry to save VRAM (model is now loaded)
            self.vae.state_dict = None
            del state_dict
            torch.cuda.empty_cache()
            
            logger.info(f"[ModelRegistry] Loaded VAE ({precision})")
            return self._loaded_vae
    
    def get_clip(self, precision: str = "bf16") -> Any:
        """
        Get or load the registered CLIP model.
        
        Uses the registered clip_type for proper model construction.
        Sets device context to CLIP_DEVICE before loading.
        """
        with self.lock:
            if self.clip is None:
                raise RuntimeError("No CLIP registered. Call register_clip first.")
            
            if self._loaded_clip is not None:
                return self._loaded_clip
            
            # Load the CLIP using comfy.sd
            import comfy.sd  # type: ignore
            
            clip_type_str = self.clip.clip_type
            components = self.clip.components
            
            # Map clip type string to comfy.sd.CLIPType enum
            clip_type_enum_map = {
                "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,  # type: ignore
                "sd3": comfy.sd.CLIPType.SD3,  # type: ignore
                "flux": comfy.sd.CLIPType.FLUX,  # type: ignore
                "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,  # type: ignore
                "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,  # type: ignore
                "lumina2": comfy.sd.CLIPType.LUMINA2,  # type: ignore
            }
            clip_type_enum = clip_type_enum_map.get(clip_type_str, comfy.sd.CLIPType.STABLE_DIFFUSION)  # type: ignore
            
            # Set device context for CLIP loading on GPU 1
            target_device = torch.device(CLIP_DEVICE)
            logger.info(f"[ModelRegistry] ========== CLIP LOADING START ==========")
            logger.info(f"[ModelRegistry] Target device: {target_device} ({CLIP_DEVICE})")
            log_all_gpu_vram("BEFORE CLIP load: ")
            logger.info(f"[ModelRegistry] About to set context and load CLIP...")
            
            # Build state_dicts with proper precision
            clip_data = []
            for name in ["clip_l", "clip_g", "t5xxl"]:
                if name in components:  # type: ignore
                    sd = components[name]  # type: ignore
                    if precision != "fp32":
                        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                        sd = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v 
                             for k, v in sd.items()}
                    clip_data.append(sd)
            
            # Set thread-local device context for CLIP
            logger.info(f"[ModelRegistry] Setting context device to: {CLIP_DEVICE}")
            _daemon_device_context.device = CLIP_DEVICE
            logger.info(f"[ModelRegistry] Context device set, now calling load_text_encoder_state_dicts()...")
            try:
                # Load CLIP - will use our context-aware patched device
                self._loaded_clip = comfy.sd.load_text_encoder_state_dicts(  # type: ignore
                    state_dicts=clip_data,
                    clip_type=clip_type_enum
                )
                logger.info(f"[ModelRegistry] load_text_encoder_state_dicts() completed")
            finally:
                # Clear context after loading
                logger.info(f"[ModelRegistry] Clearing context device")
                _daemon_device_context.device = None
            
            log_all_gpu_vram("AFTER CLIP load: ")
            
            self.clip.loaded = True
            
            # Log which device CLIP ended up on
            clip_device = "unknown"
            if hasattr(self._loaded_clip, 'cond_stage_model') and hasattr(self._loaded_clip.cond_stage_model, 'device'):
                clip_device = str(self._loaded_clip.cond_stage_model.device)
            
            # Free the state_dicts from registry to save VRAM (model is now loaded)
            self.clip.components = None
            del clip_data
            torch.cuda.empty_cache()
            
            logger.info(f"[ModelRegistry] Loaded CLIP ({precision}, type={clip_type_str}, device={clip_device})")
            
            # Log memory usage after loading
            if 'cuda' in str(clip_device):
                gpu_id = int(clip_device.split(':')[1]) if ':' in clip_device else 0
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                logger.info(f"[ModelRegistry] GPU {gpu_id} memory after CLIP load: {allocated:.2f} GB allocated")
            
            return self._loaded_clip
    
    def has_vae(self) -> bool:
        """Check if a VAE is registered"""
        return self.vae is not None
    
    def has_clip(self) -> bool:
        """Check if a CLIP is registered"""
        return self.clip is not None
    
    def register_vae_by_path(self, vae_path: str, vae_type: str) -> dict:
        """
        Register a VAE by path (loads from disk).
        
        This is more efficient than sending state dict over socket.
        
        Args:
            vae_path: Full path to VAE file (.safetensors)
            vae_type: Type string (e.g., "sdxl", "flux")
        
        Returns:
            Registration status dict
        """
        import os
        
        if not os.path.exists(vae_path):
            return {"success": False, "error": f"VAE file not found: {vae_path}"}
        
        with self.lock:
            # Clear any previously loaded model
            if self._loaded_vae is not None:
                del self._loaded_vae
                self._loaded_vae = None
                torch.cuda.empty_cache()
            
            # Store path for lazy loading
            self.vae = RegisteredModel(
                model_type=vae_type,
                clip_type="",
                path=vae_path,
                loaded=False
            )
            
            size_mb = os.path.getsize(vae_path) / 1024 / 1024
            logger.info(f"[ModelRegistry] Registered VAE by path: {os.path.basename(vae_path)} ({size_mb:.1f} MB)")
            
            return {
                "success": True,
                "vae_type": vae_type,
                "vae_path": vae_path,
                "size_mb": size_mb,
                "message": "VAE registered from path"
            }
    
    def load_vae_async(self, precision: str = "bf16"):
        """
        Load VAE in background thread (non-blocking).
        
        Called after registration to pre-warm the model so it's ready
        when the first encode/decode request comes in.
        """
        def _load():
            try:
                logger.info("[ModelRegistry] Loading VAE in background...")
                self.get_vae_model(precision)
                logger.info("[ModelRegistry] VAE loaded and ready!")
            except Exception as e:
                logger.error(f"[ModelRegistry] Background VAE load failed: {e}")
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        return thread
    
    def register_clip_by_path(self, clip_components: dict, model_type: str, clip_type: str) -> dict:
        """
        Register CLIP by paths (loads from disk).
        
        This is more efficient than sending state dicts over socket.
        
        Args:
            clip_components: Dict of {component_type: path}
            model_type: Luna model type (e.g., "SDXL", "Flux")
            clip_type: ComfyUI CLIPType string (e.g., "stable_diffusion", "flux")
        
        Returns:
            Registration status dict
        """
        import os
        import folder_paths  # type: ignore
        
        # Validate paths
        valid_components = {}
        for comp_type, path in clip_components.items():
            if os.path.exists(path):
                valid_components[comp_type] = path
            else:
                # Try folder_paths resolution
                full_path = folder_paths.get_full_path("clip", os.path.basename(path)) if folder_paths else None
                if full_path and os.path.exists(full_path):
                    valid_components[comp_type] = full_path
                else:
                    logger.warning(f"[ModelRegistry] CLIP path not found for {comp_type}: {path}")
        
        if not valid_components:
            return {"success": False, "error": "No valid CLIP paths found"}
        
        with self.lock:
            # Clear any previously loaded model
            if self._loaded_clip is not None:
                del self._loaded_clip
                self._loaded_clip = None
                torch.cuda.empty_cache()
            
            # Store paths for lazy loading
            self.clip = RegisteredModel(
                model_type=model_type,
                clip_type=clip_type,
                paths=valid_components, # Now a dict  # type: ignore
                loaded=False
            )
            
            total_size_mb = sum(os.path.getsize(p) / 1024 / 1024 for p in valid_components.values())
            logger.info(f"[ModelRegistry] Registered CLIP by path: {model_type} -> {clip_type} "
                       f"({len(valid_components)} files, {total_size_mb:.1f} MB)")
            
            return {
                "success": True,
                "model_type": model_type,
                "clip_type": clip_type,
                "components": list(valid_components.keys()),
                "size_mb": total_size_mb,
                "message": f"CLIP registered from {len(valid_components)} path(s)"
            }
    
    def load_clip_async(self, precision: str = "bf16"):
        """
        Load CLIP in background thread (non-blocking).
        
        Called after registration to pre-warm the model so it's ready
        when the first encode request comes in.
        """
        def _load():
            try:
                logger.info("[ModelRegistry] Loading CLIP in background...")
                self.get_clip_model(precision)
                logger.info("[ModelRegistry] CLIP loaded and ready!")
            except Exception as e:
                logger.error(f"[ModelRegistry] Background CLIP load failed: {e}")
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        return thread
    
    def get_vae_model(self, precision: str = "bf16") -> Any:
        """
        Get or load the registered VAE model.
        
        Supports both state_dict and path-based registration.
        Sets device context to VAE_DEVICE before loading.
        """
        with self.lock:
            if self.vae is None:
                return None
            
            if self._loaded_vae is not None:
                return self._loaded_vae
            
            import comfy.sd  # type: ignore
            import comfy.utils  # type: ignore
            
            # Set device context for VAE loading on GPU 0
            target_device = torch.device(VAE_DEVICE)
            logger.info(f"[ModelRegistry] Loading VAE on {target_device} (via device context)...")
            
            # Check if path-based or state_dict-based
            if hasattr(self.vae, 'path') and self.vae.path:
                # Load from path
                state_dict = comfy.utils.load_torch_file(self.vae.path)  # type: ignore
            elif self.vae.state_dict:
                state_dict = self.vae.state_dict
            else:
                return None
            
            if precision != "fp32":
                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                state_dict = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v 
                             for k, v in state_dict.items()}
            
            # Set thread-local device context for VAE
            _daemon_device_context.device = VAE_DEVICE
            try:
                self._loaded_vae = comfy.sd.VAE(sd=state_dict)  # type: ignore
                self.vae.loaded = True
            finally:
                # Clear context after loading
                _daemon_device_context.device = None
            
            # Free the state_dict to save VRAM
            if hasattr(self.vae, 'state_dict'):
                self.vae.state_dict = None
            del state_dict
            torch.cuda.empty_cache()
            
            logger.info(f"[ModelRegistry] Loaded VAE ({precision})")
            return self._loaded_vae
    
    def get_clip_model(self, precision: str = "bf16") -> Any:
        """
        Get or load the registered CLIP model.
        
        Supports both state_dict and path-based registration.
        Sets device context to CLIP_DEVICE before loading.
        """
        with self.lock:
            if self.clip is None:
                return None
            
            if self._loaded_clip is not None:
                return self._loaded_clip
            
            import comfy.sd  # type: ignore
            import folder_paths  # type: ignore
            
            clip_type_str = self.clip.clip_type
            
            # Map clip type string to comfy.sd.CLIPType enum
            clip_type_enum_map = {
                "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,  # type: ignore
                "sd3": comfy.sd.CLIPType.SD3,  # type: ignore
                "flux": comfy.sd.CLIPType.FLUX,  # type: ignore
                "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,  # type: ignore
                "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,  # type: ignore
                "lumina2": comfy.sd.CLIPType.LUMINA2,  # type: ignore
            }
            clip_type_enum = clip_type_enum_map.get(clip_type_str, comfy.sd.CLIPType.STABLE_DIFFUSION)  # type: ignore
            
            # Set device context for CLIP loading
            logger.info(f"[ModelRegistry] ========== CLIP LOADING START (get_clip_model) ==========")
            log_all_gpu_vram("BEFORE CLIP load: ")
            logger.info(f"[ModelRegistry] Target device: {CLIP_DEVICE}")
            logger.info(f"[ModelRegistry] Setting device context to: {CLIP_DEVICE}")
            _daemon_device_context.device = CLIP_DEVICE
            
            try:
                # Check if path-based or state_dict-based
                if hasattr(self.clip, 'paths') and self.clip.paths:
                    # Load from paths using comfy.sd.load_clip
                    # Convert paths dict to list of values (paths)
                    clip_paths_list = list(self.clip.paths.values()) if isinstance(self.clip.paths, dict) else self.clip.paths
                    logger.info(f"[ModelRegistry] Loading CLIP from paths: {len(clip_paths_list)} files")
                    self._loaded_clip = comfy.sd.load_clip(  # type: ignore
                        ckpt_paths=clip_paths_list,
                        embedding_directory=folder_paths.get_folder_paths("embeddings") if folder_paths else None,
                        clip_type=clip_type_enum
                    )
                elif self.clip.components:
                    # Load from state dicts
                    clip_data = []
                    for name in ["clip_l", "clip_g", "t5xxl"]:
                        if name in self.clip.components:
                            sd = self.clip.components[name]
                            if precision != "fp32":
                                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                                sd = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v 
                                     for k, v in sd.items()}
                            clip_data.append(sd)
                    
                    logger.info(f"[ModelRegistry] Loading CLIP from state_dicts: {len(clip_data)} components")
                    self._loaded_clip = comfy.sd.load_text_encoder_state_dicts(  # type: ignore  # type: ignore
                        state_dicts=clip_data,
                        clip_type=clip_type_enum,
                        model_options={}
                    )
                    
                    # Free the state_dicts from registry to save VRAM
                    self.clip.components = None
                    del clip_data
                    torch.cuda.empty_cache()
                else:
                    return None
            finally:
                # Clear device context
                logger.info(f"[ModelRegistry] Clearing device context")
                _daemon_device_context.device = None
            
            log_all_gpu_vram("AFTER CLIP load: ")
            
            self.clip.loaded = True
            logger.info(f"[ModelRegistry] Loaded CLIP ({precision}, type={clip_type_str})")
            return self._loaded_clip
    
    def is_vae_ready(self) -> bool:
        """Check if VAE is loaded and ready for use"""
        return self._loaded_vae is not None
    
    def is_clip_ready(self) -> bool:
        """Check if CLIP is loaded and ready for use"""
        return self._loaded_clip is not None
    
    def get_info(self) -> dict:
        """Get registry status"""
        return {
            "vae_registered": self.vae is not None,
            "vae_type": self.vae.model_type if self.vae else None,
            "vae_loaded": self._loaded_vae is not None,
            "vae_path": self.vae.path if self.vae and self.vae.path else None,
            "clip_registered": self.clip is not None,
            "clip_model_type": self.clip.model_type if self.clip else None,
            "clip_type": self.clip.clip_type if self.clip else None,
            "clip_loaded": self._loaded_clip is not None,
            "clip_paths": self.clip.paths if self.clip and self.clip.paths else None,
            "model_registered": hasattr(self, 'model') and self.model is not None,
            "model_type": self.model.model_type if hasattr(self, 'model') and self.model else None,
            "model_loaded": hasattr(self, '_loaded_model') and self._loaded_model is not None,
            "model_path": self.model.path if hasattr(self, 'model') and self.model and self.model.path else None,
        }
    
    # =========================================================================
    # MODEL (UNet) Operations - Centralized Inference Server
    # =========================================================================
    
    def register_model_by_path(self, model_path: str, model_type: str) -> dict:
        """
        Register a diffusion model (UNet) by path for daemon loading.
        
        The daemon loads the model from disk and freezes it (eval + requires_grad=False)
        for optimal VRAM usage. All inference routes through this single shared model.
        
        Args:
            model_path: Full path to checkpoint or unet file
            model_type: Type string ('flux', 'sdxl', 'sd15', etc.)
        
        Returns:
            Registration status dict
        """
        with self.lock:
            # Clear any previously loaded model
            if hasattr(self, '_loaded_model') and self._loaded_model is not None:
                del self._loaded_model
                self._loaded_model = None
                torch.cuda.empty_cache()
            
            if not hasattr(self, 'model'):
                self.model = None
            
            self.model = RegisteredModel(
                model_type=model_type,
                clip_type="",  # Not applicable for models
                path=model_path,
                loaded=False
            )
            
            import os
            size_mb = os.path.getsize(model_path) / 1024 / 1024 if os.path.exists(model_path) else 0
            
            logger.info(f"[ModelRegistry] Registered Model: {model_type} from {model_path} ({size_mb:.1f} MB)")
            
            return {
                "success": True,
                "model_type": model_type,
                "model_path": model_path,
                "size_mb": size_mb,
                "message": f"Model registered, ready for loading"
            }
    
    def register_model(self, model_type: str, state_dict: Dict[str, Any]) -> dict:
        """
        Register a model from state dict (less efficient than path-based).
        
        Args:
            model_type: Type string ('flux', 'sdxl', 'sd15', etc.)
            state_dict: The model state dict
        
        Returns:
            Registration status dict
        """
        with self.lock:
            # Clear any previously loaded model
            if hasattr(self, '_loaded_model') and self._loaded_model is not None:
                del self._loaded_model
                self._loaded_model = None
                torch.cuda.empty_cache()
            
            if not hasattr(self, 'model'):
                self.model = None
            
            self.model = RegisteredModel(
                model_type=model_type,
                clip_type="",
                state_dict=state_dict,
                loaded=False
            )
            
            size_mb = sum(t.numel() * t.element_size() for t in state_dict.values() 
                         if isinstance(t, torch.Tensor)) / 1024 / 1024
            
            logger.info(f"[ModelRegistry] Registered Model: {model_type} ({size_mb:.1f} MB)")
            
            return {
                "success": True,
                "model_type": model_type,
                "size_mb": size_mb,
                "message": f"Model registered, ready for loading"
            }
    
    def load_model_async(self):
        """Async load model in background (called after registration)"""
        # For now just log - actual loading happens on first inference
        logger.info("[ModelRegistry] Model load scheduled (lazy - will load on first forward)")
    
    def get_model(self):
        """
        Get or load the registered model.
        
        Loads on first access, returns cached model on subsequent calls.
        Model is frozen (eval + requires_grad=False) for optimal VRAM.
        """
        with self.lock:
            if not hasattr(self, 'model') or self.model is None:
                raise RuntimeError("No model registered. Call register_model_by_path first.")
            
            if hasattr(self, '_loaded_model') and self._loaded_model is not None:
                return self._loaded_model
            
            # Load the model
            import comfy.sd  # type: ignore
            import comfy.utils  # type: ignore
            import folder_paths  # type: ignore
            
            logger.info(f"[ModelRegistry] Loading model: {self.model.model_type}")
            
            # Set device context
            _daemon_device_context.device = self.device
            
            try:
                if self.model.path:
                    # Load from path (preferred)
                    logger.info(f"[ModelRegistry] Loading model from path: {self.model.path}")
                    
                    # Use comfy's load functions
                    if 'unet' in self.model.path.lower() or self.model.path.endswith('.safetensors'):
                        # Direct unet file
                        model_patcher = comfy.sd.load_diffusion_model(self.model.path)
                    else:
                        # Full checkpoint - extract model only
                        out = comfy.sd.load_checkpoint_guess_config(
                            self.model.path,
                            output_vae=False,
                            output_clip=False,
                            embedding_directory=folder_paths.get_folder_paths("embeddings")
                        )
                        model_patcher = out[0]
                else:
                    # Load from state dict (less common)
                    logger.info(f"[ModelRegistry] Loading model from state dict")
                    # TODO: Create model from state dict
                    raise NotImplementedError("State dict loading not yet implemented for models")
                
                # CRITICAL: Freeze the model for inference
                # This prevents gradient tracking and saves 60-70% VRAM
                if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
                    diff_model = model_patcher.model.diffusion_model
                    diff_model.eval()
                    param_count = 0
                    for param in diff_model.parameters():
                        param.requires_grad = False
                        param_count += 1
                    logger.info(f"[ModelRegistry] [OK] Model frozen: {param_count} parameters (eval + requires_grad=False)")
                
                # Move to daemon device
                model_patcher.load_device = torch.device(self.device)
                
                self._loaded_model = model_patcher
                self.model.loaded = True
                
                logger.info(f"[ModelRegistry] [OK] Model loaded and frozen on {self.device}")
                return self._loaded_model
                
            finally:
                # Clear device context
                _daemon_device_context.device = None
    
    def model_forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                     context: Optional[Any] = None,
                     model_type: str = 'sdxl',
                     lora_stack: Optional[List[tuple]] = None,
                     fb_cache_params: Optional[Dict[str, Any]] = None,
                     **kwargs) -> torch.Tensor:
        """
        Execute UNet forward pass with optional transient LoRA and FB cache.
        
        LoRAs and FB cache are applied as temporary patches during inference only,
        then removed. This allows different requests to have different configurations
        without persistent state pollution.
        
        Args:
            x: Noisy latents (B, C, H, W)
            timesteps: Timestep tensor (B,)
            context: Conditioning/context tensor
            model_type: Model type string
            lora_stack: Optional list of (lora_name, model_str, clip_str) tuples
            fb_cache_params: Optional dict with FB cache config:
                - enabled: bool
                - start_percent: float (0.0-1.0)
                - end_percent: float (0.0-1.0)
                - residual_diff_threshold: float
                - max_consecutive_hits: int
                - object_to_patch: str (default 'diffusion_model')
            **kwargs: Additional model-specific arguments
        
        Returns:
            Denoised output tensor
        """
        model = self.get_model()
        
        # Move inputs to daemon device
        device = torch.device(self.device)
        x = x.to(device)
        timesteps = timesteps.to(device)
        if context is not None:
            context = context.to(device)
        
        # Build FB cache config
        fb_config = None
        if fb_cache_params and FBCacheConfig:
            fb_config = FBCacheConfig(
                enabled=fb_cache_params.get('enabled', False),
                start_percent=fb_cache_params.get('start_percent', 0.0),
                end_percent=fb_cache_params.get('end_percent', 1.0),
                residual_diff_threshold=fb_cache_params.get('residual_diff_threshold', 0.1),
                max_consecutive_hits=fb_cache_params.get('max_consecutive_hits', -1),
                object_to_patch=fb_cache_params.get('object_to_patch', 'diffusion_model'),
            )
        
        # CRITICAL: Call model.model._apply_model(), NOT model.model.diffusion_model()
        # _apply_model does essential preprocessing:
        #   1. Sigma/input scaling via model_sampling.calculate_input()
        #   2. Timestep conversion via model_sampling.timestep()
        #   3. Denoising calculation via model_sampling.calculate_denoised()
        # The kwargs come in apply_model format (c_crossattn, y, control, etc.)
        # and _apply_model handles the conversion to diffusion_model format internally.
        
        # Move all tensor kwargs to device
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                processed_kwargs[key] = value.to(device)
            else:
                processed_kwargs[key] = value
        
        # If context was passed separately, add it as c_crossattn
        if context is not None and 'c_crossattn' not in processed_kwargs:
            processed_kwargs['c_crossattn'] = context
        
        with apply_fb_cache_transient(model, fb_config):
            if lora_stack:
                with self._apply_lora_transient(model, lora_stack):
                    with torch.inference_mode():
                        # Call _apply_model which handles all preprocessing
                        result = model.model._apply_model(x, timesteps, **processed_kwargs)
            else:
                # No LoRAs - direct inference
                with torch.inference_mode():
                    result = model.model._apply_model(x, timesteps, **processed_kwargs)
        
        return result
    
    def _apply_lora_transient(self, model, lora_stack):
        """
        Context manager for transient LoRA application to UNet.
        
        Loads LoRA weights, applies them as temporary patches, then removes them.
        Thread-safe for concurrent requests with different LoRA stacks.
        
        Usage:
            with self._apply_lora_transient(model, lora_stack):
                output = model(x, timesteps, context)  # LoRAs applied during forward
                # LoRAs automatically removed here
        """
        import contextlib
        import comfy.utils  # type: ignore
        import folder_paths  # type: ignore
        
        @contextlib.contextmanager
        def lora_context():
            diff_model = model.model.diffusion_model
            original_weights = {}  # Store originals for restoration
            
            try:
                # Load and apply each LoRA in stack
                for lora_name, model_strength, _ in lora_stack:
                    try:
                        # Find and load LoRA file
                        lora_list = folder_paths.get_filename_list("loras")
                        lora_file = None
                        
                        # Exact match first
                        if lora_name in lora_list:
                            lora_file = lora_name
                        else:
                            # Try with extensions or partial match
                            for f in lora_list:
                                if lora_name.lower() in f.lower():
                                    lora_file = f
                                    break
                        
                        if lora_file is None:
                            logger.warning(f"[ModelRegistry] LoRA '{lora_name}' not found")
                            continue
                        
                        # Load LoRA weights from disk
                        lora_path = folder_paths.get_full_path("loras", lora_file)
                        lora_data = comfy.utils.load_torch_file(lora_path)
                        
                        # Filter to UNet weights only (not CLIP)
                        unet_weights = {k: v for k, v in lora_data.items()
                                       if not any(p in k.lower() for p in
                                                 ['clip_l', 'clip_g', 'te1', 'te2', 'text_encoder', 'lora_te'])}
                        
                        if not unet_weights:
                            continue
                        
                        # Apply LoRA weights as patches (in-place modification)
                        for layer_key, lora_weight in unet_weights.items():
                            # Convert LoRA delta format to model weight patches
                            # Simplified: just add LoRA weighted deltas
                            # TODO: Implement proper LoRA application (low-rank decomposition)
                            try:
                                # This is a simplified approach - actual LoRA needs proper decomposition
                                if layer_key in diff_model.state_dict():
                                    # Store original weight
                                    param = dict(diff_model.named_parameters()).get(layer_key)
                                    if param is not None:
                                        original_weights[layer_key] = param.data.clone()
                                        # Apply LoRA (scaled by strength)
                                        param.data.add_(lora_weight.to(param.device), alpha=model_strength)
                            except Exception as e:
                                logger.debug(f"[ModelRegistry] Could not apply LoRA to {layer_key}: {e}")
                        
                        logger.debug(f"[ModelRegistry] Applied LoRA '{lora_file}' (strength={model_strength})")
                    
                    except Exception as e:
                        logger.warning(f"[ModelRegistry] Error applying LoRA '{lora_name}': {e}")
                
                yield
            
            finally:
                # Restore original weights
                for layer_key, original_weight in original_weights.items():
                    try:
                        param = dict(diff_model.named_parameters()).get(layer_key)
                        if param is not None:
                            param.data = original_weight
                    except Exception as e:
                        logger.debug(f"[ModelRegistry] Error restoring {layer_key}: {e}")
        
        return lora_context()


class TransientLoRAContext:
    """
    Context manager for transient LoRA injection (F-150 Method).
    
    Applies LoRA weights as forward hooks, runs inference, removes hooks.
    Uses locking to ensure only one request modifies the model at a time.
    
    Usage:
        with TransientLoRAContext(model, lora_stack, registry):
            output = model.encode(text)
    """
    
    def __init__(
        self,
        model: Any,
        lora_stack: List[Dict[str, Any]],  # [{"hash": str, "strength": float}, ...]
        registry: LoRARegistry
    ):
        self.model = model
        self.lora_stack = lora_stack
        self.registry = registry
        self.hooks: List[Any] = []
        self.original_weights: Dict[str, torch.Tensor] = {}
    
    def __enter__(self):
        """Apply LoRA weights to model"""
        if not self.lora_stack:
            return self
        
        # Get the actual model to patch
        cond_model = getattr(self.model, 'cond_stage_model', None)
        if cond_model is None:
            logger.warning("[TransientLoRA] No cond_stage_model found, skipping LoRA")
            return self
        
        # Collect all LoRA deltas
        for lora_item in self.lora_stack:
            lora_hash = lora_item.get("hash")
            strength = lora_item.get("strength", 1.0)
            
            if not lora_hash:
                continue
            
            weights = self.registry.get(lora_hash)
            if weights is None:
                logger.warning(f"[TransientLoRA] LoRA {lora_hash[:12]}... not in registry")
                continue
            
            # Apply weights to matching layers
            self._apply_lora_weights(cond_model, weights, strength)
        
        return self
    
    def _apply_lora_weights(self, model: Any, weights: Dict[str, torch.Tensor], strength: float):
        """
        Apply LoRA weights to model layers.
        
        LoRA weights are typically named like:
        - lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight
        - lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight
        
        We need to find the corresponding layer and add (up @ down) * strength to it.
        """
        # Group by layer (remove .lora_down/.lora_up suffix)
        lora_pairs = {}
        for key, tensor in weights.items():
            # Extract base key (remove lora_down/lora_up)
            if '.lora_down.' in key:
                base_key = key.replace('.lora_down.', '.')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['down'] = tensor
            elif '.lora_up.' in key:
                base_key = key.replace('.lora_up.', '.')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['up'] = tensor
            elif '.alpha' in key:
                # LoRA alpha scaling factor
                base_key = key.replace('.alpha', '.weight')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['alpha'] = tensor
        
        # Apply each LoRA pair
        for base_key, pair in lora_pairs.items():
            if 'down' not in pair or 'up' not in pair:
                continue
            
            down = pair['down']
            up = pair['up']
            alpha = pair.get('alpha', torch.tensor(down.shape[0]))
            
            # Find the target layer
            target_layer = self._find_layer(model, base_key)
            if target_layer is None:
                continue
            
            # Compute delta: (up @ down) * scale
            # LoRA scale = alpha / rank * strength
            rank = down.shape[0]
            scale = (alpha.item() / rank) * strength
            
            # Compute the weight delta
            if down.dim() == 2 and up.dim() == 2:
                delta = (up @ down) * scale
            else:
                # For conv layers or other shapes
                delta = torch.einsum('o...,i...->oi...', up, down) * scale
            
            # Store original and apply delta
            if hasattr(target_layer, 'weight'):
                weight = target_layer.weight
                weight_key = f"{id(target_layer)}_weight"
                
                if weight_key not in self.original_weights:
                    self.original_weights[weight_key] = (target_layer, weight.data.clone())  # type: ignore
                
                # Apply delta (in-place add)
                if delta.shape == weight.shape:
                    weight.data.add_(delta.to(weight.dtype))
    
    def _find_layer(self, model: Any, key: str) -> Optional[Any]:
        """Find a layer in the model by LoRA key name"""
        # Convert LoRA key format to attribute path
        # e.g., "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.weight"
        # -> model.text_model.encoder.layers[0].self_attn.q_proj
        
        # Remove common prefixes
        key = key.replace('lora_te1_', '').replace('lora_te2_', '')
        key = key.replace('lora_te_', '')
        key = key.replace('.weight', '')
        
        # Split and traverse
        parts = key.split('_')
        current = model
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Handle numeric indices (layers_0 -> layers[0])
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                attr_name = part
                idx = int(parts[i + 1])
                if hasattr(current, attr_name):
                    container = getattr(current, attr_name)
                    if hasattr(container, '__getitem__'):
                        try:
                            current = container[idx]
                            i += 2
                            continue
                        except (IndexError, KeyError):
                            pass
                i += 1
            else:
                if hasattr(current, part):
                    current = getattr(current, part)
                    i += 1
                else:
                    # Try joining with next part (e.g., self_attn -> self.attn doesn't exist)
                    if i + 1 < len(parts):
                        combined = f"{part}_{parts[i + 1]}"
                        if hasattr(current, combined):
                            current = getattr(current, combined)
                            i += 2
                            continue
                    return None
        
        return current
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original weights"""
        # Restore all modified weights
        for weight_key, (layer, original) in self.original_weights.items():
            if hasattr(layer, 'weight'):
                layer.weight.data.copy_(original)  # type: ignore
        
        self.original_weights.clear()
        return False  # Don't suppress exceptions


# ============================================================================
# Worker Classes
# ============================================================================

class ModelWorker:
    """A single worker that holds a model and processes requests"""
    
    def __init__(
        self,
        worker_id: int,
        worker_type: WorkerType,
        device: str,
        precision: str,
        request_queue: queue.Queue,
        result_queues: Dict[int, queue.Queue],
        lora_registry: Optional[LoRARegistry] = None,
        model_registry: Optional[ModelRegistry] = None
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.lora_registry = lora_registry  # For CLIP LoRA injection (F-150)
        self.model_registry = model_registry  # For dynamic model loading
        
        self.model: Any = None  # Loaded on-demand, will be VAE or CLIP model
        self.is_running = False
        self.is_loaded = False
        self.last_active = time.time()
        self.request_count = 0
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Legacy static paths (fallback if no model_registry)
        self.clip_l_path = CLIP_L_PATH
        self.clip_g_path = CLIP_G_PATH
        self.embeddings_dir = EMBEDDINGS_DIR
    
    @property
    def dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        return torch.float32
    
    def _convert_state_dict_precision(self, sd: dict) -> dict:
        """Convert state dict tensors to target precision"""
        if self.precision == "fp32":
            return sd
        
        converted = {}
        for key, value in sd.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                converted[key] = value.to(self.dtype)
            else:
                converted[key] = value
        return converted
    
    def _resolve_path(self, path_or_name: str, type_name: str) -> Optional[str]:
        """
        Resolve a path or filename to a full path.
        
        Args:
            path_or_name: Filename (e.g. "sdxl_vae.safetensors") or full path
            type_name: Type for folder_paths (e.g. "vae", "clip")
            
        Returns:
            Full path if found, else None
        """
        if not path_or_name:
            return None
            
        # 1. Check if it's an absolute path that exists
        if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
            return path_or_name
            
        # 2. Use folder_paths to find it in standard directories
        if HAS_FOLDER_PATHS:
            try:
                full_path = folder_paths.get_full_path(type_name, path_or_name)  # type: ignore
                if full_path:
                    return full_path
            except:
                pass
                
        return None

    def load_model(self):
        """
        Load the model for this worker.
        
        Uses model_registry if available (dynamic loading from client registration).
        Falls back to static config paths for backwards compatibility.
        """
        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore
        
        if self.worker_type == WorkerType.VAE:
            logger.info(f"[VAE-{self.worker_id}] Loading VAE model...")
            
            # Try model_registry first (dynamic loading)
            if self.model_registry and self.model_registry.has_vae():
                self.model = self.model_registry.get_vae_model(self.precision)
                logger.info(f"[VAE-{self.worker_id}] VAE loaded from registry ({self.precision})")
            else:
                # Fallback to static config paths
                vae_path = self._resolve_path(VAE_PATH or "", "vae")  # type: ignore
                
                if not vae_path:
                    raise RuntimeError(f"VAE model not found: {VAE_PATH}")
                    
                sd = comfy.utils.load_torch_file(vae_path)  # type: ignore
                if self.precision != "fp32":
                    sd = self._convert_state_dict_precision(sd)
                self.model = comfy.sd.VAE(sd=sd)  # type: ignore
                logger.info(f"[VAE-{self.worker_id}] VAE loaded from config path ({self.precision})")
            
        elif self.worker_type == WorkerType.CLIP:
            logger.info(f"[CLIP-{self.worker_id}] Loading CLIP model...")
            
            # Try model_registry first (dynamic loading)
            if self.model_registry and self.model_registry.has_clip():
                self.model = self.model_registry.get_clip_model(self.precision)
                clip_type = self.model_registry.clip.clip_type if self.model_registry.clip else "unknown"
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from registry ({self.precision}, type={clip_type})")
            else:
                # Fallback to static config paths
                clip_paths = []
                
                l_path = self._resolve_path(self.clip_l_path or "", "clip")  # type: ignore
                if l_path:
                    clip_paths.append(l_path)
                    
                g_path = self._resolve_path(self.clip_g_path or "", "clip")  # type: ignore
                if g_path:
                    clip_paths.append(g_path)
                
                logger.info(f"[CLIP-{self.worker_id}] Debug: clip_paths={clip_paths}")
                
                if not clip_paths:
                    # If no paths configured/found, try auto-discovery as last resort
                    if HAS_FOLDER_PATHS and (not self.clip_l_path and not self.clip_g_path):
                        logger.info(f"[CLIP-{self.worker_id}] No CLIPs configured, attempting auto-discovery...")
                        try:
                            l_candidates = folder_paths.get_filename_list("clip")  # type: ignore
                            for c in l_candidates:
                                if "clip_l" in c.lower():
                                    p = folder_paths.get_full_path("clip", c)  # type: ignore
                                    if p:
                                        clip_paths.append(p)
                                        break
                            for c in l_candidates:
                                if "clip_g" in c.lower():
                                    p = folder_paths.get_full_path("clip", c)  # type: ignore
                                    if p:
                                        clip_paths.append(p)
                                        break
                        except:
                            pass
                
                if not clip_paths:
                    raise RuntimeError("No CLIP models found. Please configure CLIP_L_PATH/CLIP_G_PATH in config.py or ensure models exist in ComfyUI/models/clip/")

                # Get CLIP type string from MODEL_TYPE via CLIP_TYPE_MAP
                clip_type_str = CLIP_TYPE_MAP.get(MODEL_TYPE, "stable_diffusion")
                
                logger.info(f"[CLIP-{self.worker_id}] Debug: MODEL_TYPE={MODEL_TYPE}, clip_type_str={clip_type_str}")

                # Map clip type string to comfy.sd.CLIPType enum
                clip_type_enum_map = {
                    "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,  # type: ignore
                    "sd3": comfy.sd.CLIPType.SD3,  # type: ignore
                    "flux": comfy.sd.CLIPType.FLUX,  # type: ignore
                    "stable_cascade": comfy.sd.CLIPType.STABLE_CASCADE,  # type: ignore
                    "stable_audio": comfy.sd.CLIPType.STABLE_AUDIO,  # type: ignore
                    "lumina2": comfy.sd.CLIPType.LUMINA2,  # type: ignore
                }
                clip_type_enum = clip_type_enum_map.get(clip_type_str, comfy.sd.CLIPType.STABLE_DIFFUSION)  # type: ignore
                
                logger.info(f"[CLIP-{self.worker_id}] Debug: clip_type_enum={clip_type_enum}")
                
                emb_dir = self.embeddings_dir if self.embeddings_dir and os.path.exists(self.embeddings_dir) else None  # type: ignore
                self.model = comfy.sd.load_clip(  # type: ignore
                    ckpt_paths=clip_paths,
                    embedding_directory=emb_dir,
                    clip_type=clip_type_enum
                )
                
                # Convert to target precision
                if self.precision != "fp32" and hasattr(self.model, 'cond_stage_model'):
                    self.model.cond_stage_model.to(self.dtype)
                
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from config path ({self.precision}, type={clip_type_str})")
        
        self.is_loaded = True
        torch.cuda.empty_cache()
    
    def unload_model(self):
        """Unload the model to free VRAM"""
        if self.model is not None:
            logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Unloading model...")
            del self.model
            self.model = None
            self.is_loaded = False
            torch.cuda.empty_cache()
    
    def process_vae_encode(self, pixels: torch.Tensor, tiled: bool = False,
                           tile_size: int = 512, overlap: int = 64) -> torch.Tensor:
        """
        Encode image pixels to latent space.
        
        Args:
            pixels: Image tensor (B, H, W, C) or (H, W, C)
            tiled: If True, use tiled encoding for large images
            tile_size: Size of tiles for tiled encoding
            overlap: Overlap between tiles
            
        Returns:
            Latent tensor
        """
        assert self.model is not None, "VAE model not loaded"
        
        log_all_gpu_vram(f"[VAE-{self.worker_id}] BEFORE encode: ")
        
        # CRITICAL: Use inference_mode to disable gradient tracking
        with torch.inference_mode():
            # Detach input to prevent holding computation graph references
            pixels = pixels.detach()
            
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
            
            if tiled:
                latents = self._encode_tiled(pixels, tile_size, overlap)
            else:
                try:
                    latents = self.model.encode(pixels)  # type: ignore
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"[VAE] OOM during encode, falling back to tiled mode")
                        torch.cuda.empty_cache()
                        latents = self._encode_tiled(pixels, tile_size, overlap)
                    else:
                        raise
        
            # Delete input tensor immediately after use
            del pixels
            
            log_all_gpu_vram(f"[VAE-{self.worker_id}] AFTER encode: ")
            
            # Detach output to break computation graph (redundant in inference_mode but safe)
            latents = latents.detach()
            result = latents.cpu()
        
        # === AGGRESSIVE GPU MEMORY CLEANUP ===
        # The VAE encoder creates intermediate activations that aren't captured by latents variable
        # We need to explicitly clear those to prevent memory from staying on GPU after encode
        del latents  # Free GPU memory
        
        # Ensure GPU work is complete before cleanup
        torch.cuda.synchronize()
        
        # First cache clear after sync
        torch.cuda.empty_cache()
        
        # If the model has cached activations, clear them
        if self.model is not None and hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        
        # Second cache clear to catch any lingering allocations
        torch.cuda.empty_cache()
        
        # Force garbage collection to free any Python objects holding GPU memory
        import gc
        gc.collect()
        
        # Final cache clear after GC
        torch.cuda.empty_cache()
        
        log_all_gpu_vram(f"[VAE-{self.worker_id}] AFTER CPU transfer: ")
        return result
    
    def _encode_tiled(self, pixels: torch.Tensor, tile_size: int = 512, 
                      overlap: int = 64) -> torch.Tensor:
        """
        Tiled VAE encoding for large images.
        
        Uses multiple tile configurations and averages results for better seam handling.
        """
        import comfy.utils  # type: ignore
        
        # Get model properties
        downscale = getattr(self.model, 'downscale_ratio', 8)
        if callable(downscale):
            downscale = 8  # Default for SDXL
        latent_channels = getattr(self.model, 'latent_channels', 4)
        
        # Prepare the encoding function
        def encode_fn(a):
            # Process input if the model has this method
            if hasattr(self.model, 'process_input'):
                a = self.model.process_input(a)  # type: ignore
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.encode(a.to(vae_dtype).to(device)).float()
        
        # Average multiple tile configurations for better seams
        output_device = torch.device('cpu')
        
        samples = comfy.utils.tiled_scale(  # type: ignore
            pixels, encode_fn, tile_size, tile_size, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples += comfy.utils.tiled_scale(  # type: ignore
            pixels, encode_fn, tile_size * 2, tile_size // 2, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples += comfy.utils.tiled_scale(  # type: ignore
            pixels, encode_fn, tile_size // 2, tile_size * 2, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples /= 3.0
        
        torch.cuda.empty_cache()  # Clean up after tiled operations
        
        return samples
    
    def process_vae_decode(self, latents: torch.Tensor, tiled: bool = False,
                           tile_size: int = 64, overlap: int = 16) -> torch.Tensor:
        """
        Decode latent space to image pixels.
        
        Args:
            latents: Latent tensor
            tiled: If True, use tiled decoding for large latents
            tile_size: Size of tiles for tiled decoding (in latent space)
            overlap: Overlap between tiles
            
        Returns:
            Pixel tensor
        """
        assert self.model is not None, "VAE model not loaded"
        
        logger.info(f"[VAE-{self.worker_id}] Starting decode (tiled={tiled}, shape={latents.shape}, tile_size={tile_size}, overlap={overlap})")
        log_all_gpu_vram(f"[VAE-{self.worker_id}] BEFORE decode: ")
        
        # CRITICAL: Use inference_mode to disable gradient tracking
        # This allows activations to be freed layer-by-layer instead of retained for backprop
        with torch.inference_mode():
            # Detach input to prevent holding computation graph references
            latents = latents.detach()
            
            if tiled:
                logger.info(f"[VAE-{self.worker_id}] Using tiled decode")
                pixels = self._decode_tiled(latents, tile_size, overlap)
            else:
                try:
                    logger.info(f"[VAE-{self.worker_id}] Using normal decode")
                    pixels = self.model.decode(latents)
                    logger.info(f"[VAE-{self.worker_id}] Normal decode completed")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"[VAE-{self.worker_id}] OOM during decode, falling back to tiled mode")
                        torch.cuda.empty_cache()
                        pixels = self._decode_tiled(latents, tile_size, overlap)
                    else:
                        raise
        
        # Delete input tensor immediately after use to free any GPU copies
        del latents
        
        log_all_gpu_vram(f"[VAE-{self.worker_id}] AFTER decode: ")
        logger.info(f"[VAE-{self.worker_id}] Decode completed, moving to CPU")
        
        # Detach output to break any remaining computation graph
        pixels = pixels.detach()
        
        # Move result to CPU
        result = pixels.cpu()
        
        # === AGGRESSIVE GPU MEMORY CLEANUP ===
        # The VAE decoder creates intermediate activations that aren't captured by pixels variable
        # We need to explicitly clear those to prevent 18GB from staying on GPU after decode
        del pixels  # Free GPU memory immediately
        
        # Ensure GPU work is complete before cleanup
        torch.cuda.synchronize()
        
        # First cache clear after sync
        torch.cuda.empty_cache()
        
        # If the model has cached activations, clear them
        if self.model is not None and hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        
        # Second cache clear to catch any lingering allocations
        torch.cuda.empty_cache()
        
        # Force garbage collection to free any Python objects holding GPU memory
        import gc
        gc.collect()
        
        # Final cache clear after GC
        torch.cuda.empty_cache()
        
        log_all_gpu_vram(f"[VAE-{self.worker_id}] AFTER CPU transfer: ")
        logger.info(f"[VAE-{self.worker_id}] CPU transfer complete")
        return result
    
    def _decode_tiled(self, latents: torch.Tensor, tile_size: int = 64,
                      overlap: int = 16) -> torch.Tensor:
        """
        Tiled VAE decoding for large latents.
        
        Uses multiple tile configurations and averages results for better seam handling.
        """
        import comfy.utils  # type: ignore
        
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: latents.shape={latents.shape}, tile_size={tile_size}, overlap={overlap}")
        
        # Get model properties
        upscale = getattr(self.model, 'upscale_ratio', 8)
        if callable(upscale):
            upscale = 8  # Default for SDXL
        
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: upscale={upscale}")
        
        # Prepare the decoding function
        def decode_fn(a):
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.decode(a.to(vae_dtype).to(device)).float()
        
        output_device = torch.device('cpu')
        
        # Average multiple tile configurations for better seams
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: Starting tiled_scale pass 1/3")
        pixels = comfy.utils.tiled_scale(  # type: ignore
            latents, decode_fn, tile_size // 2, tile_size * 2, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: Starting tiled_scale pass 2/3")
        pixels += comfy.utils.tiled_scale(  # type: ignore
            latents, decode_fn, tile_size * 2, tile_size // 2, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: Starting tiled_scale pass 3/3")
        pixels += comfy.utils.tiled_scale(  # type: ignore
            latents, decode_fn, tile_size, tile_size, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: All passes complete, averaging")
        pixels /= 3.0
        logger.info(f"[VAE-{self.worker_id}] _decode_tiled: Complete")
        
        # Process output if model has the method
        if hasattr(self.model, 'process_output'):
            pixels = self.model.process_output(pixels)
        
        torch.cuda.empty_cache()  # Clean up after tiled operations
        
        return pixels
    
    def process_clip_encode(
        self, 
        positive: str, 
        negative: str = "",
        lora_stack: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple:
        """
        Encode text to CLIP conditioning.
        
        If lora_stack is provided and lora_registry is available, applies LoRA
        weights transiently using the F-150 pattern (lock-based).
        """
        assert self.model is not None, "CLIP model not loaded"
        
        log_all_gpu_vram(f"[CLIP-{self.worker_id}] BEFORE encode: ")
        
        with self.lock:  # F-150: serialize LoRA application per-worker
            # Apply LoRAs if provided
            ctx = TransientLoRAContext(self.model, lora_stack or [], self.lora_registry) \
                  if lora_stack and self.lora_registry else None
            
            try:
                if ctx:
                    ctx.__enter__()
                
                tokens_pos = self.model.tokenize(positive)
                cond, pooled = self.model.encode_from_tokens(tokens_pos, return_pooled=True)
                
                tokens_neg = self.model.tokenize(negative if negative else "")
                uncond, pooled_neg = self.model.encode_from_tokens(tokens_neg, return_pooled=True)
                
                result = (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
                
                # Clean up GPU memory from encoding operations
                del cond, pooled, uncond, pooled_neg
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # If model has cache, clear it
                if self.model is not None and hasattr(self.model, 'cache_clear'):
                    self.model.cache_clear()
                torch.cuda.empty_cache()
                
                # Force GC
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                log_all_gpu_vram(f"[CLIP-{self.worker_id}] AFTER encode: ")
                return result
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
    
    def process_clip_encode_sdxl(
        self,
        positive: str,
        negative: str = "",
        width: int = 1024,
        height: int = 1024,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: int = 1024,
        target_height: int = 1024,
        lora_stack: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[list, list]:
        """
        Encode text with SDXL-specific size conditioning.
        
        If lora_stack is provided and lora_registry is available, applies LoRA
        weights transiently using the F-150 pattern (lock-based).
        """
        assert self.model is not None, "CLIP model not loaded"
        
        with self.lock:  # F-150: serialize LoRA application per-worker
            # Apply LoRAs if provided
            ctx = TransientLoRAContext(self.model, lora_stack or [], self.lora_registry) \
                  if lora_stack and self.lora_registry else None
            
            try:
                if ctx:
                    ctx.__enter__()
                
                tokens_pos = self.model.tokenize(positive)
                tokens_neg = self.model.tokenize(negative if negative else "")
                
                cond, pooled = self.model.encode_from_tokens(tokens_pos, return_pooled=True)
                uncond, pooled_neg = self.model.encode_from_tokens(tokens_neg, return_pooled=True)
                
                positive_out = [[
                    cond.cpu(),
                    {
                        "pooled_output": pooled.cpu(),
                        "width": width,
                        "height": height,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "target_width": target_width,
                        "target_height": target_height
                    }
                ]]
                
                negative_out = [[
                    uncond.cpu(),
                    {
                        "pooled_output": pooled_neg.cpu(),
                        "width": width,
                        "height": height,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "target_width": target_width,
                        "target_height": target_height
                    }
                ]]
                
                # Clean up GPU memory from encoding operations
                del cond, pooled, uncond, pooled_neg
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # If model has cache, clear it
                if self.model is not None and hasattr(self.model, 'cache_clear'):
                    self.model.cache_clear()
                torch.cuda.empty_cache()
                
                # Force GC
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                return (positive_out, negative_out)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
    
    def run(self):
        """Main worker loop - process requests from queue"""
        self.is_running = True
        
        # Lazy loading: Don't load on startup. Wait for first request.
        # This avoids warnings when config paths are empty.
        self.is_loaded = False
        logger.info(f"[{self.worker_type.name}-{self.worker_id}] Worker started in IDLE mode (Lazy Loading)")
        
        while self.is_running:
            try:
                # Wait for request with timeout (allows checking is_running)
                try:
                    request_id, cmd, data = self.request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.last_active = time.time()
                self.request_count += 1
                
                # Check if model is loaded before processing
                if not self.is_loaded:
                    # Try to load again if registry has been updated
                    try:
                        self.load_model()
                    except Exception as e:
                        # Still can't load - return error
                        if request_id:
                            self.result_queues[request_id].put({"error": f"Model not loaded: {e}"})
                        continue
                
                try:
                    # Process based on command type
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Processing {cmd} (request_id={request_id})")
                    
                    result: Any = {}  # Initialize to prevent unbound errors
                    
                    if self.worker_type == WorkerType.VAE:
                        if cmd == "vae_encode":
                            result = self.process_vae_encode(
                                data["pixels"],
                                tiled=data.get("tiled", False),
                                tile_size=data.get("tile_size", 512),
                                overlap=data.get("overlap", 64)
                            )
                        elif cmd == "vae_decode":
                            result = self.process_vae_decode(
                                data["latents"],
                                tiled=data.get("tiled", False),
                                tile_size=data.get("tile_size", 64),
                                overlap=data.get("overlap", 16)
                            )
                        else:
                            result = {"error": f"Unknown VAE command: {cmd}"}
                    
                    elif self.worker_type == WorkerType.CLIP:
                        # Extract lora_stack from data (F-150)
                        lora_stack = data.get("lora_stack")
                        
                        if cmd == "clip_encode":
                            result = self.process_clip_encode(
                                data["positive"],
                                data.get("negative", ""),
                                lora_stack=lora_stack
                            )
                        elif cmd == "clip_encode_sdxl":
                            result = self.process_clip_encode_sdxl(
                                data["positive"],
                                data.get("negative", ""),
                                data.get("width", 1024),
                                data.get("height", 1024),
                                data.get("crop_w", 0),
                                data.get("crop_h", 0),
                                data.get("target_width", 1024),
                                data.get("target_height", 1024),
                                lora_stack=lora_stack
                            )
                        else:
                            result = {"error": f"Unknown CLIP command: {cmd}"}
                    
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Completed {cmd} successfully (request_id={request_id})")
                    
                except Exception as e:
                    logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Error processing {cmd}: {e}")
                    import traceback
                    logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Traceback:\n{traceback.format_exc()}")
                    result = {"error": str(e)}
                
                # Send result back
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                
                # Clean up GPU memory after each request
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker error: {e}")
    
    def start(self):
        """Start the worker thread"""
        # Don't load model here - let the thread handle it in run()
        # This prevents blocking the main thread and allows lazy loading
        
        self.thread = threading.Thread(
            target=self.run,
            name=f"{self.worker_type.value}-worker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker started")
    
    def stop(self):
        """Stop the worker thread and unload model"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.unload_model()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker stopped")


# ============================================================================
# Worker Pool Manager
# ============================================================================

class WorkerPool:
    """Manages a pool of workers with dynamic scaling"""
    
    def __init__(
        self,
        worker_type: WorkerType,
        device: str,
        precision: str,
        config: ScalingConfig,
        on_scale_event: Optional[Callable[[str, dict], None]] = None,
        lora_registry: Optional[LoRARegistry] = None,
        model_registry: Optional[ModelRegistry] = None
    ):
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.config = config
        self.on_scale_event = on_scale_event  # Callback for scaling events
        self.lora_registry = lora_registry  # For CLIP LoRA support (F-150)
        self.model_registry = model_registry  # For dynamic model loading
        
        self.workers: List[ModelWorker] = []
        self.request_queue: queue.Queue = queue.Queue()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.next_request_id = 0
        self.lock = threading.Lock()
        
        self._next_worker_id = 0
        self._scaling_thread: Optional[threading.Thread] = None
        self._running = False
    
    @property
    def min_workers(self) -> int:
        if self.worker_type == WorkerType.VAE:
            return self.config.min_vae_workers
        return self.config.min_clip_workers
    
    @property
    def max_workers(self) -> int:
        if self.worker_type == WorkerType.VAE:
            return self.config.max_vae_workers
        return self.config.max_clip_workers
    
    @property
    def model_size_gb(self) -> float:
        if self.worker_type == WorkerType.VAE:
            return self.config.vae_size_gb
        return self.config.clip_size_gb
    
    def get_available_vram_gb(self) -> float:
        """Get available VRAM in GB"""
        if 'cuda' not in self.device:
            return float('inf')
        
        device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        used = torch.cuda.memory_allocated(device_idx) / 1024**3
        return total - used
    
    def can_scale_up(self) -> bool:
        """Check if we can add another worker"""
        if len(self.workers) >= self.max_workers:
            return False
        
        available = self.get_available_vram_gb()
        needed = self.model_size_gb + self.config.vram_safety_margin_gb
        return available >= needed
    
    def scale_up(self) -> Optional[ModelWorker]:
        """Add a new worker to the pool"""
        with self.lock:
            if not self.can_scale_up():
                return None
            
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            
            worker = ModelWorker(
                worker_id=worker_id,
                worker_type=self.worker_type,
                device=self.device,
                precision=self.precision,
                request_queue=self.request_queue,
                result_queues=self.result_queues,
                lora_registry=self.lora_registry,  # F-150: pass registry to CLIP workers
                model_registry=self.model_registry  # Dynamic model loading
            )
            
            worker.start()
            self.workers.append(worker)
            
            vram_available = self.get_available_vram_gb()
            logger.info(
                f"[{self.worker_type.value.upper()}] Scaled UP to {len(self.workers)} workers "
                f"(VRAM available: {vram_available:.2f} GB)"
            )
            
            # Broadcast scaling event
            if self.on_scale_event:
                self.on_scale_event("scale_up", {
                    "pool": self.worker_type.value,
                    "worker_id": worker_id,
                    "active_workers": len(self.workers),
                    "vram_available_gb": round(vram_available, 2)
                })
            
            return worker
    
    def scale_down(self, worker: ModelWorker):
        """Remove an idle worker from the pool"""
        with self.lock:
            if len(self.workers) <= self.min_workers:
                return
            
            if worker in self.workers:
                worker_id = worker.worker_id
                worker.stop()
                self.workers.remove(worker)
                
                vram_available = self.get_available_vram_gb()
                logger.info(
                    f"[{self.worker_type.value.upper()}] Scaled DOWN to {len(self.workers)} workers "
                    f"(VRAM available: {vram_available:.2f} GB)"
                )
                
                # Broadcast scaling event
                if self.on_scale_event:
                    self.on_scale_event("scale_down", {
                        "pool": self.worker_type.value,
                        "worker_id": worker_id,
                        "active_workers": len(self.workers),
                        "vram_available_gb": round(vram_available, 2)
                    })
    
    def _scaling_loop(self):
        """Background thread that monitors and scales workers"""
        last_scale_up_check = 0
        queue_was_backed_up = False
        
        while self._running:
            time.sleep(self.config.scaling_check_interval_sec)
            
            now = time.time()
            queue_depth = self.request_queue.qsize()
            
            # Scale UP check
            # 1. Immediate scale up if we have requests but no workers (Lazy Loading)
            if queue_depth > 0 and len(self.workers) == 0:
                if self.can_scale_up():
                    logger.info(f"[{self.worker_type.value.upper()}] Lazy loading triggered by request")
                    self.scale_up()
                    last_scale_up_check = now
            
            # 2. Standard queue-based scaling
            elif queue_depth > self.config.queue_threshold:
                if not queue_was_backed_up:
                    queue_was_backed_up = True
                    last_scale_up_check = now
                elif now - last_scale_up_check >= self.config.scale_up_delay_sec:
                    if self.can_scale_up():
                        self.scale_up()
                    last_scale_up_check = now
            else:
                queue_was_backed_up = False
            
            # Scale DOWN check - find idle workers
            with self.lock:
                idle_workers = [
                    w for w in self.workers
                    if now - w.last_active > self.config.idle_timeout_sec
                ]
            
            # Only scale down one at a time, and keep minimum
            for worker in idle_workers:
                if len(self.workers) > self.min_workers:
                    self.scale_down(worker)
                    break  # One at a time
    
    def submit(self, cmd: str, data: dict) -> Any:
        """Submit a request and wait for result"""
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            self.result_queues[request_id] = queue.Queue()
        
        # Submit to work queue
        self.request_queue.put((request_id, cmd, data))
        
        # Wait for result - longer timeout for VAE operations with tiled decoding
        timeout = 180.0 if cmd in ["vae_encode", "vae_decode"] else 60.0
        try:
            result = self.result_queues[request_id].get(timeout=timeout)
        finally:
            with self.lock:
                del self.result_queues[request_id]
        
        return result
    
    def start(self):
        """Start the pool with minimum workers"""
        self._running = True
        
        # Start minimum workers - DISABLED for lazy loading
        # for _ in range(self.min_workers):
        #     self.scale_up()
        
        # Start scaling monitor thread
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            name=f"{self.worker_type.value}-scaler",
            daemon=True
        )
        self._scaling_thread.start()
    
    def stop(self):
        """Stop all workers"""
        self._running = False
        
        with self.lock:
            for worker in self.workers:
                worker.stop()
            self.workers.clear()
    
    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self.lock:
            return {
                "type": self.worker_type.value,
                "active_workers": len(self.workers),
                "queue_depth": self.request_queue.qsize(),
                "total_requests": sum(w.request_count for w in self.workers),
                "worker_ids": [w.worker_id for w in self.workers],
            }


# ============================================================================
# WebSocket Server for Status Monitoring (LUNA-Narrates Compatible)
# ============================================================================

class WebSocketServer:
    """
    Simple WebSocket server for daemon status monitoring.
    Compatible with LUNA-Narrates monitoring pattern.
    
    Message Types (similar to ComfyUI):
    - {"type": "status", "data": {...}}      - Periodic status updates
    - {"type": "scaling", "data": {...}}     - Worker scale up/down events  
    - {"type": "request", "data": {...}}     - Request started/completed
    - {"type": "error", "data": {...}}       - Error events
    """
    
    def __init__(self, daemon: 'DynamicDaemon', host: str, port: int):
        self.daemon = daemon
        self.host = host
        self.port = port
        self.clients: Set[socket.socket] = set()
        self.clients_lock = threading.Lock()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._broadcast_thread: Optional[threading.Thread] = None
    
    def _create_accept_key(self, key: str) -> str:
        """Create WebSocket accept key from client key"""
        GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        sha1 = hashlib.sha1((key + GUID).encode()).digest()
        return base64.b64encode(sha1).decode()
    
    def _handshake(self, conn: socket.socket) -> bool:
        """Perform WebSocket handshake"""
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                return False
            
            # Parse headers
            headers = {}
            lines = data.split('\r\n')
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Check for WebSocket upgrade
            if headers.get('upgrade', '').lower() != 'websocket':
                return False
            
            # Get client key
            client_key = headers.get('sec-websocket-key', '')
            if not client_key:
                return False
            
            # Send handshake response
            accept_key = self._create_accept_key(client_key)
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n"
                "\r\n"
            )
            conn.sendall(response.encode())
            return True
            
        except Exception as e:
            logger.error(f"WebSocket handshake error: {e}")
            return False
    
    def _encode_frame(self, data: str) -> bytes:
        """Encode data as WebSocket text frame"""
        payload = data.encode('utf-8')
        length = len(payload)
        
        if length <= 125:
            frame = bytes([0x81, length]) + payload
        elif length <= 65535:
            frame = bytes([0x81, 126]) + struct.pack('>H', length) + payload
        else:
            frame = bytes([0x81, 127]) + struct.pack('>Q', length) + payload
        
        return frame
    
    def _decode_frame(self, conn: socket.socket) -> Optional[str]:
        """Decode incoming WebSocket frame"""
        try:
            header = conn.recv(2)
            if len(header) < 2:
                return None
            
            opcode = header[0] & 0x0F
            
            # Close frame
            if opcode == 0x08:
                return None
            
            # Ping - send pong
            if opcode == 0x09:
                conn.sendall(bytes([0x8A, 0]))
                return ""
            
            masked = (header[1] & 0x80) != 0
            length = header[1] & 0x7F
            
            if length == 126:
                length = struct.unpack('>H', conn.recv(2))[0]
            elif length == 127:
                length = struct.unpack('>Q', conn.recv(8))[0]
            
            if masked:
                mask = conn.recv(4)
                data = bytearray(conn.recv(length))
                for i in range(length):
                    data[i] ^= mask[i % 4]
                return data.decode('utf-8')
            else:
                return conn.recv(length).decode('utf-8')
                
        except Exception:
            return None
    
    def broadcast(self, message_type: str, data: dict):
        """Broadcast message to all connected clients"""
        message = json.dumps({"type": message_type, "data": data})
        frame = self._encode_frame(message)
        
        with self.clients_lock:
            dead_clients = []
            for client in self.clients:
                try:
                    client.sendall(frame)
                except Exception:
                    dead_clients.append(client)
            
            # Clean up dead connections
            for client in dead_clients:
                self.clients.discard(client)
                try:
                    client.close()
                except:
                    pass
    
    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle a single WebSocket client connection"""
        if not self._handshake(conn):
            conn.close()
            return
        
        with self.clients_lock:
            self.clients.add(conn)
        
        logger.info(f"WebSocket client connected: {addr}")
        
        # Send initial status
        try:
            status = self.daemon.get_info()
            message = json.dumps({"type": "status", "data": status})
            conn.sendall(self._encode_frame(message))
        except Exception as e:
            logger.error(f"Error sending initial status: {e}")
        
        # Keep connection alive and handle incoming messages
        try:
            while self._running:
                conn.settimeout(1.0)
                try:
                    data = self._decode_frame(conn)
                    if data is None:  # Connection closed
                        break
                    
                    # Handle client messages (e.g., explicit status request)
                    if data:
                        try:
                            msg = json.loads(data)
                            if msg.get("type") == "get_status":
                                status = self.daemon.get_info()
                                response = json.dumps({"type": "status", "data": status})
                                conn.sendall(self._encode_frame(response))
                        except json.JSONDecodeError:
                            pass
                            
                except socket.timeout:
                    continue
                except Exception:
                    break
                    
        finally:
            with self.clients_lock:
                self.clients.discard(conn)
            try:
                conn.close()
            except:
                pass
            logger.info(f"WebSocket client disconnected: {addr}")
    
    def _broadcast_loop(self):
        """Periodically broadcast status to all clients"""
        while self._running:
            time.sleep(1.0)  # Broadcast every second
            
            if self.clients:
                try:
                    status = self.daemon.get_info()
                    self.broadcast("status", status)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
    
    def start(self):
        """Start the WebSocket server"""
        self._running = True
        
        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start broadcast thread
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="ws-broadcast",
            daemon=True
        )
        self._broadcast_thread.start()
        
        # Accept loop in separate thread
        def accept_loop():
            while self._running:
                try:
                    if self._server_socket is None:
                        break
                    conn, addr = self._server_socket.accept()
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"WebSocket accept error: {e}")
        
        threading.Thread(target=accept_loop, name="ws-accept", daemon=True).start()
        logger.info(f"WebSocket monitoring server started on ws://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the WebSocket server"""
        self._running = False
        
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass


# ============================================================================
# Main Daemon Server
# ============================================================================

class DynamicDaemon:
    """
    Main daemon server with dynamic worker scaling.
    
    v1.3 Split Daemon Architecture:
    - ServiceType.FULL: Both CLIP and VAE (default, legacy mode)
    - ServiceType.CLIP_ONLY: Just CLIP encoding (for secondary GPU)
    - ServiceType.VAE_ONLY: Just VAE encode/decode (for primary GPU with IPC)
    """
    
    def __init__(
        self, 
        device: str = CLIP_DEVICE, 
        precision: str = MODEL_PRECISION,
        clip_precision: Optional[str] = None,
        vae_precision: Optional[str] = None,
        service_type: ServiceType = SERVICE_TYPE,
        port: Optional[int] = None
    ):
        self.device = device
        self.precision = precision  # Legacy fallback
        # Use separate precisions if provided, otherwise fall back to unified or config
        self.clip_precision = clip_precision or CLIP_PRECISION
        self.vae_precision = vae_precision or VAE_PRECISION
        self.service_type = service_type
        self.port = port or DAEMON_PORT
        self.config = ScalingConfig()
        
        logger.info(f"[MULTI-GPU] Context-aware device routing ready (CLIP: {CLIP_DEVICE}, VAE: {VAE_DEVICE})")
        
        # LoRA Registry (F-150) - only needed for CLIP
        if service_type in (ServiceType.FULL, ServiceType.CLIP_ONLY):
            self.lora_registry = LoRARegistry(max_size_mb=2048.0, device=device)
        else:
            self.lora_registry = None
        
        # Model Registry - for dynamic model loading from client registration
        self.model_registry = ModelRegistry(device=device)
        
        # Checkpoint tracking - stores info about UNet/checkpoint models loaded in ComfyUI instances
        # Format: {instance_id: {"name": str, "size_mb": float, "device": str, "path": str}}
        self.checkpoint_registry: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_lock = threading.Lock()
        
        # Adjust model sizes based on precision
        # VAE size
        if self.vae_precision in ("bf16", "fp16"):
            self.config.vae_size_gb = 0.082  # 164MB / 2
        else:
            self.config.vae_size_gb = 0.164
        
        # CLIP size
        if self.clip_precision in ("bf16", "fp16"):
            self.config.clip_size_gb = 1.6   # (2.72GB + 483MB) / 2
        else:
            self.config.clip_size_gb = 3.2
        
        self.vae_pool: Optional[WorkerPool] = None
        self.clip_pool: Optional[WorkerPool] = None
        self.save_pool: Optional[WorkerPool] = None  # Async image saver pool
        self.ws_server: Optional[WebSocketServer] = None
        
        # Qwen3 encoder for Z-IMAGE CLIP + LLM (unified model)
        # Loads once, serves both CLIP encoding and text generation
        self.qwen3_encoder = None
        self.qwen3_lock = threading.Lock()  # Thread-safe access
        
        self.start_time = time.time()
        self.request_count = 0
        self.work_request_count = 0  # Track only VAE/CLIP encode/decode work
        self.shutdown_requested = False  # Flag for clean shutdown
        
        logger.info(f"Daemon mode: {service_type.value}")
    
    def _on_scale_event(self, event_type: str, data: dict):
        """Callback for worker pool scaling events - broadcasts to WebSocket clients"""
        if self.ws_server:
            self.ws_server.broadcast("scaling", {
                "event": event_type,
                **data
            })
    
    def start_pools(self):
        """Initialize and start worker pools based on service type"""
        logger.info(f"Starting worker pools...")
        logger.info(f"  CLIP precision: {self.clip_precision}, VAE precision: {self.vae_precision}")
        
        # VAE pool - only for FULL or VAE_ONLY modes
        if self.service_type in (ServiceType.FULL, ServiceType.VAE_ONLY):
            # Use VAE_DEVICE for VAE pool in FULL mode if configured
            vae_device = self.device
            if self.service_type == ServiceType.FULL and VAE_DEVICE:
                vae_device = VAE_DEVICE
                
            self.vae_pool = WorkerPool(
                worker_type=WorkerType.VAE,
                device=vae_device,
                precision=self.vae_precision,
                config=self.config,
                on_scale_event=self._on_scale_event,
                model_registry=self.model_registry  # Dynamic model registration
            )
            logger.info(f"VAE pool configured on {vae_device} ({self.vae_precision})")
        
        # CLIP pool - only for FULL or CLIP_ONLY modes
        if self.service_type in (ServiceType.FULL, ServiceType.CLIP_ONLY):
            # Use CLIP_DEVICE for CLIP pool
            clip_device = self.device
            if self.service_type == ServiceType.FULL and CLIP_DEVICE:
                clip_device = CLIP_DEVICE
                
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=clip_device,  # CLIP uses the configured CLIP_DEVICE
                precision=self.clip_precision,
                config=self.config,
                on_scale_event=self._on_scale_event,
                lora_registry=self.lora_registry,  # F-150: enable LoRA for CLIP workers
                model_registry=self.model_registry  # Dynamic model registration
            )
            logger.info(f"CLIP pool configured on {clip_device} ({self.clip_precision})")
        
        # Start pools (CLIP first if present, as it's larger)
        if self.clip_pool:
            self.clip_pool.start()
            # Don't scale up immediately - wait for demand
            # self.clip_pool.scale_up()
        if self.vae_pool:
            self.vae_pool.start()
            # Don't scale up immediately - wait for demand
            # self.vae_pool.scale_up()
        
        # Async image save pool - CPU-only, always available for parallel saves
        # Uses 1-4 worker threads for disk I/O without GPU overhead
        self.save_pool = WorkerPool(
            worker_type=WorkerType.IMAGE_SAVE,
            device="cpu",
            precision="fp32",  # Not used for CPU image saver
            config=self.config,
            on_scale_event=self._on_scale_event
        )
        self.save_pool.start()
        logger.info(f"Image save pool configured on CPU (up to 4 parallel workers)")
        
        # Report initial VRAM usage for all GPUs
        log_all_gpu_vram("Initial VRAM: ")
    
    def get_info(self) -> dict:
        """Get daemon status info"""
        info = {
            "status": "ok",
            "version": "2.1-split",
            "service_type": self.service_type.value,
            "device": self.device,
            "precision": self.precision,
            "attention_mode": ATTENTION_MODE,
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.work_request_count,  # Only VAE/CLIP work requests
            "total_requests": self.request_count,  # All requests including health checks
        }
        
        if self.vae_pool:
            info["vae_pool"] = self.vae_pool.get_stats()
        if self.clip_pool:
            info["clip_pool"] = self.clip_pool.get_stats()
        
        # Add LoRA registry stats
        if self.lora_registry:
            info["lora_registry"] = self.lora_registry.get_stats()
        
        # Add model registry info (loaded models, paths, etc.)
        if self.model_registry:
            registry_info = self.model_registry.get_info()
            info.update({
                "vae_loaded": registry_info.get("vae_loaded", False),
                "clip_loaded": registry_info.get("clip_loaded", False),
                "loaded_vae": registry_info.get("vae_type"),
                "loaded_vae_path": registry_info.get("vae_path"),
                "loaded_clip": registry_info.get("clip_model_type"),
                "loaded_clip_paths": registry_info.get("clip_paths"),
            })
        
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
            reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
            total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
            
            info["vram_used_gb"] = reserved  # Use reserved (actual VRAM held by PyTorch)
            info["vram_allocated_gb"] = allocated  # Active tensors only
            info["vram_total_gb"] = total
            info["vram_percent"] = round((reserved / total) * 100, 1)
            
            # Multi-GPU monitoring - report all available GPUs
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = gpu_count
            info["gpus"] = []
            
            for gpu_id in range(gpu_count):
                try:
                    # Get process-specific memory (what this daemon allocated)
                    gpu_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    gpu_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    
                    # Get system-level memory info (shows actual GPU usage across all processes)
                    free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                    gpu_total = total_mem / 1024**3
                    gpu_used = (total_mem - free_mem) / 1024**3
                    gpu_percent = round((gpu_used / gpu_total) * 100, 1)
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    
                    info["gpus"].append({
                        "id": gpu_id,
                        "name": gpu_name,
                        "allocated_gb": round(gpu_allocated, 2),  # Daemon's allocation
                        "reserved_gb": round(gpu_used, 2),  # Actual system usage
                        "total_gb": round(gpu_total, 1),
                        "percent": gpu_percent,
                        "is_daemon_device": (gpu_id == device_idx)
                    })
                except Exception as e:
                    logger.error(f"Error reading GPU {gpu_id} stats: {e}")
        
        # Add checkpoint registry info
        with self.checkpoint_lock:
            info["checkpoints"] = list(self.checkpoint_registry.values())
        
        return info
    
    def _apply_attention_mode(self, mode: str) -> bool:
        """Apply attention mode settings dynamically at runtime
        
        Args:
            mode: Attention mode to apply ("sage", "xformers", "flash", "pytorch", "split", "auto")
            
        Returns:
            True if successfully applied, False otherwise
        """
        if mode == "auto":
            logger.info("[DynamicAttention] Set to auto mode - will use ComfyUI defaults")
            return True
        
        try:
            import comfy.model_management as mm  # type: ignore
            
            logger.info(f"[DynamicAttention] Applying {mode} attention mode...")
            
            # Map modes to their configuration
            attention_map = {
                "xformers": lambda: None,  # xformers is default if available
                "flash": lambda: setattr(mm, 'XFORMERS_IS_AVAILABLE', False) or setattr(mm, 'ENABLE_PYTORCH_ATTENTION', False),
                "sage": lambda: setattr(mm, 'XFORMERS_IS_AVAILABLE', False) or setattr(mm, 'ENABLE_PYTORCH_ATTENTION', False),
                "pytorch": lambda: setattr(mm, 'XFORMERS_IS_AVAILABLE', False) or setattr(mm, 'ENABLE_PYTORCH_ATTENTION', True),
                "split": lambda: setattr(mm, 'XFORMERS_IS_AVAILABLE', False) or setattr(mm, 'ENABLE_PYTORCH_ATTENTION', False)
            }
            
            if mode.lower() in attention_map:
                config_func = attention_map[mode.lower()]
                if config_func:
                    config_func()
                logger.info(f"[DynamicAttention] Successfully applied {mode} attention mode")
                return True
            else:
                logger.warning(f"[DynamicAttention] Unknown mode '{mode}'")
                return False
                
        except Exception as e:
            logger.error(f"[DynamicAttention] Failed to apply {mode}: {e}")
            return False
    
    def _get_gpu_id(self) -> Optional[int]:
        """Get the GPU index this daemon is using."""
        if 'cuda' not in self.device:
            return None
        try:
            if ':' in self.device:
                return int(self.device.split(':')[1])
            return 0
        except:
            return None
    
    def handle_request(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming request with Length-Prefix Protocol (Optimized)"""
        try:
            # 1. Read the Header (4 bytes = uint32 length)
            header = b""
            while len(header) < 4:
                chunk = conn.recv(4 - len(header))
                if not chunk:
                    return
                header += chunk
            
            data_len = struct.unpack('>I', header)[0]
            
            # 2. Read exact payload (collect chunks, join once - no slow accumulator)
            chunks = []
            bytes_recd = 0
            while bytes_recd < data_len:
                chunk_size = min(data_len - bytes_recd, 1048576)  # 1MB chunks
                chunk = conn.recv(chunk_size)
                if not chunk:
                    raise ConnectionError("Socket closed mid-stream")
                chunks.append(chunk)
                bytes_recd += len(chunk)
            
            data = b"".join(chunks)
            
            # 3. Unpickle
            request = pickle.loads(data)
            cmd = request.get("cmd", "unknown")
            
            self.request_count += 1
            # logger.debug commented out for throughput - stdout is blocking on Windows
            # logger.debug(f"Request #{self.request_count}: {cmd}")
            
            # Initialize result to prevent unbound errors
            result: Any = {}
            
            # Route command
            if cmd == "health":
                result = {"status": "ok", "service_type": self.service_type.value}
            elif cmd == "info":
                result = self.get_info()
            elif cmd == "set_attention_mode":
                # Dynamically change attention mode at runtime
                new_mode = request.get("mode", "auto")
                logger.info(f"[API] Received request to change attention mode to: {new_mode}")
                
                # Update the global config (for new models loaded after this)
                global ATTENTION_MODE
                ATTENTION_MODE = new_mode
                
                # Apply the new mode immediately
                success = self._apply_attention_mode(new_mode)
                result = {
                    "success": success,
                    "mode": new_mode,
                    "message": f"Attention mode {'changed to' if success else 'failed to change to'} {new_mode}"
                }
            
            # Model registration commands (dynamic loading from clients)
            elif cmd == "register_vae":
                if self.service_type == ServiceType.CLIP_ONLY:
                    result = {"error": "VAE registration not available in CLIP-only mode"}
                else:
                    vae_type = request.get("vae_type", "sdxl")
                    state_dict = request.get("state_dict", {})
                    result = self.model_registry.register_vae(vae_type, state_dict)
                    
                    # Immediately start loading in background (non-blocking)
                    if result.get("success"):
                        self.model_registry.load_vae_async(self.vae_precision)
                    
                    # If pool exists but has no workers, start one now
                    if self.vae_pool and len(self.vae_pool.workers) == 0:
                        self.vae_pool.scale_up()
                        
            elif cmd == "register_clip":
                if self.service_type == ServiceType.VAE_ONLY:
                    result = {"error": "CLIP registration not available in VAE-only mode"}
                else:
                    clip_type = request.get("clip_type", "stable_diffusion")
                    model_type = request.get("model_type", "SDXL")
                    components = request.get("components", {})
                    result = self.model_registry.register_clip(clip_type, model_type, components)
                    
                    # Immediately start loading in background (non-blocking)
                    if result.get("success"):
                        self.model_registry.load_clip_async(self.clip_precision)
                    
                    # If pool exists but has no workers, start one now
                    if self.clip_pool and len(self.clip_pool.workers) == 0:
                        self.clip_pool.scale_up()
            
            # Path-based registration (loads from disk, avoids socket serialization)
            elif cmd == "register_vae_by_path":
                if self.service_type == ServiceType.CLIP_ONLY:
                    result = {"error": "VAE registration not available in CLIP-only mode"}
                else:
                    vae_path = request.get("vae_path", "")
                    vae_type = request.get("vae_type", "sdxl")
                    result = self.model_registry.register_vae_by_path(vae_path, vae_type)
                    
                    # Immediately start loading in background (non-blocking)
                    if result.get("success"):
                        self.model_registry.load_vae_async(self.vae_precision)
                    
                    # If pool exists but has no workers, start one now
                    if self.vae_pool and len(self.vae_pool.workers) == 0:
                        self.vae_pool.scale_up()
                        
            elif cmd == "register_clip_by_path":
                if self.service_type == ServiceType.VAE_ONLY:
                    result = {"error": "CLIP registration not available in VAE-only mode"}
                else:
                    clip_components = request.get("clip_components", {})
                    # Fallback for legacy clients sending list
                    if not clip_components and "clip_paths" in request:
                        clip_paths = request.get("clip_paths", [])
                        # Best effort mapping for legacy list
                        clip_components = {f"legacy_{i}": p for i, p in enumerate(clip_paths)}
                        
                    model_type = request.get("model_type", "SDXL")
                    clip_type = request.get("clip_type", "stable_diffusion")
                    result = self.model_registry.register_clip_by_path(clip_components, model_type, clip_type)
                    
                    # Immediately start loading in background (non-blocking)
                    if result.get("success"):
                        self.model_registry.load_clip_async(self.clip_precision)
                    
                    # If pool exists but has no workers, start one now
                    if self.clip_pool and len(self.clip_pool.workers) == 0:
                        self.clip_pool.scale_up()
            
            # MODEL (UNet) Registration - centralized inference server
            elif cmd == "register_model_by_path":
                model_path = request.get("model_path", "")
                model_type = request.get("model_type", "sdxl")
                result = self.model_registry.register_model_by_path(model_path, model_type)
                
                # Immediately start loading in background (non-blocking)
                if result.get("success"):
                    self.model_registry.load_model_async()
            
            elif cmd == "register_model":
                model_type = request.get("model_type", "sdxl")
                state_dict = request.get("state_dict", {})
                result = self.model_registry.register_model(model_type, state_dict)
                
                # Immediately start loading in background (non-blocking)
                if result.get("success"):
                    self.model_registry.load_model_async()
            
            # MODEL (UNet) Inference - routes all instance sampling through daemon
            elif cmd == "model_forward":
                self.work_request_count += 1  # Count model work
                x = request.get("x")
                timesteps = request.get("timesteps")
                context = request.get("context")
                model_type = request.get("model_type", "sdxl")
                lora_stack = request.get("lora_stack", [])
                fb_cache_params = request.get("fb_cache_params")
                
                # Extract any additional kwargs (control, y, etc.) - exclude internal params
                excluded_keys = ["cmd", "x", "timesteps", "context", "model_type", "lora_stack", "fb_cache_params"]
                extra_kwargs = {k: v for k, v in request.items() if k not in excluded_keys}
                
                result = self.model_registry.model_forward(
                    x, timesteps, context, model_type, lora_stack, 
                    fb_cache_params=fb_cache_params, **extra_kwargs
                )
            
            # VAE commands - only available in FULL or VAE_ONLY mode
            elif cmd in ("vae_encode", "vae_decode"):
                self.work_request_count += 1  # Count VAE work
                if self.vae_pool is None:
                    result = {"error": f"VAE not available in {self.service_type.value} mode"}
                else:
                    result = self.vae_pool.submit(cmd, request)
            
            # CLIP commands - only available in FULL or CLIP_ONLY mode
            elif cmd in ("clip_encode", "clip_encode_sdxl"):
                self.work_request_count += 1  # Count CLIP work
                if self.clip_pool is None:
                    result = {"error": f"CLIP not available in {self.service_type.value} mode"}
                else:
                    # lora_stack is passed in request dict, workers have registry reference
                    result = self.clip_pool.submit(cmd, request)
            
            # LoRA commands (F-150) - only available when CLIP is loaded
            elif cmd == "has_lora":
                if self.lora_registry is None:
                    result = {"error": "LoRA registry not available in VAE-only mode"}
                else:
                    lora_hash = request.get("lora_hash", "")
                    result = {"exists": self.lora_registry.has(lora_hash)}
            elif cmd == "upload_lora":
                if self.lora_registry is None:
                    result = {"error": "LoRA registry not available in VAE-only mode"}
                else:
                    lora_hash = request.get("lora_hash", "")
            
            # Checkpoint tracking - register UNet/checkpoint models from ComfyUI instances
            elif cmd == "register_checkpoint":
                instance_id = request.get("instance_id", "unknown")
                checkpoint_info = {
                    "instance_id": instance_id,
                    "name": request.get("name", "unknown"),
                    "path": request.get("path", ""),
                    "size_mb": request.get("size_mb", 0),
                    "device": request.get("device", "unknown"),
                    "dtype": request.get("dtype", "unknown"),
                }
                
                with self.checkpoint_lock:
                    self.checkpoint_registry[instance_id] = checkpoint_info
                
                result = {"success": True, "message": f"Checkpoint registered for {instance_id}"}
                logger.info(f"[CheckpointRegistry] Registered: {checkpoint_info['name']} ({checkpoint_info['size_mb']:.1f} MB) on {checkpoint_info['device']}")
            
            elif cmd == "unregister_checkpoint":
                instance_id = request.get("instance_id", "unknown")
                
                with self.checkpoint_lock:
                    if instance_id in self.checkpoint_registry:
                        removed = self.checkpoint_registry.pop(instance_id)
                        result = {"success": True, "message": f"Unregistered {removed['name']}"}
                        logger.info(f"[CheckpointRegistry] Unregistered: {removed['name']} from {instance_id}")
                    else:
                        result = {"success": False, "message": f"Instance {instance_id} not found"}
            
            elif cmd == "upload_lora":
                    if self.lora_registry is None:
                        result = {"error": "LoRA registry not available in VAE-only mode"}
                    else:
                        lora_hash = request.get("lora_hash", "")
                        weights = request.get("weights", {})
                        if lora_hash:
                            success = self.lora_registry.put(lora_hash, weights)
                            result = {"success": success, "hash": lora_hash}
                        else:
                            result = {"error": "lora_hash not provided"}
            elif cmd == "register_lora":
                # Disk-based LoRA loading (preferred for Config Gateway)
                if self.lora_registry is None:
                    result = {"error": "LoRA registry not available in VAE-only mode"}
                else:
                    lora_name = request.get("lora_name", "")
                    lora_hash = self.lora_registry.register_from_file(lora_name)
                    if lora_hash:
                        result = {"success": True, "hash": lora_hash}
                    else:
                        result = {"success": False, "error": f"Failed to load LoRA: {lora_name}"}
            elif cmd == "lora_stats":
                if self.lora_registry is None:
                    result = {"cached_loras": 0, "total_size_mb": 0}
                else:
                    result = self.lora_registry.get_stats()
            elif cmd == "clear_loras":
                if self.lora_registry:
                    self.lora_registry.clear()
                result = {"success": True}
            
            # IPC Negotiation (v1.3)
            elif cmd == "negotiate_ipc":
                client_gpu_id = request.get("client_gpu_id")
                daemon_gpu_id = self._get_gpu_id()
                
                # Check if we can use IPC (same GPU)
                can_ipc = (
                    ENABLE_CUDA_IPC and
                    client_gpu_id is not None and
                    daemon_gpu_id is not None and
                    client_gpu_id == daemon_gpu_id
                )
                
                result = {
                    "ipc_enabled": can_ipc,
                    "daemon_gpu_id": daemon_gpu_id,
                    "client_gpu_id": client_gpu_id
                }
                
                if can_ipc:
                    logger.info(f"IPC enabled for client on GPU {client_gpu_id}")
            
            # VAE IPC commands (zero-copy for same GPU)
            elif cmd == "vae_encode_ipc":
                if self.vae_pool is None:
                    result = {"error": f"VAE not available in {self.service_type.value} mode"}
                else:
                    # Reconstruct tensor from shared storage
                    storage = request.get("pixels_storage")
                    shape = request.get("pixels_shape")
                    dtype_str = request.get("pixels_dtype")
                    
                    dtype = getattr(torch, dtype_str.replace("torch.", ""))
                    pixels = torch.tensor(storage, dtype=dtype).reshape(shape)
                    
                    # Process via pool (tensor is already on CUDA)
                    request["pixels"] = pixels
                    result = self.vae_pool.submit("vae_encode", request)
            
            elif cmd == "vae_decode_ipc":
                if self.vae_pool is None:
                    result = {"error": f"VAE not available in {self.service_type.value} mode"}
                else:
                    # Reconstruct tensor from shared storage
                    storage = request.get("latents_storage")
                    shape = request.get("latents_shape")
                    dtype_str = request.get("latents_dtype")
                    
                    dtype = getattr(torch, dtype_str.replace("torch.", ""))
                    latents = torch.tensor(storage, dtype=dtype).reshape(shape)
                    
                    # Process via pool (tensor is already on CUDA)
                    request["latents"] = latents
                    result = self.vae_pool.submit("vae_decode", request)
            
            # ================================================================
            # Qwen3/Z-IMAGE Commands - Unified model for CLIP + LLM
            # ================================================================
            elif cmd == "register_qwen3":
                # Register and load Qwen3-VL model (used for both CLIP and LLM)
                model_path = request.get("model_path", "")
                device = request.get("device", LLM_DEVICE or self.device)
                
                if not model_path:
                    result = {"error": "model_path is required"}
                else:
                    try:
                        from .qwen3_encoder import Qwen3VLEncoder, Qwen3VLConfig
                        
                        with self.qwen3_lock:
                            # Unload existing model if any
                            if self.qwen3_encoder is not None:
                                del self.qwen3_encoder
                                self.qwen3_encoder = None
                                torch.cuda.empty_cache()
                            
                            # Create and load new encoder
                            config = Qwen3VLConfig(
                                model_path=model_path,
                                device=device
                            )
                            self.qwen3_encoder = Qwen3VLEncoder(config)
                            success = self.qwen3_encoder.load_model(model_path)
                            
                            if success:
                                result = {
                                    "success": True,
                                    "model_path": model_path,
                                    "device": device,
                                    "zimage_compatible": self.qwen3_encoder.is_zimage_compatible,
                                    "has_vision": getattr(self.qwen3_encoder, '_has_vision', False),
                                    "message": "Qwen3-VL loaded for CLIP+LLM"
                                }
                                logger.info(f"[Qwen3] Model loaded: {model_path}")
                            else:
                                result = {"error": "Failed to load Qwen3 model"}
                                
                    except ImportError as e:
                        result = {"error": f"Qwen3 encoder not available: {e}"}
                    except Exception as e:
                        result = {"error": f"Failed to load Qwen3: {e}"}
            
            elif cmd == "zimage_encode":
                # Z-IMAGE CLIP encoding - extracts embeddings from Qwen3
                if self.qwen3_encoder is None:
                    result = {"error": "Qwen3 not loaded. Call register_qwen3 first."}
                else:
                    text = request.get("text", "")
                    negative_text = request.get("negative_text", "")
                    
                    try:
                        with self.qwen3_lock:
                            pos_emb, neg_emb = self.qwen3_encoder.encode_text_for_zimage(
                                text, negative_text
                            )
                        result = {
                            "success": True,
                            "positive": pos_emb.cpu(),
                            "negative": neg_emb.cpu(),
                            "shape": list(pos_emb.shape)
                        }
                    except Exception as e:
                        result = {"error": f"Z-IMAGE encode failed: {e}"}
            
            elif cmd == "llm_generate":
                # LLM text generation - uses same Qwen3 model as CLIP
                if self.qwen3_encoder is None:
                    result = {"error": "Qwen3 not loaded. Call register_qwen3 first."}
                else:
                    prompt = request.get("prompt", "")
                    max_tokens = request.get("max_tokens", 256)
                    temperature = request.get("temperature", 0.7)
                    system_prompt = request.get("system_prompt")
                    
                    try:
                        with self.qwen3_lock:
                            generated = self.qwen3_encoder.generate_text(
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                system_prompt=system_prompt
                            )
                        result = {
                            "success": True,
                            "text": generated
                        }
                    except Exception as e:
                        result = {"error": f"LLM generation failed: {e}"}
            
            elif cmd == "vlm_describe":
                # VLM image description - uses Qwen3 with vision
                if self.qwen3_encoder is None:
                    result = {"error": "Qwen3 not loaded. Call register_qwen3 first."}
                elif not getattr(self.qwen3_encoder, '_has_vision', False):
                    result = {"error": "Vision not available. Load model with mmproj."}
                else:
                    image = request.get("image")  # Tensor [B, H, W, C]
                    prompt = request.get("prompt", "Describe this image.")
                    max_tokens = request.get("max_tokens", 256)
                    
                    if image is None:
                        result = {"error": "image is required"}
                    else:
                        try:
                            with self.qwen3_lock:
                                description = self.qwen3_encoder.describe_image(
                                    image=image,
                                    prompt=prompt,
                                    max_tokens=max_tokens
                                )
                            result = {
                                "success": True,
                                "text": description
                            }
                        except Exception as e:
                            result = {"error": f"VLM description failed: {e}"}
            
            elif cmd == "qwen3_status":
                # Get status of loaded Qwen3 model
                if self.qwen3_encoder is None:
                    result = {"loaded": False}
                else:
                    result = {
                        "loaded": True,
                        "zimage_compatible": self.qwen3_encoder.is_zimage_compatible,
                        "has_vision": getattr(self.qwen3_encoder, '_has_vision', False),
                        "encode_count": self.qwen3_encoder._encode_count,
                        "vlm_count": self.qwen3_encoder._vlm_count
                    }
            
            elif cmd == "shutdown":
                # Request clean shutdown
                logger.info("Shutdown requested via command")
                self.shutdown_requested = True
                result = {"status": "ok", "message": "Shutdown initiated"}
            
            elif cmd == "save_images_async":
                # Asynchronous image saving - submit to worker pool and return immediately
                # This prevents workflow blocking on disk I/O
                try:
                    save_request = {
                        "save_path": request.get("save_path", ""),
                        "filename": request.get("filename", ""),
                        "model_name": request.get("model_name", ""),
                        "quality_gate": request.get("quality_gate", "disabled"),
                        "min_quality_threshold": request.get("min_quality_threshold", 0.3),
                        "png_compression": request.get("png_compression", 4),
                        "lossy_quality": request.get("lossy_quality", 90),
                        "lossless_webp": request.get("lossless_webp", False),
                        "embed_workflow": request.get("embed_workflow", True),
                        "filename_index": request.get("filename_index", 0),
                        "custom_metadata": request.get("custom_metadata", ""),
                        "metadata": request.get("metadata", {}),
                        "prompt": request.get("prompt"),
                        "extra_pnginfo": request.get("extra_pnginfo"),
                        "images": request.get("images", []),
                        "output_dir": request.get("output_dir", ""),
                        "timestamp": request.get("timestamp", ""),
                    }
                    
                    # Generate unique job ID
                    import uuid
                    job_id = str(uuid.uuid4())[:8]
                    
                    # Submit to async worker pool (fire and forget)
                    # Worker will save images in background without blocking
                    if hasattr(self, 'save_pool') and self.save_pool:
                        self.save_pool.submit("save_images", save_request)
                        result = {
                            "success": True,
                            "job_id": job_id,
                            "message": f"Image save job submitted (ID: {job_id})",
                            "num_images": len(save_request.get("images", []))
                        }
                        logger.debug(f"[Image Save] Job {job_id}: {result['num_images']} images submitted")
                    else:
                        # No save pool available - still accept but log warning
                        result = {
                            "warning": "Save pool not initialized, images may not be saved",
                            "job_id": job_id,
                            "success": False
                        }
                        logger.warning("[Image Save] No save pool available")
                        
                except Exception as e:
                    result = {"error": f"Image save submission failed: {str(e)}"}
                    logger.error(f"[Image Save] Error: {str(e)}")
            
            else:
                result = {"error": f"Unknown command: {cmd}"}
            
            # Send response with Length-Prefix
            response_data = pickle.dumps(result)
            conn.sendall(struct.pack('>I', len(response_data)) + response_data)
            
        except Exception as e:
            import traceback
            logger.error(f"Error handling request: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            try:
                # Send error with length-prefix
                err_data = pickle.dumps({"error": str(e)})
                conn.sendall(struct.pack('>I', len(err_data)) + err_data)
            except:
                pass
        finally:
            conn.close()
    
    def run(self):
        """Main server loop"""
        # Start worker pools
        self.start_pools()
        
        # Start WebSocket server for monitoring
        self.ws_server = WebSocketServer(self, DAEMON_HOST, DAEMON_WS_PORT)
        self.ws_server.start()
        
        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((DAEMON_HOST, self.port))
            server.listen(MAX_WORKERS * 2)
            
            logger.info(f"Socket server (work): {DAEMON_HOST}:{self.port}")
            logger.info(f"WebSocket server (monitoring): ws://{DAEMON_HOST}:{DAEMON_WS_PORT}")
            logger.info(f"Service type: {self.service_type.value}")
            
            if self.vae_pool:
                logger.info(f"  VAE: {self.config.min_vae_workers}-{self.config.max_vae_workers} workers")
            if self.clip_pool:
                logger.info(f"  CLIP: {self.config.min_clip_workers}-{self.config.max_clip_workers} workers")
            logger.info(f"  Idle timeout: {self.config.idle_timeout_sec}s")
            logger.info("Ready to accept connections!")
            logger.info("Press Ctrl+C to stop")
            
            # Set socket to non-blocking so we can check shutdown flag
            server.settimeout(1.0)
            
            while not self.shutdown_requested:
                try:
                    conn, addr = server.accept()
                    thread = threading.Thread(
                        target=self.handle_request,
                        args=(conn, addr),
                        daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    # Check shutdown flag
                    continue
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.ws_server:
                self.ws_server.stop()
            if self.vae_pool:
                self.vae_pool.stop()
            if self.clip_pool:
                self.clip_pool.stop()
            if self.save_pool:
                self.save_pool.stop()
            server.close()


def configure_attention_mode():
    """Configure ComfyUI attention mechanism based on ATTENTION_MODE setting.
    
    Call this before creating the DynamicDaemon to ensure the correct attention
    mechanism is used (sage, xformers, pytorch, etc.).
    
    Performs live detection of ComfyUI's attention mode when ATTENTION_MODE is "auto".
    
    Returns:
        bool: True if attention mode was successfully configured, False otherwise
    """
    try:
        import comfy.model_management as mm  # type: ignore
        
        # Detect current mode from ComfyUI (via subprocess, detection functions may not work)
        # The LUNA_ATTENTION_MODE env var is set by the parent ComfyUI process
        detected_mode = "pytorch"  # Default fallback
        
        # If auto mode, use whatever was passed via environment variable (already in ATTENTION_MODE)
        if ATTENTION_MODE == "auto":
            logger.info(f"[Auto-detect] Luna daemon using: auto (will match ComfyUI)")
            return True
        
        # For non-auto modes, just log and proceed
        logger.info(f"[Config] Luna daemon configured for: {ATTENTION_MODE} attention")
        logger.info(f"[Config] Environment LUNA_ATTENTION_MODE was set by parent ComfyUI process")
        
        # No need to override - the config already has the right mode from env var
        return True
    except Exception as e:
        logger.warning(f"Failed to configure attention mode: {e}")
        return False


def main():
    """Entry point with CLI argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Luna VAE/CLIP Daemon v2.1")
    parser.add_argument(
        "--service-type", "-t",
        choices=["full", "clip", "vae"],
        default="full",
        help="Service type: full (both), clip (CLIP only), vae (VAE only)"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help=f"CUDA device (default: {SHARED_DEVICE} for clip, {VAE_DEVICE} for vae)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help=f"Port to listen on (default: {DAEMON_PORT} for clip, {DAEMON_VAE_PORT} for vae)"
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help=f"Model precision for all models (overrides --clip-precision and --vae-precision)"
    )
    parser.add_argument(
        "--clip-precision",
        choices=["bf16", "fp16", "fp32"],
        default=CLIP_PRECISION,
        help=f"CLIP precision (default: {CLIP_PRECISION})"
    )
    parser.add_argument(
        "--vae-precision",
        choices=["bf16", "fp16", "fp32"],
        default=VAE_PRECISION,
        help=f"VAE precision (default: {VAE_PRECISION}). bf16 recommended to avoid fp16 NaN issues."
    )
    
    args = parser.parse_args()
    
    # Map string to ServiceType
    service_map = {
        "full": ServiceType.FULL,
        "clip": ServiceType.CLIP_ONLY,
        "vae": ServiceType.VAE_ONLY
    }
    service_type = service_map[args.service_type]
    
    # Set defaults based on service type
    if args.device is None:
        device = VAE_DEVICE if service_type == ServiceType.VAE_ONLY else SHARED_DEVICE
    else:
        device = args.device
    
    if args.port is None:
        port = DAEMON_VAE_PORT if service_type == ServiceType.VAE_ONLY else DAEMON_PORT
    else:
        port = args.port
    
    # Handle precision - unified --precision overrides individual settings
    clip_precision = args.precision if args.precision else args.clip_precision
    vae_precision = args.precision if args.precision else args.vae_precision
    
    # Apply attention mode configuration
    configure_attention_mode()
    
    # Print banner
    print("=" * 60)
    print("  Luna VAE/CLIP Daemon v2.1 - Split Architecture")
    print("=" * 60)
    print(f"  Service Type: {service_type.value.upper()}")
    print(f"  Device: {device}")
    print(f"  CLIP Precision: {clip_precision}")
    print(f"  VAE Precision: {vae_precision}")
    print(f"  Attention Mode: {ATTENTION_MODE}")
    print(f"  Socket: {DAEMON_HOST}:{port}")
    print(f"  WebSocket: ws://{DAEMON_HOST}:{DAEMON_WS_PORT}")
    print("=" * 60)
    print()
    
    daemon = DynamicDaemon(
        device=device,
        clip_precision=clip_precision,
        vae_precision=vae_precision,
        service_type=service_type,
        port=port
    )
    daemon.run()


if __name__ == "__main__":
    main()
