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
import asyncio
import hashlib
import base64
import struct
from typing import Any, Dict, Tuple, Optional, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

# Add ComfyUI to path if needed
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

# Import folder_paths for LoRA disk loading
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False
    folder_paths = None
    
# Import safetensors for LoRA loading
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    load_safetensors = None

# Add daemon folder to path for direct script execution
daemon_path = os.path.dirname(os.path.abspath(__file__))
if daemon_path not in sys.path:
    sys.path.insert(0, daemon_path)

# Try relative import first (when used as module), fall back to direct import
try:
    from .config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_VAE_PORT, DAEMON_WS_PORT, SHARED_DEVICE, VAE_DEVICE,
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
except ImportError:
    from config import (
        DAEMON_HOST, DAEMON_PORT, DAEMON_VAE_PORT, DAEMON_WS_PORT, SHARED_DEVICE, VAE_DEVICE,
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

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [Daemon] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
        if not HAS_FOLDER_PATHS or not HAS_SAFETENSORS:
            logger.error("[LoRARegistry] folder_paths or safetensors not available")
            return None
        
        # Find the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            # Try with/without extensions
            base_name = lora_name
            for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin'):
                if base_name.lower().endswith(ext):
                    base_name = base_name[:-len(ext)]
                    break
            
            # Try common extensions
            for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin'):
                test_name = base_name + ext
                lora_path = folder_paths.get_full_path("loras", test_name)
                if lora_path and os.path.exists(lora_path):
                    break
            else:
                logger.warning(f"[LoRARegistry] LoRA not found: {lora_name}")
                return None
        
        logger.info(f"[LoRARegistry] Loading LoRA from disk: {os.path.basename(lora_path)}")
        
        try:
            # Load the full LoRA file
            all_weights = load_safetensors(lora_path)
            
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
                    self.original_weights[weight_key] = (target_layer, weight.data.clone())
                
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
                layer.weight.data.copy_(original)
        
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
        lora_registry: Optional[LoRARegistry] = None
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.lora_registry = lora_registry  # For CLIP LoRA injection (F-150)
        
        self.model = None
        self.is_running = False
        self.is_loaded = False
        self.last_active = time.time()
        self.request_count = 0
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # For CLIP, we also need paths
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
    
    def load_model(self):
        """Load the model for this worker"""
        import comfy.sd
        import comfy.utils
        
        if self.worker_type == WorkerType.VAE:
            logger.info(f"[VAE-{self.worker_id}] Loading VAE model...")
            sd = comfy.utils.load_torch_file(VAE_PATH)
            if self.precision != "fp32":
                sd = self._convert_state_dict_precision(sd)
            self.model = comfy.sd.VAE(sd=sd)
            logger.info(f"[VAE-{self.worker_id}] VAE loaded ({self.precision})")
            
        elif self.worker_type == WorkerType.CLIP:
            logger.info(f"[CLIP-{self.worker_id}] Loading CLIP model...")
            clip_paths = []
            if os.path.exists(self.clip_l_path):
                clip_paths.append(self.clip_l_path)
            if os.path.exists(self.clip_g_path):
                clip_paths.append(self.clip_g_path)
            
            emb_dir = self.embeddings_dir if os.path.exists(self.embeddings_dir) else None
            self.model = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=emb_dir
            )
            
            # Convert to target precision
            if self.precision != "fp32" and hasattr(self.model, 'cond_stage_model'):
                self.model.cond_stage_model.to(self.dtype)
            
            logger.info(f"[CLIP-{self.worker_id}] CLIP loaded ({self.precision})")
        
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
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        
        if tiled:
            latents = self._encode_tiled(pixels, tile_size, overlap)
        else:
            try:
                latents = self.model.encode(pixels)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"[VAE] OOM during encode, falling back to tiled mode")
                    torch.cuda.empty_cache()
                    latents = self._encode_tiled(pixels, tile_size, overlap)
                else:
                    raise
        return latents.cpu()
    
    def _encode_tiled(self, pixels: torch.Tensor, tile_size: int = 512, 
                      overlap: int = 64) -> torch.Tensor:
        """
        Tiled VAE encoding for large images.
        
        Uses multiple tile configurations and averages results for better seam handling.
        """
        import comfy.utils
        
        # Get model properties
        downscale = getattr(self.model, 'downscale_ratio', 8)
        if callable(downscale):
            downscale = 8  # Default for SDXL
        latent_channels = getattr(self.model, 'latent_channels', 4)
        
        # Prepare the encoding function
        def encode_fn(a):
            # Process input if the model has this method
            if hasattr(self.model, 'process_input'):
                a = self.model.process_input(a)
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.encode(a.to(vae_dtype).to(device)).float()
        
        # Average multiple tile configurations for better seams
        output_device = torch.device('cpu')
        
        samples = comfy.utils.tiled_scale(
            pixels, encode_fn, tile_size, tile_size, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples += comfy.utils.tiled_scale(
            pixels, encode_fn, tile_size * 2, tile_size // 2, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples += comfy.utils.tiled_scale(
            pixels, encode_fn, tile_size // 2, tile_size * 2, overlap,
            upscale_amount=(1.0/downscale), out_channels=latent_channels,
            output_device=output_device
        )
        samples /= 3.0
        
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
        if tiled:
            pixels = self._decode_tiled(latents, tile_size, overlap)
        else:
            try:
                pixels = self.model.decode(latents)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"[VAE] OOM during decode, falling back to tiled mode")
                    torch.cuda.empty_cache()
                    pixels = self._decode_tiled(latents, tile_size, overlap)
                else:
                    raise
        return pixels.cpu()
    
    def _decode_tiled(self, latents: torch.Tensor, tile_size: int = 64,
                      overlap: int = 16) -> torch.Tensor:
        """
        Tiled VAE decoding for large latents.
        
        Uses multiple tile configurations and averages results for better seam handling.
        """
        import comfy.utils
        
        # Get model properties
        upscale = getattr(self.model, 'upscale_ratio', 8)
        if callable(upscale):
            upscale = 8  # Default for SDXL
        
        # Prepare the decoding function
        def decode_fn(a):
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.decode(a.to(vae_dtype).to(device)).float()
        
        output_device = torch.device('cpu')
        
        # Average multiple tile configurations for better seams
        pixels = comfy.utils.tiled_scale(
            latents, decode_fn, tile_size // 2, tile_size * 2, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        pixels += comfy.utils.tiled_scale(
            latents, decode_fn, tile_size * 2, tile_size // 2, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        pixels += comfy.utils.tiled_scale(
            latents, decode_fn, tile_size, tile_size, overlap,
            upscale_amount=upscale, output_device=output_device
        )
        pixels /= 3.0
        
        # Process output if model has the method
        if hasattr(self.model, 'process_output'):
            pixels = self.model.process_output(pixels)
        
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
                
                return (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
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
                
                return (positive_out, negative_out)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
    
    def run(self):
        """Main worker loop - process requests from queue"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Wait for request with timeout (allows checking is_running)
                try:
                    request_id, cmd, data = self.request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.last_active = time.time()
                self.request_count += 1
                
                try:
                    # Process based on command type
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
                    
                except Exception as e:
                    logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Error: {e}")
                    result = {"error": str(e)}
                
                # Send result back
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker error: {e}")
    
    def start(self):
        """Start the worker thread"""
        if not self.is_loaded:
            self.load_model()
        
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
        lora_registry: Optional[LoRARegistry] = None
    ):
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.config = config
        self.on_scale_event = on_scale_event  # Callback for scaling events
        self.lora_registry = lora_registry  # For CLIP LoRA support (F-150)
        
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
                lora_registry=self.lora_registry  # F-150: pass registry to CLIP workers
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
            if queue_depth > self.config.queue_threshold:
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
        
        # Wait for result
        try:
            result = self.result_queues[request_id].get(timeout=60.0)
        finally:
            with self.lock:
                del self.result_queues[request_id]
        
        return result
    
    def start(self):
        """Start the pool with minimum workers"""
        self._running = True
        
        # Start minimum workers
        for _ in range(self.min_workers):
            self.scale_up()
        
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
        logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
    
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
        device: str = SHARED_DEVICE, 
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
        
        # LoRA Registry (F-150) - only needed for CLIP
        if service_type in (ServiceType.FULL, ServiceType.CLIP_ONLY):
            self.lora_registry = LoRARegistry(max_size_mb=2048.0, device=device)
        else:
            self.lora_registry = None
        
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
        self.ws_server: Optional[WebSocketServer] = None
        
        self.start_time = time.time()
        self.request_count = 0
        
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
        logger.info(f"Starting worker pools on {self.device}...")
        logger.info(f"  CLIP precision: {self.clip_precision}, VAE precision: {self.vae_precision}")
        
        # VAE pool - only for FULL or VAE_ONLY modes
        if self.service_type in (ServiceType.FULL, ServiceType.VAE_ONLY):
            self.vae_pool = WorkerPool(
                worker_type=WorkerType.VAE,
                device=self.device,
                precision=self.vae_precision,
                config=self.config,
                on_scale_event=self._on_scale_event
            )
            logger.info(f"VAE pool configured ({self.vae_precision})")
        
        # CLIP pool - only for FULL or CLIP_ONLY modes
        if self.service_type in (ServiceType.FULL, ServiceType.CLIP_ONLY):
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=self.device,
                precision=self.clip_precision,
                config=self.config,
                on_scale_event=self._on_scale_event,
                lora_registry=self.lora_registry  # F-150: enable LoRA for CLIP workers
            )
            logger.info(f"CLIP pool configured ({self.clip_precision})")
        
        # Start pools (CLIP first if present, as it's larger)
        if self.clip_pool:
            self.clip_pool.start()
        if self.vae_pool:
            self.vae_pool.start()
        
        # Report VRAM usage
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            used = torch.cuda.memory_allocated(device_idx) / 1024**3
            total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
            logger.info(f"Initial VRAM usage: {used:.2f} / {total:.2f} GB")
    
    def get_info(self) -> dict:
        """Get daemon status info"""
        info = {
            "status": "ok",
            "version": "2.1-split",
            "service_type": self.service_type.value,
            "device": self.device,
            "precision": self.precision,
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_count,
        }
        
        if self.vae_pool:
            info["vae_pool"] = self.vae_pool.get_stats()
        if self.clip_pool:
            info["clip_pool"] = self.clip_pool.get_stats()
        
        # Add LoRA registry stats
        if self.lora_registry:
            info["lora_registry"] = self.lora_registry.get_stats()
        
        if 'cuda' in self.device:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            info["vram_used_gb"] = torch.cuda.memory_allocated(device_idx) / 1024**3
            info["vram_total_gb"] = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        
        return info
    
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
            
            # Route command
            if cmd == "health":
                result = {"status": "ok", "service_type": self.service_type.value}
            elif cmd == "info":
                result = self.get_info()
            
            # VAE commands - only available in FULL or VAE_ONLY mode
            elif cmd in ("vae_encode", "vae_decode"):
                if self.vae_pool is None:
                    result = {"error": f"VAE not available in {self.service_type.value} mode"}
                else:
                    result = self.vae_pool.submit(cmd, request)
            
            # CLIP commands - only available in FULL or CLIP_ONLY mode
            elif cmd in ("clip_encode", "clip_encode_sdxl"):
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
                    weights = request.get("weights", {})
                    success = self.lora_registry.put(lora_hash, weights)
                    result = {"success": success, "hash": lora_hash}
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
            
            else:
                result = {"error": f"Unknown command: {cmd}"}
            
            # Send response with Length-Prefix
            response_data = pickle.dumps(result)
            conn.sendall(struct.pack('>I', len(response_data)) + response_data)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
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
            
            logger.info(f"Socket server: {DAEMON_HOST}:{self.port}")
            logger.info(f"WebSocket monitor: ws://{DAEMON_HOST}:{DAEMON_WS_PORT}")
            logger.info(f"Service type: {self.service_type.value}")
            
            if self.vae_pool:
                logger.info(f"  VAE: {self.config.min_vae_workers}-{self.config.max_vae_workers} workers")
            if self.clip_pool:
                logger.info(f"  CLIP: {self.config.min_clip_workers}-{self.config.max_clip_workers} workers")
            logger.info(f"  Idle timeout: {self.config.idle_timeout_sec}s")
            logger.info("Ready to accept connections!")
            logger.info("Press Ctrl+C to stop")
            
            while True:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=self.handle_request,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.ws_server:
                self.ws_server.stop()
            if self.vae_pool:
                self.vae_pool.stop()
            if self.clip_pool:
                self.clip_pool.stop()
            server.close()


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
    
    # Print banner
    print("=" * 60)
    print("  Luna VAE/CLIP Daemon v2.1 - Split Architecture")
    print("=" * 60)
    print(f"  Service Type: {service_type.value.upper()}")
    print(f"  Device: {device}")
    print(f"  CLIP Precision: {clip_precision}")
    print(f"  VAE Precision: {vae_precision}")
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
