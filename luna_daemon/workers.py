"""
Luna Daemon Workers - VAE and CLIP Worker Pools

Provides dynamic scaling worker pools for VAE and CLIP operations.
Workers are spawned/despawned based on queue depth and available VRAM.

Architecture:
- WorkerPool: Manages a pool of ModelWorker instances
- ModelWorker: Individual worker that holds a model and processes requests
- Dynamic scaling: Scale up when queue backs up, scale down when idle
- Lazy loading: Workers don't load models until first request
"""

import os
import gc
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import torch

# Try relative import first, fallback to direct
try:
    from .core import logger
except (ImportError, ValueError):
    # Fallback: load core.py directly
    import importlib.util
    core_path = os.path.join(os.path.dirname(__file__), "core.py")
    spec = importlib.util.spec_from_file_location("luna_daemon_core", core_path)
    if spec and spec.loader:
        core_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_mod)
        logger = core_mod.logger
    else:
        import logging
        logger = logging.getLogger("LunaWorkers")

# Import folder_paths for model resolution
try:
    import folder_paths  # type: ignore
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False
    folder_paths = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScalingConfig:
    """Configuration for worker pool scaling behavior."""
    # VAE workers
    min_vae_workers: int = 0  # Lazy loading: start with 0
    max_vae_workers: int = 2
    vae_size_gb: float = 0.5  # VAE model size in GB
    
    # CLIP workers  
    min_clip_workers: int = 0  # Lazy loading: start with 0
    max_clip_workers: int = 2
    clip_size_gb: float = 1.5  # CLIP model size in GB
    
    # VRAM management
    vram_safety_margin_gb: float = 2.0
    
    # Scaling thresholds
    queue_threshold: int = 2  # Scale up if queue depth exceeds this
    scale_up_delay_sec: float = 1.0  # Wait before scaling up
    idle_timeout_sec: float = 30.0  # Scale down after idle this long
    scaling_check_interval_sec: float = 0.25


# =============================================================================
# Worker Type Enum
# =============================================================================

class WorkerType(Enum):
    """Types of workers in the pool."""
    VAE = "vae"
    CLIP = "clip"
    IMAGE_SAVE = "image_save"


# =============================================================================
# VRAM Monitoring
# =============================================================================

def log_all_gpu_vram(prefix: str = ""):
    """Log VRAM usage for all available GPUs."""
    if not torch.cuda.is_available():
        return
    
    gpu_count = torch.cuda.device_count()
    vram_info = []
    
    for gpu_id in range(gpu_count):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            used_gb = (total_mem - free_mem) / 1024**3
            total_gb = total_mem / 1024**3
            percent = round((used_gb / total_gb) * 100, 1)
            vram_info.append(f"GPU{gpu_id}: {used_gb:.2f}/{total_gb:.1f}GB ({percent}%)")
        except Exception:
            vram_info.append(f"GPU{gpu_id}: ERROR")
    
    logger.debug(f"[VRAM] {prefix}{' | '.join(vram_info)}")


def get_available_vram_gb(device: str) -> float:
    """Get available VRAM in GB for a device."""
    if 'cuda' not in device:
        return float('inf')
    
    device_idx = int(device.split(':')[1]) if ':' in device else 0
    total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
    used = torch.cuda.memory_allocated(device_idx) / 1024**3
    return total - used


# =============================================================================
# Model Worker
# =============================================================================

class ModelWorker:
    """
    A single worker that holds a model and processes requests.
    
    Supports lazy loading - model is loaded on first request, not at startup.
    """
    
    def __init__(
        self,
        worker_id: int,
        worker_type: WorkerType,
        device: str,
        precision: str,
        request_queue: queue.Queue,
        result_queues: Dict[int, queue.Queue],
        model_registry: Optional[Any] = None,
        lora_registry: Optional[Any] = None,
        config_paths: Optional[Dict[str, str]] = None
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.model_registry = model_registry
        self.lora_registry = lora_registry
        self.config_paths = config_paths or {}
        
        self.model: Any = None
        self.is_running = False
        self.is_loaded = False
        self.last_active = time.time()
        self.request_count = 0
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    @property
    def dtype(self) -> torch.dtype:
        """Get torch dtype from precision string."""
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        return torch.float32
    
    def _resolve_path(self, path_or_name: str, type_name: str) -> Optional[str]:
        """Resolve a path or filename to a full path."""
        if not path_or_name:
            return None
            
        if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
            return path_or_name
            
        if HAS_FOLDER_PATHS:
            try:
                full_path = folder_paths.get_full_path(type_name, path_or_name)
                if full_path:
                    return full_path
            except:
                pass
                
        return None
    
    def _convert_state_dict_precision(self, sd: dict) -> dict:
        """Convert state dict tensors to target precision."""
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
        """Load the model for this worker."""
        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore
        
        if self.worker_type == WorkerType.VAE:
            logger.info(f"[VAE-{self.worker_id}] Loading VAE model...")
            
            if self.model_registry and hasattr(self.model_registry, 'has_vae') and self.model_registry.has_vae():
                self.model = self.model_registry.get_vae_model(self.precision)
                logger.info(f"[VAE-{self.worker_id}] VAE loaded from registry ({self.precision})")
            else:
                vae_path = self._resolve_path(self.config_paths.get('vae', ''), "vae")
                if not vae_path:
                    raise RuntimeError(f"VAE model not found")
                    
                sd = comfy.utils.load_torch_file(vae_path)
                if self.precision != "fp32":
                    sd = self._convert_state_dict_precision(sd)
                self.model = comfy.sd.VAE(sd=sd)
                logger.info(f"[VAE-{self.worker_id}] VAE loaded from path ({self.precision})")
            
        elif self.worker_type == WorkerType.CLIP:
            logger.info(f"[CLIP-{self.worker_id}] Loading CLIP model...")
            
            if self.model_registry and hasattr(self.model_registry, 'has_clip') and self.model_registry.has_clip():
                self.model = self.model_registry.get_clip_model(self.precision)
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from registry ({self.precision})")
            else:
                clip_paths = []
                for key in ['clip_l', 'clip_g']:
                    path = self._resolve_path(self.config_paths.get(key, ''), "clip")
                    if path:
                        clip_paths.append(path)
                
                if not clip_paths:
                    raise RuntimeError("No CLIP models found")
                
                clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
                emb_dir = self.config_paths.get('embeddings')
                if emb_dir and not os.path.exists(emb_dir):
                    emb_dir = None
                    
                self.model = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=emb_dir,
                    clip_type=clip_type
                )
                
                if self.precision != "fp32" and hasattr(self.model, 'cond_stage_model'):
                    self.model.cond_stage_model.to(self.dtype)
                
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from path ({self.precision})")
        
        self.is_loaded = True
        torch.cuda.empty_cache()
    
    def unload_model(self):
        """Unload the model to free VRAM."""
        if self.model is not None:
            logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Unloading model...")
            del self.model
            self.model = None
            self.is_loaded = False
            torch.cuda.empty_cache()
    
    def _cleanup_gpu_memory(self):
        """Aggressive GPU memory cleanup after operations."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        if self.model is not None and hasattr(self.model, 'cache_clear'):
            self.model.cache_clear()
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    
    # =========================================================================
    # VAE Operations
    # =========================================================================
    
    def process_vae_encode(
        self, 
        pixels: torch.Tensor, 
        tiled: bool = False,
        tile_size: int = 512, 
        overlap: int = 64
    ) -> torch.Tensor:
        """Encode image pixels to latent space."""
        assert self.model is not None, "VAE model not loaded"
        
        with torch.inference_mode():
            pixels = pixels.detach()
            
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
            
            del pixels
            latents = latents.detach()
            result = latents.cpu()
        
        del latents
        self._cleanup_gpu_memory()
        return result
    
    def _encode_tiled(
        self, 
        pixels: torch.Tensor, 
        tile_size: int = 512, 
        overlap: int = 64
    ) -> torch.Tensor:
        """Tiled VAE encoding for large images."""
        import comfy.utils  # type: ignore
        
        downscale = getattr(self.model, 'downscale_ratio', 8)
        if callable(downscale):
            downscale = 8
        latent_channels = getattr(self.model, 'latent_channels', 4)
        
        def encode_fn(a):
            if hasattr(self.model, 'process_input'):
                a = self.model.process_input(a)
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.encode(a.to(vae_dtype).to(device)).float()
        
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
        
        torch.cuda.empty_cache()
        return samples
    
    def process_vae_decode(
        self, 
        latents: torch.Tensor, 
        tiled: bool = False,
        tile_size: int = 64, 
        overlap: int = 16
    ) -> torch.Tensor:
        """Decode latent space to image pixels."""
        assert self.model is not None, "VAE model not loaded"
        
        logger.info(f"[VAE-{self.worker_id}] Starting decode (tiled={tiled}, shape={latents.shape})")
        
        with torch.inference_mode():
            latents = latents.detach()
            
            if tiled:
                pixels = self._decode_tiled(latents, tile_size, overlap)
            else:
                try:
                    pixels = self.model.decode(latents)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"[VAE-{self.worker_id}] OOM during decode, falling back to tiled mode")
                        torch.cuda.empty_cache()
                        pixels = self._decode_tiled(latents, tile_size, overlap)
                    else:
                        raise
        
        del latents
        pixels = pixels.detach()
        result = pixels.cpu()
        
        del pixels
        self._cleanup_gpu_memory()
        
        logger.info(f"[VAE-{self.worker_id}] Decode complete")
        return result
    
    def _decode_tiled(
        self, 
        latents: torch.Tensor, 
        tile_size: int = 64,
        overlap: int = 16
    ) -> torch.Tensor:
        """Tiled VAE decoding for large latents."""
        import comfy.utils  # type: ignore
        
        upscale = getattr(self.model, 'upscale_ratio', 8)
        if callable(upscale):
            upscale = 8
        
        def decode_fn(a):
            vae_dtype = getattr(self.model, 'vae_dtype', torch.float32)
            device = next(self.model.first_stage_model.parameters()).device
            return self.model.first_stage_model.decode(a.to(vae_dtype).to(device)).float()
        
        output_device = torch.device('cpu')
        
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
        
        if hasattr(self.model, 'process_output'):
            pixels = self.model.process_output(pixels)
        
        torch.cuda.empty_cache()
        return pixels
    
    # =========================================================================
    # CLIP Operations
    # =========================================================================
    
    def process_clip_encode(
        self, 
        positive: str, 
        negative: str = "",
        lora_stack: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple:
        """Encode text to CLIP conditioning."""
        assert self.model is not None, "CLIP model not loaded"
        
        with self.lock:
            # TODO: Apply LoRAs if provided (lora_stack + lora_registry)
            
            tokens_pos = self.model.tokenize(positive)
            cond, pooled = self.model.encode_from_tokens(tokens_pos, return_pooled=True)
            
            tokens_neg = self.model.tokenize(negative if negative else "")
            uncond, pooled_neg = self.model.encode_from_tokens(tokens_neg, return_pooled=True)
            
            result = (cond.cpu(), pooled.cpu(), uncond.cpu(), pooled_neg.cpu())
            
            del cond, pooled, uncond, pooled_neg
            self._cleanup_gpu_memory()
            
            return result
    
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
        """Encode text with SDXL-specific size conditioning."""
        assert self.model is not None, "CLIP model not loaded"
        
        with self.lock:
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
            
            del cond, pooled, uncond, pooled_neg
            self._cleanup_gpu_memory()
            
            return (positive_out, negative_out)
    
    # =========================================================================
    # Worker Thread
    # =========================================================================
    
    def run(self):
        """Main worker loop - process requests from queue."""
        self.is_running = True
        self.is_loaded = False
        logger.info(f"[{self.worker_type.name}-{self.worker_id}] Worker started (lazy loading)")
        
        while self.is_running:
            try:
                try:
                    request_id, cmd, data = self.request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.last_active = time.time()
                self.request_count += 1
                
                # Lazy load model on first request
                if not self.is_loaded:
                    try:
                        self.load_model()
                    except Exception as e:
                        if request_id in self.result_queues:
                            self.result_queues[request_id].put({"error": f"Model not loaded: {e}"})
                        continue
                
                try:
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Processing {cmd}")
                    
                    result: Any = {}
                    
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
                    
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Completed {cmd}")
                    
                except Exception as e:
                    logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    result = {"error": str(e)}
                
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(result)
                
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker error: {e}")
    
    def start(self):
        """Start the worker thread."""
        self.thread = threading.Thread(
            target=self.run,
            name=f"{self.worker_type.value}-worker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker started")
    
    def stop(self):
        """Stop the worker thread and unload model."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.unload_model()
        logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Worker stopped")


# =============================================================================
# Worker Pool
# =============================================================================

class WorkerPool:
    """
    Manages a pool of workers with dynamic scaling.
    
    Scales up when queue backs up, scales down when workers are idle.
    VRAM-aware: won't scale beyond available memory.
    """
    
    def __init__(
        self,
        worker_type: WorkerType,
        device: str,
        precision: str,
        config: ScalingConfig,
        on_scale_event: Optional[Callable[[str, dict], None]] = None,
        model_registry: Optional[Any] = None,
        lora_registry: Optional[Any] = None,
        config_paths: Optional[Dict[str, str]] = None
    ):
        self.worker_type = worker_type
        self.device = device
        self.precision = precision
        self.config = config
        self.on_scale_event = on_scale_event
        self.model_registry = model_registry
        self.lora_registry = lora_registry
        self.config_paths = config_paths or {}
        
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
    
    def can_scale_up(self) -> bool:
        """Check if we can add another worker."""
        if len(self.workers) >= self.max_workers:
            return False
        
        available = get_available_vram_gb(self.device)
        needed = self.model_size_gb + self.config.vram_safety_margin_gb
        return available >= needed
    
    def scale_up(self) -> Optional[ModelWorker]:
        """Add a new worker to the pool."""
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
                model_registry=self.model_registry,
                lora_registry=self.lora_registry,
                config_paths=self.config_paths
            )
            
            worker.start()
            self.workers.append(worker)
            
            vram_available = get_available_vram_gb(self.device)
            logger.info(
                f"[{self.worker_type.value.upper()}] Scaled UP to {len(self.workers)} workers "
                f"(VRAM: {vram_available:.2f} GB)"
            )
            
            if self.on_scale_event:
                self.on_scale_event("scale_up", {
                    "pool": self.worker_type.value,
                    "worker_id": worker_id,
                    "active_workers": len(self.workers),
                    "vram_available_gb": round(vram_available, 2)
                })
            
            return worker
    
    def scale_down(self, worker: ModelWorker):
        """Remove an idle worker from the pool."""
        with self.lock:
            if len(self.workers) <= self.min_workers:
                return
            
            if worker in self.workers:
                worker_id = worker.worker_id
                worker.stop()
                self.workers.remove(worker)
                
                vram_available = get_available_vram_gb(self.device)
                logger.info(
                    f"[{self.worker_type.value.upper()}] Scaled DOWN to {len(self.workers)} workers "
                    f"(VRAM: {vram_available:.2f} GB)"
                )
                
                if self.on_scale_event:
                    self.on_scale_event("scale_down", {
                        "pool": self.worker_type.value,
                        "worker_id": worker_id,
                        "active_workers": len(self.workers),
                        "vram_available_gb": round(vram_available, 2)
                    })
    
    def _scaling_loop(self):
        """Background thread that monitors and scales workers."""
        last_scale_up_check = 0
        queue_was_backed_up = False
        
        while self._running:
            time.sleep(self.config.scaling_check_interval_sec)
            
            now = time.time()
            queue_depth = self.request_queue.qsize()
            
            # Scale UP: Immediate if queue has items but no workers (lazy loading)
            if queue_depth > 0 and len(self.workers) == 0:
                if self.can_scale_up():
                    logger.info(f"[{self.worker_type.value.upper()}] Lazy loading triggered")
                    self.scale_up()
                    last_scale_up_check = now
            
            # Scale UP: Standard queue-based scaling
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
            
            # Scale DOWN: Find idle workers
            with self.lock:
                idle_workers = [
                    w for w in self.workers
                    if now - w.last_active > self.config.idle_timeout_sec
                ]
            
            for worker in idle_workers:
                if len(self.workers) > self.min_workers:
                    self.scale_down(worker)
                    break  # One at a time
    
    def submit(self, cmd: str, data: dict) -> Any:
        """Submit a request and wait for result."""
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            self.result_queues[request_id] = queue.Queue()
        
        self.request_queue.put((request_id, cmd, data))
        
        # Longer timeout for VAE operations
        timeout = 180.0 if cmd in ["vae_encode", "vae_decode"] else 60.0
        try:
            result = self.result_queues[request_id].get(timeout=timeout)
        finally:
            with self.lock:
                del self.result_queues[request_id]
        
        return result
    
    def start(self):
        """Start the pool (lazy loading - no workers created initially)."""
        self._running = True
        
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            name=f"{self.worker_type.value}-scaler",
            daemon=True
        )
        self._scaling_thread.start()
    
    def stop(self):
        """Stop all workers."""
        self._running = False
        
        with self.lock:
            for worker in self.workers:
                worker.stop()
            self.workers.clear()
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.lock:
            return {
                "type": self.worker_type.value,
                "active_workers": len(self.workers),
                "queue_depth": self.request_queue.qsize(),
                "total_requests": sum(w.request_count for w in self.workers),
                "worker_ids": [w.worker_id for w in self.workers],
            }
