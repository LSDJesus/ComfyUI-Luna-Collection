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
    CLIP_VISION = "clip_vision"  # CLIP-ViT image encoder
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
        config_paths: Optional[Dict[str, Optional[str]]] = None
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
        
        # Track what models this worker has loaded
        self.loaded_model_paths: Dict[str, str] = {}  # {component_name: full_path}
        
        self.model: Any = None
        self.qwen3_encoder: Any = None  # For Z-IMAGE/VLM functionality
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
                full_path = folder_paths.get_full_path(type_name, path_or_name)  # type: ignore
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
        
        # Force attention mode if set by parent process
        if "LUNA_ATTENTION_MODE" in os.environ:
            try:
                import comfy.model_management as mm
                attention_mode = os.environ["LUNA_ATTENTION_MODE"]
                # Force the attention mode
                if attention_mode == "sage":
                    mm.attention_function = mm.attention_sage_masked
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Forced attention mode: sage")
                elif attention_mode == "sdp":
                    mm.attention_function = mm.attention_pytorch
                    logger.info(f"[{self.worker_type.value.upper()}-{self.worker_id}] Forced attention mode: pytorch")
            except Exception as e:
                logger.warning(f"[{self.worker_type.value.upper()}-{self.worker_id}] Could not set attention mode: {e}")
        
        if self.worker_type == WorkerType.VAE:
            logger.info(f"[VAE-{self.worker_id}] Loading VAE model...")
            
            if self.model_registry and hasattr(self.model_registry, 'has_vae') and self.model_registry.has_vae():
                self.model = self.model_registry.get_vae_model(self.precision)
                logger.info(f"[VAE-{self.worker_id}] VAE loaded from registry ({self.precision})")
            else:
                vae_path = self._resolve_path(self.config_paths.get('vae') or '', "vae")
                
                if not vae_path:
                    raise RuntimeError(
                        "VAE model not configured. "
                        "Use load_vae_model() to specify which VAE to load, or set VAE_PATH in config.py"
                    )
                    
                sd = comfy.utils.load_torch_file(vae_path)
                if self.precision != "fp32":
                    sd = self._convert_state_dict_precision(sd)
                
                # Create VAE and explicitly move to target device
                self.model = comfy.sd.VAE(sd=sd)
                if hasattr(self.model, 'first_stage_model'):
                    self.model.first_stage_model.to(self.device)
                
                # Track loaded model
                self.loaded_model_paths['vae'] = vae_path
                logger.info(f"[VAE-{self.worker_id}] VAE loaded: {vae_path} ({self.precision})")
            
        elif self.worker_type == WorkerType.CLIP:
            logger.info(f"[CLIP-{self.worker_id}] Loading CLIP model...")
            
            if self.model_registry and hasattr(self.model_registry, 'has_clip') and self.model_registry.has_clip():
                self.model = self.model_registry.get_clip_model(self.precision)
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from registry ({self.precision})")
            else:
                clip_paths = []
                
                # Try configured paths
                for key in ['clip_l', 'clip_g']:
                    path = self._resolve_path(self.config_paths.get(key) or '', "clip")
                    if path:
                        clip_paths.append(path)
                
                if not clip_paths:
                    raise RuntimeError(
                        "CLIP model not configured. "
                        "Use load_clip_model() to specify which CLIP to load, or set CLIP_L_PATH/CLIP_G_PATH in config.py"
                    )
                
                clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
                emb_dir = self.config_paths.get('embeddings')
                if emb_dir and not os.path.exists(emb_dir):
                    emb_dir = None
                
                # Force device context before loading
                # ComfyUI's load_clip() will use the current default device
                import comfy.model_management
                old_device = comfy.model_management.get_torch_device()
                try:
                    # Temporarily set default device to our target device
                    comfy.model_management.set_vram_to = self.device
                    
                    self.model = comfy.sd.load_clip(
                        ckpt_paths=clip_paths,
                        embedding_directory=emb_dir,
                        clip_type=clip_type
                    )
                    
                    # Explicitly move model to our device
                    if hasattr(self.model, 'cond_stage_model'):
                        self.model.cond_stage_model.to(self.device)
                    
                finally:
                    # Restore original device
                    comfy.model_management.set_vram_to = old_device
                
                # Track loaded models
                for i, path in enumerate(clip_paths):
                    if i == 0:
                        self.loaded_model_paths['clip_l'] = path
                    elif i == 1:
                        self.loaded_model_paths['clip_g'] = path
                
                if self.precision != "fp32" and hasattr(self.model, 'cond_stage_model'):
                    self.model.cond_stage_model.to(self.dtype)
                
                logger.info(f"[CLIP-{self.worker_id}] CLIP loaded from path ({self.precision})")
        
        elif self.worker_type == WorkerType.CLIP_VISION:
            logger.info(f"[CLIP_VISION-{self.worker_id}] Loading CLIP Vision model...")
            
            # Try to load CLIP Vision model
            vision_path = self._resolve_path(self.config_paths.get('clip_vision') or '', "clip_vision")
            
            if not vision_path:
                raise RuntimeError(
                    "CLIP Vision model not configured. "
                    "Use load_clip_vision() to specify which vision model to load, or set CLIP_VISION_PATH in config.py"
                )
            
            # Load using ComfyUI's clip_vision loader
            import comfy.clip_vision  # type: ignore
            self.model = comfy.clip_vision.load(vision_path)
            
            # Move to target device
            if hasattr(self.model, 'model'):
                self.model.model.to(self.device)
            
            if self.precision != "fp32":
                if hasattr(self.model, 'model'):
                    self.model.model.to(self.dtype)
            
            self.loaded_model_paths['clip_vision'] = vision_path
            logger.info(f"[CLIP_VISION-{self.worker_id}] Vision model loaded: {vision_path} ({self.precision})")
        
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
    
    def reload_model(self):
        """Reload the model (e.g., after config paths change)."""
        self.unload_model()
        self.load_model()
    
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
    
    def process_clip_tokenize(self, text: str, return_word_ids: bool = False):
        """Tokenize text with CLIP."""
        assert self.model is not None, "CLIP model not loaded"
        
        with torch.inference_mode():
            tokens = self.model.tokenize(text, return_word_ids=return_word_ids)
            return tokens
    
    def process_clip_encode_from_tokens(
        self, 
        tokens, 
        return_pooled: bool = False,
        return_dict: bool = False,
        lora_stack = None
    ):
        """Encode pre-tokenized input."""
        assert self.model is not None, "CLIP model not loaded"
        
        with torch.inference_mode():
            result = self.model.encode_from_tokens(
                tokens,
                return_pooled=return_pooled,
                return_dict=return_dict
            )
            
            # If return_dict is True, ensure we return a proper dict format
            if return_dict and isinstance(result, tuple):
                # Result is (cond, pooled) - convert to dict
                cond, pooled = result
                return {"cond": cond, "pooled_output": pooled}
            
            return result
    
    def process_register_lora(self, lora_name: str, clip_strength: float, model_strength: float):
        """Register and apply LoRA to CLIP."""
        # TODO: Implement LoRA application to CLIP
        # For now, return success (LoRAs would be applied via add_patches)
        return {"success": True, "message": "LoRA registration not yet implemented in workers"}
    
    def process_clear_loras(self):
        """Clear all applied LoRAs."""
        # TODO: Implement LoRA clearing
        return {"success": True, "message": "LoRA clearing not yet implemented in workers"}
    
    # =========================================================================
    # CLIP Vision Operations (Structural Anchoring)
    # =========================================================================
    
    def process_vision_encode(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single image with CLIP Vision model.
        
        Args:
            image: Image tensor [1, H, W, 3] or [H, W, 3]
        
        Returns:
            Vision embedding [1, 257, vision_dim]
        """
        assert self.model is not None, "CLIP Vision model not loaded"
        
        with torch.inference_mode():
            # Ensure proper shape
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Move to device
            image = image.to(self.device)
            
            # Encode using ComfyUI's clip_vision
            output = self.model.encode_image(image)
            
            # Extract embedding
            if hasattr(output, 'last_hidden_state'):
                embedding = output.last_hidden_state
            elif hasattr(output, 'image_embeds'):
                embedding = output.image_embeds
            else:
                embedding = output
            
            result = embedding.cpu()
            
            del image, output
            self._cleanup_gpu_memory()
            
            return result
    
    def process_vision_encode_batch(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch encode multiple images with CLIP Vision model.
        
        Args:
            images: Image tensor [N, H, W, 3]
        
        Returns:
            Vision embeddings [N, 257, vision_dim]
        """
        assert self.model is not None, "CLIP Vision model not loaded"
        
        with torch.inference_mode():
            images = images.to(self.device)
            
            # Batch encode
            output = self.model.encode_image(images)
            
            if hasattr(output, 'last_hidden_state'):
                embedding = output.last_hidden_state
            elif hasattr(output, 'image_embeds'):
                embedding = output.image_embeds
            else:
                embedding = output
            
            result = embedding.cpu()
            
            del images, output
            self._cleanup_gpu_memory()
            
            return result
    
    def process_vision_encode_crops(
        self,
        full_image: torch.Tensor,
        crop_coords: list,
        tile_size: int = 1024
    ) -> list:
        """
        Crop full image and batch encode crops with CLIP Vision.
        
        This is the most efficient method - receives one full image,
        crops on GPU, then batch encodes all crops at once.
        
        Args:
            full_image: Full canvas [1, H, W, 3] in BHWC format
            crop_coords: List of (x1, y1, x2, y2) crop regions
            tile_size: Target crop size for encoding
        
        Returns:
            List of vision embeddings, one per crop
        """
        assert self.model is not None, "CLIP Vision model not loaded"
        
        import torch.nn.functional as F
        
        with torch.inference_mode():
            full_image = full_image.to(self.device)
            
            # Crop on GPU
            crops = []
            for (x1, y1, x2, y2) in crop_coords:
                crop = full_image[:, y1:y2, x1:x2, :]  # [1, crop_h, crop_w, 3]
                
                # Resize to tile_size if needed
                if crop.shape[1] != tile_size or crop.shape[2] != tile_size:
                    crop_bchw = crop.permute(0, 3, 1, 2)  # BHWC → BCHW
                    crop_bchw = F.interpolate(
                        crop_bchw, size=(tile_size, tile_size),
                        mode='bicubic', align_corners=False
                    )
                    crop = crop_bchw.permute(0, 2, 3, 1)  # BCHW → BHWC
                
                crops.append(crop)
            
            # Stack into batch
            batch = torch.cat(crops, dim=0)  # [N, tile_size, tile_size, 3]
            
            logger.info(f"[CLIP_VISION-{self.worker_id}] Encoding {len(crops)} crops")
            
            # Batch encode
            output = self.model.encode_image(batch)
            
            if hasattr(output, 'last_hidden_state'):
                embeddings = output.last_hidden_state
            elif hasattr(output, 'image_embeds'):
                embeddings = output.image_embeds
            else:
                embeddings = output
            
            # Split back into list
            result = [embeddings[i:i+1].cpu() for i in range(len(crop_coords))]
            
            del full_image, crops, batch, output, embeddings
            self._cleanup_gpu_memory()
            
            logger.info(f"[CLIP_VISION-{self.worker_id}] Encoding complete: {len(result)} embeddings")
            
            return result

    def process_zimage_encode(self, text: str):
        """Encode text with Z-IMAGE (Qwen3-VL CLIP)."""
        if self.qwen3_encoder is None:
            return {"error": "Qwen3 not loaded. Call register_qwen3_transformers first."}
        
        try:
            with torch.inference_mode():
                embeddings = self.qwen3_encoder.encode_text(text)
                return embeddings.cpu()
        except Exception as e:
            return {"error": f"Z-IMAGE encoding failed: {e}"}
    
    def process_register_qwen3(self, model_path: str, device: Optional[str], dtype: Optional[str]):
        """Load Qwen3-VL model using transformers."""
        try:
            from .qwen3_encoder import Qwen3VLEncoder, Qwen3VLConfig
            
            # Use provided device or fall back to worker device
            target_device = device or self.device
            target_dtype = dtype or self.precision
            
            # Create config
            config = Qwen3VLConfig(
                model_path=model_path,
                device=target_device,
                dtype=target_dtype
            )
            
            # Unload existing encoder if any
            if self.qwen3_encoder is not None:
                del self.qwen3_encoder
                self.qwen3_encoder = None
                torch.cuda.empty_cache()
            
            # Load new encoder
            self.qwen3_encoder = Qwen3VLEncoder(config)
            success = self.qwen3_encoder.load_model(model_path)
            
            if success:
                logger.info(f"[CLIP-{self.worker_id}] Qwen3-VL loaded: {model_path}")
                return {
                    "success": True,
                    "model_path": model_path,
                    "device": target_device,
                    "dtype": target_dtype,
                    "zimage_compatible": getattr(self.qwen3_encoder, 'is_zimage_compatible', True),
                    "message": "Qwen3-VL loaded for Z-IMAGE/VLM"
                }
            else:
                return {"error": "Failed to load Qwen3 model"}
                
        except ImportError as e:
            return {"error": f"Qwen3 encoder module not available: {e}"}
        except Exception as e:
            logger.error(f"[CLIP-{self.worker_id}] Qwen3 loading error: {e}")
            return {"error": f"Failed to load Qwen3: {e}"}
    
    def process_qwen3_status(self):
        """Get Qwen3-VL model status."""
        if self.qwen3_encoder is None:
            return {
                "loaded": False,
                "message": "Qwen3-VL not loaded"
            }
        
        return {
            "loaded": True,
            "model_path": getattr(self.qwen3_encoder.config, 'model_path', 'unknown'),
            "device": getattr(self.qwen3_encoder.config, 'device', 'unknown'),
            "zimage_compatible": getattr(self.qwen3_encoder, 'is_zimage_compatible', False),
            "has_vision": getattr(self.qwen3_encoder, 'has_vision', False)
        }
    
    def process_vlm_generate(self, data: dict):
        """Generate text with VLM."""
        if self.qwen3_encoder is None:
            return {"error": "Qwen3 not loaded. Call register_qwen3_transformers first."}
        
        try:
            prompt = data.get("prompt", "")
            image = data.get("image")
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            
            with torch.inference_mode():
                if image is not None:
                    # VLM with image
                    response = self.qwen3_encoder.describe_image(
                        image,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    # Text-only generation
                    response = self.qwen3_encoder.generate_text(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                
                return {
                    "success": True,
                    "text": response
                }
        except Exception as e:
            return {"error": f"VLM generation failed: {e}"}
    
    def process_encode_vision(self, image):
        """Encode image with vision model."""
        if self.qwen3_encoder is None:
            return {"error": "Qwen3 not loaded. Call register_qwen3_transformers first."}
        
        if not getattr(self.qwen3_encoder, 'has_vision', False):
            return {"error": "Loaded model does not have vision capabilities"}
        
        try:
            with torch.inference_mode():
                embeddings = self.qwen3_encoder.encode_image(image)
                return embeddings.cpu()
        except Exception as e:
            return {"error": f"Vision encoding failed: {e}"}
    
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
                        elif cmd == "preload":
                            # Preload command - model is already loaded by lazy load above
                            result = {"success": True, "preloaded": True}
                        else:
                            result = {"error": f"Unknown VAE command: {cmd}"}
                    
                    elif self.worker_type == WorkerType.CLIP:
                        lora_stack = data.get("lora_stack")
                        
                        if cmd == "preload":
                            # Preload command - model is already loaded by lazy load above
                            result = {"success": True, "preloaded": True}
                        elif cmd == "clip_encode":
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
                        elif cmd == "clip_tokenize":
                            result = self.process_clip_tokenize(
                                data["text"],
                                return_word_ids=data.get("return_word_ids", False)
                            )
                        elif cmd == "clip_encode_from_tokens":
                            result = self.process_clip_encode_from_tokens(
                                data["tokens"],
                                return_pooled=data.get("return_pooled", False),
                                return_dict=data.get("return_dict", False),
                                lora_stack=lora_stack
                            )
                        elif cmd == "register_lora":
                            result = self.process_register_lora(
                                data["lora_name"],
                                data.get("clip_strength", 1.0),
                                data.get("model_strength", 1.0)
                            )
                        elif cmd == "clear_loras":
                            result = self.process_clear_loras()
                        elif cmd == "zimage_encode":
                            result = self.process_zimage_encode(data["text"])
                        elif cmd == "register_qwen3_transformers":
                            result = self.process_register_qwen3(
                                data["model_path"],
                                data.get("device"),
                                data.get("dtype")
                            )
                        elif cmd == "qwen3_status":
                            result = self.process_qwen3_status()
                        elif cmd == "vlm_generate":
                            result = self.process_vlm_generate(data)
                        elif cmd == "encode_vision":
                            result = self.process_encode_vision(data["image"])
                        else:
                            result = {"error": f"Unknown CLIP command: {cmd}"}
                    
                    elif self.worker_type == WorkerType.CLIP_VISION:
                        if cmd == "preload":
                            result = {"success": True, "preloaded": True}
                        elif cmd == "vision_encode":
                            result = self.process_vision_encode(data["image"])
                        elif cmd == "vision_encode_batch":
                            result = self.process_vision_encode_batch(data["images"])
                        elif cmd == "vision_encode_crops":
                            result = self.process_vision_encode_crops(
                                data["full_image"],
                                data["crop_coords"],
                                data.get("tile_size", 1024)
                            )
                        elif cmd == "load_model":
                            # Reload with new config
                            self.config_paths.update(data)
                            self.reload_model()
                            result = {"success": True, "reloaded": True}
                        else:
                            result = {"error": f"Unknown CLIP_VISION command: {cmd}"}
                    
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
        config_paths: Optional[Dict[str, Optional[str]]] = None
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
    
    def set_device(self, new_device: str) -> bool:
        """
        Change the device for all workers.
        
        Requires stopping and restarting all workers as they hold model references.
        """
        old_device = self.device
        self.device = new_device
        
        logger.info(f"[{self.worker_type.value.upper()}] Device change requested: {old_device} -> {new_device}")
        
        # Stop all current workers
        with self.lock:
            for worker in self.workers:
                worker.stop()
            self.workers.clear()
        
        # Briefly wait for workers to shut down
        time.sleep(0.5)
        
        # Workers will be recreated on demand with the new device
        logger.info(f"[{self.worker_type.value.upper()}] Device changed to {new_device}. Workers will be recreated on next request.")
        
        return True
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.lock:
            # Collect loaded model paths from workers
            loaded_models = set()
            for worker in self.workers:
                if worker.loaded_model_paths:
                    for component, path in worker.loaded_model_paths.items():
                        if path:
                            # Get just the filename for display
                            import os
                            loaded_models.add(os.path.basename(path))
            
            return {
                "type": self.worker_type.value,
                "active_workers": len(self.workers),
                "queue_depth": self.request_queue.qsize(),
                "total_requests": sum(w.request_count for w in self.workers),
                "worker_ids": [w.worker_id for w in self.workers],
                "workers_count": len(self.workers),
                "loaded_models": list(loaded_models),
            }
