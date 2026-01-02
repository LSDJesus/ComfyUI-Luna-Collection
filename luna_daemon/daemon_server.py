"""
Luna Daemon Server - Main Orchestrator

Simplified daemon server that coordinates:
- VAE/CLIP worker pools
- WebSocket monitoring
- LoRA RAM cache
- Socket communication with ComfyUI clients

This is the main entry point for the daemon.
"""

import os
import sys

# CRITICAL: Set attention mode BEFORE any ComfyUI imports
# Check for LUNA_ATTENTION_MODE env var (set by ComfyUI startup)
if "LUNA_ATTENTION_MODE" in os.environ:
    attention_mode = os.environ["LUNA_ATTENTION_MODE"]
    # ComfyUI uses ATTN_PRECISION internally
    os.environ["ATTN_PRECISION"] = attention_mode
    print(f"[Luna.Daemon] Using attention mode: {attention_mode}")

# CRITICAL: Force PyTorch to use legacy CUDA allocator for IPC support
# The options in this PyTorch version are 'native' and 'cudaMallocAsync'
# 'native' is the standard caching allocator which supports legacy IPC
os.environ["PYTORCH_ALLOC_CONF"] = "backend:native"

import socket
import pickle
import threading
import time
import logging
from typing import Any, Dict, Optional, Set

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LunaDaemon")

# =============================================================================
# Import Configuration
# =============================================================================

from enum import Enum, auto

# Try to load config - first as relative import, then as fallback
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283
DAEMON_WS_PORT = 19284
CLIP_DEVICE = "cuda:1"
VAE_DEVICE = "cuda:0"
CLIP_L_PATH = ""
CLIP_G_PATH = ""
EMBEDDINGS_DIR = ""
MODEL_PRECISION = "fp16"
CLIP_PRECISION = "fp16"
MAX_CLIP_WORKERS = 2
MIN_CLIP_WORKERS = 0
QUEUE_THRESHOLD = 2
SCALE_UP_DELAY_SEC = 1.0
IDLE_TIMEOUT_SEC = 30.0
SERVICE_TYPE = "full"
CLIP_VISION_PATH = None
CLIP_VISION_DEVICE = None

config_loaded = False
config_source = "hardcoded defaults"

try:
    # Try relative import first (package context)
    logger.info("[Config] Attempting package import from .config")
    from .config import (
        DAEMON_HOST,
        DAEMON_PORT,
        DAEMON_WS_PORT,
        CLIP_DEVICE,
        VAE_DEVICE,
        CLIP_L_PATH,
        CLIP_G_PATH,
        EMBEDDINGS_DIR,
        MODEL_PRECISION,
        CLIP_PRECISION,
        MAX_CLIP_WORKERS,
        MIN_CLIP_WORKERS,
        QUEUE_THRESHOLD,
        SCALE_UP_DELAY_SEC,
        IDLE_TIMEOUT_SEC,
        SERVICE_TYPE,
        ServiceType,
        CLIP_VISION_PATH,
        CLIP_VISION_DEVICE,
    )
    logger.info("[Config] [OK] Loaded config via package import (.config)")
    config_loaded = True
    config_source = "package import"
except (ImportError, ValueError) as e:
    logger.warning(f"[Config] Package import failed: {e}")
    logger.info("[Config] Attempting fallback import from config file...")
    
    try:
        import importlib.util
        daemon_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(daemon_dir, "config.py")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        spec = importlib.util.spec_from_file_location("luna_daemon_config", config_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not create module spec for {config_path}")
        
        config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_mod)
        
        # Extract all values from loaded config
        DAEMON_HOST = getattr(config_mod, "DAEMON_HOST", DAEMON_HOST)
        DAEMON_PORT = getattr(config_mod, "DAEMON_PORT", DAEMON_PORT)
        DAEMON_WS_PORT = getattr(config_mod, "DAEMON_WS_PORT", DAEMON_WS_PORT)
        CLIP_DEVICE = getattr(config_mod, "CLIP_DEVICE", CLIP_DEVICE)
        VAE_DEVICE = getattr(config_mod, "VAE_DEVICE", VAE_DEVICE)
        CLIP_L_PATH = getattr(config_mod, "CLIP_L_PATH", CLIP_L_PATH)
        CLIP_G_PATH = getattr(config_mod, "CLIP_G_PATH", CLIP_G_PATH)
        EMBEDDINGS_DIR = getattr(config_mod, "EMBEDDINGS_DIR", EMBEDDINGS_DIR)
        MODEL_PRECISION = getattr(config_mod, "MODEL_PRECISION", MODEL_PRECISION)
        CLIP_PRECISION = getattr(config_mod, "CLIP_PRECISION", CLIP_PRECISION)
        MAX_CLIP_WORKERS = getattr(config_mod, "MAX_CLIP_WORKERS", MAX_CLIP_WORKERS)
        MIN_CLIP_WORKERS = getattr(config_mod, "MIN_CLIP_WORKERS", MIN_CLIP_WORKERS)
        QUEUE_THRESHOLD = getattr(config_mod, "QUEUE_THRESHOLD", QUEUE_THRESHOLD)
        SCALE_UP_DELAY_SEC = getattr(config_mod, "SCALE_UP_DELAY_SEC", SCALE_UP_DELAY_SEC)
        IDLE_TIMEOUT_SEC = getattr(config_mod, "IDLE_TIMEOUT_SEC", IDLE_TIMEOUT_SEC)
        SERVICE_TYPE = getattr(config_mod, "SERVICE_TYPE", SERVICE_TYPE)
        
        # Try to get ServiceType class
        try:
            ServiceType = getattr(config_mod, "ServiceType")
        except AttributeError:
            logger.warning("[Config] ServiceType not found in config, using default")
        
        logger.info("[Config] [OK] Loaded config via fallback import (importlib)")
        config_loaded = True
        config_source = "fallback import"
    except Exception as e:
        logger.error(f"[Config] Fallback import failed: {e}", exc_info=True)
        logger.warning("[Config] [WARN] Using hardcoded defaults - this may not match your config.py")

logger.info(f"[Config] Source: {config_source}")
logger.info(f"[Config] CLIP_DEVICE={CLIP_DEVICE}, VAE_DEVICE={VAE_DEVICE}")
logger.info(f"[Config] DAEMON_HOST={DAEMON_HOST}:{DAEMON_PORT}")

# =============================================================================
# Import Modules
# =============================================================================

# Handle imports - try relative first, then absolute
try:
    from .core import ServiceType as CoreServiceType
    from .workers import WorkerPool, WorkerType, ScalingConfig
    from .monitoring import WebSocketServer
    from .lora_cache import LoRACache, get_lora_cache
except (ImportError, ValueError):
    # Fallback: load modules directly
    import importlib.util
    daemon_dir = os.path.dirname(os.path.abspath(__file__))
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        raise ImportError(f"Could not load {name}")
    
    core_mod = load_module("luna_daemon_core", os.path.join(daemon_dir, "core.py"))
    CoreServiceType = getattr(core_mod, "ServiceType", ServiceType)
    
    workers_mod = load_module("luna_daemon_workers", os.path.join(daemon_dir, "workers.py"))
    WorkerPool = workers_mod.WorkerPool
    WorkerType = workers_mod.WorkerType
    ScalingConfig = workers_mod.ScalingConfig
    
    monitoring_mod = load_module("luna_daemon_monitoring", os.path.join(daemon_dir, "monitoring.py"))
    WebSocketServer = monitoring_mod.WebSocketServer
    
    cache_mod = load_module("luna_daemon_cache", os.path.join(daemon_dir, "lora_cache.py"))
    LoRACache = cache_mod.LoRACache
    get_lora_cache = cache_mod.get_lora_cache


# =============================================================================
# Main Daemon Server
# =============================================================================

class LunaDaemon:
    """
    Main daemon server with CLIP worker pool and monitoring.
    
    Simplified architecture:
    - Workers handle CLIP encoding
    - WebSocket broadcasts status to JS panel
    - LoRA cache stores state_dicts in RAM
    - Socket server handles client requests
    """
    
    def __init__(
        self,
        host: str = DAEMON_HOST,
        port: int = DAEMON_PORT,
        ws_port: int = DAEMON_WS_PORT,
        clip_device: str = CLIP_DEVICE,
        service_type: Any = SERVICE_TYPE,
        # Backward compatibility with old tray app parameters
        device: Optional[str] = None,
        clip_precision: Optional[str] = None,
        **kwargs
    ):
        # Handle old tray app parameter names
        if device is not None:
            clip_device = device
        
        self.host = host
        self.port = port
        self.ws_port = ws_port
        self.clip_device = clip_device
        self.vae_device = VAE_DEVICE  # From config.py
        self.llm_device = getattr(globals(), 'LLM_DEVICE', clip_device)  # Optional LLM device, default to CLIP device
        self.service_type = service_type
        
        # Scaling configuration
        self.scaling_config = ScalingConfig(
            min_clip_workers=MIN_CLIP_WORKERS,
            max_clip_workers=MAX_CLIP_WORKERS,
            queue_threshold=QUEUE_THRESHOLD,
            scale_up_delay_sec=SCALE_UP_DELAY_SEC,
            idle_timeout_sec=IDLE_TIMEOUT_SEC
        )
        
        # Config paths for workers
        self.config_paths: Dict[str, Optional[str]] = {
            'clip_l': CLIP_L_PATH,
            'clip_g': CLIP_G_PATH,
            'embeddings': EMBEDDINGS_DIR,
            'clip_vision': CLIP_VISION_PATH
        }
        
        # Vision device (defaults to CLIP device)
        self.vision_device = CLIP_VISION_DEVICE or clip_device
        
        # Weight registry for CUDA IPC weight sharing
        try:
            from .weight_registry import WeightRegistry
        except (ImportError, ValueError):
            # Fallback: direct import when run as __main__
            import importlib.util
            daemon_dir = os.path.dirname(os.path.abspath(__file__))
            registry_path = os.path.join(daemon_dir, "weight_registry.py")
            spec = importlib.util.spec_from_file_location("luna_weight_registry", registry_path)
            if spec and spec.loader:
                registry_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(registry_mod)
                WeightRegistry = registry_mod.WeightRegistry
            else:
                raise ImportError("Could not load weight_registry module")
        
        self.weight_registry = WeightRegistry(device=clip_device)
        logger.info(f"[Daemon] Weight registry initialized on {clip_device}")
        
        # Worker pools
        self.clip_pool: Any = None
        self.vision_pool: Any = None  # CLIP Vision worker pool
        self.vision_device: str = clip_device  # Share device with CLIP by default
        
        # WebSocket monitoring
        self.ws_server: Any = None
        
        # LoRA cache
        self.lora_cache = get_lora_cache()
        
        # Model state tracking for workflow multiplexing
        self.loaded_models: Dict[str, Any] = {}  # {path: loaded_model_object}
        self.workflow_model_sets: Dict[str, Dict[str, Any]] = {}  # {workflow_id: {model_type, clip_l, clip_g, vae}}
        self.current_workflow_id: Optional[str] = None  # Track which workflow's models are in workers
        
        # SAM3 model registry (shared across all instances)
        self.sam3_model: Optional[Any] = None
        self.sam3_processor: Optional[Any] = None  # Sam3Processor instance
        self.sam3_model_name: Optional[str] = None
        self.sam3_device: str = clip_device  # Use same device as CLIP by default
        self.models_lock = threading.Lock()
        
        # Request counters
        self._vision_request_count = 0
        
        # Socket server
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._client_threads: Set[threading.Thread] = set()
        
        # Stats
        self._start_time = time.time()
        self._request_count = 0
        self._clip_request_count = 0
        self._vae_request_count = 0  # Track VAE requests separately
        
        # Configuration state
        self._attention_mode = "auto"  # Current attention mode setting
    
    def _on_scale_event(self, event_type: str, data: dict):
        """Callback for worker pool scaling events."""
        if self.ws_server:
            self.ws_server.broadcast("scaling", data)
    
    def get_info(self) -> Dict[str, Any]:
        """Get daemon status info for monitoring."""
        info = {
            "status": "running" if self._running else "stopped",
            "uptime_sec": time.time() - self._start_time,
            "request_count": self._request_count,
            "clip_request_count": self._clip_request_count,
            "vae_request_count": self._vae_request_count,  # Add VAE request count
            "service_type": self.service_type.name if hasattr(self.service_type, 'name') else str(self.service_type),
            "attention_mode": self._attention_mode,
            "devices": {
                "clip": self.clip_device,
                "vae": self.vae_device if hasattr(self, 'vae_device') else self.clip_device,
                "llm": self.llm_device if hasattr(self, 'llm_device') else self.clip_device
            }
        }
        if self.clip_pool:
            info["clip_pool"] = self.clip_pool.get_stats()
        
        # VRAM info for ALL GPUs
        if torch.cuda.is_available():
            info["vram"] = {}
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                info["vram"][f"cuda:{i}"] = {
                    "used_gb": round((total - free) / 1024**3, 2),
                    "total_gb": round(total / 1024**3, 2)
                }
        
        # Weight registry model details
        if hasattr(self, 'weight_registry') and self.weight_registry:
            info["weight_registry_models"] = self.weight_registry.get_model_details()
        else:
            info["weight_registry_models"] = []
        
        return info
    
    def _delayed_shutdown(self):
        """Shut down daemon after a brief delay to allow response to be sent."""
        import time
        time.sleep(0.5)  # Give time for response to be sent
        self.stop()
    
    def _handle_request(self, cmd: str, data: dict) -> Any:
        """Route request to appropriate handler."""
        # Skip counting ping/health check/status requests
        if cmd not in ("ping", "get_info", "status"):
            self._request_count += 1
        
        # CLIP commands
        if cmd == "clip_encode":
            self._clip_request_count += 1
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        elif cmd == "clip_encode_sdxl":
            self._clip_request_count += 1
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        elif cmd == "clip_tokenize":
            self._clip_request_count += 1
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        elif cmd == "clip_encode_from_tokens":
            self._clip_request_count += 1
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            return self.clip_pool.submit(cmd, data)
        
        # Z-IMAGE commands
        elif cmd == "register_qwen3_transformers":
            # Load Qwen3-VL model using transformers
            model_path = data.get("model_path")
            device = data.get("device")
            dtype = data.get("dtype")
            
            if not model_path:
                return {"error": "model_path required"}
            
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            
            # Submit to CLIP pool for Qwen3 loading
            return self.clip_pool.submit("register_qwen3_transformers", {
                "model_path": model_path,
                "device": device or self.clip_device,
                "dtype": dtype
            })
        
        elif cmd == "qwen3_status":
            # Get Qwen3 model status
            if self.clip_pool is None:
                return {"loaded": False, "error": "CLIP pool not available"}
            
            return self.clip_pool.submit("qwen3_status", {})
        
        elif cmd == "zimage_encode":
            self._clip_request_count += 1  # Count as CLIP request
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            # Route to CLIP pool - Z-IMAGE uses modified CLIP architecture
            return self.clip_pool.submit(cmd, data)
        
        # Vision/VLM commands
        elif cmd == "vlm_generate":
            self._clip_request_count += 1  # Count as CLIP/vision request
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            # Route to CLIP pool - VLM shares infrastructure
            return self.clip_pool.submit(cmd, data)
        
        elif cmd == "encode_vision":
            self._clip_request_count += 1  # Count as vision request
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            # Route to CLIP pool - vision encoder shares infrastructure
            return self.clip_pool.submit(cmd, data)
        
        # CLIP Vision (Structural Anchoring) commands
        elif cmd == "vision_encode":
            self._vision_request_count += 1
            if self.vision_pool is None:
                return {"error": "CLIP Vision pool not available"}
            return self.vision_pool.submit("vision_encode", data)
        
        elif cmd == "vision_encode_batch":
            self._vision_request_count += 1
            if self.vision_pool is None:
                return {"error": "CLIP Vision pool not available"}
            return self.vision_pool.submit("vision_encode_batch", data)
        
        elif cmd == "vision_encode_crops":
            self._vision_request_count += 1
            if self.vision_pool is None:
                return {"error": "CLIP Vision pool not available"}
            return self.vision_pool.submit("vision_encode_crops", data)
        
        elif cmd == "load_clip_vision":
            # Load CLIP Vision model
            model_path = data.get("model_path")
            device = data.get("device")
            
            if not model_path:
                return {"error": "model_path required"}
            
            if self.vision_pool is None:
                return {"error": "CLIP Vision pool not available"}
            
            # Update config and trigger reload
            return self.vision_pool.submit("load_model", {
                "clip_vision": model_path,
                "device": device or self.vision_device
            })
        
        elif cmd == "vision_status":
            # Get CLIP Vision model status
            if self.vision_pool is None:
                return {"loaded": False, "available": False}
            
            return {
                "available": True,
                "loaded": self.vision_pool.has_loaded_workers(),
                "worker_count": self.vision_pool.active_worker_count(),
                "pending_requests": self.vision_pool.pending_count()
            }
        
        # SAM3 commands
        elif cmd == "load_sam3":
            # Load SAM3 model using vendored luna_sam3 library
            model_name = data.get("model_name")
            device = data.get("device", self.sam3_device)
            
            if not model_name:
                return {"error": "model_name required"}
            
            try:
                # Check if already loaded
                if self.sam3_model is not None and self.sam3_model_name == model_name:
                    logger.info(f"[Daemon] SAM3 model '{model_name}' already loaded")
                    return {"success": True, "status": "already_loaded", "model_name": model_name}
                
                logger.info(f"[Daemon] Loading SAM3 model '{model_name}' on {device}")
                
                # Import vendored SAM3
                from pathlib import Path
                import sys
                
                # Add ComfyUI-Luna-Collection to sys.path so luna_sam3 can be imported
                luna_collection_root = Path(__file__).parent.parent
                if str(luna_collection_root) not in sys.path:
                    sys.path.insert(0, str(luna_collection_root))
                
                from luna_sam3.sam3_video_predictor import Sam3VideoPredictor
                from luna_sam3.model.sam3_image_processor import Sam3Processor
                
                # Get BPE vocabulary path from vendored luna_sam3
                bpe_path = luna_collection_root / "luna_sam3" / "bpe_simple_vocab_16e6.txt.gz"
                if not bpe_path.exists():
                    return {"error": f"BPE vocabulary file not found at {bpe_path}"}
                
                # Get model checkpoint path using ComfyUI's folder_paths (respects extra_model_paths)
                import folder_paths
                
                # Register sam3 as a model folder if not already registered
                if "sam3" not in folder_paths.folder_names_and_paths:
                    sam3_paths = [os.path.join(folder_paths.models_dir, "sam3")]
                    folder_paths.folder_names_and_paths["sam3"] = (sam3_paths, {".pt", ".pth", ".safetensors"})
                
                checkpoint_path = None
                if model_name:
                    # Use get_full_path which searches all registered paths including extra_model_paths
                    checkpoint_path = folder_paths.get_full_path("sam3", model_name)
                    
                    if checkpoint_path is None or not os.path.exists(checkpoint_path):
                        # Fallback: check common locations
                        fallback_paths = [
                            os.path.join(folder_paths.models_dir, "sam3", model_name),
                        ]
                        for fp in fallback_paths:
                            if os.path.exists(fp):
                                checkpoint_path = fp
                                break
                    
                    if checkpoint_path is None or not os.path.exists(checkpoint_path):
                        # List available models for helpful error
                        try:
                            available = folder_paths.get_filename_list("sam3")
                        except:
                            available = []
                        return {"error": f"SAM3 checkpoint not found: {model_name}. Available: {available}"}
                    
                    logger.info(f"[Daemon] Using SAM3 checkpoint: {checkpoint_path}")
                
                # Build SAM3 video predictor (contains detector for image segmentation)
                logger.info(f"[Daemon] Building SAM3 model with checkpoint: {checkpoint_path}")
                
                # CRITICAL: Set the default CUDA device for SAM3 operations
                # This ensures ALL tensor operations default to cuda:1, not cuda:0
                device_index = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(device_index)
                logger.info(f"[Daemon] Set default CUDA device to: {torch.cuda.current_device()}")
                
                video_predictor = Sam3VideoPredictor(
                    checkpoint_path=checkpoint_path,
                    bpe_path=str(bpe_path),
                    enable_inst_interactivity=True,  # Enable point/box prompts
                )
                
                # Extract image detector and move to target device
                sam3_model = video_predictor.model.detector  # Sam3Image instance
                sam3_model = sam3_model.to(device)
                sam3_model.eval()
                
                # Create processor
                processor = Sam3Processor(sam3_model)
                processor.device = device
                # Sync processor device state with actual model device
                processor.sync_device_with_model()
                
                # Store references (keep video_predictor alive for model lifecycle)
                self.sam3_model = sam3_model
                self.sam3_processor = processor
                self.sam3_model_name = model_name
                self.sam3_device = device
                self._sam3_video_predictor = video_predictor  # Keep reference
                
                logger.info(f"[Daemon] SAM3 model loaded successfully on {device}: {model_name}")
                return {"success": True, "status": "loaded", "model_name": model_name}
                
            except Exception as e:
                logger.error(f"[Daemon] Error loading SAM3: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
        
        elif cmd == "sam3_detect":
            # Run SAM3 grounding detection using official sam3 API
            image_data = data.get("image")  # PIL image (pickled)
            text_prompt = data.get("text_prompt")
            threshold = data.get("threshold", 0.25)
            
            if image_data is None or not text_prompt:
                return {"error": "image and text_prompt required"}
            
            if self.sam3_model is None or self.sam3_processor is None:
                return {"error": "SAM3 model not loaded. Call load_sam3 first."}
            
            try:
                # Deserialize PIL image
                import pickle as pkl
                pil_image = pkl.loads(image_data)
                
                # Run SAM3 grounding using official API
                logger.info(f"[Daemon] Running SAM3 detection for '{text_prompt}'")
                
                with torch.inference_mode():
                    # Set the image
                    inference_state = self.sam3_processor.set_image(pil_image)
                    
                    # Prompt with text
                    output = self.sam3_processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt
                    )
                
                # Extract results
                masks = output.get("masks", [])
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])
                
                if len(scores) == 0:
                    logger.info(f"[Daemon] No detections found for '{text_prompt}'")
                    return {"success": True, "detections": []}
                
                # Convert to tensors if needed for filtering
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
                
                # Filter by threshold
                keep_mask = scores > threshold
                
                # Convert to serializable format
                detections = []
                for i in range(len(scores)):
                    if keep_mask[i]:
                        # Handle mask - could be tensor or numpy
                        mask_data = masks[i]
                        if isinstance(mask_data, torch.Tensor):
                            mask_data = mask_data.cpu().numpy()
                        
                        # Handle box
                        box_data = boxes[i]
                        if isinstance(box_data, torch.Tensor):
                            box_data = box_data.cpu().tolist()
                        elif hasattr(box_data, 'tolist'):
                            box_data = box_data.tolist()
                        
                        # Handle score
                        score_val = scores[i]
                        if isinstance(score_val, torch.Tensor):
                            score_val = score_val.cpu().item()
                        
                        detections.append({
                            "bbox": box_data,
                            "mask": mask_data.tolist() if hasattr(mask_data, 'tolist') else mask_data,
                            "confidence": float(score_val)
                        })
                
                logger.info(f"[Daemon] SAM3 found {len(detections)} detections")
                return {
                    "success": True,
                    "detections": detections,
                    "num_detections": len(detections)
                }
                
            except Exception as e:
                logger.error(f"[Daemon] Error in SAM3 detection: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
        
        elif cmd == "sam3_detect_batch":
            # Run SAM3 grounding detection for MULTIPLE prompts efficiently
            # Reuses backbone features across prompts for speed
            image_data = data.get("image")  # PIL image (pickled)
            prompts = data.get("prompts")   # List of {"prompt": str, "threshold": float, "label": str}
            default_threshold = data.get("threshold", 0.25)
            
            if image_data is None or not prompts:
                return {"error": "image and prompts list required"}
            
            if self.sam3_model is None or self.sam3_processor is None:
                return {"error": "SAM3 model not loaded. Call load_sam3 first."}
            
            try:
                import pickle as pkl
                pil_image = pkl.loads(image_data)
                
                logger.info(f"[Daemon] Running SAM3 batch detection for {len(prompts)} prompts")
                
                # CRITICAL: Force ALL tensor operations to use the daemon's SAM3 device
                # This ensures no tensors are accidentally created on cuda:0 (ComfyUI's device)
                device = self.sam3_device
                with torch.cuda.device(device):
                    with torch.inference_mode():
                        # Set the image ONCE (computes backbone features)
                        inference_state = self.sam3_processor.set_image(pil_image)
                        
                        all_results = {}
                        
                        # Run each prompt (reuses backbone features)
                        for prompt_info in prompts:
                            if isinstance(prompt_info, str):
                                # Simple string prompt
                                prompt_text = prompt_info
                                threshold = default_threshold
                                label = prompt_text
                            else:
                                # Dict with prompt, threshold, label
                                prompt_text = prompt_info.get("prompt", prompt_info.get("text", ""))
                                threshold = prompt_info.get("threshold", default_threshold)
                                label = prompt_info.get("label", prompt_text)
                            
                            if not prompt_text:
                                continue
                        
                        # Run detection for this prompt
                        output = self.sam3_processor.set_text_prompt(
                            state=inference_state,
                            prompt=prompt_text
                        )
                        
                        masks = output.get("masks", [])
                        boxes = output.get("boxes", [])
                        scores = output.get("scores", [])
                        
                        # Convert and filter
                        if not isinstance(scores, torch.Tensor):
                            scores = torch.tensor(scores) if len(scores) > 0 else torch.tensor([])
                        if not isinstance(boxes, torch.Tensor):
                            boxes = torch.tensor(boxes) if len(boxes) > 0 else torch.tensor([])
                        
                        detections = []
                        for i in range(len(scores)):
                            score_val = scores[i].item() if isinstance(scores[i], torch.Tensor) else float(scores[i])
                            if score_val > threshold:
                                mask_data = masks[i]
                                if isinstance(mask_data, torch.Tensor):
                                    mask_data = mask_data.cpu().numpy()
                                
                                box_data = boxes[i]
                                if isinstance(box_data, torch.Tensor):
                                    box_data = box_data.cpu().tolist()
                                elif hasattr(box_data, 'tolist'):
                                    box_data = box_data.tolist()
                                
                                detections.append({
                                    "bbox": box_data,
                                    "mask": mask_data.tolist() if hasattr(mask_data, 'tolist') else mask_data,
                                    "confidence": score_val,
                                    "label": label
                                })
                        
                        all_results[label] = detections
                        logger.info(f"[Daemon] SAM3 '{prompt_text}': {len(detections)} detections")
                
                # Flatten all detections with labels
                all_detections = []
                for label, dets in all_results.items():
                    all_detections.extend(dets)
                
                return {
                    "success": True,
                    "results_by_label": all_results,
                    "detections": all_detections,
                    "num_detections": len(all_detections)
                }
                
            except Exception as e:
                logger.error(f"[Daemon] Error in SAM3 batch detection: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
        
        elif cmd == "unload_sam3":
            # Unload SAM3 model to free VRAM
            try:
                if self.sam3_model is not None:
                    model_name = self.sam3_model_name
                    del self.sam3_model
                    del self.sam3_processor
                    if hasattr(self, '_sam3_video_predictor'):
                        del self._sam3_video_predictor
                    self.sam3_model = None
                    self.sam3_processor = None
                    self.sam3_model_name = None
                    torch.cuda.empty_cache()
                    logger.info(f"[Daemon] SAM3 model '{model_name}' unloaded")
                    return {"success": True, "status": "unloaded", "model_name": model_name}
                else:
                    return {"success": True, "status": "not_loaded"}
            except Exception as e:
                logger.error(f"[Daemon] Error unloading SAM3: {e}")
                return {"error": str(e)}
        
        # Async task commands
        elif cmd == "submit_async":
            # Handle async tasks (like image saving)
            task_name = data.get("task_name")
            task_data = data.get("task_data", {})
            
            if not task_name:
                return {"error": "task_name required"}
            
            # Generate job ID
            import uuid
            job_id = str(uuid.uuid4())[:8]
            
            # Handle specific task types
            if task_name == "save_images_async":
                # Execute image saving in background thread
                try:
                    import threading
                    
                    num_images = len(task_data.get("images", []))
                    
                    def save_worker():
                        """Background worker for saving images"""
                        try:
                            self._save_images_worker(task_data, job_id)
                        except Exception as e:
                            logger.error(f"[Daemon] Image save worker error (Job {job_id}): {e}", exc_info=True)
                    
                    # Launch background thread
                    thread = threading.Thread(target=save_worker, daemon=True, name=f"ImgSave-{job_id}")
                    thread.start()
                    
                    logger.info(f"[Daemon] Image save job {job_id} submitted ({num_images} images)")
                    
                    return {
                        "success": True,
                        "job_id": job_id,
                        "num_images": num_images,
                        "message": f"Image save job submitted (ID: {job_id})"
                    }
                except Exception as e:
                    logger.error(f"[Daemon] Failed to submit image save job: {e}", exc_info=True)
                    return {"success": False, "error": str(e)}
            else:
                return {"error": f"Unknown task type: {task_name}"}
        
        # Model loading commands
        elif cmd == "get_model_proxies":
            """
            Request CLIP/VAE proxies for a workflow with specific models.
            
            Daemon strategy:
            1. Check if requested models are already loaded in any worker
            2. If yes, route workflow to use those workers
            3. If missing models, spin up new workers to load them
            4. Track workflow→worker mappings
            
            This enables model sharing across workflows while keeping all models loaded.
            """
            workflow_id = data.get("workflow_id")
            model_type = data.get("model_type")
            models = data.get("models", {})  # Dict of {component: path}
            
            if not workflow_id or not model_type or not models:
                return {"error": "workflow_id, model_type, and models dict required"}
            
            try:
                with self.models_lock:
                    # Check if this workflow already has a mapping
                    if workflow_id in self.workflow_model_sets:
                        existing_set = self.workflow_model_sets[workflow_id]
                        # Verify models match (same paths)
                        if existing_set.get("model_type") == model_type:
                            existing_models = existing_set.get("models", {})
                            if existing_models == models:
                                logger.info(f"[Daemon] Reusing existing model set for workflow {workflow_id}")
                                return {
                                    "success": True,
                                    "status": "cached",
                                    "clip_type": model_type.lower(),
                                    "vae_type": model_type.lower(),
                                    "workflow_id": workflow_id
                                }
                    
                    # Need to find/load workers with requested models
                    # Collect required components and paths
                    required_components = {}
                    for component, path in models.items():
                        if path and path != "None":
                            required_components[component] = path
                    
                    # Find or create workers for this workflow's models
                    logger.info(f"[Daemon] Setting up models for workflow {workflow_id}: {list(required_components.keys())}")
                    
                    # Update config_paths so new workers will load the right models
                    for component, path in required_components.items():
                        if component in ["clip_l", "clip_g"]:
                            self.config_paths[component] = path
                    
                    logger.info(f"[Daemon] Updated config_paths: {self.config_paths}")
                    
                    # Store vision model path if provided
                    if "vision" in required_components:
                        self.config_paths["clip_vision"] = required_components["vision"]
                        logger.info(f"[Daemon] Vision model path set: {required_components['vision']}")
                    
                    # Ensure workers exist (they'll use the updated config_paths)
                    if not self.clip_pool and any(c in required_components for c in ["clip_l", "clip_g"]):
                        self.clip_pool = WorkerPool(
                            worker_type=WorkerType.CLIP,
                            device=CLIP_DEVICE,
                            precision=CLIP_PRECISION,
                            config=self.scaling_config,
                            config_paths=self.config_paths.copy()  # Pass a copy to ensure workers get current state
                        )
                        self.clip_pool.start()
                        logger.info("[Daemon] CLIP pool started")
                        
                        # EAGER LOADING: Force workers to load models NOW (not lazily)
                        # This catches errors early in ModelRouter instead of later in ConfigGateway
                        logger.info("[Daemon] Preloading CLIP models (eager loading)...")
                        if not self.clip_pool.preload_models(timeout=60.0):
                            return {"error": "Failed to preload CLIP models - check daemon logs for details"}
                        logger.info("[Daemon] [OK] CLIP models preloaded successfully")
                    
                    # Store this workflow's model set
                    self.workflow_model_sets[workflow_id] = {
                        "model_type": model_type,
                        "models": required_components,
                        "created_at": time.time()
                    }
                    
                    logger.info(f"[Daemon] [OK] Models ready for workflow {workflow_id} ({model_type})")
                    return {
                        "success": True,
                        "status": "loaded",
                        "clip_type": model_type.lower(),
                        "vae_type": model_type.lower(),
                        "workflow_id": workflow_id
                    }
            except Exception as e:
                logger.error(f"[Daemon] get_model_proxies error: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
        

        
        elif cmd == "load_clip_model":
            clip_path = data.get("clip_path")
            clip_type = data.get("clip_type", "sdxl")
            
            if not clip_path:
                return {"error": "clip_path required"}
            
            # For SDXL and multi-component models, clip_path is typically the CLIP_G
            # Store both clip_l and clip_g as the same path (worker will handle)
            self.config_paths['clip_g'] = clip_path
            self.config_paths['clip_l'] = clip_path  # Fallback if only one CLIP needed
            logger.info(f"[Daemon] Updated CLIP paths: {clip_path} (type: {clip_type})")
            return {"success": True, "clip_path": clip_path, "clip_type": clip_type}
        
        # LoRA cache commands
        elif cmd == "lora_cache_get":
            lora_name = data.get("lora_name")
            if not lora_name:
                return {"error": "lora_name required"}
            state_dict = self.lora_cache.get(lora_name)
            if state_dict is None:
                return {"cached": False}
            return {"cached": True, "state_dict": state_dict}
        
        elif cmd == "lora_cache_put":
            lora_name = data.get("lora_name")
            state_dict = data.get("state_dict")
            if not lora_name or state_dict is None:
                return {"error": "lora_name and state_dict required"}
            success = self.lora_cache.put(lora_name, state_dict)
            return {"success": success}
        
        elif cmd == "lora_cache_check":
            lora_name = data.get("lora_name")
            if not lora_name:
                return {"error": "lora_name required"}
            return {"cached": self.lora_cache.contains(lora_name)}
        
        elif cmd == "lora_cache_stats":
            return self.lora_cache.get_stats()
        
        elif cmd == "register_lora":
            # Apply LoRA to CLIP with strength
            lora_name = data.get("lora_name")
            clip_strength = data.get("clip_strength", 1.0)
            model_strength = data.get("model_strength", 1.0)
            
            if not lora_name:
                return {"error": "lora_name required"}
            
            # Check if CLIP pool is available
            if self.clip_pool is None:
                return {"error": "CLIP pool not available"}
            
            # Submit LoRA registration to CLIP pool
            # The worker will handle applying the LoRA to CLIP
            return self.clip_pool.submit("register_lora", {
                "lora_name": lora_name,
                "clip_strength": clip_strength,
                "model_strength": model_strength
            })
        
        elif cmd == "clear_loras":
            # Clear all LoRAs from cache
            cleared_count = len(self.lora_cache._cache)
            self.lora_cache.clear()
            
            # Also notify CLIP pool to clear applied LoRAs
            if self.clip_pool:
                self.clip_pool.submit("clear_loras", {})
            
            return {
                "success": True,
                "cleared": cleared_count,
                "message": f"Cleared {cleared_count} LoRAs from cache"
            }
        
        # Status/info commands
        elif cmd == "ping":
            return {"pong": True, "time": time.time()}
        
        elif cmd == "get_info":
            return self.get_info()
        
        elif cmd == "get_status":
            return self.get_info()
        
        elif cmd == "negotiate_ipc":
            # IPC negotiation - check if client and daemon are on compatible GPUs
            client_gpu_id = data.get("client_gpu_id")
            daemon_gpu_id = int(self.clip_device.split(":")[-1]) if ":" in self.clip_device else 0
            
            # Enable IPC if client is on same GPU as daemon (or can do peer access)
            ipc_enabled = (client_gpu_id == daemon_gpu_id) if client_gpu_id is not None else False
            
            return {
                "ipc_enabled": ipc_enabled,
                "daemon_gpu_id": daemon_gpu_id
            }
        
        elif cmd == "load_clip_weights":
            # Load CLIP into weight registry and return IPC handles
            clip_l_path = data.get("clip_l_path")
            clip_g_path = data.get("clip_g_path")
            model_key = data.get("model_key")
            
            if not (clip_l_path or clip_g_path):
                return {"success": False, "error": "clip_l_path or clip_g_path required"}
            
            try:
                key = self.weight_registry.load_clip(clip_l_path, clip_g_path, model_key)
                handles_result = self.weight_registry.get_handles(key)
                if handles_result:
                    response = {"success": True, "model_key": key}
                    response.update(handles_result)
                    return response
                else:
                    return {"success": False, "error": "Failed to get handles"}
            except Exception as e:
                logger.error(f"[Daemon] load_clip_weights error: {e}")
                return {"success": False, "error": str(e)}
        
        elif cmd == "get_weight_handles":
            # Get IPC handles for previously loaded model
            model_key = data.get("model_key")
            
            if not model_key:
                return {"success": False, "error": "model_key required"}
            
            handles = self.weight_registry.get_handles(str(model_key))
            
            if handles:
                response = {"success": True}
                response.update(handles)
                return response
            else:
                return {"success": False, "error": f"Model not found: {model_key}"}
        
        elif cmd == "list_loaded_weights":
            # List all models in weight registry
            models = self.weight_registry.list_loaded_models()
            return {"success": True, "models": models}
        
        elif cmd == "unload_weights":
            # Unload model from weight registry
            model_key = data.get("model_key")
            
            if not model_key:
                return {"success": False, "error": "model_key required"}
            
            success = self.weight_registry.unload(str(model_key))
            return {"success": success}
        
        elif cmd == "shutdown":
            # Shutdown command - stop the daemon gracefully
            logger.info("Shutdown command received")
            # Stop in a separate thread to allow response to be sent
            import threading
            threading.Thread(target=self._delayed_shutdown, daemon=True).start()
            return {"success": True, "message": "Daemon shutting down"}
        
        elif cmd == "set_attention_mode":
            # Set attention mechanism mode
            mode = data.get("mode", "auto")
            valid_modes = ["auto", "flash", "split", "pytorch"]
            
            if mode not in valid_modes:
                return {
                    "success": False,
                    "error": f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
                }
            
            self._attention_mode = mode
            logger.info(f"Attention mode set to: {mode}")
            
            # Note: Attention mode will be applied to workers on next model load
            # Existing loaded models continue using their current attention mode
            return {
                "success": True,
                "mode": mode,
                "message": f"Attention mode set to {mode} (will apply to newly loaded models)"
            }
        
        elif cmd == "unload_models":
            # Unload models from VRAM
            logger.info("Unload models command received")
            unloaded = []
            
            if self.clip_pool:
                # CLIP pool will stop workers which unloads models
                self.clip_pool.stop()
                self.clip_pool = None
                unloaded.append("CLIP")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "unloaded": unloaded,
                "message": f"Unloaded: {', '.join(unloaded)}" if unloaded else "No models were loaded"
            }
        
        # Device configuration commands
        elif cmd == "get_devices":
            """Get current device configuration and available GPUs."""
            logger.info("[Daemon] Handling get_devices request")
            available_gpus = []
            if torch.cuda.is_available():
                available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            
            response = {
                "success": True,
                "devices": {
                    "clip": self.clip_device,
                    "vae": self.vae_device,
                    "llm": getattr(self, "llm_device", self.clip_device),  # Fallback to clip_device if not set
                },
                "available_gpus": available_gpus,
                "has_cuda": torch.cuda.is_available(),
            }
            logger.info(f"[Daemon] Returning get_devices response: {response}")
            return response
        
        elif cmd == "set_clip_device":
            """Change CLIP device at runtime."""
            device = data.get("device")
            if not device:
                return {"error": "device parameter required"}
            
            # Validate device
            if device.startswith("cuda:"):
                try:
                    device_id = int(device.split(":")[-1])
                    if device_id >= torch.cuda.device_count():
                        return {"error": f"Invalid CUDA device: {device}. Available: 0-{torch.cuda.device_count()-1}"}
                except (ValueError, IndexError):
                    return {"error": f"Invalid device format: {device}"}
            elif device != "cpu":
                return {"error": f"Invalid device: {device}. Use 'cpu' or 'cuda:N'"}
            
            old_device = self.clip_device
            self.clip_device = device
            
            # Try to move CLIP models if loaded
            if self.clip_pool:
                try:
                    self.clip_pool.set_device(device)
                    logger.info(f"[Daemon] CLIP device changed: {old_device} -> {device}")
                    return {
                        "success": True,
                        "old_device": old_device,
                        "new_device": device,
                        "message": f"CLIP device changed to {device}"
                    }
                except Exception as e:
                    self.clip_device = old_device  # Revert on error
                    return {"error": f"Failed to change CLIP device: {e}"}
            
            return {
                "success": True,
                "old_device": old_device,
                "new_device": device,
                "message": f"CLIP device set to {device} (will apply on next model load)"
            }
        
        elif cmd == "set_vae_device":
            """Change VAE device at runtime."""
            device = data.get("device")
            if not device:
                return {"error": "device parameter required"}
            
            # Validate device
            if device.startswith("cuda:"):
                try:
                    device_id = int(device.split(":")[-1])
                    if device_id >= torch.cuda.device_count():
                        return {"error": f"Invalid CUDA device: {device}. Available: 0-{torch.cuda.device_count()-1}"}
                except (ValueError, IndexError):
                    return {"error": f"Invalid device format: {device}"}
            elif device != "cpu":
                return {"error": f"Invalid device: {device}. Use 'cpu' or 'cuda:N'"}
            
            old_device = self.vae_device
            self.vae_device = device
            
            logger.info(f"[Daemon] VAE device changed: {old_device} -> {device}")
            return {
                "success": True,
                "old_device": old_device,
                "new_device": device,
                "message": f"VAE device set to {device} (will apply on next operation)"
            }
        
        elif cmd == "set_llm_device":
            """Change LLM device at runtime."""
            device = data.get("device")
            if not device:
                return {"error": "device parameter required"}
            
            # Validate device
            if device.startswith("cuda:"):
                try:
                    device_id = int(device.split(":")[-1])
                    if device_id >= torch.cuda.device_count():
                        return {"error": f"Invalid CUDA device: {device}. Available: 0-{torch.cuda.device_count()-1}"}
                except (ValueError, IndexError):
                    return {"error": f"Invalid device format: {device}"}
            elif device != "cpu":
                return {"error": f"Invalid device: {device}. Use 'cpu' or 'cuda:N'"}
            
            old_device = getattr(self, "llm_device", self.clip_device)
            self.llm_device = device
            
            logger.info(f"[Daemon] LLM device changed: {old_device} -> {device}")
            return {
                "success": True,
                "old_device": old_device,
                "new_device": device,
                "message": f"LLM device set to {device} (will apply on next model load)"
            }
        
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    def _handle_client(self, conn: socket.socket, addr):
        """Handle a client connection."""
        logger.debug(f"Client connected: {addr}")
        
        try:
            while self._running:
                # Receive length prefix
                length_data = conn.recv(8)
                if not length_data or len(length_data) < 8:
                    break
                
                msg_length = int.from_bytes(length_data, 'big')
                
                # Receive message
                data = b''
                while len(data) < msg_length:
                    chunk = conn.recv(min(msg_length - len(data), 65536))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < msg_length:
                    break
                
                # Deserialize request
                try:
                    request = pickle.loads(data)
                except Exception as e:
                    logger.error(f"Failed to deserialize request: {e}")
                    continue
                
                cmd = request.get("cmd", "")
                
                # Process request
                try:
                    result = self._handle_request(cmd, request)
                except Exception as e:
                    logger.error(f"Error handling {cmd}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    result = {"error": str(e)}
                
                # Serialize and send response
                try:
                    response_data = pickle.dumps(result)
                    length_prefix = len(response_data).to_bytes(8, 'big')
                    conn.sendall(length_prefix + response_data)
                except Exception as e:
                    logger.error(f"Failed to send response: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
            logger.debug(f"Client disconnected: {addr}")
    
    def start(self):
        """Start the daemon server."""
        logger.info("=" * 60)
        logger.info("Luna Daemon Starting...")
        logger.info("=" * 60)
        
        self._running = True
        self._start_time = time.time()
        
        # Start worker pools based on service type
        service = self.service_type
        if hasattr(service, 'value'):
            service = service  # Already enum
        
        if service in (ServiceType.FULL, CoreServiceType.FULL) or str(service) == "ServiceType.FULL":
            # Start CLIP pool (VAE loads locally)
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=self.clip_device,
                precision=CLIP_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.clip_pool.start()
            logger.info(f"[CLIP] Worker pool started on {self.clip_device}")
        
        elif service in (ServiceType.CLIP_ONLY, CoreServiceType.CLIP_ONLY) or "CLIP" in str(service):
            self.clip_pool = WorkerPool(
                worker_type=WorkerType.CLIP,
                device=self.clip_device,
                precision=CLIP_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.clip_pool.start()
            logger.info(f"[CLIP] Worker pool started on {self.clip_device}")
        
        # Start CLIP Vision pool if vision path is configured
        vision_path = self.config_paths.get('clip_vision')
        if vision_path:
            self.vision_pool = WorkerPool(
                worker_type=WorkerType.CLIP_VISION,
                device=self.vision_device,
                precision=CLIP_PRECISION,
                config=self.scaling_config,
                on_scale_event=self._on_scale_event,
                config_paths=self.config_paths
            )
            self.vision_pool.start()
            logger.info(f"[CLIP_VISION] Worker pool started on {self.vision_device}")
        else:
            logger.info("[CLIP_VISION] No vision model configured, pool not started")
        
        # Start WebSocket monitoring
        self.ws_server = WebSocketServer(
            status_provider=self.get_info,
            host=self.host,
            port=self.ws_port
        )
        self.ws_server.start()
        logger.info(f"[WS] Monitoring server started on ws://{self.host}:{self.ws_port}")
        
        # Start socket server
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._server_socket.settimeout(1.0)
        
        logger.info(f"[Socket] Server listening on {self.host}:{self.port}")
        logger.info("=" * 60)
        logger.info("Luna Daemon Ready!")
        logger.info("=" * 60)
        
        # Accept loop
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
                self._client_threads.add(thread)
                
                # Clean up finished threads
                self._client_threads = {t for t in self._client_threads if t.is_alive()}
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _save_images_worker(self, task_data: Dict[str, Any], job_id: str):
        """Background worker that actually saves images to disk"""
        try:
            import json
            import numpy as np
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo
            from datetime import datetime
            
            # Try to import piexif for EXIF support
            try:
                import piexif
                import piexif.helper
                has_piexif = True
            except ImportError:
                has_piexif = False
            
            # Extract all parameters from task_data
            save_path = task_data.get("save_path", "")
            filename_template = task_data.get("filename", "")
            model_name_raw = task_data.get("model_name", "")
            quality_gate = task_data.get("quality_gate", "disabled")
            min_quality_threshold = task_data.get("min_quality_threshold", 0.3)
            png_compression = task_data.get("png_compression", 4)
            lossy_quality = task_data.get("lossy_quality", 90)
            lossless_webp = task_data.get("lossless_webp", False)
            embed_workflow = task_data.get("embed_workflow", True)
            filename_index = task_data.get("filename_index", 0)
            custom_metadata = task_data.get("custom_metadata", "")
            metadata = task_data.get("metadata", {})
            prompt = task_data.get("prompt")
            extra_pnginfo = task_data.get("extra_pnginfo", {})
            images = task_data.get("images", [])
            output_dir = task_data.get("output_dir", "")
            timestamp_str = task_data.get("timestamp", datetime.now().isoformat())
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.now()
            
            saved_count = 0
            
            for img_data in images:
                image_np = img_data.get("image")
                affix = img_data.get("affix", "IMAGE")
                fmt = img_data.get("format", "png")
                use_subdir = img_data.get("subdir", True)
                
                if image_np is None:
                    continue
                
                # Ensure numpy array
                if not isinstance(image_np, np.ndarray):
                    image_np = np.array(image_np)
                
                # Ensure proper range [0-255] uint8
                if image_np.dtype != np.uint8:
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image_np.astype(np.uint8)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(image_np)
                
                # Build save directory
                if save_path:
                    if use_subdir:
                        save_dir = os.path.join(output_dir, save_path, affix)
                    else:
                        save_dir = os.path.join(output_dir, save_path)
                else:
                    if use_subdir:
                        save_dir = os.path.join(output_dir, affix)
                    else:
                        save_dir = output_dir
                
                os.makedirs(save_dir, exist_ok=True)
                
                # Build filename
                batch_timestamp = timestamp.strftime("%Y_%m_%d_%H%M%S")
                
                # Parse model name
                model_name = os.path.basename(model_name_raw)
                for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'):
                    if model_name.lower().endswith(ext):
                        model_name = model_name[:-len(ext)]
                        break
                
                # Simple filename generation
                if filename_template:
                    filename_base = filename_template.replace("%model_name%", model_name)
                    filename_base = filename_base.replace("%index%", str(filename_index))
                else:
                    filename_base = f"{batch_timestamp}_{model_name}"
                
                ext = fmt.lower()
                if ext == "jpeg":
                    ext = "jpg"
                
                final_filename = f"{filename_base}_{affix}.{ext}"
                file_path = os.path.join(save_dir, final_filename)
                
                # Save with metadata
                if ext == 'png':
                    png_metadata = None
                    if embed_workflow:
                        png_metadata = PngInfo()
                        if prompt is not None:
                            png_metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for key in extra_pnginfo:
                                png_metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                        if metadata:
                            png_metadata.add_text("luna_metadata", json.dumps(metadata))
                    
                    pil_img.save(file_path, compress_level=png_compression, pnginfo=png_metadata)
                
                elif ext == 'webp':
                    if lossless_webp:
                        pil_img.save(file_path, lossless=True)
                    else:
                        pil_img.save(file_path, quality=lossy_quality)
                
                elif ext == 'jpg':
                    # JPEG with EXIF metadata
                    exif_data = None
                    if embed_workflow and has_piexif:
                        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
                        if metadata:
                            user_comment = json.dumps(metadata)
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment)
                        exif_data = piexif.dump(exif_dict)
                    
                    pil_img.save(file_path, quality=lossy_quality, exif=exif_data if exif_data else None)
                
                saved_count += 1
                logger.debug(f"[Daemon] Saved image: {file_path}")
            
            logger.info(f"[Daemon] Image save job {job_id} complete ({saved_count} images saved)")
            
        except Exception as e:
            logger.error(f"[Daemon] Image save worker error (Job {job_id}): {e}", exc_info=True)
    
    def stop(self):
        """Stop the daemon server."""
        logger.info("Luna Daemon stopping...")
        self._running = False
        
        # Stop pools
        if self.clip_pool:
            self.clip_pool.stop()
        
        # Stop WebSocket
        if self.ws_server:
            self.ws_server.stop()
        
        # Close socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
        
        logger.info("Luna Daemon stopped")
    
    def run(self):
        """Alias for start() - for backward compatibility with tray app."""
        self.start()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    daemon = LunaDaemon()
    
    try:
        daemon.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
