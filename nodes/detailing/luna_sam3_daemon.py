"""
Luna SAM3 Daemon - Persistent SAM3 Model Manager

Manages SAM3 model lifecycle to prevent redundant VRAM loading across multiple
detector nodes. Uses singleton pattern similar to Luna Daemon for CLIP/VAE sharing.

Architecture:
- Single global instance holds SAM3 model in VRAM
- Multiple detector nodes share the same model reference
- Configurable offload device (CPU/GPU) for memory management
- Thread-safe access for multi-instance ComfyUI workflows
"""

import os
import gc
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import comfy.model_management
from folder_paths import models_dir

# Global registry for SAM3 models
_SAM3_MODELS: Dict[str, Any] = {}
_SAM3_LOCK = threading.Lock()


def get_sam3_model_path(model_name: str) -> Optional[str]:
    """
    Locate SAM3 model file in ComfyUI's models directory.
    
    Searches in: models/sam3/
    Supports: .pt, .safetensors
    
    Args:
        model_name: Name of model file (with or without extension)
        
    Returns:
        Absolute path to model file, or None if not found
    """
    sam3_dir = Path(models_dir) / "sam3"
    
    if not sam3_dir.exists():
        return None
    
    # Try exact match first
    if (sam3_dir / model_name).exists():
        return str(sam3_dir / model_name)
    
    # Try with extensions
    for ext in [".pt", ".safetensors"]:
        model_path = sam3_dir / f"{model_name}{ext}"
        if model_path.exists():
            return str(model_path)
    
    return None


class LunaSAM3Daemon:
    """
    Singleton SAM3 model manager.
    
    Provides persistent model storage across node executions to minimize
    VRAM churn and loading overhead.
    
    Usage:
        predictor = LunaSAM3Daemon.get_predictor(model_name="sam3_h.safetensors")
        # Use predictor for detection
        LunaSAM3Daemon.release_model(model_name)  # Optional explicit cleanup
    """
    
    @classmethod
    def get_predictor(
        cls,
        model_name: str = "sam3_h.safetensors",
        device: Optional[str] = None,
        offload_device: str = "cpu"
    ) -> Any:
        """
        Get or load SAM3 predictor.
        
        Thread-safe model loading with automatic device management.
        
        Args:
            model_name: Model filename in models/sam3/
            device: Target device (None = auto-select based on torch)
            offload_device: Device for model offloading when not in use
            
        Returns:
            SAM3 unified model instance (supports both image/video)
            
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model loading fails
        """
        with _SAM3_LOCK:
            # Check if model already loaded
            if model_name in _SAM3_MODELS:
                print(f"[LunaSAM3Daemon] Reusing loaded model: {model_name}")
                return _SAM3_MODELS[model_name]
            
            # Locate model file
            model_path = get_sam3_model_path(model_name)
            if model_path is None:
                raise FileNotFoundError(
                    f"[LunaSAM3Daemon] SAM3 model '{model_name}' not found in models/sam3/. "
                    f"Please download the model and place it in: {Path(models_dir) / 'sam3'}"
                )
            
            # Determine device
            if device is None:
                device = comfy.model_management.get_torch_device()
                device = str(device)
            
            print(f"[LunaSAM3Daemon] Loading SAM3 model: {model_path}")
            print(f"[LunaSAM3Daemon] Load device: {device}, Offload device: {offload_device}")
            
            try:
                # Import SAM3 components from installed comfyui-sam3 package
                import sys
                from pathlib import Path
                
                # Find comfyui-sam3 installation
                sam3_path = Path(models_dir).parent.parent / "custom_nodes" / "comfyui-sam3"
                if sam3_path.exists() and str(sam3_path) not in sys.path:
                    sys.path.insert(0, str(sam3_path))
                
                from nodes.sam3_lib.model_builder import build_sam3_video_predictor
                from nodes.sam3_lib.sam3_image_processor import Sam3Processor
                from nodes.load_model import SAM3UnifiedModel
                
                # Build video predictor (unified model for image/video)
                video_predictor = build_sam3_video_predictor(
                    model_path,
                    device=device,
                    offload_device=offload_device
                )
                
                # Create processor for image segmentation
                processor = Sam3Processor()
                
                # Wrap in unified model (following comfyui-sam3 pattern)
                unified_model = SAM3UnifiedModel(
                    video_predictor=video_predictor,
                    processor=processor,
                    load_device=device,
                    offload_device=offload_device
                )
                
                # Cache model
                _SAM3_MODELS[model_name] = unified_model
                
                print(f"[LunaSAM3Daemon] Model loaded successfully: {model_name}")
                return unified_model
                
            except ImportError as e:
                raise RuntimeError(
                    f"[LunaSAM3Daemon] Failed to import SAM3 components. "
                    f"Is comfyui-sam3 installed? Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[LunaSAM3Daemon] Failed to load SAM3 model '{model_name}': {e}"
                )
    
    @classmethod
    def release_model(cls, model_name: str) -> bool:
        """
        Explicitly release a SAM3 model from memory.
        
        Optional - models are automatically managed by ComfyUI's system.
        Use this for explicit cleanup when switching models frequently.
        
        Args:
            model_name: Model to release
            
        Returns:
            True if model was released, False if not loaded
        """
        with _SAM3_LOCK:
            if model_name not in _SAM3_MODELS:
                return False
            
            model = _SAM3_MODELS.pop(model_name)
            
            # Move to CPU and clear cache
            try:
                if hasattr(model, 'model'):
                    model.model.cpu()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[LunaSAM3Daemon] Released model: {model_name}")
                return True
            except Exception as e:
                print(f"[LunaSAM3Daemon] Error releasing model: {e}")
                return False
    
    @classmethod
    def get_loaded_models(cls) -> list:
        """Get list of currently loaded model names."""
        with _SAM3_LOCK:
            return list(_SAM3_MODELS.keys())
    
    @classmethod
    def clear_all(cls):
        """Release all loaded models. Use with caution."""
        with _SAM3_LOCK:
            model_names = list(_SAM3_MODELS.keys())
            for name in model_names:
                cls.release_model(name)
            print("[LunaSAM3Daemon] All models cleared")


# Cleanup on module unload
def _cleanup():
    """Cleanup function called on module unload."""
    LunaSAM3Daemon.clear_all()

import atexit
atexit.register(_cleanup)
