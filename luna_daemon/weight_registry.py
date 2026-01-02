"""
Luna Daemon Weight Registry

Manages shared model weights via CUDA IPC handles.

Architecture:
- Load models once on daemon GPU
- Generate IPC handles to weight tensors
- Distribute handles to client instances
- Clients access shared GPU memory directly (zero copy)

This eliminates socket overhead - clients run encode/decode locally
using shared weights instead of sending requests to the daemon.
"""

import torch
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import pickle
import comfy.utils  # type: ignore

logger = logging.getLogger(__name__)


class WeightRegistry:
    """Registry of shared model weights accessible via CUDA IPC."""
    
    def __init__(self, device: str = "cuda:1"):
        """
        Args:
            device: GPU device where shared weights are stored
        """
        self.device = torch.device(device)
        self.weights: Dict[str, Dict[str, Any]] = {}  # model_key -> {model, ipc_handles, metadata}
        
        logger.info(f"[WeightRegistry] Initialized on {device}")
    
    def load_vae(self, vae_path: str, model_key: Optional[str] = None) -> str:
        """
        Load VAE model and create IPC handles to its weights.
        
        Args:
            vae_path: Path to VAE safetensors file
            model_key: Optional key for registry (defaults to path)
        
        Returns:
            model_key for future reference
        """
        if model_key is None:
            model_key = f"vae:{vae_path}"
        
        # Check if already loaded
        if model_key in self.weights:
            logger.info(f"[WeightRegistry] VAE already loaded: {model_key}")
            return model_key
        
        logger.info(f"[WeightRegistry] Loading VAE from {vae_path}")
        
        try:
            # Import ComfyUI's VAE loader
            import folder_paths
            from comfy.sd import VAE
            
            # Load VAE model
            vae_sd = comfy.utils.load_torch_file(vae_path)
            vae = VAE(sd=vae_sd)
            vae.first_stage_model.to(self.device)
            vae.first_stage_model.eval()
            
            # Generate IPC handles for model parameters
            ipc_handles = self._create_ipc_handles(vae.first_stage_model)
            
            # Store in registry
            self.weights[model_key] = {
                "model": vae,
                "ipc_handles": ipc_handles,
                "type": "vae",
                "path": vae_path,
                "device": str(self.device)
            }
            
            logger.info(f"[WeightRegistry] [OK] VAE loaded with {len(ipc_handles)} weight tensors")
            return model_key
            
        except Exception as e:
            logger.error(f"[WeightRegistry] Failed to load VAE: {e}")
            raise
    
    def load_clip(self, clip_l_path: Optional[str] = None, clip_g_path: Optional[str] = None, 
                  model_key: Optional[str] = None) -> str:
        """
        Load CLIP model(s) and create IPC handles to weights.
        
        Args:
            clip_l_path: Path to CLIP-L safetensors
            clip_g_path: Path to CLIP-G safetensors (for SDXL)
            model_key: Optional key for registry
        
        Returns:
            model_key for future reference
        """
        if model_key is None:
            paths = [p for p in [clip_l_path, clip_g_path] if p]
            model_key = f"clip:{':'.join(paths)}"
        
        # Check if already loaded
        if model_key in self.weights:
            logger.info(f"[WeightRegistry] CLIP already loaded: {model_key}")
            return model_key
        
        logger.info(f"[WeightRegistry] Loading CLIP (L={clip_l_path}, G={clip_g_path})")
        
        try:
            # Import ComfyUI's CLIP loader
            from comfy.sd import load_clip
            
            # Load CLIP model(s)
            clip_paths = []
            if clip_l_path:
                clip_paths.append(clip_l_path)
            if clip_g_path:
                clip_paths.append(clip_g_path)
            
            clip = load_clip(ckpt_paths=clip_paths, embedding_directory=None)
            
            # Move to device
            clip.cond_stage_model.to(self.device)
            clip.cond_stage_model.eval()
            
            # Generate IPC handles
            ipc_handles = self._create_ipc_handles(clip.cond_stage_model)
            
            # Store in registry
            self.weights[model_key] = {
                "model": clip,
                "ipc_handles": ipc_handles,
                "type": "clip",
                "clip_l_path": clip_l_path,
                "clip_g_path": clip_g_path,
                "device": str(self.device)
            }
            
            logger.info(f"[WeightRegistry] [OK] CLIP loaded with {len(ipc_handles)} weight tensors")
            return model_key
            
        except Exception as e:
            logger.error(f"[WeightRegistry] Failed to load CLIP: {e}")
            raise
    
    def _create_ipc_handles(self, model: torch.nn.Module) -> Dict[str, Dict]:
        """
        Create CUDA IPC handles for all parameters in a model.
        
        Args:
            model: PyTorch model with parameters on GPU
        
        Returns:
            Dict mapping parameter name to IPC handle metadata
        """
        ipc_handles = {}
        
        for name, param in model.named_parameters():
            if param.is_cuda and param.device == self.device:
                try:
                    # For PyTorch CUDA IPC, we share storage, not tensor directly
                    # _share_cuda_() returns a tuple: (storage_type, device, handle, size, ...)
                    storage = param.storage()
                    ipc_tuple = storage._share_cuda_()  # Returns full tuple of IPC metadata
                    
                    # Store the complete IPC tuple plus tensor shape info
                    ipc_handles[name] = {
                        "ipc_tuple": ipc_tuple,  # Store the entire tuple
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "requires_grad": param.requires_grad,
                        "offset": param.storage_offset()
                    }
                except Exception as e:
                    logger.warning(f"[WeightRegistry] Could not create IPC handle for {name}: {e}")
        
        return ipc_handles
    
    def get_handles(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Get IPC handles for a loaded model.
        
        Args:
            model_key: Key from load_vae or load_clip
        
        Returns:
            Dict with ipc_handles and metadata, or None if not found
        """
        if model_key not in self.weights:
            logger.warning(f"[WeightRegistry] Model not found: {model_key}")
            return None
        
        entry = self.weights[model_key]
        return {
            "ipc_handles": entry["ipc_handles"],
            "type": entry["type"],
            "device": entry["device"],
            "path": entry.get("path") or entry.get("clip_l_path")
        }
    
    def _get_tensor_memory_mb(self, ipc_handles: Dict[str, Dict]) -> float:
        """
        Calculate total memory usage of a model from its IPC handles.
        
        Args:
            ipc_handles: Dict of IPC handle metadata
        
        Returns:
            Memory usage in MB
        """
        total_bytes = 0
        for name, handle_data in ipc_handles.items():
            shape = handle_data.get("shape", [])
            dtype_str = handle_data.get("dtype", "torch.float32")
            
            # Calculate element count
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            
            # Get bytes per element based on dtype
            dtype_sizes = {
                "torch.float32": 4,
                "torch.float16": 2,
                "torch.bfloat16": 2,
                "torch.int32": 4,
                "torch.int64": 8,
                "torch.int8": 1,
                "torch.uint8": 1,
            }
            bytes_per_element = dtype_sizes.get(dtype_str, 4)  # Default to 4 if unknown
            
            total_bytes += num_elements * bytes_per_element
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def list_loaded_models(self) -> List[str]:
        """Get list of all loaded model keys."""
        return list(self.weights.keys())
    
    def get_model_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all loaded models including VRAM usage.
        
        Returns:
            List of dicts with model metadata and memory info
        """
        models = []
        for key, entry in self.weights.items():
            memory_mb = self._get_tensor_memory_mb(entry["ipc_handles"])
            
            model_info = {
                "key": key,
                "type": entry["type"],
                "device": entry["device"],
                "memory_mb": round(memory_mb, 2),
                "memory_gb": round(memory_mb / 1024, 3),
                "tensor_count": len(entry["ipc_handles"])
            }
            
            # Add path information
            if entry["type"] == "vae":
                model_info["path"] = entry.get("path", "unknown")
            elif entry["type"] == "clip":
                model_info["clip_l_path"] = entry.get("clip_l_path", "unknown")
                model_info["clip_g_path"] = entry.get("clip_g_path", "unknown")
            
            models.append(model_info)
        
        return models
    
    def unload(self, model_key: str) -> bool:
        """
        Unload a model and free its resources.
        
        Args:
            model_key: Key to unload
        
        Returns:
            True if unloaded, False if not found
        """
        if model_key not in self.weights:
            return False
        
        entry = self.weights[model_key]
        
        # Close IPC handles
        for name, handle_data in entry["ipc_handles"].items():
            try:
                # cudaIpcCloseMemHandle would be called here if needed
                pass
            except:
                pass
        
        # Delete model
        del self.weights[model_key]
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        logger.info(f"[WeightRegistry] Unloaded {model_key}")
        return True
    
    def unload_all(self):
        """Unload all models."""
        keys = list(self.weights.keys())
        for key in keys:
            self.unload(key)
        
        logger.info("[WeightRegistry] Unloaded all models")
