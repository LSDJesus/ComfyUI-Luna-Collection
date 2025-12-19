"""
CUDA IPC Model Wrapper

Wraps shared model weights accessed via CUDA IPC handles.
Allows local execution with zero-copy shared GPU memory.

This module reconstructs models using IPC-shared weights instead of
loading them from disk, enabling multi-instance VRAM deduplication.
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def open_ipc_handle_to_tensor(ipc_data: Dict[str, Any]) -> torch.Tensor:
    """
    Open CUDA IPC handle and reconstruct tensor.
    
    Args:
        ipc_data: Dict with ipc_tuple, shape, dtype, etc.
    
    Returns:
        Tensor pointing to shared GPU memory
    """
    try:
        # Parse dtype
        dtype_str = ipc_data["dtype"].replace("torch.", "")
        dtype = getattr(torch, dtype_str)
        
        # Map dtype string to PyTorch storage class names
        dtype_to_storage = {
            "float32": "FloatStorage",
            "float16": "HalfStorage",
            "bfloat16": "BFloat16Storage",
            "float64": "DoubleStorage",
            "int32": "IntStorage",
            "int64": "LongStorage",
            "int16": "ShortStorage",
            "int8": "CharStorage",
            "uint8": "ByteStorage",
        }
        
        storage_name = dtype_to_storage.get(dtype_str)
        if not storage_name:
            raise ValueError(f"Unsupported dtype for IPC: {dtype_str}")
        
        # Get storage class from torch.cuda
        storage_cls = getattr(torch.cuda, storage_name)
        
        # Reconstruct storage from full IPC tuple (not just handle bytes)
        # _new_shared_cuda expects the complete tuple from _share_cuda_()
        shared_storage = storage_cls._new_shared_cuda(*ipc_data["ipc_tuple"])
        
        # Create tensor from shared storage
        tensor = torch.tensor([], dtype=dtype, device='cuda').set_(
            shared_storage,
            ipc_data.get("offset", 0),
            ipc_data["shape"]
        )
        tensor.requires_grad = ipc_data.get("requires_grad", False)
        
        return tensor
        
    except Exception as e:
        logger.error(f"[IPC] Failed to open handle: {e}")
        raise


class SharedWeightModel:
    """
    Base class for models using IPC-shared weights.
    
    Reconstructs model structure locally but references shared GPU memory
    instead of loading weights from disk.
    """
    
    def __init__(self, ipc_handles: Dict[str, Dict], model_type: str):
        """
        Args:
            ipc_handles: Dict of parameter name -> IPC handle metadata
            model_type: "vae" or "clip"
        """
        self.ipc_handles = ipc_handles
        self.model_type = model_type
        self.shared_tensors: Dict[str, torch.Tensor] = {}
        
        # Open all IPC handles
        self._open_handles()
    
    def _open_handles(self):
        """Open all IPC handles to get shared weight tensors."""
        logger.info(f"[IPC] Opening {len(self.ipc_handles)} shared weight tensors")
        
        for name, handle_data in self.ipc_handles.items():
            try:
                tensor = open_ipc_handle_to_tensor(handle_data)
                self.shared_tensors[name] = tensor
            except Exception as e:
                logger.warning(f"[IPC] Could not open handle for {name}: {e}")
        
        logger.info(f"[IPC] [OK] Opened {len(self.shared_tensors)} shared tensors")
    
    def close_handles(self):
        """Close IPC handles (call on cleanup)."""
        for name, tensor in self.shared_tensors.items():
            try:
                # Close memory handle
                torch.cuda.cudart().cudaIpcCloseMemHandle(tensor.data_ptr())
            except:
                pass
        
        self.shared_tensors.clear()
        logger.info(f"[IPC] Closed shared weight handles")


class SharedVAE(SharedWeightModel):
    """
    VAE model using IPC-shared weights.
    
    Executes encode/decode locally using shared GPU memory.
    """
    
    def __init__(self, ipc_handles: Dict[str, Dict], vae_path: str):
        super().__init__(ipc_handles, "vae")
        self.vae_path = vae_path
        
        # Build local VAE structure pointing to shared weights
        self._build_vae_model()
    
    def _build_vae_model(self):
        """
        Reconstruct VAE model structure with shared weights.
        
        This creates a local VAE instance but injects the IPC-shared
        weight tensors instead of loading from disk.
        """
        try:
            from comfy.sd import VAE
            import comfy.utils
            
            # Load VAE architecture (metadata only, no weights)
            vae_sd = comfy.utils.load_torch_file(self.vae_path)
            
            # Create VAE instance
            self.vae = VAE(sd=vae_sd)
            
            # Inject shared weights into model
            model_dict = self.vae.first_stage_model.state_dict()
            
            for name in model_dict.keys():
                if name in self.shared_tensors:
                    # Replace with shared tensor
                    model_dict[name] = self.shared_tensors[name]
            
            # Load the modified state dict
            self.vae.first_stage_model.load_state_dict(model_dict, strict=False)
            self.vae.first_stage_model.eval()
            
            logger.info(f"[IPC] [OK] VAE model built with shared weights")
            
        except Exception as e:
            logger.error(f"[IPC] Failed to build VAE model: {e}")
            raise
    
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latents using shared weights."""
        with torch.inference_mode():
            return self.vae.encode(pixels)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixels using shared weights."""
        with torch.inference_mode():
            return self.vae.decode(latents)


class SharedCLIP(SharedWeightModel):
    """
    CLIP model using IPC-shared weights.
    
    Executes text encoding locally using shared GPU memory.
    """
    
    def __init__(self, ipc_handles: Dict[str, Dict], clip_l_path: Optional[str] = None, 
                 clip_g_path: Optional[str] = None):
        super().__init__(ipc_handles, "clip")
        self.clip_l_path = clip_l_path
        self.clip_g_path = clip_g_path
        
        # Build local CLIP structure pointing to shared weights
        self._build_clip_model()
    
    def _build_clip_model(self):
        """
        Reconstruct CLIP model structure with shared weights.
        
        Similar to VAE - creates local instance with IPC-shared weights.
        """
        try:
            from comfy.sd import load_clip
            
            # Load CLIP architecture
            clip_paths = []
            if self.clip_l_path:
                clip_paths.append(self.clip_l_path)
            if self.clip_g_path:
                clip_paths.append(self.clip_g_path)
            
            self.clip = load_clip(ckpt_paths=clip_paths, embedding_directory=None)
            
            # Inject shared weights
            model_dict = self.clip.cond_stage_model.state_dict()
            
            for name in model_dict.keys():
                if name in self.shared_tensors:
                    model_dict[name] = self.shared_tensors[name]
            
            self.clip.cond_stage_model.load_state_dict(model_dict, strict=False)
            self.clip.cond_stage_model.eval()
            
            logger.info(f"[IPC] [OK] CLIP model built with shared weights")
            
        except Exception as e:
            logger.error(f"[IPC] Failed to build CLIP model: {e}")
            raise
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text."""
        return self.clip.tokenize(text)
    
    def encode_from_tokens(self, tokens: torch.Tensor, return_pooled: bool = False) -> Any:
        """Encode tokens using shared weights."""
        with torch.inference_mode():
            return self.clip.encode_from_tokens(tokens, return_pooled=return_pooled)
