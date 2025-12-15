"""
Luna Daemon Proxy Objects

Proxy VAE and CLIP classes that route operations to the Luna Daemon.
These can be passed to ANY node that expects VAE or CLIP.

Also exports InferenceModeWrapper for model inference optimization.

Architecture:
    - DaemonVAE: Routes encode/decode to daemon's VAE workers
    - DaemonCLIP: Routes text encoding to daemon's CLIP workers  
    - InferenceModeWrapper: Wraps local models with inference_mode()
    
The key insight: VAE and CLIP are simple request/response patterns that
work well over sockets. Model inference (diffusion) is too complex and
should stay local with just an inference_mode() wrapper.
"""

import os
import torch
import hashlib
from typing import Optional, Dict, Any, List

# Try relative imports first, fallback to direct
try:
    from . import client as daemon_client
    from .client import DaemonConnectionError, ModelMismatchError
    from .inference_wrapper import InferenceModeWrapper, wrap_model_for_inference
except (ImportError, ValueError):
    # Fallback: load modules directly
    import importlib.util
    daemon_dir = os.path.dirname(__file__)
    
    spec = importlib.util.spec_from_file_location("luna_client", os.path.join(daemon_dir, "client.py"))
    if spec and spec.loader:
        daemon_client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daemon_client)
        DaemonConnectionError = daemon_client.DaemonConnectionError
        ModelMismatchError = daemon_client.ModelMismatchError
    else:
        raise ImportError("Could not load client module")
    
    spec = importlib.util.spec_from_file_location("luna_wrapper", os.path.join(daemon_dir, "inference_wrapper.py"))
    if spec and spec.loader:
        wrapper_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper_mod)
        InferenceModeWrapper = wrapper_mod.InferenceModeWrapper
        wrap_model_for_inference = wrapper_mod.wrap_model_for_inference
    else:
        raise ImportError("Could not load inference_wrapper module")


# =============================================================================
# Model Family Detection
# =============================================================================

def detect_clip_type(clip) -> str:
    """Detect the type of CLIP model from a CLIP object."""
    try:
        if clip is None:
            return 'unknown'
        
        cond_model = getattr(clip, 'cond_stage_model', None)
        if cond_model is None and hasattr(clip, 'patcher'):
            cond_model = getattr(clip.patcher, 'model', None)
        
        if cond_model is not None:
            class_name = type(cond_model).__name__
            
            if 'SDXL' in class_name or hasattr(cond_model, 'clip_g'):
                return 'sdxl'
            if 'Flux' in class_name or 't5xxl' in class_name.lower():
                return 'flux'
            if 'SD3' in class_name:
                return 'sd3'
            if 'SD1' in class_name or 'SDClip' in class_name:
                return 'sd15'
        
        clip_class = type(clip).__name__
        if 'SDXL' in clip_class:
            return 'sdxl'
        elif 'Flux' in clip_class:
            return 'flux'
        elif 'SD3' in clip_class:
            return 'sd3'
        
        return 'sdxl'  # Default assumption
        
    except Exception as e:
        print(f"[DaemonProxy] Warning: Could not detect CLIP type: {e}")
        return 'unknown'


def detect_vae_type(vae) -> str:
    """Detect the VAE type/family from a VAE object."""
    try:
        if vae is None:
            return 'unknown'
        
        first_stage = getattr(vae, 'first_stage_model', None)
        if first_stage is None:
            first_stage = vae
        
        if hasattr(first_stage, 'config'):
            config = first_stage.config
            latent_channels = getattr(config, 'latent_channels', None)
            if latent_channels == 16:
                return 'flux'
            elif latent_channels == 4:
                return 'sdxl'
        
        if hasattr(first_stage, 'encoder'):
            encoder = first_stage.encoder
            if hasattr(encoder, 'conv_out'):
                out_channels = encoder.conv_out.out_channels
                if out_channels == 32:
                    return 'flux'
                elif out_channels == 8:
                    return 'sdxl'
        
        class_name = type(first_stage).__name__
        if 'Flux' in class_name:
            return 'flux'
        elif 'SD3' in class_name:
            return 'sd3'
        
        return 'sdxl'
        
    except Exception as e:
        print(f"[DaemonProxy] Warning: Could not detect VAE type: {e}")
        return 'unknown'


# =============================================================================
# Daemon VAE Proxy
# =============================================================================

class DaemonVAE:
    """
    A VAE proxy that routes all encode/decode operations to the Luna Daemon.
    
    This object can be passed to any ComfyUI node that expects a VAE,
    including third-party nodes like FaceDetailer, UltimateSDUpscale, etc.
    """
    
    def __init__(
        self, 
        source_vae: Optional[Any] = None,
        vae_type: Optional[str] = None,
        use_existing: bool = False
    ):
        self.source_vae = source_vae
        self.use_existing = use_existing
        self._registered = False
        self._memory_used = 0
        
        if vae_type is None and source_vae is not None:
            self.vae_type = detect_vae_type(source_vae)
        else:
            self.vae_type = vae_type or 'unknown'
        
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.dtype = torch.float32
    
    def _ensure_registered(self):
        """Ensure VAE is registered with daemon before use."""
        if self._registered or self.use_existing:
            return
        
        if self.source_vae is None:
            raise RuntimeError("DaemonVAE has no source VAE and use_existing=False")
        
        try:
            daemon_client.register_vae(self.source_vae, self.vae_type)
            self._registered = True
        except Exception as e:
            raise RuntimeError(f"Failed to register VAE with daemon: {e}")
    
    def _check_daemon(self):
        """Verify daemon is running."""
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel."
            )
    
    def encode(self, pixel_samples: torch.Tensor, auto_tile: bool = False) -> torch.Tensor:
        """Encode pixels to latent space via daemon."""
        print(f"[DaemonVAE] encode() - type={self.vae_type}, shape={pixel_samples.shape}")
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_encode(
                pixel_samples, 
                self.vae_type,
                tiled=auto_tile,
                tile_size=512,
                overlap=64
            )
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def decode(self, samples_in: torch.Tensor, vae_options: Optional[Dict] = None,
               auto_tile: bool = False) -> torch.Tensor:
        """Decode latents to pixels via daemon."""
        print(f"[DaemonVAE] decode() - type={self.vae_type}, shape={samples_in.shape}")
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_decode(
                samples_in, 
                self.vae_type,
                tiled=auto_tile,
                tile_size=64,
                overlap=16
            )
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode_tiled(self, pixel_samples: torch.Tensor, 
                     tile_x: int = 512, tile_y: int = 512, 
                     overlap: int = 64, **kwargs) -> torch.Tensor:
        """Tiled encoding for large images."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_encode(
                pixel_samples,
                self.vae_type,
                tiled=True,
                tile_size=min(tile_x, tile_y),
                overlap=overlap
            )
        except (ModelMismatchError, DaemonConnectionError) as e:
            raise RuntimeError(str(e))
    
    def decode_tiled(self, samples: torch.Tensor,
                     tile_x: int = 64, tile_y: int = 64,
                     overlap: int = 16, **kwargs) -> torch.Tensor:
        """Tiled decoding for large latents."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_decode(
                samples,
                self.vae_type,
                tiled=True,
                tile_size=min(tile_x, tile_y),
                overlap=overlap
            )
        except (ModelMismatchError, DaemonConnectionError) as e:
            raise RuntimeError(str(e))
    
    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        return self.encode_tiled(pixel_samples, tile_x, tile_y, overlap)
    
    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        return self.decode_tiled(samples, tile_x, tile_y, overlap)
    
    def vae_encode_crop_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Crop pixels to be divisible by 8 before encoding."""
        dims = pixels.shape[1:3]
        for i in range(len(dims)):
            x = dims[i]
            if x % 8 != 0:
                x = (x // 8) * 8
                if i == 0:
                    pixels = pixels[:, :x, :, :]
                else:
                    pixels = pixels[:, :, :x, :]
        return pixels
    
    def get_sd(self) -> Dict:
        return {}
    
    def throw_exception_if_invalid(self):
        self._check_daemon()
    
    def spacial_compression_decode(self) -> int:
        return 8
    
    def spacial_compression_encode(self) -> int:
        return 8
    
    def temporal_compression_decode(self) -> int:
        return 1
    
    def temporal_compression_encode(self) -> int:
        return 1


# =============================================================================
# Daemon CLIP Proxy
# =============================================================================

class DaemonCLIP:
    """
    A CLIP proxy that routes all tokenization and encoding to the Luna Daemon.
    
    This object can be passed to any ComfyUI node that expects a CLIP.
    """
    
    def __init__(
        self, 
        source_clip: Optional[Any] = None,
        clip_type: Optional[str] = None,
        use_existing: bool = False
    ):
        self.source_clip = source_clip
        self.use_existing = use_existing
        self._registered = False
        self._layer_idx = None
        
        # LoRA stack: list of {"hash": str, "strength": float}
        self.lora_stack: List[Dict[str, Any]] = []
        
        if clip_type is None and source_clip is not None:
            self.clip_type = detect_clip_type(source_clip)
        else:
            self.clip_type = clip_type or 'unknown'
        
        self.components = self._get_required_components()
        
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cond_stage_model = None
        self.patcher = None
    
    def _get_required_components(self) -> List[str]:
        component_map = {
            'sd15': ['clip_l'],
            'sdxl': ['clip_l', 'clip_g'],
            'flux': ['clip_l', 't5xxl'],
            'sd3': ['clip_l', 'clip_g', 't5xxl'],
        }
        return component_map.get(self.clip_type, ['clip_l'])
    
    def _ensure_registered(self):
        if self._registered or self.use_existing:
            return
        
        if self.source_clip is None:
            raise RuntimeError("DaemonCLIP has no source CLIP and use_existing=False")
        
        try:
            daemon_client.register_clip(self.source_clip, self.clip_type)
            self._registered = True
        except Exception as e:
            raise RuntimeError(f"Failed to register CLIP with daemon: {e}")
    
    def _check_daemon(self):
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel."
            )
    
    def clone(self):
        """Clone this CLIP proxy."""
        cloned = DaemonCLIP(
            source_clip=self.source_clip,
            clip_type=self.clip_type,
            use_existing=True
        )
        cloned._registered = self._registered
        cloned._layer_idx = self._layer_idx
        cloned.lora_stack = self.lora_stack.copy()
        return cloned
    
    def tokenize(self, text: str, return_word_ids: bool = False):
        """Tokenize text via daemon."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.clip_tokenize(text, self.clip_type, return_word_ids)
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode_from_tokens(self, tokens, return_pooled: bool = False, return_dict: bool = False):
        """Encode tokens via daemon."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.clip_encode_from_tokens(
                tokens, 
                self.clip_type,
                return_pooled=return_pooled,
                return_dict=return_dict,
                lora_stack=self.lora_stack if self.lora_stack else None
            )
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode(self, text: str):
        """Convenience method: tokenize and encode text."""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)
    
    def set_clip_options(self, options: Dict):
        """Store CLIP options."""
        self._clip_options = options
    
    def get_key_patches(self, *args, **kwargs):
        """Return empty patches - LoRAs handled separately."""
        return {}
    
    def add_patches(self, patches: Dict, strength_patch: float = 1.0, 
                    strength_model: float = 1.0) -> List:
        """
        Intercept patch adding for LoRA support.
        
        Extracts CLIP-specific patches and uploads to daemon.
        """
        if not patches:
            return []
        
        clip_patches = {}
        for key, value in patches.items():
            if any(x in key.lower() for x in ['clip', 'te1', 'te2', 'text_encoder']):
                clip_patches[key] = value
        
        if not clip_patches:
            return []
        
        # Compute hash
        hash_data = []
        for key in sorted(clip_patches.keys()):
            val = clip_patches[key]
            if isinstance(val, tuple):
                for t in val:
                    if isinstance(t, torch.Tensor):
                        hash_data.append(t.cpu().numpy().tobytes())
            elif isinstance(val, torch.Tensor):
                hash_data.append(val.cpu().numpy().tobytes())
        
        if not hash_data:
            return []
        
        lora_hash = hashlib.md5(b''.join(hash_data)).hexdigest()
        
        # Upload to daemon if not already cached
        try:
            if not daemon_client.has_lora(lora_hash):
                daemon_client.upload_lora(lora_hash, clip_patches)
        except Exception as e:
            print(f"[DaemonCLIP] Warning: Failed to upload LoRA: {e}")
        
        # Add to stack
        self.lora_stack.append({
            "hash": lora_hash,
            "strength": strength_patch
        })
        
        return list(patches.keys())
    
    def load_model(self):
        """No-op for proxy."""
        pass
    
    def get_sd(self) -> Dict:
        return {}
    
    def clip_layer(self, layer_idx: int):
        """Set CLIP layer for encoding."""
        self._layer_idx = layer_idx


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DaemonVAE',
    'DaemonCLIP',
    'InferenceModeWrapper',
    'wrap_model_for_inference',
    'detect_vae_type',
    'detect_clip_type',
]
