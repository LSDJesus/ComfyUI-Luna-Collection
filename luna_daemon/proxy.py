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
# Daemon VAE Proxy (CUDA IPC Weight Sharing Version)
# =============================================================================

class DaemonVAE:
    """
    A VAE proxy that uses CUDA IPC shared weights from Luna Daemon.
    
    Instead of sending encode/decode requests over sockets, this loads
    IPC handles to shared GPU weights and executes operations locally.
    
    Zero overhead - multiple instances share the same VAE weights in GPU memory.
    """
    
    def __init__(
        self, 
        vae_path: str,
        source_vae: Optional[Any] = None,
        vae_type: Optional[str] = None,
        use_ipc: bool = True
    ):
        self.vae_path = vae_path
        self.source_vae = source_vae
        self.use_ipc = use_ipc
        self._model_key: Optional[str] = None
        self._shared_vae: Optional[Any] = None
        
        if vae_type is None and source_vae is not None:
            self.vae_type = detect_vae_type(source_vae)
        else:
            self.vae_type = vae_type or 'unknown'
        
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.dtype = torch.float32
        
        # Try to load shared weights on init
        if use_ipc:
            self._setup_shared_weights()
    
    def _setup_shared_weights(self):
        """Load VAE via IPC shared weights."""
        try:
            # Ask daemon to load VAE and get IPC handles
            print(f"[DaemonVAE] Loading shared weights for {self.vae_path}")
            result = daemon_client.load_vae_weights(self.vae_path)
            
            if result.get("success"):
                self._model_key = result["model_key"]
                ipc_handles = result["ipc_handles"]
                
                # Create local VAE using shared weights
                try:
                    from .ipc_models import SharedVAE
                except (ImportError, ValueError):
                    # Fallback: direct import
                    import importlib.util
                    daemon_dir = os.path.dirname(os.path.abspath(__file__))
                    ipc_path = os.path.join(daemon_dir, "ipc_models.py")
                    spec = importlib.util.spec_from_file_location("luna_ipc_models", ipc_path)
                    if spec and spec.loader:
                        ipc_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(ipc_mod)
                        SharedVAE = ipc_mod.SharedVAE
                    else:
                        raise ImportError("Could not load ipc_models module")
                
                self._shared_vae = SharedVAE(ipc_handles, self.vae_path)
                
                print(f"[DaemonVAE] [OK] Using shared weights (key={self._model_key})")
            else:
                print(f"[DaemonVAE] Failed to load shared weights: {result.get('error')}")
                self.use_ipc = False
                
        except Exception as e:
            print(f"[DaemonVAE] IPC setup failed, falling back to socket mode: {e}")
            self.use_ipc = False
    
    def _check_daemon(self):
        """Verify daemon is running."""
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel."
            )
    
    def encode(self, pixel_samples: torch.Tensor, auto_tile: bool = False) -> torch.Tensor:
        """Encode pixels to latent space using shared weights."""
        if self.use_ipc and self._shared_vae:
            # Local execution with shared weights - ZERO socket overhead!
            return self._shared_vae.encode(pixel_samples)
        else:
            # Fallback: socket-based request/response (old slow method)
            self._check_daemon()
            try:
                return daemon_client.vae_encode(
                    pixel_samples, 
                    self.vae_type,
                    tiled=auto_tile,
                    tile_size=512,
                    overlap=64
                )
            except (ModelMismatchError, DaemonConnectionError) as e:
                raise RuntimeError(f"Daemon error: {e}")
    
    def decode(self, samples_in: torch.Tensor, vae_options: Optional[Dict] = None,
               auto_tile: bool = False) -> torch.Tensor:
        """Decode latents to pixels using shared weights."""
        if self.use_ipc and self._shared_vae:
            # Local execution with shared weights - ZERO socket overhead!
            return self._shared_vae.decode(samples_in)
        else:
            # Fallback: socket-based request/response (old slow method)
            self._check_daemon()
            try:
                return daemon_client.vae_decode(
                    samples_in, 
                    self.vae_type,
                    tiled=auto_tile,
                    tile_size=64,
                    overlap=16
                )
            except (ModelMismatchError, DaemonConnectionError) as e:
                raise RuntimeError(f"Daemon error: {e}")
    
    def encode_tiled(self, pixel_samples: torch.Tensor, 
                     tile_x: int = 512, tile_y: int = 512, 
                     overlap: int = 64, **kwargs) -> torch.Tensor:
        """Tiled encoding for large images."""
        # For now, call regular encode - tiling can be added to SharedVAE if needed
        return self.encode(pixel_samples, auto_tile=True)
    
    def decode_tiled(self, samples: torch.Tensor,
                     tile_x: int = 64, tile_y: int = 64,
                     overlap: int = 16, **kwargs) -> torch.Tensor:
        """Tiled decoding for large latents."""
        # For now, call regular decode - tiling can be added to SharedVAE if needed
        return self.decode(samples, auto_tile=True)

        
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
    
    def encode_from_tokens_scheduled(self, tokens, **kwargs):
        """Encode tokens via daemon with scheduling support (ComfyUI 0.3.0+)."""
        # Call encode_from_tokens with return_dict=True to get both cond and pooled
        self._check_daemon()
        self._ensure_registered()
        
        try:
            result = daemon_client.clip_encode_from_tokens(
                tokens, 
                self.clip_type,
                return_pooled=True,
                return_dict=True,
                lora_stack=self.lora_stack if self.lora_stack else None
            )
            
            # Result should be a dict with 'cond' and 'pooled_output'
            # Convert to ComfyUI's expected format: [[cond, {"pooled_output": pooled}]]
            if isinstance(result, dict):
                cond = result.get('cond')
                pooled = result.get('pooled_output')
                if cond is not None and pooled is not None:
                    return [[cond, {"pooled_output": pooled}]]
            
            # Fallback: if result is already in the right format, return it
            return result
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
    
    def add_lora_by_name(self, lora_name: str, model_strength: float = 1.0, clip_strength: float = 1.0):
        """
        Add LoRA by filename - daemon loads from disk and applies.
        
        Args:
            lora_name: LoRA filename (relative to loras folder)
            model_strength: Strength for model (unused for CLIP-only proxy)
            clip_strength: Strength for CLIP
        
        Returns:
            Self (for chaining)
        """
        try:
            # Send to daemon to apply LoRA
            result = daemon_client.register_lora(lora_name, clip_strength, model_strength)
            if not result.get("success"):
                print(f"[DaemonCLIP] Warning: Failed to register LoRA '{lora_name}': {result.get('error')}")
        except Exception as e:
            print(f"[DaemonCLIP] Warning: Failed to apply LoRA '{lora_name}': {e}")
        
        return self
    
    def load_model(self):
        """No-op for proxy."""
        pass
    
    def get_sd(self) -> Dict:
        return {}
    
    def clip_layer(self, layer_idx: int):
        """Set CLIP layer for encoding."""
        self._layer_idx = layer_idx


# =============================================================================
# DaemonSAM3 - Proxy for SAM3 Object Detection
# =============================================================================

class DaemonSAM3:
    """
    Proxy for SAM3 model running on Luna Daemon.
    
    Routes detection requests to the daemon's SAM3 model, which stays loaded
    and shared across all ComfyUI instances.
    
    Architecture:
    - Image (PIL) is serialized and sent to daemon
    - Daemon runs SAM3 grounding on its GPU
    - Returns lightweight detection data (coordinates, masks)
    - Image is discarded after detection (one-way transfer)
    
    Usage:
        sam3 = DaemonSAM3("sam3_h.safetensors", device="cuda:1")
        detections = sam3.ground(pil_image, "face", threshold=0.25)
    """
    
    def __init__(self, model_name: str = "sam3_h.safetensors", device: str = "cuda:1"):
        """
        Initialize SAM3 proxy.
        
        Args:
            model_name: SAM3 model filename in models/sam3/
            device: Device for SAM3 on daemon (cuda:0, cuda:1, cpu)
        """
        self.model_name = model_name
        self.device = device
        self._loaded = False
    
    def load_model(self) -> bool:
        """
        Load SAM3 model on daemon (if not already loaded).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            result = daemon_client.load_sam3(
                model_name=self.model_name,
                device=self.device
            )
            
            if result.get("success"):
                self._loaded = True
                print(f"[DaemonSAM3] Model loaded: {self.model_name} on {self.device}")
                return True
            else:
                print(f"[DaemonSAM3] Failed to load model: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"[DaemonSAM3] Error loading model: {e}")
            return False
    
    def ground(
        self,
        image,  # PIL Image
        text_prompt: str,
        threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Run SAM3 grounding detection.
        
        Args:
            image: PIL Image to detect objects in
            text_prompt: Text description of what to find (e.g., "face", "hands")
            threshold: Confidence threshold (0.0 to 1.0)
        
        Returns:
            List of detection dicts with keys:
            - bbox: [x1, y1, x2, y2] in pixel coordinates
            - mask: 2D array (H, W) binary mask
            - confidence: float score
        """
        # Ensure model is loaded
        if not self._loaded:
            if not self.load_model():
                return []
        
        try:
            # Serialize PIL image
            import pickle
            image_bytes = pickle.dumps(image)
            
            # Send to daemon using the dedicated function
            result = daemon_client.sam3_detect(
                image_bytes=image_bytes,
                text_prompt=text_prompt,
                threshold=threshold
            )
            
            if result.get("success"):
                detections = result.get("detections", [])
                
                # Deserialize masks (they come as nested lists)
                import numpy as np
                for det in detections:
                    if "mask" in det:
                        det["mask"] = np.array(det["mask"])
                
                return detections
            else:
                print(f"[DaemonSAM3] Detection failed: {result.get('error')}")
                return []
                
        except Exception as e:
            print(f"[DaemonSAM3] Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def ground_batch(
        self,
        image,  # PIL Image
        prompts: List[Dict[str, Any]],
        default_threshold: float = 0.25
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run SAM3 grounding detection for multiple prompts efficiently.
        
        Reuses backbone features across prompts for significant speedup.
        
        Args:
            image: PIL Image to detect objects in
            prompts: List of prompt configs, each can be:
                - str: Just the prompt text
                - dict: {"prompt": str, "threshold": float, "label": str}
            default_threshold: Default confidence threshold
        
        Returns:
            Dict mapping label → list of detections
        """
        # Ensure model is loaded
        if not self._loaded:
            if not self.load_model():
                return {}
        
        try:
            import pickle
            import numpy as np
            image_bytes = pickle.dumps(image)
            
            result = daemon_client.sam3_detect_batch(
                image_bytes=image_bytes,
                prompts=prompts,
                threshold=default_threshold
            )
            
            if result.get("success"):
                results_by_label = result.get("results_by_label", {})
                
                # Deserialize masks
                for label, detections in results_by_label.items():
                    for det in detections:
                        if "mask" in det:
                            det["mask"] = np.array(det["mask"])
                
                return results_by_label
            else:
                print(f"[DaemonSAM3] Batch detection failed: {result.get('error')}")
                return {}
                
        except Exception as e:
            print(f"[DaemonSAM3] Error during batch detection: {e}")
            import traceback
            traceback.print_exc()
            return {}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DaemonVAE',
    'DaemonCLIP',
    'DaemonSAM3',
    'InferenceModeWrapper',
    'wrap_model_for_inference',
    'detect_vae_type',
    'detect_clip_type',
]
