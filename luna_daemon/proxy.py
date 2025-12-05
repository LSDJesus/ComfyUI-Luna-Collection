"""
Luna Daemon Proxy Objects

Proxy VAE and CLIP classes that look like real ComfyUI models
but route all operations to the Luna Daemon.

These can be passed to ANY node that expects VAE or CLIP,
including third-party nodes like FaceDetailer, UltimateSDUpscale, etc.

Architecture:
    The LunaCheckpointTunnel node receives actual VAE/CLIP objects from
    a standard checkpoint loader. The proxy classes wrap these objects and:
    
    1. On first use: Register the model with the daemon (daemon stores it)
    2. On subsequent calls: Route all operations through the daemon
    3. For matching components: Share already-loaded daemon models
    
    Component-based sharing:
    - CLIP components (clip_l, clip_g, t5xxl) can be shared across model families
    - VAE components are family-specific (sdxl_vae, flux_vae, etc.)
    
    LoRA Support (F-150 Architecture):
    - DaemonCLIP intercepts add_patches() calls from LoraLoader
    - Extracts CLIP-specific LoRA weights and hashes them
    - Uploads unique LoRAs to daemon's LoRARegistry
    - Carries lora_stack with encode requests
    - Daemon applies LoRAs transiently per-request
    
    This allows maximum VRAM sharing across ComfyUI instances.
"""

import torch
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Union

from . import client as daemon_client
from .client import DaemonConnectionError, ModelMismatchError


# =============================================================================
# Model Family Detection
# =============================================================================

def detect_clip_type(clip) -> str:
    """
    Detect the type of CLIP model from a CLIP object.
    
    Returns one of: 'sdxl', 'sd15', 'flux', 'sd3', 'unknown'
    """
    try:
        if clip is None:
            return 'unknown'
        
        # Check for cond_stage_model attribute (the actual CLIP model)
        cond_model = getattr(clip, 'cond_stage_model', None)
        if cond_model is None and hasattr(clip, 'patcher'):
            # It might be a model patcher wrapping the CLIP
            cond_model = getattr(clip.patcher, 'model', None)
        
        if cond_model is not None:
            class_name = type(cond_model).__name__
            
            # SDXL uses SDXLClipModel which has clip_l and clip_g
            if 'SDXL' in class_name or hasattr(cond_model, 'clip_g'):
                return 'sdxl'
            
            # Flux uses a different structure with t5xxl
            if 'Flux' in class_name or 't5xxl' in class_name.lower():
                return 'flux'
            
            # SD3 has triple clip
            if 'SD3' in class_name:
                return 'sd3'
            
            # SD1.5 is simpler
            if 'SD1' in class_name or 'SDClip' in class_name:
                return 'sd15'
        
        # Fallback: check the clip object itself
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
    """
    Detect the VAE type/family from a VAE object.
    
    Returns one of: 'sdxl', 'sd15', 'flux', 'sd3', 'unknown'
    
    Key differences:
    - SD1.5/SDXL: 4-channel latent space
    - Flux/SD3: 16-channel latent space
    """
    try:
        if vae is None:
            return 'unknown'
        
        # Try to get the first stage model
        first_stage = getattr(vae, 'first_stage_model', None)
        if first_stage is None:
            first_stage = vae
        
        # Check latent channels - this is the key differentiator
        # Flux/SD3 use 16 channels, SDXL/SD1.5 use 4
        if hasattr(first_stage, 'config'):
            config = first_stage.config
            latent_channels = getattr(config, 'latent_channels', None)
            if latent_channels == 16:
                return 'flux'  # Could also be SD3
            elif latent_channels == 4:
                return 'sdxl'  # Could also be SD1.5
        
        # Check encoder output channels
        if hasattr(first_stage, 'encoder'):
            encoder = first_stage.encoder
            if hasattr(encoder, 'conv_out'):
                out_channels = encoder.conv_out.out_channels
                if out_channels == 32:  # 16 * 2 for mean/var
                    return 'flux'
                elif out_channels == 8:  # 4 * 2 for mean/var
                    return 'sdxl'
        
        # Fallback: check class name
        class_name = type(first_stage).__name__
        if 'Flux' in class_name:
            return 'flux'
        elif 'SD3' in class_name:
            return 'sd3'
        
        return 'sdxl'  # Default assumption
        
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
    including third-party nodes. All encoding/decoding happens on the
    daemon's GPU, not the local instance.
    
    Usage:
        # From LunaCheckpointTunnel - wrapping an actual VAE
        proxy_vae = DaemonVAE(actual_vae_object, vae_type='sdxl')
        
        # For sharing existing daemon models
        proxy_vae = DaemonVAE(None, vae_type='sdxl', use_existing=True)
    """
    
    def __init__(
        self, 
        source_vae: Optional[Any] = None,
        vae_type: Optional[str] = None,
        use_existing: bool = False
    ):
        """
        Create a VAE proxy.
        
        Args:
            source_vae: The actual VAE object from checkpoint loader (for registration)
            vae_type: VAE type string ('sdxl', 'flux', 'sd3', 'sd15')
            use_existing: If True, use already-loaded daemon VAE (no source needed)
        """
        self.source_vae = source_vae
        self.use_existing = use_existing
        self._registered = False
        self._memory_used = 0
        
        # Auto-detect type if not provided
        if vae_type is None and source_vae is not None:
            self.vae_type = detect_vae_type(source_vae)
        else:
            self.vae_type = vae_type or 'unknown'
        
        # These match ComfyUI VAE attributes
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.dtype = torch.float32
    
    def _ensure_registered(self):
        """Ensure VAE is registered with daemon before use."""
        if self._registered or self.use_existing:
            return
        
        if self.source_vae is None:
            raise RuntimeError(
                "DaemonVAE has no source VAE and use_existing=False.\n"
                "Cannot register with daemon."
            )
        
        # Register with daemon
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
                "Start it from the Luna Daemon panel or run:\n"
                "  python -m luna_daemon.server"
            )
        
    def encode(self, pixel_samples: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latent space via daemon."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_encode(pixel_samples, self.vae_type)
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def decode(self, samples_in: torch.Tensor, vae_options: Optional[Dict] = None) -> torch.Tensor:
        """Decode latents to pixels via daemon."""
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.vae_decode(samples_in, self.vae_type)
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode_tiled(self, pixel_samples: torch.Tensor, 
                     tile_x: int = 512, tile_y: int = 512, 
                     overlap: int = 64, **kwargs) -> torch.Tensor:
        """Tiled encoding for large images."""
        # TODO: Add tiled support to daemon protocol
        return self.encode(pixel_samples)
    
    def decode_tiled(self, samples: torch.Tensor,
                     tile_x: int = 64, tile_y: int = 64,
                     overlap: int = 16, **kwargs) -> torch.Tensor:
        """Tiled decoding for large latents."""
        # TODO: Add tiled support to daemon protocol
        return self.decode(samples)
    
    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        """Legacy tiled encode method."""
        return self.encode_tiled(pixel_samples, tile_x, tile_y, overlap)
    
    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        """Legacy tiled decode method."""
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
        """Get state dict - not applicable for daemon proxy."""
        return {}
    
    def throw_exception_if_invalid(self):
        """Validate the VAE is usable."""
        self._check_daemon()
    
    # Compression ratio methods
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
    
    This object can be passed to any ComfyUI node that expects a CLIP,
    including third-party nodes. All encoding happens on the daemon's GPU.
    
    Component-based architecture:
        CLIP models are composed of components that can be shared:
        - clip_l: Used by SDXL, Flux, SD3
        - clip_g: Used by SDXL, SD3
        - t5xxl: Used by Flux, SD3
        
        The daemon tracks which components are loaded and shares them.
    
    LoRA Support (F-150 Architecture):
        When add_patches() is called (e.g., by LoraLoader), this proxy:
        1. Extracts CLIP-specific weights from the patches
        2. Computes a deterministic hash of the weights
        3. Uploads to daemon if not already cached
        4. Stores {hash, strength} in lora_stack
        5. Sends lora_stack with encode requests
        
        The daemon applies LoRAs transiently per-request using locking.
    
    Usage:
        # From LunaCheckpointTunnel - wrapping an actual CLIP
        proxy_clip = DaemonCLIP(actual_clip_object, clip_type='sdxl')
        
        # For sharing existing daemon CLIPs
        proxy_clip = DaemonCLIP(None, clip_type='sdxl', use_existing=True)
    """
    
    def __init__(
        self, 
        source_clip: Optional[Any] = None,
        clip_type: Optional[str] = None,
        use_existing: bool = False
    ):
        """
        Create a CLIP proxy.
        
        Args:
            source_clip: The actual CLIP object from checkpoint loader (for registration)
            clip_type: CLIP type string ('sdxl', 'flux', 'sd3', 'sd15')
            use_existing: If True, use already-loaded daemon CLIP components
        """
        self.source_clip = source_clip
        self.use_existing = use_existing
        self._registered = False
        self._layer_idx = None
        
        # LoRA stack: list of {"hash": str, "strength": float}
        self.lora_stack: List[Dict[str, Any]] = []
        
        # Auto-detect type if not provided
        if clip_type is None and source_clip is not None:
            self.clip_type = detect_clip_type(source_clip)
        else:
            self.clip_type = clip_type or 'unknown'
        
        # Map type to required components
        self.components = self._get_required_components()
        
        # Match ComfyUI CLIP attributes
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cond_stage_model = None
        self.patcher = None
    
    def _get_required_components(self) -> List[str]:
        """Get the CLIP components required for this type."""
        component_map = {
            'sd15': ['clip_l'],
            'sdxl': ['clip_l', 'clip_g'],
            'flux': ['clip_l', 't5xxl'],
            'sd3': ['clip_l', 'clip_g', 't5xxl'],
        }
        return component_map.get(self.clip_type, ['clip_l'])
    
    def _ensure_registered(self):
        """Ensure CLIP components are registered with daemon."""
        if self._registered or self.use_existing:
            return
        
        if self.source_clip is None:
            raise RuntimeError(
                "DaemonCLIP has no source CLIP and use_existing=False.\n"
                "Cannot register with daemon."
            )
        
        # Register with daemon
        try:
            daemon_client.register_clip(self.source_clip, self.clip_type)
            self._registered = True
        except Exception as e:
            raise RuntimeError(f"Failed to register CLIP with daemon: {e}")
    
    def _check_daemon(self):
        """Verify daemon is running."""
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel."
            )
    
    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs):
        """
        Tokenize text - stores text for later encoding.
        
        The actual tokenization happens on the daemon side during encode.
        We return a wrapper object that encode_from_tokens* methods recognize.
        """
        return DaemonTokens(text, self.clip_type, return_word_ids, kwargs)
    
    def encode_from_tokens(self, tokens, return_pooled: bool = False, 
                           return_dict: bool = False) -> Any:
        """Encode tokens to conditioning via daemon."""
        self._check_daemon()
        self._ensure_registered()
        
        # Handle our token wrapper
        if isinstance(tokens, DaemonTokens):
            text = tokens.text
            clip_type = tokens.clip_type
        elif isinstance(tokens, dict) and "text" in tokens:
            text = tokens["text"]
            clip_type = self.clip_type
        else:
            raise RuntimeError(
                "DaemonCLIP received incompatible tokens.\n"
                "Tokens must come from DaemonCLIP.tokenize(), not another CLIP."
            )
        
        try:
            # Pass lora_stack to daemon for transient application
            cond, pooled, _, _ = daemon_client.clip_encode(
                text, "", clip_type, 
                lora_stack=self.lora_stack
            )
            
            if return_dict:
                return {"cond": cond, "pooled_output": pooled}
            elif return_pooled:
                return cond, pooled
            else:
                return cond
                
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode_from_tokens_scheduled(self, tokens, unprojected: bool = False, 
                                      add_dict: Optional[Dict[str, Any]] = None,
                                      show_pbar: bool = True) -> List:
        """
        Scheduled encoding - this is what CLIPTextEncode uses.
        Returns conditioning in ComfyUI's expected format.
        """
        if add_dict is None:
            add_dict = {}
        
        self._check_daemon()
        self._ensure_registered()
        
        # Handle our token wrapper
        if isinstance(tokens, DaemonTokens):
            text = tokens.text
            clip_type = tokens.clip_type
        elif isinstance(tokens, dict) and "text" in tokens:
            text = tokens["text"]
            clip_type = self.clip_type
        else:
            raise RuntimeError(
                "DaemonCLIP received incompatible tokens.\n"
                "Use DaemonCLIP for the entire CLIP workflow."
            )
        
        try:
            # Pass lora_stack to daemon for transient application
            cond, pooled, _, _ = daemon_client.clip_encode(
                text, "", clip_type,
                lora_stack=self.lora_stack
            )
            
            # Return in ComfyUI conditioning format
            cond_dict = {"pooled_output": pooled}
            cond_dict.update(add_dict)
            
            return [[cond, cond_dict]]
                
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def encode(self, text: str):
        """Convenience method: tokenize + encode."""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)
    
    def clip_layer(self, layer_idx: int):
        """Set CLIP layer for output (affects encoding)."""
        self._layer_idx = layer_idx
        # TODO: Send layer preference to daemon
    
    def clone(self):
        """Create a copy of this CLIP proxy, preserving lora_stack."""
        new_clip = DaemonCLIP(
            source_clip=self.source_clip,
            clip_type=self.clip_type,
            use_existing=self.use_existing or self._registered
        )
        new_clip._layer_idx = self._layer_idx
        new_clip._registered = self._registered
        # Copy the lora_stack so patches accumulate correctly
        new_clip.lora_stack = self.lora_stack.copy()
        return new_clip
    
    def load_model(self):
        """Load model - daemon handles this on first request."""
        pass
    
    def get_sd(self) -> Dict:
        """Get state dict - not applicable for daemon proxy."""
        return {}
    
    def get_key_patches(self) -> Dict:
        """Get patches for model - not applicable."""
        return {}
    
    def add_patches(self, patches: Dict, strength_patch: float = 1.0, strength_model: float = 1.0):
        """
        Add patches (e.g., LoRA) to CLIP - routes to daemon.
        
        F-150 Architecture:
        1. Extract CLIP-specific weights from patches
        2. Hash the weights for cache key
        3. Upload to daemon if not cached
        4. Add {hash, strength} to lora_stack
        5. Return cloned self with updated stack
        
        Args:
            patches: Dict of model patches (from LoraLoader)
            strength_patch: LoRA strength
            strength_model: Model strength (usually 1.0)
        
        Returns:
            New DaemonCLIP instance with LoRA added to stack
        """
        # Extract CLIP-specific patches
        clip_patches = self._extract_clip_patches(patches)
        
        if not clip_patches:
            print("[DaemonCLIP] No CLIP patches found in LoRA, skipping")
            return self.clone()
        
        # Compute deterministic hash
        lora_hash = self._compute_lora_hash(clip_patches)
        
        # Check if daemon has this LoRA cached
        try:
            if not daemon_client.has_lora(lora_hash):
                # Upload to daemon
                print(f"[DaemonCLIP] Uploading LoRA {lora_hash[:12]}... to daemon")
                daemon_client.upload_lora(lora_hash, clip_patches)
            else:
                print(f"[DaemonCLIP] LoRA {lora_hash[:12]}... already cached in daemon")
        except DaemonConnectionError as e:
            print(f"[DaemonCLIP] Warning: Could not upload LoRA to daemon: {e}")
            print("[DaemonCLIP] LoRA will be ignored for this encoding")
            return self.clone()
        
        # Clone self and add to lora_stack
        new_clip = self.clone()
        new_clip.lora_stack.append({
            "hash": lora_hash,
            "strength": strength_patch * strength_model
        })
        
        print(f"[DaemonCLIP] Added LoRA to stack (strength={strength_patch:.2f}), "
              f"stack size: {len(new_clip.lora_stack)}")
        
        return new_clip
    
    def _extract_clip_patches(self, patches: Dict) -> Dict[str, torch.Tensor]:
        """
        Extract CLIP-specific patches from LoRA patch dict.
        
        LoRA patches use keys like:
        - 'clip_l.transformer.text_model...'
        - 'clip_g.transformer.text_model...'
        - 'lora_te1_...' (CLIP text encoder 1)
        - 'lora_te2_...' (CLIP text encoder 2)
        
        We filter for these and return flattened tensors.
        """
        clip_patches = {}
        
        for key, patch_data in patches.items():
            # Check if this is a CLIP-related key
            is_clip = any(pattern in key.lower() for pattern in [
                'clip_l', 'clip_g', 'te1', 'te2', 'text_encoder',
                'text_model', 'lora_te', 'clip.'
            ])
            
            if not is_clip:
                continue
            
            # patch_data format varies:
            # - Could be (tensor,) tuple
            # - Could be tensor directly
            # - Could be (tensor, function) for alpha
            if isinstance(patch_data, tuple):
                tensor = patch_data[0] if patch_data else None
            elif isinstance(patch_data, torch.Tensor):
                tensor = patch_data
            else:
                continue
            
            if tensor is not None and isinstance(tensor, torch.Tensor):
                # Store on CPU for transport
                clip_patches[key] = tensor.cpu().clone()
        
        return clip_patches
    
    def _compute_lora_hash(self, weights: Dict[str, torch.Tensor]) -> str:
        """
        Compute deterministic hash of LoRA weights.
        
        Uses SHA256 of concatenated tensor bytes for uniqueness.
        """
        hasher = hashlib.sha256()
        
        # Sort keys for deterministic ordering
        for key in sorted(weights.keys()):
            tensor = weights[key]
            # Hash key name
            hasher.update(key.encode('utf-8'))
            # Hash tensor shape
            hasher.update(str(tensor.shape).encode('utf-8'))
            # Hash tensor data (sample for speed if large)
            if tensor.numel() > 10000:
                # Sample every Nth element for large tensors
                step = tensor.numel() // 10000
                sample = tensor.flatten()[::step]
                hasher.update(sample.numpy().tobytes())
            else:
                hasher.update(tensor.numpy().tobytes())
        
        return hasher.hexdigest()
    
    def load_sd(self, sd, full_model=False):
        """Load state dict - not applicable."""
        pass
    
    def set_tokenizer_option(self, option_name: str, value: Any):
        """Set tokenizer option - stored for daemon to use."""
        # TODO: Send to daemon
        pass
    
    def add_hooks_to_dict(self, pooled_dict: Dict[str, Any]):
        """Add hooks to pooled dict - pass through."""
        return pooled_dict


# =============================================================================
# Token Wrapper
# =============================================================================

class DaemonTokens:
    """
    Wrapper for tokenized text that will be sent to daemon.
    
    This mimics the structure that encode_from_tokens expects,
    but stores the original text for daemon-side tokenization.
    """
    
    def __init__(self, text: str, clip_type: str, 
                 return_word_ids: bool = False, kwargs: Optional[Dict] = None):
        self.text = text
        self.clip_type = clip_type
        self.return_word_ids = return_word_ids
        self.kwargs = kwargs or {}
    
    def __repr__(self):
        return f"DaemonTokens({self.clip_type}: {self.text[:50]}...)"
