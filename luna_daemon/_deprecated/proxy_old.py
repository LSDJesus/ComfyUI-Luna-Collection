"""
Luna Daemon Proxy Objects

Proxy VAE and CLIP classes that look like real ComfyUI models
but route all operations to the Luna Daemon.

These can be passed to ANY node that expects VAE or CLIP,
including third-party nodes like FaceDetailer, UltimateSDUpscale, etc.

Architecture:
    The LunaModelRouter node loads actual VAE/CLIP objects. The proxy 
    classes wrap these objects and:
    
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
from typing import Optional, Dict, Any, List

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
    
    def _should_tile(self, tensor: torch.Tensor, is_latent: bool = False) -> bool:
        """
        Determine if tiling should be used based on tensor size.
        
        Args:
            tensor: Input tensor (pixels or latents)
            is_latent: True if tensor is latent space (smaller dimensions)
            
        Returns:
            True if tiling is recommended
        """
        # Get spatial dimensions
        if is_latent:
            # Latents: (B, C, H, W) - check H, W
            if tensor.dim() >= 4:
                h, w = tensor.shape[2], tensor.shape[3]
            else:
                return False
            # Latent threshold: 96x96 = 768x768 pixel equivalent
            threshold = 96
        else:
            # Pixels: (B, H, W, C) - check H, W
            if tensor.dim() >= 4:
                h, w = tensor.shape[1], tensor.shape[2]
            elif tensor.dim() == 3:
                h, w = tensor.shape[0], tensor.shape[1]
            else:
                return False
            # Pixel threshold: 1536x1536 (larger than typical 1024x1024)
            threshold = 1536
        
        return h > threshold or w > threshold
        
    def encode(self, pixel_samples: torch.Tensor, auto_tile: bool = False) -> torch.Tensor:
        """
        Encode pixels to latent space via daemon.
        
        Args:
            pixel_samples: Image tensor (B, H, W, C)
            auto_tile: If True, force tiled encoding. By default (False), the daemon
                      will attempt full encode and fall back to tiled on OOM.
            
        Returns:
            Latent tensor
        """
        print(f"[DaemonVAE] encode() called - type={self.vae_type}, shape={pixel_samples.shape}")
        self._check_daemon()
        self._ensure_registered()
        
        # Only use tiling if explicitly requested
        use_tiled = auto_tile
        
        try:
            print(f"[DaemonVAE] Sending encode request to daemon (tiled={use_tiled})")
            return daemon_client.vae_encode(
                pixel_samples, 
                self.vae_type,
                tiled=use_tiled,
                tile_size=512,
                overlap=64
            )
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def decode(self, samples_in: torch.Tensor, vae_options: Optional[Dict] = None,
               auto_tile: bool = False) -> torch.Tensor:
        """
        Decode latents to pixels via daemon.
        
        Args:
            samples_in: Latent tensor
            vae_options: Optional VAE options (for compatibility)
            auto_tile: If True, force tiled decoding. By default (False), the daemon
                      will attempt full decode and fall back to tiled on OOM.
            
        Returns:
            Pixel tensor
        """
        print(f"[DaemonVAE] decode() called - type={self.vae_type}, shape={samples_in.shape}")
        self._check_daemon()
        self._ensure_registered()
        
        # Only use tiling if explicitly requested
        use_tiled = auto_tile
        
        try:
            print(f"[DaemonVAE] Sending decode request to daemon (tiled={use_tiled})")
            return daemon_client.vae_decode(
                samples_in, 
                self.vae_type,
                tiled=use_tiled,
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
        """
        Tiled encoding for large images.
        
        Explicitly requests tiled encoding from the daemon.
        """
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
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
    def decode_tiled(self, samples: torch.Tensor,
                     tile_x: int = 64, tile_y: int = 64,
                     overlap: int = 16, **kwargs) -> torch.Tensor:
        """
        Tiled decoding for large latents.
        
        Explicitly requests tiled decoding from the daemon.
        """
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
        except ModelMismatchError as e:
            raise RuntimeError(str(e))
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error: {e}")
    
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
    
    def add_lora_by_name(
        self, 
        lora_name: str, 
        model_strength: float = 1.0, 
        clip_strength: float = 1.0
    ) -> "DaemonCLIP":
        """
        Add LoRA by filename - daemon loads from disk (no socket serialization).
        
        This is the preferred method for Config Gateway workflows. The daemon
        loads the LoRA directly from disk using folder_paths, extracts CLIP
        weights, and caches them.
        
        Args:
            lora_name: LoRA filename (e.g., "my_lora.safetensors")
            model_strength: UNet strength (stored for reference, applied by model)
            clip_strength: CLIP strength for text encoder
        
        Returns:
            New DaemonCLIP instance with LoRA added to stack
        """
        try:
            # Tell daemon to load LoRA from disk
            result = daemon_client.register_lora(lora_name, clip_strength)
            
            if not result.get("success"):
                error = result.get("error", "Unknown error")
                print(f"[DaemonCLIP] Failed to register LoRA '{lora_name}': {error}")
                return self.clone()
            
            lora_hash = result.get("hash")
            if not lora_hash:
                print(f"[DaemonCLIP] No hash returned for LoRA '{lora_name}'")
                return self.clone()
            
            # Clone self and add to lora_stack
            new_clip = self.clone()
            new_clip.lora_stack.append({
                "hash": lora_hash,
                "strength": clip_strength,
                "name": lora_name  # Keep name for debugging
            })
            
            print(f"[DaemonCLIP] Added LoRA '{lora_name}' (hash={lora_hash[:12]}..., "
                  f"clip_str={clip_strength:.2f}), stack size: {len(new_clip.lora_stack)}")
            
            return new_clip
            
        except DaemonConnectionError as e:
            print(f"[DaemonCLIP] Warning: Could not register LoRA with daemon: {e}")
            return self.clone()
        except Exception as e:
            print(f"[DaemonCLIP] Error adding LoRA by name: {e}")
            return self.clone()
    
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


# =============================================================================
# MODEL (UNet) Proxy
# =============================================================================

class DaemonModel:
    """
    Proxy for diffusion models (UNets) that routes inference to Luna Daemon.
    
    This allows centralized UNet hosting on the daemon's GPU with:
    - Frozen weights (no gradient tracking → 60-70% VRAM reduction)
    - Shared model across multiple ComfyUI instances
    - LoRA application in daemon (same pattern as DaemonCLIP)
    
    Usage:
        # In Luna Model Router when daemon is enabled:
        model = DaemonModel(actual_model, model_type='flux')
        
        # ComfyUI samplers will call model() and it routes to daemon
        latents = model(x, timestep, context)  # → daemon inference
    """
    
    def __init__(
        self,
        source_model: Optional[Any] = None,
        model_type: Optional[str] = None,
        use_existing: bool = False
    ):
        """
        Create a UNet proxy.
        
        Args:
            source_model: The actual ModelPatcher from checkpoint loader (for registration)
            model_type: Model type string ('flux', 'sdxl', 'sd15', etc.)
            use_existing: If True, use already-loaded daemon model
        """
        self.source_model = source_model
        self.use_existing = use_existing
        self._registered = False
        self.model_type = model_type or 'sdxl'
        
        # LoRA stack: list of (lora_name, model_strength, clip_strength)
        self.lora_stack: List[tuple] = []
        
        # FB cache params: dict with caching configuration
        self.fb_cache_params: Optional[Dict[str, Any]] = None
        
        # Match ComfyUI ModelPatcher attributes
        self.device = torch.device("cpu")  # Proxy doesn't hold weights
        self._model_dtype = torch.float16
        self.load_device = torch.device("cuda:0")  # Where ComfyUI thinks it loads
        self.offload_device = torch.device("cpu")
        
        # Model structure (mimics ModelPatcher)
        self.model = self  # Self-reference for compatibility
        self.model_options = {
            "transformer_options": {},
        }
        self.model_keys = set()
    
    @property
    def model_sampling(self):
        """Forward model_sampling from source model (always fresh lookup)."""
        if self.source_model is not None:
            # Try direct attribute first
            if hasattr(self.source_model, 'model_sampling'):
                return getattr(self.source_model, 'model_sampling', None)
            # Try through .model if it exists
            if hasattr(self.source_model, 'model'):
                model_obj = getattr(self.source_model, 'model', None)
                if model_obj and hasattr(model_obj, 'model_sampling'):
                    return getattr(model_obj, 'model_sampling', None)
        return None
    
    @property
    def sampling_type(self):
        """Forward sampling_type from source model (always fresh lookup)."""
        if self.source_model is not None:
            if hasattr(self.source_model, 'sampling_type'):
                return getattr(self.source_model, 'sampling_type', None)
            if hasattr(self.source_model, 'model'):
                model_obj = getattr(self.source_model, 'model', None)
                if model_obj and hasattr(model_obj, 'sampling_type'):
                    return getattr(model_obj, 'sampling_type', None)
        return None
    
    @property
    def model_config(self):
        """Forward model_config from source model (always fresh lookup)."""
        if self.source_model is not None:
            if hasattr(self.source_model, 'model_config'):
                return getattr(self.source_model, 'model_config', None)
            if hasattr(self.source_model, 'model'):
                model_obj = getattr(self.source_model, 'model', None)
                if model_obj and hasattr(model_obj, 'model_config'):
                    return getattr(model_obj, 'model_config', None)
        return None
    
    @property
    def latent_format(self):
        """Forward latent_format from source model (needed for latent preview)."""
        if self.source_model is not None:
            # Try to get latent_format from source model
            if hasattr(self.source_model, 'model') and hasattr(self.source_model.model, 'latent_format'):
                return self.source_model.model.latent_format
            # Fallback to direct attribute
            return getattr(self.source_model, 'latent_format', None)
        return None
        
    def _ensure_registered(self):
        """Ensure model is registered with daemon."""
        if self._registered or self.use_existing:
            return
        
        # For now, skip registration if the daemon should already have it
        # The model is forwarded via tensors only, not the model object itself
        # Actual model persistence on daemon is handled by the daemon's model manager
        self._registered = True
        return
    
    def _check_daemon(self):
        """Verify daemon is running."""
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel or run:\n"
                "  python -m luna_daemon.server"
            )
    
    def model_dtype(self):
        """Return the model dtype (ComfyUI calls this as a method)."""
        return self._model_dtype
    
    def extra_conds_shapes(self, **kwargs):
        """Return extra conditioning shapes for memory estimation."""
        # For proxy, we don't have actual shapes, so return empty dict
        # (daemon will handle actual memory management)
        return {}
    
    def memory_required(self, shape: List[int], **kwargs) -> int:
        """
        Estimate memory required for model.
        
        ComfyUI calls this during memory estimation. We return a dummy value
        since the actual model runs on the daemon's GPU.
        
        Args:
            shape: Model input shape
            **kwargs: Additional arguments (cond_shapes, etc.) for compatibility
            
        Returns:
            Estimated memory in bytes (dummy value for proxy)
        """
        # Return minimal dummy value - daemon manages actual memory
        # This satisfies ComfyUI's memory estimation without needing actual values
        return 1024 * 1024 * 100  # 100MB dummy estimate
    
    @property
    def current_patcher(self):
        """Return self as current patcher (daemon model acts as its own patcher)."""
        return self
    
    def prepare_state(self, timestep):
        """Prepare state for sampling (no-op, daemon handles this)."""
        pass
    
    def apply_model(self, x: torch.Tensor, timestep: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """
        Apply model inference (called by ComfyUI samplers).
        
        This is the actual inference call during denoising.
        
        Args:
            x: Input tensor (latents)
            timestep: Timestep tensor
            **kwargs: Additional arguments (conditioning, etc.)
            
        Returns:
            Model output tensor
        """
        # Route to daemon via __call__
        return self(x, timestep, **kwargs)
    
    def __call__(self, x: torch.Tensor, timesteps: torch.Tensor, 
                 context: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass through UNet via daemon.
        
        This is called by ComfyUI's samplers during denoising.
        
        Args:
            x: Noisy latents (B, C, H, W)
            timesteps: Timestep tensor (B,)
            context: Conditioning/context tensor (B, seq_len, dim)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Denoised output tensor
        """
        self._check_daemon()
        self._ensure_registered()
        
        try:
            return daemon_client.model_forward(
                x=x,
                timesteps=timesteps,
                context=context,
                model_type=self.model_type,
                lora_stack=self.lora_stack,
                fb_cache_params=self.fb_cache_params,
                **kwargs
            )
        except DaemonConnectionError as e:
            raise RuntimeError(f"Daemon error during model forward: {e}")
    
    def add_lora(self, lora_name: str, model_strength: float, clip_strength: float):
        """
        Add a LoRA to the stack (will be applied in daemon).
        
        Args:
            lora_name: LoRA filename
            model_strength: UNet strength multiplier
            clip_strength: CLIP strength (ignored here, handled by DaemonCLIP)
        """
        self.lora_stack.append((lora_name, model_strength, clip_strength))
        print(f"[DaemonModel] Added LoRA '{lora_name}' (model_str={model_strength}), stack size: {len(self.lora_stack)}")
    
    def clear_loras(self):
        """Clear all LoRAs from the stack."""
        self.lora_stack = []
        print("[DaemonModel] Cleared LoRA stack")
    
    # ModelPatcher compatibility methods
    def clone(self):
        """Clone the proxy (returns new instance with same config)."""
        cloned = DaemonModel(self.source_model, self.model_type, self.use_existing)
        cloned.lora_stack = self.lora_stack.copy()
        cloned.fb_cache_params = self.fb_cache_params.copy() if self.fb_cache_params else None
        cloned._registered = self._registered
        return cloned
    
    def patch_model(self, *args, **kwargs):
        """Compatibility method - LoRAs handled via add_lora()."""
        pass
    
    def unpatch_model(self, *args, **kwargs):
        """Compatibility method - daemon manages unpatching."""
        pass
    
    def get_model_object(self, name: str):
        """Get model object by name (for compatibility)."""
        if name == "model_sampling" and self.model_sampling is not None:
            return self.model_sampling
        # For other names, try source_model
        if self.source_model is not None and hasattr(self.source_model, 'get_model_object'):
            return self.source_model.get_model_object(name)
        return self
    
    def model_patches_to(self, device):
        """Move patches to device (no-op for proxy)."""
        pass
    
    def __getattr__(self, name: str):
        """
        Forward unknown attributes to source model for compatibility.
        
        This allows third-party nodes to access attributes like model_sampling.percent_to_sigma()
        without explicitly implementing them. Falls back to source_model for attribute lookup.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute from source model if available, else raises AttributeError
        """
        # Avoid infinite recursion on initialization
        if name.startswith('_'):
            raise AttributeError(f"'DaemonModel' object has no attribute '{name}'")
        
        # Try to get from source_model if available
        source_model = object.__getattribute__(self, 'source_model')
        if source_model is not None:
            try:
                return getattr(source_model, name)
            except AttributeError:
                pass
        
        # Standard Python behavior for missing attributes
        raise AttributeError(f"'DaemonModel' object has no attribute '{name}'")
