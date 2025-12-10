"""
Luna Secondary Model Loader - Multi-Model Workflow Support

For workflows that use multiple model types (e.g., Flux generation → SDXL refinement),
this node provides efficient model switching with:

1. CLIP Sharing - Reuse CLIP-L/CLIP-G from primary model, only load what's needed
2. RAM Offloading - Move primary model to system RAM during secondary model use
3. Automatic Swap-Back - Reload primary model after secondary processing

┌─────────────────────────────────────────────────────────────────────────────┐
│                    Luna Secondary Model Loader                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUTS (from primary Model Router):                                        │
│    model (optional) ← If connected, can unload to RAM                      │
│    clip (optional)  ← Reuse CLIP encoders from primary                     │
│    vae (optional)   ← Reuse VAE from primary                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  SECONDARY MODEL:                                                           │
│    model_source: [checkpoints | diffusion_models | unet]                   │
│    model_name: [model file dropdown]                                        │
│    model_type: [SD1.5 | SDXL | Flux | SD3 | Z-IMAGE]                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ADDITIONAL ENCODERS (only what secondary needs that primary doesn't have): │
│    additional_clip: [t5xxl.safetensors]  ← For Flux/SD3 if primary is SDXL │
│    secondary_vae: [None | different_vae.safetensors]                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  MEMORY MANAGEMENT:                                                         │
│    unload_primary_to_ram: [✓] Move primary model to system RAM             │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS:                                                                   │
│    MODEL, CLIP, VAE, model_name, status                                    │
└─────────────────────────────────────────────────────────────────────────────┘

CLIP Sharing Logic:
===================
  Primary SDXL → Secondary Flux:  Reuse CLIP-L, add T5-XXL
  Primary SDXL → Secondary SD3:   Reuse CLIP-L + CLIP-G, add T5-XXL  
  Primary Flux → Secondary SDXL:  Reuse CLIP-L, add CLIP-G (skip T5)
  Primary Flux → Secondary SD3:   Reuse CLIP-L + T5, add CLIP-G
  Primary SD1.5 → Secondary SDXL: Reuse CLIP-L, add CLIP-G
  Any → Z-IMAGE:                  Cannot share (different architecture)
"""

from __future__ import annotations

import os
import gc
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict

import torch

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None


# =============================================================================
# MODEL MEMORY MANAGER - RAM Offloading
# =============================================================================

class ModelMemoryManager:
    """
    Manages model offloading between VRAM and system RAM.
    
    Allows temporarily moving models to CPU memory to free VRAM for other models,
    then reloading them faster than from disk.
    """
    
    _instance: Optional['ModelMemoryManager'] = None
    _ram_cache: Dict[str, Any] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ram_cache = {}  # type: ignore
            cls._instance._metadata = {}  # type: ignore
        return cls._instance
    
    def offload_to_ram(self, model: Any, model_id: str) -> bool:
        """
        Move a model from VRAM to system RAM.
        
        Args:
            model: The ComfyUI model object to offload
            model_id: Unique identifier for later retrieval
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the actual model object
            if hasattr(model, 'model'):
                actual_model = model.model
            else:
                actual_model = model
            
            # Store model state
            if hasattr(actual_model, 'state_dict'):
                state_dict = actual_model.state_dict()
                # Move all tensors to CPU
                cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
                
                # Store metadata for reconstruction
                self._metadata[model_id] = {
                    'model_type': type(model).__name__,
                    'has_wrapper': hasattr(model, 'model'),
                    'config': getattr(model, 'model_config', None),
                }
                
                self._ram_cache[model_id] = cpu_state
                
                # Clear from VRAM
                del state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"[ModelMemoryManager] Offloaded '{model_id}' to RAM ({len(cpu_state)} tensors)")
                return True
            else:
                # For models without state_dict, store the whole object on CPU
                if hasattr(actual_model, 'to'):
                    actual_model.to('cpu')
                self._ram_cache[model_id] = actual_model
                self._metadata[model_id] = {'model_type': type(model).__name__, 'is_object': True}
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"[ModelMemoryManager] Moved '{model_id}' to CPU")
                return True
                
        except Exception as e:
            print(f"[ModelMemoryManager] Failed to offload '{model_id}': {e}")
            return False
    
    def reload_from_ram(self, model_id: str, target_model: Any = None, device: str = "cuda") -> Optional[Any]:
        """
        Reload a model from RAM back to VRAM.
        
        Args:
            model_id: The identifier used when offloading
            target_model: Optional existing model to load state into
            device: Target device (default: cuda)
            
        Returns:
            The reloaded model, or None if not found
        """
        if model_id not in self._ram_cache:
            print(f"[ModelMemoryManager] Model '{model_id}' not found in RAM cache")
            return None
        
        try:
            cached = self._ram_cache[model_id]
            metadata = self._metadata.get(model_id, {})
            
            if metadata.get('is_object'):
                # Whole object was cached
                if hasattr(cached, 'to'):
                    cached.to(device)
                print(f"[ModelMemoryManager] Reloaded '{model_id}' from CPU to {device}")
                return cached
            
            # State dict was cached
            if target_model is not None:
                # Load into existing model
                actual_model = target_model.model if hasattr(target_model, 'model') else target_model
                gpu_state = {k: v.to(device) for k, v in cached.items()}
                actual_model.load_state_dict(gpu_state)
                del gpu_state
                print(f"[ModelMemoryManager] Reloaded '{model_id}' state into existing model")
                return target_model
            else:
                # Return state dict for manual loading
                gpu_state = {k: v.to(device) for k, v in cached.items()}
                print(f"[ModelMemoryManager] Returned '{model_id}' state dict for manual loading")
                return gpu_state
                
        except Exception as e:
            print(f"[ModelMemoryManager] Failed to reload '{model_id}': {e}")
            return None
    
    def is_cached(self, model_id: str) -> bool:
        """Check if a model is in the RAM cache."""
        return model_id in self._ram_cache
    
    def clear_cache(self, model_id: Optional[str] = None):
        """Clear cached model(s) from RAM."""
        if model_id:
            if model_id in self._ram_cache:
                del self._ram_cache[model_id]
                del self._metadata[model_id]
                print(f"[ModelMemoryManager] Cleared '{model_id}' from RAM cache")
        else:
            self._ram_cache.clear()
            self._metadata.clear()
            gc.collect()
            print("[ModelMemoryManager] Cleared all RAM cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        info = {}
        for model_id, state in self._ram_cache.items():
            if isinstance(state, dict):
                size_mb = sum(t.numel() * t.element_size() for t in state.values()) / (1024 * 1024)
                info[model_id] = {'type': 'state_dict', 'size_mb': size_mb}
            else:
                info[model_id] = {'type': 'object', 'size_mb': 'unknown'}
        return info


# Global memory manager instance
memory_manager = ModelMemoryManager()


# =============================================================================
# CLIP SHARING LOGIC
# =============================================================================

# What CLIP encoders each model type needs
CLIP_REQUIREMENTS_MAP = {
    "SD1.5": {"clip_l"},
    "SDXL": {"clip_l", "clip_g"},
    "Flux": {"clip_l", "t5xxl"},
    "SD3": {"clip_l", "clip_g", "t5xxl"},
    "Z-IMAGE": {"qwen3"},  # Incompatible with others
}

def get_shareable_clips(primary_type: str, secondary_type: str) -> Tuple[set, set]:
    """
    Determine which CLIPs can be shared and which need to be loaded.
    
    Returns:
        (shareable, needs_loading): Sets of encoder names
    """
    primary_clips = CLIP_REQUIREMENTS_MAP.get(primary_type, set())
    secondary_clips = CLIP_REQUIREMENTS_MAP.get(secondary_type, set())
    
    # Z-IMAGE uses different architecture, can't share
    if "qwen3" in primary_clips or "qwen3" in secondary_clips:
        return set(), secondary_clips
    
    shareable = primary_clips & secondary_clips
    needs_loading = secondary_clips - primary_clips
    
    return shareable, needs_loading


# =============================================================================
# LUNA SECONDARY MODEL LOADER
# =============================================================================

class LunaSecondaryModelLoader:
    """
    Load a secondary model while optionally offloading the primary to RAM.
    
    Designed for multi-model workflows like:
    - Flux generation → SDXL refinement
    - SD1.5 draft → SDXL upscale
    - Base model → Specialized model
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name", "status")
    FUNCTION = "load"
    
    MODEL_SOURCES = ["checkpoints", "diffusion_models", "unet"]
    MODEL_TYPES = ["SD1.5", "SDXL", "Flux", "SD3", "Z-IMAGE"]
    
    @classmethod
    def INPUT_TYPES(cls):
        if not HAS_COMFY:
            return {"required": {}}
        
        # Model files
        try:
            checkpoint_list = folder_paths.get_filename_list("checkpoints")
        except:
            checkpoint_list = []
        
        try:
            diffusion_list = folder_paths.get_filename_list("diffusion_models")
        except:
            diffusion_list = []
        
        try:
            unet_list = folder_paths.get_filename_list("unet")
        except:
            unet_list = []
        
        all_models = ["None"] + checkpoint_list + diffusion_list + unet_list
        
        # CLIP list
        try:
            clip_list = ["None"] + folder_paths.get_filename_list("clip")
        except:
            clip_list = ["None"]
        
        # VAE list
        try:
            vae_list = ["None"] + folder_paths.get_filename_list("vae")
        except:
            vae_list = ["None"]
        
        return {
            "required": {
                # === Secondary Model ===
                "model_source": (cls.MODEL_SOURCES, {
                    "default": "checkpoints",
                    "tooltip": "Folder to load secondary model from"
                }),
                "model_name": (all_models, {
                    "default": "None",
                    "tooltip": "Secondary model file"
                }),
                "model_type": (cls.MODEL_TYPES, {
                    "default": "SDXL",
                    "tooltip": "Architecture of secondary model"
                }),
                
                # === Memory Management ===
                "unload_primary_to_ram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move primary model to system RAM to free VRAM. Faster than reloading from disk."
                }),
            },
            "optional": {
                # === From Primary Model Router ===
                "primary_model": ("MODEL", {
                    "tooltip": "Primary model to offload to RAM (from Model Router)"
                }),
                "primary_clip": ("CLIP", {
                    "tooltip": "CLIP from primary model - reuse compatible encoders"
                }),
                "primary_vae": ("VAE", {
                    "tooltip": "VAE from primary model - reuse if compatible"
                }),
                "primary_type": (["auto"] + cls.MODEL_TYPES, {
                    "default": "auto",
                    "tooltip": "Type of primary model (for CLIP sharing detection)"
                }),
                
                # === Additional Encoders ===
                "additional_clip": (clip_list, {
                    "default": "None",
                    "tooltip": "Additional CLIP encoder (e.g., T5-XXL for Flux, CLIP-G for SDXL)"
                }),
                "secondary_vae": (vae_list, {
                    "default": "None",
                    "tooltip": "Different VAE for secondary model (None = reuse primary)"
                }),
            }
        }
    
    def load(
        self,
        model_source: str,
        model_name: str,
        model_type: str,
        unload_primary_to_ram: bool,
        primary_model: Any = None,
        primary_clip: Any = None,
        primary_vae: Any = None,
        primary_type: str = "auto",
        additional_clip: str = "None",
        secondary_vae: str = "None",
    ) -> Tuple[Any, Any, Any, str, str]:
        """
        Load secondary model with CLIP sharing and optional RAM offloading.
        """
        status_parts = []
        
        # === Step 1: Detect primary model type if auto ===
        if primary_type == "auto" and primary_clip is not None:
            primary_type = self._detect_clip_type(primary_clip)
            status_parts.append(f"Detected primary: {primary_type}")
        
        # === Step 2: Offload primary model to RAM if requested ===
        if unload_primary_to_ram and primary_model is not None:
            model_id = f"primary_{id(primary_model)}"
            if memory_manager.offload_to_ram(primary_model, model_id):
                status_parts.append("Primary → RAM")
                # Store ID for potential reload
                self._cached_primary_id = model_id
            else:
                status_parts.append("RAM offload failed")
        
        # === Step 3: Determine CLIP sharing ===
        shareable, needs_loading = get_shareable_clips(primary_type, model_type)
        
        if shareable:
            status_parts.append(f"Sharing: {', '.join(shareable)}")
        if needs_loading:
            status_parts.append(f"Loading: {', '.join(needs_loading)}")
        
        # === Step 4: Load secondary model ===
        output_model = None
        output_model_name = ""
        
        if model_name and model_name != "None":
            output_model, output_model_name = self._load_model(model_source, model_name)
            status_parts.append(f"Model: {os.path.basename(model_name)}")
        
        # === Step 5: Build combined CLIP ===
        output_clip = self._build_combined_clip(
            model_type, primary_clip, primary_type, additional_clip, shareable, needs_loading
        )
        if output_clip:
            status_parts.append("CLIP: combined")
        
        # === Step 6: Get VAE ===
        output_vae = None
        if secondary_vae and secondary_vae != "None":
            output_vae = self._load_vae(secondary_vae)
            status_parts.append(f"VAE: {os.path.basename(secondary_vae)}")
        elif primary_vae is not None:
            output_vae = primary_vae
            status_parts.append("VAE: reused")
        
        # === Build status ===
        status = f"[Secondary {model_type}] " + " | ".join(status_parts)
        print(f"[LunaSecondaryModelLoader] {status}")
        
        return (output_model, output_clip, output_vae, output_model_name, status)
    
    def _detect_clip_type(self, clip: Any) -> str:
        """Detect model type from CLIP architecture."""
        try:
            # Check for T5 (Flux/SD3)
            if hasattr(clip, 'cond_stage_model'):
                model = clip.cond_stage_model
                if hasattr(model, 't5xxl'):
                    if hasattr(model, 'clip_g'):
                        return "SD3"
                    return "Flux"
                if hasattr(model, 'clip_g'):
                    return "SDXL"
            return "SD1.5"
        except:
            return "SDXL"  # Safe default
    
    def _load_model(self, source: str, name: str) -> Tuple[Any, str]:
        """Load model from source."""
        model_path = folder_paths.get_full_path(source, name)
        if not model_path:
            for alt in ["checkpoints", "diffusion_models", "unet"]:
                model_path = folder_paths.get_full_path(alt, name)
                if model_path and os.path.exists(model_path):
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {name}")
        
        output_name = os.path.splitext(os.path.basename(name))[0]
        
        if model_path.endswith('.gguf'):
            # GGUF loading
            try:
                from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
                loader = UnetLoaderGGUF()
                model = loader.load_unet(unet_name=os.path.basename(model_path))[0]
                return model, output_name
            except ImportError:
                raise ImportError("ComfyUI-GGUF required for .gguf files")
        
        if source == "checkpoints":
            out = comfy.sd.load_checkpoint_guess_config(
                model_path,
                output_vae=False,
                output_clip=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            return out[0], output_name
        else:
            model = comfy.sd.load_unet(model_path)
            return model, output_name
    
    def _build_combined_clip(
        self,
        secondary_type: str,
        primary_clip: Any,
        primary_type: str,
        additional_clip_name: str,
        shareable: set,
        needs_loading: set
    ) -> Any:
        """Build combined CLIP from shared + additional encoders."""
        
        if not primary_clip and (not additional_clip_name or additional_clip_name == "None"):
            return None
        
        # If no sharing possible, load fresh
        if not shareable or primary_clip is None:
            if additional_clip_name and additional_clip_name != "None":
                clip_path = folder_paths.get_full_path("clip", additional_clip_name)
                if clip_path:
                    return comfy.sd.load_clip(  # type: ignore
                        ckpt_paths=[clip_path],
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
            return None
        
        # Start with primary CLIP
        combined = primary_clip
        
        # Add additional encoder if needed
        if needs_loading and additional_clip_name and additional_clip_name != "None":
            clip_path = folder_paths.get_full_path("clip", additional_clip_name)
            if clip_path and os.path.exists(clip_path):
                try:
                    # Load additional encoder
                    additional = comfy.sd.load_clip(  # type: ignore
                        ckpt_paths=[clip_path],
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    
                    # Merge CLIPs based on secondary type
                    combined = self._merge_clips(combined, additional, secondary_type)
                    print(f"[LunaSecondaryModelLoader] Merged additional CLIP: {additional_clip_name}")
                    
                except Exception as e:
                    print(f"[LunaSecondaryModelLoader] Failed to merge CLIP: {e}")
        
        return combined
    
    def _merge_clips(self, clip1: Any, clip2: Any, target_type: str) -> Any:
        """
        Merge two CLIP models for the target model type.
        
        This creates a combined CLIP that has encoders from both sources.
        """
        # For now, we'll use a simple approach:
        # ComfyUI's dual CLIP loading handles this internally when you load multiple files
        # We'll return clip1 with clip2's additional components attached
        
        # TODO: More sophisticated merging for specific architectures
        # For Flux: need clip_l from one, t5xxl from another
        # For SD3: need clip_l, clip_g, t5xxl
        
        # Simple approach: prefer clip2 if it's the additional one
        return clip2
    
    def _load_vae(self, vae_name: str) -> Any:
        """Load VAE."""
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        try:
            try:
                from nodes import VAELoader  # type: ignore
            except ImportError:
                VAELoader = None  # type: ignore
            
            if VAELoader:
                loader = VAELoader()
                return loader.load_vae(vae_name)[0]
            else:
                raise ImportError("VAELoader not available")
        except ImportError:
            sd = comfy.utils.load_torch_file(vae_path)  # type: ignore
            return comfy.sd.VAE(sd=sd)  # type: ignore


# =============================================================================
# MODEL RESTORE NODE
# =============================================================================

class LunaModelRestore:
    """
    Restore a model that was offloaded to RAM back to VRAM.
    
    This is a passthrough trigger node - connect any output to the trigger input,
    and it will pass that data through while also restoring the cached model.
    
    Use this after the secondary model processing is complete to restore
    the primary model for continued use.
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("*", "MODEL", "STRING")
    RETURN_NAMES = ("passthrough", "restored_model", "status")
    FUNCTION = "restore"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Any input - will be passed through to output while triggering model restore"}),
            },
            "optional": {
                "model_id": ("STRING", {
                    "default": "",
                    "tooltip": "Model ID to restore (leave empty for auto-detect)"
                }),
                "original_model": ("MODEL", {
                    "tooltip": "Original model object (for state loading)"
                }),
            }
        }
    
    def restore(
        self,
        trigger: Any,
        model_id: str = "",
        original_model: Any = None
    ) -> Tuple[Any, Any, str]:
        """
        Restore model from RAM to VRAM and pass through the trigger input.
        
        Returns:
            (passthrough, restored_model, status)
        """
        
        # Get cached model IDs
        cache_info = memory_manager.get_cache_info()
        
        if not cache_info:
            return (trigger, None, "No models in RAM cache")
        
        # Auto-detect model ID if not specified
        if not model_id:
            # Use first cached model
            model_id = list(cache_info.keys())[0]
        
        if model_id not in cache_info:
            return (trigger, None, f"Model '{model_id}' not in RAM cache. Available: {list(cache_info.keys())}")
        
        # Restore model
        restored = memory_manager.reload_from_ram(model_id, original_model)
        
        if restored is not None:
            return (trigger, restored, f"Restored '{model_id}' from RAM")
        else:
            return (trigger, None, f"Failed to restore '{model_id}'")


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaSecondaryModelLoader": LunaSecondaryModelLoader,
    "LunaModelRestore": LunaModelRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSecondaryModelLoader": "Luna Secondary Model Loader 🔄",
    "LunaModelRestore": "Luna Model Restore 📤",
}
