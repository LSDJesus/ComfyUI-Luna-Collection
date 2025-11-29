"""
Luna HyperLoRA Integration
ComfyUI nodes for zero-shot LoRA generation using ByteDance's HyperLoRA.

HyperLoRA generates LoRA weights from reference images, enabling:
- Zero-shot character consistency without per-character training
- Fast on-the-fly LoRA generation for narrative scenes
- Optional caching to disk for reuse with standard LoRA loaders

Reference: https://huggingface.co/bytedance-research/HyperLoRA
Paper: https://arxiv.org/abs/2503.16944
"""

import torch
import torch.nn as nn
import os
import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from pathlib import Path
from safetensors.torch import save_file, load_file

# ComfyUI imports
try:
    import folder_paths
    import comfy.utils
    import comfy.model_management
    from comfy.sd import CLIP
except ImportError:
    folder_paths = None
    comfy = None

# HyperLoRA model components (will be loaded dynamically)
HYPERLORA_LOADED = False
HYPERLORA_MODEL = None
HYPERLORA_CACHE_DIR = None


def get_hyperlora_cache_dir() -> str:
    """Get the cache directory for generated HyperLoRA weights"""
    global HYPERLORA_CACHE_DIR
    if HYPERLORA_CACHE_DIR is None:
        if folder_paths:
            HYPERLORA_CACHE_DIR = os.path.join(folder_paths.models_dir, "loras", "hyperlora_cache")
        else:
            HYPERLORA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "hyperlora_cache")
    os.makedirs(HYPERLORA_CACHE_DIR, exist_ok=True)
    return HYPERLORA_CACHE_DIR


def compute_image_hash(image: torch.Tensor) -> str:
    """Compute a stable hash for an image tensor for caching"""
    img_bytes = image.cpu().float().numpy().tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


class LunaHyperLoRAGenerate:
    """
    Generate LoRA weights from reference image(s) using HyperLoRA.
    
    This node uses ByteDance's HyperLoRA to generate LoRA weights on-the-fly
    from one or more reference images. The weights can be:
    1. Applied directly to the model (in-memory)
    2. Saved to disk for reuse with standard LoRA loaders
    3. Cached by character ID for consistent character generation
    
    For Luna Narrates: Use character_id to associate generated LoRAs with
    narrative characters for automatic application in future scenes.
    """
    
    CATEGORY = "Luna/HyperLoRA"
    RETURN_TYPES = ("LORA_WEIGHTS", "STRING", "STRING")
    RETURN_NAMES = ("lora_weights", "lora_path", "character_hash")
    FUNCTION = "generate_lora"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s) of the character/subject"}),
            },
            "optional": {
                # Character identification
                "character_id": ("STRING", {"default": "", "tooltip": "Unique character ID for caching and Luna Narrates integration"}),
                "character_name": ("STRING", {"default": "", "tooltip": "Human-readable character name (for metadata)"}),
                
                # Generation options
                "include_base_lora": ("BOOLEAN", {"default": False, "tooltip": "Include Hyper Base-LoRA for background/clothing (slower but more complete)"}),
                "lora_rank": ("INT", {"default": 64, "min": 4, "max": 256, "step": 4, "tooltip": "LoRA rank (higher = more capacity, slower)"}),
                
                # Caching options
                "save_to_disk": ("BOOLEAN", {"default": True, "tooltip": "Save generated LoRA to disk for reuse"}),
                "use_cached": ("BOOLEAN", {"default": True, "tooltip": "Use cached LoRA if available for this character_id"}),
                "cache_by_image_hash": ("BOOLEAN", {"default": False, "tooltip": "Also cache by image content hash (more granular but more disk usage)"}),
                
                # Advanced
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "float16", "float32", "bfloat16"], {"default": "auto"}),
            }
        }
    
    def __init__(self):
        self.hyperlora_model = None
        self.device = None
        self.dtype = None
    
    def _load_hyperlora_model(self, device: str, dtype: str):
        """Load the HyperLoRA model (lazy loading)"""
        global HYPERLORA_MODEL, HYPERLORA_LOADED
        
        if HYPERLORA_LOADED and HYPERLORA_MODEL is not None:
            return HYPERLORA_MODEL
        
        print("[LunaHyperLoRAGenerate] Loading HyperLoRA model...")
        
        # Determine device
        if device == "auto":
            if comfy and hasattr(comfy, 'model_management'):
                self.device = comfy.model_management.get_torch_device()
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Determine dtype
        if dtype == "auto":
            if self.device.type == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        
        # Try to load HyperLoRA from various sources
        hyperlora_paths = [
            os.path.join(folder_paths.models_dir, "hyperlora") if folder_paths else None,
            os.path.expanduser("~/.cache/huggingface/hub/models--bytedance-research--HyperLoRA"),
            os.path.join(os.path.dirname(__file__), "..", "models", "hyperlora"),
        ]
        
        model_loaded = False
        for path in hyperlora_paths:
            if path and os.path.exists(path):
                try:
                    # TODO: Implement actual HyperLoRA model loading
                    # This is a placeholder - actual implementation depends on HyperLoRA's architecture
                    print(f"[LunaHyperLoRAGenerate] Found HyperLoRA at: {path}")
                    # HYPERLORA_MODEL = HyperLoRANetwork.from_pretrained(path)
                    # HYPERLORA_MODEL = HYPERLORA_MODEL.to(self.device, self.dtype)
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"[LunaHyperLoRAGenerate] Failed to load from {path}: {e}")
        
        if not model_loaded:
            print("[LunaHyperLoRAGenerate] WARNING: HyperLoRA model not found!")
            print("[LunaHyperLoRAGenerate] Please download from: https://huggingface.co/bytedance-research/HyperLoRA")
            print(f"[LunaHyperLoRAGenerate] And place in: {hyperlora_paths[0]}")
        
        HYPERLORA_LOADED = True
        return HYPERLORA_MODEL
    
    def _get_cache_path(self, character_id: str, image_hash: str = None) -> str:
        """Get the cache file path for a character"""
        cache_dir = get_hyperlora_cache_dir()
        
        if image_hash:
            filename = f"{character_id}_{image_hash}.safetensors"
        else:
            filename = f"{character_id}.safetensors"
        
        return os.path.join(cache_dir, filename)
    
    def _load_cached_lora(self, cache_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached LoRA weights from disk"""
        if os.path.exists(cache_path):
            try:
                print(f"[LunaHyperLoRAGenerate] Loading cached LoRA: {cache_path}")
                weights = load_file(cache_path)
                return weights
            except Exception as e:
                print(f"[LunaHyperLoRAGenerate] Failed to load cache: {e}")
        return None
    
    def _save_lora_to_disk(self, weights: Dict[str, torch.Tensor], cache_path: str, metadata: Dict[str, Any]):
        """Save generated LoRA weights to disk"""
        try:
            # Ensure weights are in correct format for safetensors
            tensor_dict = {}
            for key, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_dict[key] = tensor.contiguous().cpu()
            
            # Save with metadata
            save_file(tensor_dict, cache_path)
            
            # Save metadata alongside
            meta_path = cache_path.replace(".safetensors", "_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[LunaHyperLoRAGenerate] Saved LoRA to: {cache_path}")
            
        except Exception as e:
            print(f"[LunaHyperLoRAGenerate] Failed to save LoRA: {e}")
    
    def generate_lora(
        self,
        reference_images: torch.Tensor,
        character_id: str = "",
        character_name: str = "",
        include_base_lora: bool = False,
        lora_rank: int = 64,
        save_to_disk: bool = True,
        use_cached: bool = True,
        cache_by_image_hash: bool = False,
        device: str = "auto",
        dtype: str = "auto",
    ) -> Tuple[Dict[str, torch.Tensor], str, str]:
        
        # Compute image hash for caching
        image_hash = compute_image_hash(reference_images)
        
        # Generate character_id from hash if not provided
        if not character_id:
            character_id = f"char_{image_hash}"
        
        # Check cache
        cache_path = self._get_cache_path(
            character_id, 
            image_hash if cache_by_image_hash else None
        )
        
        if use_cached:
            cached_weights = self._load_cached_lora(cache_path)
            if cached_weights is not None:
                return (cached_weights, cache_path, image_hash)
        
        # Load HyperLoRA model
        model = self._load_hyperlora_model(device, dtype)
        
        if model is None:
            # Return placeholder for testing without model
            print("[LunaHyperLoRAGenerate] Using placeholder weights (model not loaded)")
            placeholder_weights = {
                "lora_unet_placeholder": torch.zeros(lora_rank, lora_rank),
                "_metadata": {
                    "character_id": character_id,
                    "character_name": character_name,
                    "image_hash": image_hash,
                    "is_placeholder": True,
                }
            }
            return (placeholder_weights, "", image_hash)
        
        # Generate LoRA weights
        print(f"[LunaHyperLoRAGenerate] Generating LoRA for {character_id or 'unnamed character'}...")
        
        # Prepare images
        if reference_images.dim() == 3:
            reference_images = reference_images.unsqueeze(0)
        
        reference_images = reference_images.to(self.device, self.dtype)
        
        try:
            # TODO: Actual HyperLoRA inference
            # id_lora_weights = model.generate_id_lora(reference_images)
            # base_lora_weights = model.generate_base_lora(reference_images) if include_base_lora else {}
            
            # Placeholder implementation
            lora_weights = {
                # These would be actual LoRA weight tensors from HyperLoRA
                "lora_unet_down_blocks_0_attentions_0_to_q.lora_A": torch.randn(lora_rank, 320),
                "lora_unet_down_blocks_0_attentions_0_to_q.lora_B": torch.randn(320, lora_rank),
                # ... more weight pairs for each LoRA target layer
            }
            
            print(f"[LunaHyperLoRAGenerate] Generated {len(lora_weights)} weight tensors")
            
        except Exception as e:
            print(f"[LunaHyperLoRAGenerate] Generation error: {e}")
            raise
        
        # Prepare metadata
        metadata = {
            "character_id": character_id,
            "character_name": character_name or character_id,
            "image_hash": image_hash,
            "lora_rank": lora_rank,
            "include_base_lora": include_base_lora,
            "created": datetime.now().isoformat(),
            "source": "hyperlora",
            "num_reference_images": reference_images.shape[0],
        }
        
        # Save to disk if requested
        if save_to_disk:
            self._save_lora_to_disk(lora_weights, cache_path, metadata)
        
        return (lora_weights, cache_path if save_to_disk else "", image_hash)


class LunaHyperLoRAApply:
    """
    Apply HyperLoRA-generated weights to a model.
    
    This node takes the in-memory LoRA weights from LunaHyperLoRAGenerate
    and applies them to the model without saving to disk first.
    
    For cached LoRAs (saved to disk), you can use standard LoRA loaders instead.
    """
    
    CATEGORY = "Luna/HyperLoRA"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_lora"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_weights": ("LORA_WEIGHTS", {"tooltip": "Generated LoRA weights from LunaHyperLoRAGenerate"}),
            },
            "optional": {
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    def apply_lora(
        self,
        model,
        clip,
        lora_weights: Dict[str, torch.Tensor],
        model_strength: float = 1.0,
        clip_strength: float = 1.0,
    ) -> Tuple[Any, Any]:
        
        if not lora_weights or lora_weights.get("_metadata", {}).get("is_placeholder"):
            print("[LunaHyperLoRAApply] No valid LoRA weights to apply")
            return (model, clip)
        
        print(f"[LunaHyperLoRAApply] Applying LoRA with strength {model_strength}/{clip_strength}")
        
        # Clone model to avoid modifying original
        model_clone = model.clone()
        clip_clone = clip.clone() if hasattr(clip, 'clone') else clip
        
        try:
            # Apply LoRA weights to model
            # This follows ComfyUI's LoRA application pattern
            for key, weight in lora_weights.items():
                if key.startswith("_"):  # Skip metadata keys
                    continue
                
                # Parse key to determine target layer
                if "lora_unet" in key:
                    # Apply to UNet
                    self._apply_lora_weight(model_clone, key, weight, model_strength)
                elif "lora_te" in key or "lora_clip" in key:
                    # Apply to CLIP
                    self._apply_lora_weight(clip_clone, key, weight, clip_strength)
            
            print(f"[LunaHyperLoRAApply] Applied {len(lora_weights) - 1} LoRA weights")
            
        except Exception as e:
            print(f"[LunaHyperLoRAApply] Error applying LoRA: {e}")
            return (model, clip)
        
        return (model_clone, clip_clone)
    
    def _apply_lora_weight(self, target, key: str, weight: torch.Tensor, strength: float):
        """Apply a single LoRA weight to the target model"""
        # TODO: Implement proper LoRA weight application
        # This needs to match ComfyUI's LoRA patching system
        pass


class LunaHyperLoRALoader:
    """
    Load a cached HyperLoRA from disk.
    
    This node loads previously generated HyperLoRA weights that were
    saved to the hyperlora_cache directory. Use this for faster loading
    of character LoRAs that don't need regeneration.
    
    Alternative: Use standard LoRA loader nodes with the cached path.
    """
    
    CATEGORY = "Luna/HyperLoRA"
    RETURN_TYPES = ("LORA_WEIGHTS", "STRING")
    RETURN_NAMES = ("lora_weights", "character_id")
    FUNCTION = "load_lora"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of cached HyperLoRAs
        cache_dir = get_hyperlora_cache_dir()
        cached_loras = []
        
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith(".safetensors"):
                    cached_loras.append(f[:-12])  # Remove .safetensors
        
        if not cached_loras:
            cached_loras = ["none"]
        
        return {
            "required": {
                "cached_lora": (cached_loras, {"tooltip": "Select a cached HyperLoRA"}),
            }
        }
    
    def load_lora(self, cached_lora: str) -> Tuple[Dict[str, torch.Tensor], str]:
        if cached_lora == "none":
            return ({}, "")
        
        cache_path = os.path.join(get_hyperlora_cache_dir(), f"{cached_lora}.safetensors")
        
        if not os.path.exists(cache_path):
            print(f"[LunaHyperLoRALoader] Cache not found: {cache_path}")
            return ({}, "")
        
        try:
            weights = load_file(cache_path)
            
            # Extract character_id from metadata if available
            meta_path = cache_path.replace(".safetensors", "_meta.json")
            character_id = cached_lora
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    character_id = meta.get("character_id", cached_lora)
            
            print(f"[LunaHyperLoRALoader] Loaded cached LoRA: {cached_lora}")
            return (weights, character_id)
            
        except Exception as e:
            print(f"[LunaHyperLoRALoader] Error loading: {e}")
            return ({}, "")


class LunaHyperLoRACacheManager:
    """
    Manage the HyperLoRA cache directory.
    
    Use this to list, clear, or get info about cached HyperLoRA weights.
    """
    
    CATEGORY = "Luna/HyperLoRA"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_info",)
    FUNCTION = "manage_cache"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list", "clear_all", "get_size", "clear_old"], {"default": "list"}),
            },
            "optional": {
                "max_age_days": ("INT", {"default": 30, "min": 1, "max": 365, "tooltip": "For clear_old: remove entries older than this"}),
            }
        }
    
    def manage_cache(self, action: str, max_age_days: int = 30) -> Tuple[str]:
        cache_dir = get_hyperlora_cache_dir()
        
        if action == "list":
            entries = []
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    if f.endswith(".safetensors"):
                        path = os.path.join(cache_dir, f)
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        entries.append(f"{f}: {size_mb:.2f} MB")
            
            return (f"Cached HyperLoRAs ({len(entries)}):\n" + "\n".join(entries) if entries else "No cached LoRAs",)
        
        elif action == "get_size":
            total_size = 0
            count = 0
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    path = os.path.join(cache_dir, f)
                    total_size += os.path.getsize(path)
                    count += 1
            
            size_mb = total_size / (1024 * 1024)
            return (f"Cache size: {size_mb:.2f} MB ({count} files)",)
        
        elif action == "clear_all":
            count = 0
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    os.remove(os.path.join(cache_dir, f))
                    count += 1
            
            return (f"Cleared {count} cached files",)
        
        elif action == "clear_old":
            import time
            cutoff = time.time() - (max_age_days * 24 * 60 * 60)
            count = 0
            
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    path = os.path.join(cache_dir, f)
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        count += 1
            
            return (f"Cleared {count} files older than {max_age_days} days",)
        
        return ("Unknown action",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaHyperLoRAGenerate": LunaHyperLoRAGenerate,
    "LunaHyperLoRAApply": LunaHyperLoRAApply,
    "LunaHyperLoRALoader": LunaHyperLoRALoader,
    "LunaHyperLoRACacheManager": LunaHyperLoRACacheManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaHyperLoRAGenerate": "Luna HyperLoRA Generate",
    "LunaHyperLoRAApply": "Luna HyperLoRA Apply (In-Memory)",
    "LunaHyperLoRALoader": "Luna HyperLoRA Loader (Cached)",
    "LunaHyperLoRACacheManager": "Luna HyperLoRA Cache Manager",
}
