"""
Luna Universal Model Router - The Ultimate Model Loader

A single node that replaces ALL other model loading nodes:
- Luna Checkpoint Tunnel
- Luna UNet Tunnel  
- Luna Dynamic Model Loader
- Luna Daemon VAE/CLIP Loaders
- Standard checkpoint loaders
- GGUF UNet loaders

Features:
=========

1. **Universal Input Detection**
   - Accepts checkpoints, UNets, GGUF files, or pre-loaded MODEL/CLIP/VAE
   - Auto-detects model type (SDXL, SD1.5, Flux, SD3, Z-IMAGE)
   - Handles all precision formats (fp16, bf16, fp8, gguf)

2. **Smart Routing**
   - Daemon running? Route CLIP/VAE through daemon for sharing
   - Daemon not running? Load locally with fallback
   - Lazy evaluation: only load what's connected

3. **Precision Control**
   - Convert checkpoints to optimized UNets on first use
   - Store locally for fast subsequent loads
   - Support for bf16, fp8, GGUF quantization

4. **Z-IMAGE Auto-Detection**
   - Detects Qwen3-4B CLIP architecture automatically
   - Routes to Qwen3-VL encoder on daemon
   - No manual configuration needed

Architecture:
============

┌─────────────────────────────────────────────────────────────────────┐
│                    Luna Universal Model Router                       │
├─────────────────────────────────────────────────────────────────────┤
│  INPUTS (any combination):                                          │
│    ckpt_name    → Load from checkpoint file                         │
│    unet_name    → Load from UNet/GGUF file                          │
│    model        → Pass through existing MODEL                        │
│    clip         → Pass through or override CLIP                      │
│    vae          → Pass through or override VAE                       │
├─────────────────────────────────────────────────────────────────────┤
│  AUTO-DETECTION:                                                     │
│    • Model type: SDXL, SD1.5, Flux, SD3, Z-IMAGE                    │
│    • CLIP type: Standard CLIP-L/G, T5-XXL, Qwen3                    │
│    • VAE type: 4-channel (SDXL/SD1.5), 16-channel (Flux/SD3)        │
├─────────────────────────────────────────────────────────────────────┤
│  ROUTING LOGIC:                                                      │
│    ┌──────────────┐                                                 │
│    │ Daemon       │ YES → Route CLIP/VAE to daemon (shared)         │
│    │ Running?     │ NO  → Load locally                              │
│    └──────────────┘                                                 │
│    ┌──────────────┐                                                 │
│    │ Z-IMAGE      │ YES → Use DaemonZImageCLIP (Qwen3-VL)           │
│    │ Detected?    │ NO  → Use standard DaemonCLIP                   │
│    └──────────────┘                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUTS:                                                           │
│    MODEL  → UNet (local or from checkpoint)                         │
│    CLIP   → Proxied to daemon or loaded locally                     │
│    VAE    → Proxied to daemon or loaded locally                     │
│    STATUS → Detailed routing status string                          │
└─────────────────────────────────────────────────────────────────────┘

Usage Examples:
===============

1. **Standard Checkpoint → Daemon Sharing**
   - Select checkpoint, daemon routes CLIP/VAE
   
2. **GGUF UNet → Daemon CLIP/VAE**
   - Select GGUF file, daemon provides matching CLIP/VAE
   
3. **Pre-loaded MODEL + Custom CLIP**
   - Connect MODEL input, optionally override CLIP
   
4. **Z-IMAGE Workflow**
   - Auto-detected, routes to Qwen3-VL encoder

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict, List, Union

import torch

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None

# GGUF support
try:
    GGUF_NODE_PATHS = [
        "custom_nodes.ComfyUI-GGUF.nodes",
        "ComfyUI-GGUF.nodes",
    ]
    UnetLoaderGGUF = None
    for path in GGUF_NODE_PATHS:
        try:
            module = __import__(path, fromlist=['UnetLoaderGGUF'])
            UnetLoaderGGUF = getattr(module, 'UnetLoaderGGUF', None)
            if UnetLoaderGGUF:
                break
        except:
            continue
    HAS_GGUF = UnetLoaderGGUF is not None
except:
    HAS_GGUF = False
    UnetLoaderGGUF = None

# Daemon support
try:
    from ..luna_daemon.proxy import DaemonVAE, DaemonCLIP, detect_vae_type, detect_clip_type
    from ..luna_daemon.zimage_proxy import (
        DaemonZImageCLIP,
        detect_clip_architecture,
        is_zimage_clip,
        create_clip_proxy
    )
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonVAE = None
    DaemonCLIP = None
    DaemonZImageCLIP = None
    daemon_client = None
    DaemonConnectionError = Exception
    
    def detect_vae_type(vae) -> str: return 'unknown'
    def detect_clip_type(clip) -> str: return 'unknown'
    def detect_clip_architecture(clip) -> dict: return {'type': 'unknown', 'is_qwen': False}
    def is_zimage_clip(clip) -> bool: return False
    def create_clip_proxy(source_clip, use_existing=False, force_type=None): return source_clip


# =============================================================================
# Model Type Detection
# =============================================================================

def detect_model_type_from_path(path: str) -> str:
    """Detect model type from filename patterns."""
    name_lower = os.path.basename(path).lower()
    
    if any(x in name_lower for x in ['sdxl', 'xl', 'pony', 'realvis']):
        return 'sdxl'
    elif any(x in name_lower for x in ['flux', 'dev', 'schnell']):
        return 'flux'
    elif any(x in name_lower for x in ['sd3', 'stable-diffusion-3']):
        return 'sd3'
    elif any(x in name_lower for x in ['zimage', 'qwen', 'z-image']):
        return 'zimage'
    elif any(x in name_lower for x in ['sd15', '1.5', 'sd-1', 'realistic', 'deliberate']):
        return 'sd15'
    
    # Default to SDXL for modern models
    return 'sdxl'


def detect_model_type_from_state_dict(state_dict: Dict[str, Any]) -> str:
    """Detect model type from state dict keys and shapes."""
    keys = list(state_dict.keys())[:100]  # Sample first 100 keys
    keys_str = ' '.join(keys).lower()
    
    # Check for specific architectures
    if 'model.diffusion_model.joint_blocks' in keys_str:
        return 'sd3'
    elif 'double_blocks' in keys_str or 'single_blocks' in keys_str:
        return 'flux'
    elif 'conditioner.embedders.1' in keys_str:
        return 'sdxl'
    elif 'cond_stage_model.transformer' in keys_str:
        return 'sd15'
    
    # Check tensor shapes for SDXL vs SD1.5
    for key, tensor in state_dict.items():
        if 'model.diffusion_model.input_blocks.0.0.weight' in key:
            if hasattr(tensor, 'shape'):
                if tensor.shape[1] == 4:
                    return 'sdxl'  # Could be SD1.5 too
                elif tensor.shape[1] == 16:
                    return 'flux'
    
    return 'sdxl'  # Default


def detect_model_type_from_model(model) -> str:
    """Detect model type from a loaded MODEL object."""
    if model is None:
        return 'unknown'
    
    try:
        # Check model class
        class_name = type(model).__name__ if hasattr(model, '__class__') else ''
        
        if 'Flux' in class_name:
            return 'flux'
        elif 'SD3' in class_name:
            return 'sd3'
        elif 'SDXL' in class_name:
            return 'sdxl'
        
        # Check inner model
        inner = getattr(model, 'model', None)
        if inner is not None:
            inner_class = type(inner).__name__
            if 'Flux' in inner_class:
                return 'flux'
            elif 'SD3' in inner_class:
                return 'sd3'
        
        # Check diffusion model structure
        diffusion = getattr(model, 'diffusion_model', None)
        if diffusion is None and inner is not None:
            diffusion = getattr(inner, 'diffusion_model', None)
        
        if diffusion is not None:
            if hasattr(diffusion, 'joint_blocks'):
                return 'sd3'
            elif hasattr(diffusion, 'double_blocks'):
                return 'flux'
    
    except Exception:
        pass
    
    return 'sdxl'  # Default


# =============================================================================
# Luna Universal Model Router
# =============================================================================

class LunaUniversalModelRouter:
    """
    The Ultimate Model Loader - One node to rule them all.
    
    Replaces:
    - Luna Checkpoint Tunnel
    - Luna UNet Tunnel
    - Luna Dynamic Model Loader
    - Luna Daemon VAE/CLIP Loaders
    - Standard checkpoint loaders
    - GGUF loaders
    
    Auto-detects everything and routes intelligently!
    """
    
    CATEGORY = "Luna/Core"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "status")
    FUNCTION = "route"
    
    # Precision options for on-demand conversion
    PRECISION_OPTIONS = [
        "auto (use source precision)",
        "bf16 (universal)",
        "fp16 (legacy)",
        "fp8_e4m3fn (Ada/Blackwell)",
        "gguf_Q8_0 (Ampere INT8)",
        "gguf_Q4_K_M (Blackwell INT4)",
    ]
    
    # Model type hints
    MODEL_TYPES = ["auto", "sdxl", "sd15", "flux", "sd3", "zimage"]
    
    # Daemon modes
    DAEMON_MODES = [
        "auto (use daemon if running)",
        "force_daemon (require daemon)",
        "force_local (never use daemon)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        if not HAS_COMFY:
            return {"required": {}}
        
        # Build file lists
        ckpt_list = ["None"] + folder_paths.get_filename_list("checkpoints")
        unet_list = ["None"] + folder_paths.get_filename_list("diffusion_models") if \
                    hasattr(folder_paths, 'get_filename_list') else ["None"]
        vae_list = ["None"] + folder_paths.get_filename_list("vae")
        
        # Try to get CLIP list
        try:
            clip_list = ["None"] + folder_paths.get_filename_list("clip")
        except:
            clip_list = ["None"]
        
        return {
            "required": {},  # All inputs optional!
            "optional": {
                # === File-based inputs ===
                "ckpt_name": (ckpt_list, {
                    "default": "None",
                    "tooltip": "Load from checkpoint file (includes MODEL, CLIP, VAE)"
                }),
                "unet_name": (unet_list, {
                    "default": "None",
                    "tooltip": "Load UNet from diffusion_models folder (GGUF or safetensors)"
                }),
                "vae_name": (vae_list, {
                    "default": "None",
                    "tooltip": "Override VAE with specific file"
                }),
                "clip_name": (clip_list, {
                    "default": "None",
                    "tooltip": "Override CLIP with specific file"
                }),
                
                # === Pre-loaded inputs (from other nodes) ===
                "model": ("MODEL", {
                    "tooltip": "Pre-loaded MODEL to pass through or route"
                }),
                "clip": ("CLIP", {
                    "tooltip": "Pre-loaded CLIP to pass through or route"
                }),
                "vae": ("VAE", {
                    "tooltip": "Pre-loaded VAE to pass through or route"
                }),
                
                # === Configuration ===
                "model_type": (cls.MODEL_TYPES, {
                    "default": "auto",
                    "tooltip": "Model architecture (auto-detected if not specified)"
                }),
                "precision": (cls.PRECISION_OPTIONS, {
                    "default": cls.PRECISION_OPTIONS[0],
                    "tooltip": "UNet precision (for conversion on first load)"
                }),
                "daemon_mode": (cls.DAEMON_MODES, {
                    "default": cls.DAEMON_MODES[0],
                    "tooltip": "How to handle daemon routing"
                }),
                
                # === Advanced ===
                "local_weights_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory for optimized UNet cache (default: models/unet/optimized)"
                }),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
    
    def route(
        self,
        ckpt_name: str = "None",
        unet_name: str = "None",
        vae_name: str = "None",
        clip_name: str = "None",
        model: Any = None,
        clip: Any = None,
        vae: Any = None,
        model_type: str = "auto",
        precision: str = "auto (use source precision)",
        daemon_mode: str = "auto (use daemon if running)",
        local_weights_dir: str = "",
        dynprompt=None,
        unique_id=None
    ) -> Tuple[Any, Any, Any, str]:
        """
        Universal routing logic.
        
        Priority:
        1. Pre-loaded inputs (model/clip/vae) take precedence
        2. File inputs (ckpt_name/unet_name) load if no pre-loaded
        3. Daemon provides CLIP/VAE if not otherwise specified
        """
        status_parts = []
        output_model = model
        output_clip = clip
        output_vae = vae
        detected_type = model_type
        
        # Parse daemon mode
        use_daemon = "force_local" not in daemon_mode
        require_daemon = "force_daemon" in daemon_mode
        
        # Check daemon availability
        daemon_running = DAEMON_AVAILABLE and daemon_client is not None and daemon_client.is_daemon_running()
        
        # Helper to check if already a daemon proxy
        def is_daemon_clip(obj):
            if DaemonCLIP is not None and isinstance(obj, DaemonCLIP):
                return True
            if DaemonZImageCLIP is not None and isinstance(obj, DaemonZImageCLIP):
                return True
            return False
        
        if require_daemon and not daemon_running:
            raise RuntimeError(
                "Daemon mode is 'force_daemon' but Luna Daemon is not running!\n"
                "Start the daemon or change mode to 'auto' or 'force_local'."
            )
        
        # === STEP 1: Determine what needs loading ===
        need_model = output_model is None
        need_clip = output_clip is None
        need_vae = output_vae is None
        
        # Check lazy evaluation for connected outputs
        if dynprompt is not None and unique_id is not None:
            try:
                need_clip = need_clip and self._is_output_connected(dynprompt, unique_id, 1)
                need_vae = need_vae and self._is_output_connected(dynprompt, unique_id, 2)
            except:
                pass  # If we can't determine, assume needed
        
        # === STEP 2: Load from files if needed ===
        
        # Load from checkpoint
        if need_model and ckpt_name and ckpt_name != "None":
            output_model, loaded_clip, loaded_vae, ckpt_type = self._load_checkpoint(
                ckpt_name, precision, local_weights_dir, need_clip, need_vae
            )
            
            if need_clip and loaded_clip is not None:
                output_clip = loaded_clip
            if need_vae and loaded_vae is not None:
                output_vae = loaded_vae
            
            if detected_type == "auto":
                detected_type = ckpt_type
            
            status_parts.append(f"MODEL: checkpoint ({ckpt_type})")
            need_model = False
        
        # Load from UNet file
        if need_model and unet_name and unet_name != "None":
            output_model, unet_type = self._load_unet(unet_name)
            
            if detected_type == "auto":
                detected_type = unet_type
            
            status_parts.append(f"MODEL: UNet ({unet_type})")
            need_model = False
        
        # Load specific VAE file
        if need_vae and vae_name and vae_name != "None":
            output_vae = self._load_vae(vae_name)
            status_parts.append("VAE: file")
            need_vae = False
        
        # Load specific CLIP file
        if need_clip and clip_name and clip_name != "None":
            output_clip = self._load_clip(clip_name)
            status_parts.append("CLIP: file")
            need_clip = False
        
        # === STEP 3: Auto-detect model type ===
        if detected_type == "auto":
            if output_model is not None:
                detected_type = detect_model_type_from_model(output_model)
            elif output_clip is not None:
                clip_arch = detect_clip_architecture(output_clip)
                if clip_arch['is_qwen']:
                    detected_type = 'zimage'
                else:
                    detected_type = clip_arch['type']
            else:
                detected_type = 'sdxl'  # Default
        
        is_zimage = detected_type == 'zimage'
        
        # === STEP 4: Route through daemon if applicable ===
        if use_daemon and daemon_running:
            # Route VAE through daemon
            if need_vae or (output_vae is not None and DaemonVAE is not None and not isinstance(output_vae, DaemonVAE)):
                vae_type = detect_vae_type(output_vae) if output_vae else detected_type
                try:
                    if output_vae is not None:
                        # Register with daemon
                        proxy_vae = DaemonVAE(source_vae=output_vae, vae_type=vae_type, use_existing=False)
                        status_parts.append(f"VAE: daemon (registered {vae_type})")
                    else:
                        # Use existing from daemon
                        proxy_vae = DaemonVAE(source_vae=None, vae_type=vae_type, use_existing=True)
                        status_parts.append(f"VAE: daemon (shared {vae_type})")
                    output_vae = proxy_vae
                    need_vae = False
                except Exception as e:
                    status_parts.append(f"VAE: daemon failed ({e})")
            
            # Route CLIP through daemon
            if need_clip or (output_clip is not None and DaemonCLIP is not None and not is_daemon_clip(output_clip)):
                try:
                    if is_zimage:
                        # Use Z-IMAGE proxy (Qwen3-VL)
                        if output_clip is not None:
                            proxy_clip = DaemonZImageCLIP(source_clip=output_clip, use_existing=False)
                            status_parts.append("CLIP: daemon (Z-IMAGE/Qwen3)")
                        else:
                            proxy_clip = DaemonZImageCLIP.create_for_daemon()
                            status_parts.append("CLIP: daemon (Z-IMAGE/Qwen3 shared)")
                    else:
                        # Use standard CLIP proxy
                        clip_type = detect_clip_type(output_clip) if output_clip else detected_type
                        if output_clip is not None:
                            proxy_clip = create_clip_proxy(output_clip, use_existing=False)
                            status_parts.append(f"CLIP: daemon (registered {clip_type})")
                        else:
                            proxy_clip = DaemonCLIP(source_clip=None, clip_type=clip_type, use_existing=True)
                            status_parts.append(f"CLIP: daemon (shared {clip_type})")
                    output_clip = proxy_clip
                    need_clip = False
                except Exception as e:
                    status_parts.append(f"CLIP: daemon failed ({e})")
        
        # === STEP 5: Handle model passthrough ===
        if model is not None and output_model is None:
            output_model = model
            status_parts.insert(0, "MODEL: passthrough")
        
        # === STEP 6: Handle remaining CLIP/VAE (local fallback) ===
        if need_clip and output_clip is None:
            status_parts.append("CLIP: not available (connect input or start daemon)")
        if need_vae and output_vae is None:
            status_parts.append("VAE: not available (connect input or start daemon)")
        
        # Build status string
        daemon_status = "daemon ✓" if daemon_running else "local"
        zimage_status = " [Z-IMAGE]" if is_zimage else ""
        status = f"[{detected_type}{zimage_status}] {daemon_status} | " + " | ".join(status_parts)
        
        print(f"[LunaRouter] {status}")
        
        return (output_model, output_clip, output_vae, status)
    
    def _is_output_connected(self, dynprompt, node_id, output_idx) -> bool:
        """Check if output is connected to any downstream node."""
        try:
            for other_id in dynprompt.all_node_ids():
                try:
                    other_node = dynprompt.get_node(other_id)
                except:
                    continue
                
                inputs = other_node.get("inputs", {})
                for input_val in inputs.values():
                    if isinstance(input_val, list) and len(input_val) >= 2:
                        if str(input_val[0]) == str(node_id) and input_val[1] == output_idx:
                            return True
            return False
        except:
            return True  # Assume connected if we can't determine
    
    def _load_checkpoint(
        self,
        ckpt_name: str,
        precision: str,
        local_weights_dir: str,
        load_clip: bool,
        load_vae: bool
    ) -> Tuple[Any, Any, Any, str]:
        """Load checkpoint with optional precision conversion."""
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")
        
        # Detect model type from filename
        model_type = detect_model_type_from_path(ckpt_path)
        
        # Check if we need precision conversion
        precision_key = precision.split()[0] if "auto" not in precision else None
        
        if precision_key and precision_key not in ["auto", "bf16", "fp16"]:
            # Need to convert - use dynamic loader logic
            model, clip, vae = self._load_with_conversion(
                ckpt_path, precision_key, local_weights_dir, load_clip, load_vae
            )
        else:
            # Standard load
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=load_vae,
                output_clip=load_clip,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            model = out[0]
            clip = out[1] if load_clip else None
            vae = out[2] if load_vae else None
        
        return model, clip, vae, model_type
    
    def _load_with_conversion(
        self,
        ckpt_path: str,
        precision_key: str,
        local_weights_dir: str,
        load_clip: bool,
        load_vae: bool
    ) -> Tuple[Any, Any, Any]:
        """Load with precision conversion (from Luna Dynamic Loader)."""
        from safetensors.torch import load_file, save_file
        
        is_gguf = "gguf" in precision_key
        
        # Build optimized path
        if local_weights_dir and os.path.isdir(local_weights_dir):
            weights_root = local_weights_dir
        else:
            weights_root = os.path.join(folder_paths.models_dir, "unet", "optimized")
        os.makedirs(weights_root, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        
        if is_gguf:
            quant_type = precision_key.replace("gguf_", "")
            unet_filename = f"{base_name}_{quant_type}.gguf"
        else:
            unet_filename = f"{base_name}_{precision_key}_unet.safetensors"
        
        unet_path = os.path.join(weights_root, unet_filename)
        
        # Convert if needed
        if not os.path.exists(unet_path):
            print(f"[LunaRouter] Converting UNet to {precision_key}...")
            # Import conversion functions from dynamic loader
            from .luna_dynamic_loader import convert_to_precision, convert_to_gguf
            
            if is_gguf:
                convert_to_gguf(ckpt_path, unet_path, quant_type)
            else:
                convert_to_precision(ckpt_path, unet_path, precision_key, strip_components=True)
        
        # Load optimized UNet
        if is_gguf:
            model = self._load_gguf_unet(unet_path)
        else:
            model = comfy.sd.load_unet(unet_path)
        
        # Load CLIP/VAE from original if needed
        clip = None
        vae = None
        
        if load_clip or load_vae:
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=load_vae,
                output_clip=load_clip,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            clip = out[1] if load_clip else None
            vae = out[2] if load_vae else None
            # Discard the model reference, we loaded optimized
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return model, clip, vae
    
    def _load_unet(self, unet_name: str) -> Tuple[Any, str]:
        """Load UNet from diffusion_models folder."""
        # Try different paths
        unet_path = None
        for folder in ["diffusion_models", "unet", "checkpoints"]:
            try:
                path = folder_paths.get_full_path(folder, unet_name)
                if path and os.path.exists(path):
                    unet_path = path
                    break
            except:
                continue
        
        if not unet_path:
            raise FileNotFoundError(f"UNet not found: {unet_name}")
        
        model_type = detect_model_type_from_path(unet_path)
        
        if unet_path.endswith('.gguf'):
            model = self._load_gguf_unet(unet_path)
        else:
            model = comfy.sd.load_unet(unet_path)
        
        return model, model_type
    
    def _load_gguf_unet(self, path: str) -> Any:
        """Load GGUF UNet."""
        if not HAS_GGUF or UnetLoaderGGUF is None:
            raise ImportError(
                "ComfyUI-GGUF required for .gguf files.\n"
                "Install from: https://github.com/city96/ComfyUI-GGUF"
            )
        
        loader = UnetLoaderGGUF()
        result = loader.load_unet(unet_name=os.path.basename(path))
        return result[0]
    
    def _load_vae(self, vae_name: str) -> Any:
        """Load VAE from file."""
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        # Load VAE using ComfyUI's standard node approach
        try:
            # Try using nodes.VAELoader if available
            from nodes import VAELoader
            loader = VAELoader()
            return loader.load_vae(vae_name)[0]
        except ImportError:
            # Fallback: load checkpoint and extract VAE
            out = comfy.sd.load_checkpoint_guess_config(
                vae_path, output_model=False, output_vae=True, output_clip=False
            )
            return out[2] if len(out) > 2 else None
    
    def _load_clip(self, clip_name: str) -> Any:
        """Load CLIP from file."""
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if not clip_path or not os.path.exists(clip_path):
            raise FileNotFoundError(f"CLIP not found: {clip_name}")
        
        # Load CLIP using ComfyUI's standard node approach
        try:
            # Try using nodes.CLIPLoader if available
            from nodes import CLIPLoader
            loader = CLIPLoader()
            return loader.load_clip(clip_name)[0]
        except ImportError:
            # Fallback: load checkpoint and extract CLIP
            out = comfy.sd.load_checkpoint_guess_config(
                clip_path, output_model=False, output_vae=False, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            return out[1] if len(out) > 1 else None


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaUniversalModelRouter": LunaUniversalModelRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaUniversalModelRouter": "Luna Universal Model Router ⚡",
}
