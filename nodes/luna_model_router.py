"""
Luna Model Router - The Ultimate Unified Model Loader

A single node that handles all model loading scenarios with explicit user control:

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Luna Model Router                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  MODEL SOURCE:     [checkpoint ▼] [diffusion_models ▼] [unet (gguf) ▼]     │
│  MODEL NAME:       [ponyDiffusionV6XL.safetensors ▼]  ← populated by source │
│  MODEL TYPE:       [SD1.5] [SDXL] [SDXL+Vision] [Flux] [Flux+Vision] [SD3] [Z-IMAGE] │
├─────────────────────────────────────────────────────────────────────────────┤
│  DYNAMIC LOADER:   [✓ Enable] → [fp8_e4m3fn ▼] [Q8_0 ▼] [Q4_K_M ▼]         │
│                    Auto-converts and caches optimized UNet                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLIP 1:          [clip_l.safetensors ▼]     ← Required for all types      │
│  CLIP 2:          [clip_g.safetensors ▼]     ← SDXL, SD3                    │
│  CLIP 3:          [t5xxl_fp16.safetensors ▼] ← Flux, SD3                    │
│  CLIP 4:          [siglip_vision.safetensors ▼] ← Vision models             │
│                   Runtime validation based on model_type                    │
│                                                                             │
│  Z-IMAGE Note: clip_1 should be full Qwen3-VL model (.safetensors/.gguf)   │
│                For vision, mmproj auto-loads if in same folder as model    │
├─────────────────────────────────────────────────────────────────────────────┤
│  VAE:             [sdxl_vae.safetensors ▼]                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  DAEMON MODE:     [auto ▼] [force_daemon ▼] [force_local ▼]                 │
└─────────────────────────────────────────────────────────────────────────────┘

OUTPUTS:
  MODEL       → UNet/Diffusion model
  CLIP        → Combined CLIP (all text encoders merged)
  VAE         → VAE for encode/decode
  LLM         → Full LLM for prompt generation (Z-IMAGE: Qwen3-VL decoder)
  CLIP_VISION → Vision encoder for image→embedding (CLIP-H/SigLIP/mmproj)
  model_name  → String for Config Gateway integration
  status      → Detailed loading status

CLIP Requirements by Model Type:
================================
  SD1.5:        clip_1 (CLIP-L)
  SDXL:         clip_1 (CLIP-L) + clip_2 (CLIP-G)
  SDXL+Vision:  clip_1 (CLIP-L) + clip_2 (CLIP-G) + clip_4 (SigLIP/CLIP-H)
  Flux:         clip_1 (CLIP-L) + clip_3 (T5-XXL)
  Flux+Vision:  clip_1 (CLIP-L) + clip_3 (T5-XXL) + clip_4 (SigLIP)
  SD3:          clip_1 (CLIP-L) + clip_2 (CLIP-G) + clip_3 (T5-XXL)
  Z-IMAGE:      clip_1 (Full Qwen3-VL) → Full model required for hidden state extraction

Smart Loading (Z-IMAGE):
========================
  - CLIP output connected: Load full Qwen3 model (required for conditioning)
  - LLM output connected: Same model used for prompt generation
  - CLIP_VISION connected: Also load mmproj from same folder as Qwen3 model
  - keep_model_loaded=False in VLM node: Unloads decoder weights, keeps encoder
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict, List

import torch
from aiohttp import web

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    from server import PromptServer
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None
    PromptServer = None


# =============================================================================
# WEB API ENDPOINTS (for dynamic model list filtering)
# =============================================================================

# Route handlers - registered lazily when PromptServer is available
async def _get_model_list_by_source(request):
    """Return list of models for a specific source folder."""
    source = request.match_info.get("source", "checkpoints")
    
    valid_sources = ["checkpoints", "diffusion_models", "unet"]
    if source not in valid_sources:
        return web.json_response({"error": f"Invalid source: {source}"}, status=400)
    
    try:
        models = folder_paths.get_filename_list(source)
        return web.json_response({"source": source, "models": models})
    except Exception as e:
        return web.json_response({"error": str(e), "models": []})

async def _get_clip_requirements(request):
    """Return CLIP requirements for a model type."""
    model_type = request.match_info.get("model_type", "SDXL")
    
    if model_type in CLIP_REQUIREMENTS:
        return web.json_response({
            "model_type": model_type,
            "requirements": CLIP_REQUIREMENTS[model_type]
        })
    else:
        return web.json_response({"error": f"Unknown model type: {model_type}"}, status=400)


def register_routes():
    """Register web routes - called after PromptServer is initialized."""
    if HAS_COMFY and PromptServer is not None:
        try:
            server = PromptServer.instance
            if server is not None:
                server.routes.get("/luna/model_list/{source}")(_get_model_list_by_source)
                server.routes.get("/luna/clip_requirements/{model_type}")(_get_clip_requirements)
                print("[LunaModelRouter] Web routes registered")
        except Exception as e:
            print(f"[LunaModelRouter] Failed to register routes: {e}")


# Try to register routes now (may work if loaded after server init)
# Also exported for __init__.py to call after full init
try:
    register_routes()
except:
    pass  # Will be called later by __init__.py

# GGUF support
try:
    GGUF_NODE_PATHS = [
        "custom_nodes.ComfyUI-GGUF.nodes",
        "ComfyUI-GGUF.nodes",
    ]
    UnetLoaderGGUF = None
    CLIPLoaderGGUF = None
    gguf_sd_loader = None
    GGMLOps = None
    GGUFModelPatcher = None
    
    for path in GGUF_NODE_PATHS:
        try:
            module = __import__(path, fromlist=['UnetLoaderGGUF', 'CLIPLoaderGGUF', 'GGMLOps', 'GGUFModelPatcher'])
            UnetLoaderGGUF = getattr(module, 'UnetLoaderGGUF', None)
            CLIPLoaderGGUF = getattr(module, 'CLIPLoaderGGUF', None)
            GGMLOps = getattr(module, 'GGMLOps', None)
            GGUFModelPatcher = getattr(module, 'GGUFModelPatcher', None)
            if UnetLoaderGGUF:
                # Also get the loader function
                loader_path = path.replace('.nodes', '.loader')
                try:
                    loader_module = __import__(loader_path, fromlist=['gguf_sd_loader'])
                    gguf_sd_loader = getattr(loader_module, 'gguf_sd_loader', None)
                except:
                    pass
                break
        except:
            continue
    HAS_GGUF = UnetLoaderGGUF is not None and gguf_sd_loader is not None
except:
    HAS_GGUF = False
    UnetLoaderGGUF = None
    CLIPLoaderGGUF = None
    gguf_sd_loader = None
    GGMLOps = None
    GGUFModelPatcher = None

# Daemon support
try:
    from ..luna_daemon.proxy import DaemonVAE, DaemonCLIP, detect_vae_type
    from ..luna_daemon.zimage_proxy import DaemonZImageCLIP, detect_clip_architecture
    from ..luna_daemon import client as daemon_client
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonVAE = None
    DaemonCLIP = None
    DaemonZImageCLIP = None
    daemon_client = None
    def detect_vae_type(vae): return 'unknown'
    def detect_clip_architecture(clip): return {'type': 'unknown', 'is_qwen': False}


# =============================================================================
# CLIP Type Mapping (Model Type → ComfyUI CLIPType)
# =============================================================================

# Maps Luna model types to the clip_type string for ComfyUI's load_clip()
# These correspond to comfy.sd.CLIPType enum values
CLIP_TYPE_MAP = {
    "SD1.5": "stable_diffusion",
    "SDXL": "stable_diffusion",      # SDXL uses same CLIPType as SD1.5
    "SDXL + Vision": "stable_diffusion",
    "Flux": "flux",
    "Flux + Vision": "flux",
    "SD3": "sd3",
    "Z-IMAGE": "lumina2",            # Qwen3-VL uses Lumina2 CLIP type
}


# =============================================================================
# CLIP Requirements by Model Type
# =============================================================================

CLIP_REQUIREMENTS = {
    # model_type: (required_clips, optional_clips, descriptions)
    "SD1.5": {
        "required": ["clip_1"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L (e.g., clip_l.safetensors)",
        }
    },
    "SDXL": {
        "required": ["clip_1", "clip_2"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L (e.g., clip_l.safetensors)",
            "clip_2": "CLIP-G (e.g., clip_g.safetensors)",
        }
    },
    "SDXL + Vision": {
        "required": ["clip_1", "clip_2", "clip_4"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_2": "CLIP-G", 
            "clip_4": "Vision encoder (SigLIP or CLIP-H)",
        }
    },
    "Flux": {
        "required": ["clip_1", "clip_3"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_3": "T5-XXL",
        }
    },
    "Flux + Vision": {
        "required": ["clip_1", "clip_3", "clip_4"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_3": "T5-XXL",
            "clip_4": "Vision encoder (SigLIP)",
        }
    },
    "SD3": {
        "required": ["clip_1", "clip_2", "clip_3"],
        "optional": [],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_2": "CLIP-G",
            "clip_3": "T5-XXL",
        }
    },
    "Z-IMAGE": {
        "required": ["clip_1"],
        "optional": [],
        "descriptions": {
            "clip_1": "Full Qwen3-VL model (.safetensors/.gguf) - entire model needed for hidden state extraction",
        },
        "notes": "For vision features, mmproj loads automatically if in same folder as Qwen3 model."
    },
}


# =============================================================================
# Luna Model Router
# =============================================================================

class LunaModelRouter:
    """
    The Ultimate Unified Model Loader.
    
    Provides explicit control over model source, type, and precision with
    runtime validation of CLIP configuration.
    
    New in this version:
    - LLM output: Full language model for prompt generation (Z-IMAGE: Qwen3-VL)
    - CLIP_VISION output: Vision encoder for image→embedding conversion
    - Smart loading: Vision components only load if CLIP_VISION is connected
    - mmproj auto-detection: Loads from same folder as Qwen3 model
    """
    
    CATEGORY = "Luna/Core"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LLM", "CLIP_VISION", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "llm", "clip_vision", "model_name", "status")
    FUNCTION = "load"
    OUTPUT_NODE = False
    
    # Source folders
    MODEL_SOURCES = ["checkpoints", "diffusion_models", "unet"]
    
    # Model architectures
    MODEL_TYPES = ["SD1.5", "SDXL", "SDXL + Vision", "Flux", "Flux + Vision", "SD3", "Z-IMAGE"]
    
    # Precision options for dynamic loading
    # None: Use source precision as-is
    # bf16: Recommended default - fp32 range, native on Ampere+, stable
    # fp16: Legacy - slightly more precision but limited range
    # fp8: 75% VRAM reduction, native on Ada/Blackwell
    # GGUF: Integer quantization using GPU tensor cores
    PRECISION_OPTIONS = ["None", "bf16", "fp16", "fp8_e4m3fn", "gguf_Q8_0", "gguf_Q4_K_M"]
    
    # Daemon routing modes
    DAEMON_MODES = ["auto", "force_daemon", "force_local"]
    
    @classmethod
    def INPUT_TYPES(cls):
        if not HAS_COMFY:
            return {"required": {}}
        
        # Build file lists for each source
        # Note: The actual list will be dynamically updated by JS based on model_source
        checkpoint_list = ["None"] + folder_paths.get_filename_list("checkpoints")
        
        # Try to get diffusion_models and unet lists
        try:
            diffusion_list = folder_paths.get_filename_list("diffusion_models")
        except:
            diffusion_list = []
        
        try:
            unet_list = folder_paths.get_filename_list("unet")
        except:
            unet_list = []
        
        # Combine all model files for the dropdown (JS will filter)
        all_models = ["None"] + checkpoint_list[1:] + diffusion_list + unet_list
        
        # CLIP list (safetensors + gguf)
        try:
            clip_list = ["None"] + folder_paths.get_filename_list("clip")
        except:
            clip_list = ["None"]
        
        # VAE list
        vae_list = ["None"] + folder_paths.get_filename_list("vae")
        
        return {
            "required": {
                # === Model Selection ===
                "model_source": (cls.MODEL_SOURCES, {
                    "default": "checkpoints",
                    "tooltip": "Folder to load model from. Changes which models appear in the dropdown."
                }),
                "model_name": (all_models, {
                    "default": "None",
                    "tooltip": "Model file to load (filtered by model_source)"
                }),
                "model_type": (cls.MODEL_TYPES, {
                    "default": "SDXL",
                    "tooltip": "Model architecture - determines CLIP requirements"
                }),
                
                # === Dynamic Loader ===
                "dynamic_precision": (cls.PRECISION_OPTIONS, {
                    "default": "None",
                    "tooltip": "Enable to auto-convert UNet to optimized precision. 'None' = use source precision."
                }),
                
                # === CLIP Selection (4 slots) ===
                "clip_1": (clip_list, {
                    "default": "None",
                    "tooltip": "Primary CLIP encoder (CLIP-L for most, Qwen3 for Z-IMAGE)"
                }),
                "clip_2": (clip_list, {
                    "default": "None",
                    "tooltip": "Secondary CLIP encoder (CLIP-G for SDXL/SD3)"
                }),
                "clip_3": (clip_list, {
                    "default": "None",
                    "tooltip": "Tertiary CLIP encoder (T5-XXL for Flux/SD3)"
                }),
                "clip_4": (clip_list, {
                    "default": "None",
                    "tooltip": "Vision encoder (SigLIP/CLIP-H for vision models)"
                }),
                
                # === VAE Selection ===
                "vae_name": (vae_list, {
                    "default": "None",
                    "tooltip": "VAE for encoding/decoding. 'None' uses VAE from checkpoint."
                }),
                
                # === Daemon Mode ===
                "daemon_mode": (cls.DAEMON_MODES, {
                    "default": "auto",
                    "tooltip": "auto: use daemon if running | force_daemon: require daemon | force_local: never use daemon"
                }),
            },
            "optional": {
                "local_weights_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory for converted UNet cache (default: models/unet/optimized)"
                }),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
    
    def load(
        self,
        model_source: str,
        model_name: str,
        model_type: str,
        dynamic_precision: str,
        clip_1: str,
        clip_2: str,
        clip_3: str,
        clip_4: str,
        vae_name: str,
        daemon_mode: str,
        local_weights_dir: str = "",
        dynprompt=None,
        unique_id=None
    ) -> Tuple[Any, Any, Any, Any, Any, str, str]:
        """
        Load model with explicit configuration and runtime CLIP validation.
        
        Returns:
            model: UNet/Diffusion model
            clip: Combined CLIP for text conditioning
            vae: VAE for encode/decode
            llm: Full LLM for prompt generation (Z-IMAGE only, None for others)
            clip_vision: Vision encoder for image→embedding (if vision model type)
            model_name: String for Config Gateway
            status: Detailed loading status
        """
        status_parts = []
        
        # === STEP 1: Validate CLIP configuration ===
        clip_config = {
            "clip_1": clip_1 if clip_1 != "None" else None,
            "clip_2": clip_2 if clip_2 != "None" else None,
            "clip_3": clip_3 if clip_3 != "None" else None,
            "clip_4": clip_4 if clip_4 != "None" else None,
        }
        
        self._validate_clip_config(model_type, clip_config)
        
        # === STEP 2: Check daemon availability ===
        use_daemon = daemon_mode != "force_local"
        require_daemon = daemon_mode == "force_daemon"
        daemon_running = DAEMON_AVAILABLE and daemon_client is not None and daemon_client.is_daemon_running()
        
        if require_daemon and not daemon_running:
            raise RuntimeError(
                "Daemon mode is 'force_daemon' but Luna Daemon is not running!\n"
                "Start the daemon or change mode to 'auto' or 'force_local'."
            )
        
        # === STEP 3: Load MODEL ===
        output_model = None
        output_model_name = ""
        
        if model_name and model_name != "None":
            output_model, output_model_name = self._load_model(
                model_source, model_name, dynamic_precision, local_weights_dir
            )
            precision_str = f" → {dynamic_precision}" if dynamic_precision != "None" else ""
            status_parts.append(f"MODEL: {model_source}/{os.path.basename(model_name)}{precision_str}")
        else:
            status_parts.append("MODEL: None (no model selected)")
        
        # === STEP 4: Load CLIP and LLM ===
        output_clip = None
        output_llm = None
        
        if model_type == "Z-IMAGE":
            # Z-IMAGE: Full Qwen3 model for CLIP (hidden state extraction)
            # The same model can also be used for LLM output
            output_clip, output_llm = self._load_zimage_clip_and_llm(
                clip_config, daemon_running, use_daemon
            )
            status_parts.append("CLIP: Z-IMAGE (Qwen3 full model)")
            if output_llm is not None:
                status_parts.append("LLM: Qwen3-VL ready")
        else:
            # Standard CLIP loading
            output_clip = self._load_standard_clip(model_type, clip_config, daemon_running, use_daemon)
            clip_count = len([c for c in clip_config.values() if c is not None])
            status_parts.append(f"CLIP: {clip_count} encoder(s)")
        
        # === STEP 5: Load CLIP_VISION (for vision model types) ===
        output_clip_vision = None
        is_vision_type = model_type in ["SDXL + Vision", "Flux + Vision", "Z-IMAGE"]
        
        if is_vision_type:
            output_clip_vision = self._load_clip_vision(
                model_type, clip_config, daemon_running, use_daemon
            )
            if output_clip_vision is not None:
                if model_type == "Z-IMAGE":
                    status_parts.append("VISION: mmproj loaded")
                else:
                    status_parts.append("VISION: CLIP-H/SigLIP loaded")
        
        # === STEP 6: Load VAE ===
        output_vae = None
        
        if vae_name and vae_name != "None":
            output_vae = self._load_vae(vae_name, daemon_running, use_daemon)
            status_parts.append(f"VAE: {os.path.basename(vae_name)}")
        else:
            # Try to get VAE from checkpoint if it was a full checkpoint
            if model_source == "checkpoints" and output_model is not None:
                # VAE might have been loaded with checkpoint
                status_parts.append("VAE: from checkpoint")
            else:
                status_parts.append("VAE: None")
        
        # === Build status ===
        daemon_status = "daemon ✓" if daemon_running and use_daemon else "local"
        status = f"[{model_type}] {daemon_status} | " + " | ".join(status_parts)
        
        print(f"[LunaModelRouter] {status}")
        
        return (output_model, output_clip, output_vae, output_llm, output_clip_vision, output_model_name, status)
    
    def _validate_clip_config(self, model_type: str, clip_config: Dict[str, Optional[str]]) -> None:
        """
        Validate CLIP configuration for the selected model type.
        Raises RuntimeError if required CLIPs are missing.
        """
        if model_type not in CLIP_REQUIREMENTS:
            raise RuntimeError(f"Unknown model type: {model_type}")
        
        requirements = CLIP_REQUIREMENTS[model_type]
        missing = []
        
        for clip_slot in requirements["required"]:
            if clip_config.get(clip_slot) is None:
                desc = requirements["descriptions"].get(clip_slot, clip_slot)
                missing.append(f"{clip_slot}: {desc}")
        
        if missing:
            raise RuntimeError(
                f"Missing required CLIP encoders for {model_type}:\n" +
                "\n".join(f"  • {m}" for m in missing) +
                f"\n\nRequired: {', '.join(requirements['required'])}"
            )
    
    def _load_model(
        self,
        source: str,
        name: str,
        precision: str,
        local_weights_dir: str
    ) -> Tuple[Any, str]:
        """Load model from specified source with optional precision conversion."""
        
        # Get full path
        model_path = folder_paths.get_full_path(source, name)
        if not model_path:
            # Try alternate sources
            for alt_source in ["checkpoints", "diffusion_models", "unet"]:
                model_path = folder_paths.get_full_path(alt_source, name)
                if model_path and os.path.exists(model_path):
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {name} in {source}")
        
        # Extract model name for output
        output_name = os.path.splitext(os.path.basename(name))[0]
        
        # Check if dynamic precision conversion is needed
        if precision != "None":
            model = self._load_with_conversion(model_path, precision, local_weights_dir)
            return model, output_name
        
        # Load based on file type
        if model_path.endswith('.gguf'):
            model = self._load_gguf_model(model_path)
        elif source == "checkpoints":
            # Full checkpoint - extract just the model
            out = comfy.sd.load_checkpoint_guess_config(
                model_path,
                output_vae=False,
                output_clip=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            model = out[0]
        else:
            # UNet/diffusion model only
            model = comfy.sd.load_unet(model_path)
        
        return model, output_name
    
    def _load_with_conversion(
        self,
        model_path: str,
        precision: str,
        local_weights_dir: str
    ) -> Any:
        """Load model with precision conversion (cached).
        
        Output locations:
        - GGUF: models/unet/unet-converted/<relative_subfolders>/filename_Q4_K_M.gguf
          (preserves folder hierarchy from checkpoints, unet-converted is typically symlinked to NVMe)
        - bf16/fp8: models/checkpoints/checkpoints_converted/filename_fp8.safetensors
          (flat structure, typically on fast NVMe via symlink)
        """
        from safetensors.torch import load_file, save_file
        
        is_gguf = "gguf" in precision
        
        # Get the base filename
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Determine relative subfolder path from checkpoints root
        # e.g., "Illustrious/Realistic" from ".../checkpoints/Illustrious/Realistic/model.safetensors"
        checkpoints_root = folder_paths.get_folder_paths("checkpoints")[0] if folder_paths.get_folder_paths("checkpoints") else None
        relative_subpath = ""
        
        if checkpoints_root:
            # Normalize paths for comparison
            norm_model_path = os.path.normpath(model_path)
            norm_checkpoints = os.path.normpath(checkpoints_root)
            
            if norm_model_path.startswith(norm_checkpoints):
                # Extract relative path (e.g., "Illustrious/Realistic/model.safetensors")
                rel_path = os.path.relpath(norm_model_path, norm_checkpoints)
                # Get just the folder part (e.g., "Illustrious/Realistic")
                relative_subpath = os.path.dirname(rel_path)
        
        # Build output path based on conversion type
        if is_gguf:
            # GGUF → models/unet/unet-converted/<relative_subpath>/filename_Q4_K_M.gguf
            quant_type = precision.replace("gguf_", "")
            cache_filename = f"{base_name}_{quant_type}.gguf"
            
            unet_root = folder_paths.get_folder_paths("unet")[0] if folder_paths.get_folder_paths("unet") else os.path.join(folder_paths.models_dir, "unet")
            
            # Always put in unet-converted subfolder, then mirror the checkpoint hierarchy
            if relative_subpath:
                cache_dir = os.path.join(unet_root, "unet-converted", relative_subpath)
            else:
                cache_dir = os.path.join(unet_root, "unet-converted")
        else:
            # bf16/fp8 → models/checkpoints/checkpoints_converted/<relative_subpath>/filename_fp8.safetensors
            cache_filename = f"{base_name}_{precision}_unet.safetensors"
            
            if checkpoints_root:
                if relative_subpath:
                    cache_dir = os.path.join(checkpoints_root, "checkpoints_converted", relative_subpath)
                else:
                    cache_dir = os.path.join(checkpoints_root, "checkpoints_converted")
            else:
                cache_dir = os.path.join(folder_paths.models_dir, "checkpoints", "checkpoints_converted")
        
        # Override with explicit local_weights_dir if provided
        if local_weights_dir and os.path.isdir(local_weights_dir):
            cache_dir = local_weights_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # Convert if not cached
        if not os.path.exists(cache_path):
            print(f"[LunaModelRouter] Converting to {precision}...")
            print(f"[LunaModelRouter] Output: {cache_path}")
            
            try:
                from .luna_dynamic_loader import convert_to_precision, convert_to_gguf
                
                if is_gguf:
                    convert_to_gguf(model_path, cache_path, quant_type)
                else:
                    convert_to_precision(model_path, cache_path, precision, strip_components=True)
                    
                print(f"[LunaModelRouter] Conversion complete!")
            except ImportError:
                raise RuntimeError(
                    "Dynamic precision conversion requires luna_dynamic_loader module.\n"
                    "Set precision to 'None' to load without conversion."
                )
        else:
            print(f"[LunaModelRouter] Using cached: {cache_path}")
        
        # Load converted model
        if is_gguf:
            return self._load_gguf_model(cache_path)
        else:
            return comfy.sd.load_unet(cache_path)
    
    def _load_gguf_model(self, path: str) -> Any:
        """Load GGUF model file directly from path.
        
        Uses the underlying gguf_sd_loader directly to support custom paths,
        rather than going through folder_paths which only searches standard dirs.
        """
        if not HAS_GGUF or gguf_sd_loader is None:
            raise ImportError(
                "ComfyUI-GGUF required for .gguf files.\n"
                "Install from: https://github.com/city96/ComfyUI-GGUF"
            )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"GGUF model not found: {path}")
        
        print(f"[LunaModelRouter] Loading GGUF from: {path}")
        
        # Create ops with default settings
        ops = GGMLOps()
        
        # Load state dict directly from path
        sd = gguf_sd_loader(path)
        
        # Create model with GGML ops
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        
        if model is None:
            raise RuntimeError(f"Could not detect model type of: {path}")
        
        # Wrap with GGUF patcher
        if GGUFModelPatcher is not None:
            model = GGUFModelPatcher.clone(model)
        
        return model
    
    def _load_standard_clip(
        self,
        model_type: str,
        clip_config: Dict[str, Optional[str]],
        daemon_running: bool,
        use_daemon: bool
    ) -> Any:
        """Load and combine standard CLIP encoders."""
        
        # Collect CLIP paths
        clip_paths = []
        clip_types = []
        
        # Map slot to encoder type
        slot_to_type = {
            "clip_1": "clip_l",
            "clip_2": "clip_g",
            "clip_3": "t5xxl",
            "clip_4": "vision",
        }
        
        for slot, path in clip_config.items():
            if path is not None:
                full_path = folder_paths.get_full_path("clip", path)
                if full_path and os.path.exists(full_path):
                    clip_paths.append(full_path)
                    clip_types.append(slot_to_type.get(slot, "clip_l"))
        
        if not clip_paths:
            return None
        
        # Get clip_type string for daemon/ComfyUI
        clip_type_str = CLIP_TYPE_MAP.get(model_type, "stable_diffusion")
        
        # Route through daemon if available (path-based for efficiency)
        if daemon_running and use_daemon and DaemonCLIP is not None and daemon_client is not None:
            try:
                # Register CLIP by path (daemon loads from disk)
                result = daemon_client.register_clip_by_path(clip_paths, model_type, clip_type_str)
                
                if result.get("success"):
                    print(f"[LunaModelRouter] Registered CLIP with daemon: {model_type} -> {clip_type_str}")
                    # Return DaemonCLIP proxy (no source_clip needed - daemon loads from path)
                    daemon_clip_type = {
                        "SD1.5": "sd15",
                        "SDXL": "sdxl",
                        "SDXL + Vision": "sdxl",
                        "Flux": "flux",
                        "Flux + Vision": "flux",
                        "SD3": "sd3",
                    }.get(model_type, "sdxl")
                    return DaemonCLIP(source_clip=None, clip_type=daemon_clip_type, use_existing=True)
                else:
                    print(f"[LunaModelRouter] Daemon CLIP registration failed: {result.get('error')}")
            except Exception as e:
                print(f"[LunaModelRouter] Daemon CLIP failed, using local: {e}")
        
        # Load locally as fallback
        clip_type_map = {
            "SD1.5": comfy.sd.CLIPType.STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,
            "SDXL": comfy.sd.CLIPType.STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,
            "SDXL + Vision": comfy.sd.CLIPType.STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,
            "Flux": comfy.sd.CLIPType.FLUX if hasattr(comfy.sd, 'CLIPType') else None,
            "Flux + Vision": comfy.sd.CLIPType.FLUX if hasattr(comfy.sd, 'CLIPType') else None,
            "SD3": comfy.sd.CLIPType.SD3 if hasattr(comfy.sd, 'CLIPType') else None,
        }
        
        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_map.get(model_type)
            )
        except Exception as e:
            print(f"[LunaModelRouter] Multi-CLIP load failed, trying single: {e}")
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_paths[0]],
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
        
        return clip
    
    def _load_zimage_clip_and_llm(
        self,
        clip_config: Dict[str, Optional[str]],
        daemon_running: bool,
        use_daemon: bool
    ) -> Tuple[Any, Any]:
        """
        Load Z-IMAGE CLIP and LLM from full Qwen3 model.
        
        Z-IMAGE uses the full Qwen3 model for CLIP conditioning:
        - Prompt is tokenized, encoded, and run through full transformer
        - Hidden states are extracted before final token projection
        - These hidden states become the conditioning signal for the UNet
        
        The same model can also be used for LLM text generation.
        
        Returns:
            (clip, llm) - Both reference the same underlying model
        """
        clip_1_path = clip_config.get("clip_1")
        
        if not clip_1_path:
            raise RuntimeError(
                "Z-IMAGE requires clip_1 to be set to the full Qwen3 model.\n"
                "Use a Qwen3-VL .safetensors or .gguf file."
            )
        
        # Get full path to Qwen3 model
        full_path = folder_paths.get_full_path("clip", clip_1_path)
        if not full_path or not os.path.exists(full_path):
            raise FileNotFoundError(f"Qwen3 model not found: {clip_1_path}")
        
        # For Z-IMAGE, we prefer daemon's Qwen3-VL if available
        if daemon_running and use_daemon and DaemonZImageCLIP is not None:
            try:
                # Load the model and create daemon proxy
                local_clip = comfy.sd.load_clip(
                    ckpt_paths=[full_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
                daemon_clip = DaemonZImageCLIP(source_clip=local_clip, use_existing=False)
                
                # For LLM, we create a reference to the same model
                # The VLM Prompt Generator node will handle the generation
                llm = self._create_llm_reference(full_path, daemon_running)
                
                return daemon_clip, llm
            except Exception as e:
                print(f"[LunaModelRouter] Daemon Qwen3 setup failed: {e}")
                # Fall through to local loading
        
        # Local-only Z-IMAGE loading
        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[full_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            llm = self._create_llm_reference(full_path, daemon_running=False)
            return clip, llm
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3 model: {e}")
    
    def _create_llm_reference(self, model_path: str, daemon_running: bool) -> Dict[str, Any]:
        """
        Create an LLM reference object that can be passed to VLM nodes.
        
        This doesn't load the full LLM weights immediately - that happens
        when the VLM Prompt Generator node executes.
        """
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        
        # Check for mmproj in same directory
        mmproj_path = None
        for filename in os.listdir(model_dir):
            if "mmproj" in filename.lower() and (filename.endswith('.gguf') or filename.endswith('.safetensors')):
                mmproj_path = os.path.join(model_dir, filename)
                print(f"[LunaModelRouter] Found mmproj: {filename}")
                break
        
        return {
            "type": "qwen3_vl",
            "model_path": model_path,
            "model_name": model_name,
            "mmproj_path": mmproj_path,
            "use_daemon": daemon_running,
            "loaded": False,  # Will be loaded by VLM node on demand
        }
    
    def _load_clip_vision(
        self,
        model_type: str,
        clip_config: Dict[str, Optional[str]],
        daemon_running: bool,
        use_daemon: bool
    ) -> Any:
        """
        Load vision encoder for image→embedding conversion.
        
        For SDXL+Vision/Flux+Vision: Load CLIP-H or SigLIP from clip_4
        For Z-IMAGE: Load mmproj from same folder as Qwen3 model
        """
        if model_type == "Z-IMAGE":
            # For Z-IMAGE, mmproj should be in same folder as clip_1
            clip_1_path = clip_config.get("clip_1")
            if not clip_1_path:
                return None
            
            full_path = folder_paths.get_full_path("clip", clip_1_path)
            if not full_path:
                return None
            
            model_dir = os.path.dirname(full_path)
            
            # Look for mmproj file
            mmproj_path = None
            for filename in os.listdir(model_dir):
                if "mmproj" in filename.lower():
                    mmproj_path = os.path.join(model_dir, filename)
                    break
            
            if not mmproj_path:
                print(f"[LunaModelRouter] No mmproj found in {model_dir}")
                print("[LunaModelRouter] Vision features will not be available")
                return None
            
            # Return mmproj reference (loaded by LunaVisionNode)
            return {
                "type": "qwen3_mmproj",
                "mmproj_path": mmproj_path,
                "model_path": full_path,  # Reference to main model
                "use_daemon": daemon_running and use_daemon,
            }
        
        else:
            # Standard vision encoder (CLIP-H/SigLIP) from clip_4
            clip_4_path = clip_config.get("clip_4")
            if not clip_4_path:
                return None
            
            full_path = folder_paths.get_full_path("clip", clip_4_path)
            if not full_path or not os.path.exists(full_path):
                print(f"[LunaModelRouter] Vision encoder not found: {clip_4_path}")
                return None
            
            try:
                # Load vision encoder
                # ComfyUI uses CLIPVision for this
                from nodes import CLIPVisionLoader
                loader = CLIPVisionLoader()
                vision_model = loader.load_clip(clip_4_path)[0]
                return vision_model
            except Exception as e:
                print(f"[LunaModelRouter] Failed to load vision encoder: {e}")
                return None
    
    def _load_vae(
        self,
        vae_name: str,
        daemon_running: bool,
        use_daemon: bool
    ) -> Any:
        """Load VAE with optional daemon routing."""
        
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        # Route through daemon if available (path-based for efficiency)
        if daemon_running and use_daemon and DaemonVAE is not None and daemon_client is not None:
            try:
                # Register VAE by path (daemon loads from disk)
                vae_type = self._detect_vae_type_from_path(vae_path)
                result = daemon_client.register_vae_by_path(vae_path, vae_type)
                
                if result.get("success"):
                    print(f"[LunaModelRouter] Registered VAE with daemon: {vae_type}")
                    # Return DaemonVAE proxy (no source_vae needed - daemon loads from path)
                    return DaemonVAE(source_vae=None, vae_type=vae_type, use_existing=True)
                else:
                    print(f"[LunaModelRouter] Daemon VAE registration failed: {result.get('error')}")
            except Exception as e:
                print(f"[LunaModelRouter] Daemon VAE failed, using local: {e}")
        
        # Load VAE locally as fallback
        try:
            from nodes import VAELoader
            loader = VAELoader()
            vae = loader.load_vae(vae_name)[0]
        except ImportError:
            # Fallback
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
        
        return vae
    
    def _detect_vae_type_from_path(self, vae_path: str) -> str:
        """Detect VAE type from path/filename."""
        basename = os.path.basename(vae_path).lower()
        
        if "flux" in basename:
            return "flux"
        elif "sd3" in basename:
            return "sd3"
        elif "sdxl" in basename or "xl" in basename:
            return "sdxl"
        elif "sd15" in basename or "sd1.5" in basename:
            return "sd15"
        else:
            # Default to SDXL (most common)
            return "sdxl"


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaModelRouter": LunaModelRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaModelRouter": "Luna Model Router ⚡",
}
