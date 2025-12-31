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
│  Z-IMAGE Note: clip_1 should be full Qwen3-VL model (.safetensors)         │
│                GGUF support pending llama-cpp-python stable release         │
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
  Flux:         clip_1 (CLIP-L) + clip_2 (CLIP-G) + clip_3 (T5-XXL)
  Flux+Vision:  clip_1 (CLIP-L) + clip_2 (CLIP-G) + clip_3 (T5-XXL) + clip_4 (SigLIP)
  SD3:          clip_1 (CLIP-L) + clip_2 (CLIP-G) + clip_3 (T5-XXL)
  Z-IMAGE:      clip_1 (Qwen2-VL) → Uses full Qwen model as text encoder (no CLIP/T5)

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
from typing import Tuple, Optional, Any, Dict, List

from aiohttp import web

# Import centralized path constants from Luna Collection
# NOTE: sys.path is configured centrally in __init__.py, so imports work
try:
    # Direct import - Luna __init__.py sets up sys.path
    from __init__ import COMFY_PATH, LUNA_PATH
except (ImportError, ModuleNotFoundError, AttributeError):
    # Fallback: construct paths if Luna constants aren't available
    COMFY_PATH = None
    LUNA_PATH = None

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
    gguf_clip_loader = None
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
                # Also get the loader functions
                loader_path = path.replace('.nodes', '.loader')
                try:
                    loader_module = __import__(loader_path, fromlist=['gguf_sd_loader', 'gguf_clip_loader'])
                    gguf_sd_loader = getattr(loader_module, 'gguf_sd_loader', None)
                    gguf_clip_loader = getattr(loader_module, 'gguf_clip_loader', None)
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
    gguf_clip_loader = None
    GGMLOps = None
    GGUFModelPatcher = None

# Daemon support
DAEMON_AVAILABLE = False
DaemonVAE = None
DaemonCLIP = None
DaemonZImageCLIP = None
daemon_client = None
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283

def detect_vae_type(vae): 
    return 'unknown'
    
def detect_clip_architecture(clip): 
    return {'type': 'unknown', 'is_qwen': False}

# Try to import daemon modules with fallback
try:
    print("[Luna.ModelRouter] Attempting to import daemon modules...")
    import sys
    from pathlib import Path
    
    # NOTE: sys.path is configured centrally in __init__.py
    # PACKAGE_ROOT is already available via sys.path setup at import time
    # This import block can directly attempt imports without path manipulation
    
    # Try absolute imports
    from luna_daemon.proxy import DaemonVAE as _DaemonVAE, DaemonCLIP as _DaemonCLIP, detect_vae_type as _detect_vae_type
    print("[Luna.ModelRouter] [OK] Imported proxy modules")
    from luna_daemon.zimage_proxy import DaemonZImageCLIP as _DaemonZImageCLIP, detect_clip_architecture as _detect_clip_architecture
    print("[Luna.ModelRouter] [OK] Imported zimage_proxy")
    from luna_daemon import client as _daemon_client
    print("[Luna.ModelRouter] [OK] Imported daemon client")
    from luna_daemon.config import DAEMON_HOST as _DAEMON_HOST, DAEMON_PORT as _DAEMON_PORT
    print("[Luna.ModelRouter] [OK] Imported daemon config")
    
    # Assign to module variables
    DaemonVAE = _DaemonVAE
    DaemonCLIP = _DaemonCLIP
    DaemonZImageCLIP = _DaemonZImageCLIP
    daemon_client = _daemon_client
    DAEMON_HOST = _DAEMON_HOST
    DAEMON_PORT = _DAEMON_PORT
    detect_vae_type = _detect_vae_type  # type: ignore
    detect_clip_architecture = _detect_clip_architecture
    DAEMON_AVAILABLE = True
    print("[Luna.ModelRouter] [OK] Daemon imports successful!")
except Exception as e:
    print(f"[Luna.ModelRouter] ✗ Daemon import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


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
    "Z-IMAGE": "lumina2",            # Workflow uses 'lumina2' type in CLIPLoader for Qwen model
}


# =============================================================================
# Helper Functions
# =============================================================================

def _get_clip_model_list() -> List[str]:
    """
    Get CLIP models from multiple search locations:
    1. Standard clip folder (models/clip)
    2. models/clip_gguf (GGUF-format CLIP models)
    3. models/clip_vision (Vision-specific CLIPs)
    4. models/LLM (LLM models that can be used as CLIP for Z-IMAGE)
    """
    clip_models = set()
    
    # Standard clip folder
    try:
        clip_models.update(folder_paths.get_filename_list("clip"))
    except Exception as e:
        print(f"[Luna.ModelRouter] Warning: Could not get standard clip folder: {e}")
    
    # Find models directory - try multiple methods
    models_dir = None
    
    # Method 1: Direct attribute
    if hasattr(folder_paths, 'models_dir'):
        models_dir = folder_paths.models_dir
    
    # Method 2: From checkpoints folder
    if not models_dir:
        try:
            checkpoints_dir = folder_paths.get_full_path("checkpoints", "")
            if checkpoints_dir:
                models_dir = os.path.dirname(checkpoints_dir)
        except:
            pass
    
    # Method 3: Scan common paths using centralized COMFY_PATH
    if not models_dir:
        potential_paths = []
        
        # Try COMFY_PATH if available (from centralized __init__.py)
        if COMFY_PATH:
            potential_paths.append(os.path.join(COMFY_PATH, "models"))
        
        # Fallback options
        potential_paths.extend([
            os.path.join(os.getcwd(), "models"),
            os.path.join(COMFY_PATH, "models") if COMFY_PATH else None,
        ])
        potential_paths = [p for p in potential_paths if p]  # Remove None entries
        
        for potential_path in potential_paths:
            if os.path.isdir(potential_path):
                models_dir = potential_path
                break
    
    # If we found the models directory, search for additional CLIP folders
    if models_dir and os.path.exists(models_dir):
        print(f"[Luna.ModelRouter] Scanning models directory: {models_dir}")
        custom_clip_folders = ["clip_gguf", "clip_vision", "LLM"]
        
        for folder in custom_clip_folders:
            folder_path = os.path.join(models_dir, folder)
            if os.path.isdir(folder_path):
                try:
                    files_found = 0
                    # Recursively walk the folder to find all model files
                    for root, dirs, files in os.walk(folder_path):
                        for filename in files:
                            # Check for model file extensions
                            if filename.lower().endswith(('.safetensors', '.gguf', '.ckpt', '.pt', '.pth', '.bin')):
                                file_path = os.path.join(root, filename)
                                # Get relative path from the custom folder
                                rel_path = os.path.relpath(file_path, models_dir)
                                # Normalize path separators to forward slashes for consistency
                                rel_path = rel_path.replace(os.sep, '/')
                                clip_models.add(rel_path)
                                files_found += 1
                    if files_found > 0:
                        print(f"[Luna.ModelRouter] Found {files_found} models in {folder}/ (including subdirs)")
                except Exception as e:
                    print(f"[Luna.ModelRouter] Error scanning {folder}: {e}")
            else:
                print(f"[Luna.ModelRouter] Folder not found: {folder_path}")
    else:
        print(f"[Luna.ModelRouter] Models directory not found")
    
    return ["None"] + sorted(list(clip_models))


def _resolve_clip_path(clip_name: str) -> Optional[str]:
    """
    Resolve a CLIP model name to full path, handling custom folders and subdirectories.
    
    Examples:
        "clip_l.safetensors" → /path/to/models/clip/clip_l.safetensors
        "clip_gguf/qwen3.gguf" → /path/to/models/clip_gguf/qwen3.gguf
        "LLM/Qwen3-VL-4B/model.gguf" → /path/to/models/LLM/Qwen3-VL-4B/model.gguf
    
    Returns:
        Full path to model file, or None if not found
    """
    if not clip_name or clip_name == "None":
        return None
    
    # Check if it has a folder prefix
    if "/" in clip_name:
        parts = clip_name.split("/")
        folder = parts[0]
        relative_path = "/".join(parts[1:])
        
        # Try to find in custom folders
        try:
            models_dir = None
            
            # Method 1: Direct attribute
            if hasattr(folder_paths, 'models_dir'):
                models_dir = folder_paths.models_dir
            
            # Method 2: From checkpoints folder
            if not models_dir:
                checkpoints_dir = folder_paths.get_full_path("checkpoints", "")
                if checkpoints_dir:
                    models_dir = os.path.dirname(checkpoints_dir)
            
            # Method 3: Scan common paths using centralized COMFY_PATH
            if not models_dir:
                potential_paths = []
                
                # Try COMFY_PATH if available (from centralized __init__.py)
                if COMFY_PATH:
                    potential_paths.append(os.path.join(COMFY_PATH, "models"))
                
                # Fallback options
                potential_paths.extend([
                    os.path.join(os.getcwd(), "models"),
                    os.path.join(COMFY_PATH, "models") if COMFY_PATH else None,
                ])
                potential_paths = [p for p in potential_paths if p]  # Remove None entries
                
                for potential_path in potential_paths:
                    if os.path.isdir(potential_path):
                        models_dir = potential_path
                        break
            
            if models_dir:
                full_path = os.path.join(models_dir, folder, relative_path)
                if os.path.exists(full_path):
                    print(f"[Luna.ModelRouter] Resolved {clip_name} → {full_path}")
                    return full_path
                else:
                    print(f"[Luna.ModelRouter] Path not found: {full_path}")
        except Exception as e:
            print(f"[Luna.ModelRouter] Error resolving {clip_name}: {e}")
    else:
        # Try standard clip folder first
        try:
            full_path = folder_paths.get_full_path("clip", clip_name)
            if full_path and os.path.exists(full_path):
                return full_path
        except:
            pass
    
    print(f"[Luna.ModelRouter] Could not resolve CLIP model: {clip_name}")
    return None


# =============================================================================
# CLIP Requirements by Model Type
# =============================================================================

CLIP_REQUIREMENTS = {
    # model_type: (required_clips, optional_clips, descriptions)
    # Note: clip_4 (vision encoder) is always optional - just set it to enable IP-Adapter anchoring
    "SD1.5": {
        "required": ["clip_1"],
        "optional": ["clip_4"],
        "descriptions": {
            "clip_1": "CLIP-L (e.g., clip_l.safetensors)",
            "clip_4": "Vision encoder (optional, for IP-Adapter)",
        }
    },
    "SDXL": {
        "required": ["clip_1", "clip_2"],
        "optional": ["clip_4"],
        "descriptions": {
            "clip_1": "CLIP-L (e.g., clip_l.safetensors)",
            "clip_2": "CLIP-G (e.g., clip_g.safetensors)",
            "clip_4": "Vision encoder (optional, for IP-Adapter - SigLIP or CLIP-H)",
        }
    },
    "Flux": {
        "required": ["clip_1", "clip_3"],
        "optional": ["clip_4"],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_3": "T5-XXL",
            "clip_4": "Vision encoder (optional, for IP-Adapter - SigLIP)",
        }
    },
    "SD3": {
        "required": ["clip_1", "clip_2", "clip_3"],
        "optional": ["clip_4"],
        "descriptions": {
            "clip_1": "CLIP-L",
            "clip_2": "CLIP-G",
            "clip_3": "T5-XXL",
            "clip_4": "Vision encoder (optional, for IP-Adapter)",
        }
    },
    "Z-IMAGE": {
        "required": ["clip_1"],
        "optional": [],
        "descriptions": {
            "clip_1": "Qwen2-VL Model (loaded as Lumina2 CLIP)",
        },
        "notes": "Loads Qwen2-VL using Lumina2 CLIP type, which extracts hidden states for generation."
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
    - IP_ADAPTER output: IP-Adapter model for structural anchoring in LSD pipeline
    - Smart loading: Vision components only load if CLIP_VISION is connected
    - mmproj auto-detection: Loads from same folder as Qwen3 model
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LLM", "CLIP_VISION", "IPADAPTER", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "llm", "clip_vision", "ip_adapter", "model_name", "status")
    FUNCTION = "load"
    OUTPUT_NODE = False
    
    # Source folders
    MODEL_SOURCES = ["checkpoints", "diffusion_models", "unet"]
    
    # Model architectures (vision is auto-enabled when clip_4 is set)
    MODEL_TYPES = ["SD1.5", "SDXL", "Flux", "SD3", "Z-IMAGE"]
    
    # Precision options for dynamic loading
    # None: Use source precision as-is
    # bf16: Recommended default - fp32 range, native on Ampere+, stable
    # fp16: Legacy - slightly more precision but limited range
    # fp8: 75% VRAM reduction, native on Ada/Blackwell
    # BnB: QLoRA-compatible quantization, widely used for fine-tuning
    # GGUF: Integer quantization using GPU tensor cores
    PRECISION_OPTIONS = ["None", "bf16", "fp16", "fp8_e4m3fn", "fp8_e4m3fn_scaled", "fp8_e5m2", "nf4", "gguf_Q8_0", "gguf_Q4_K_M"]
    
    # Daemon routing modes
    DAEMON_MODES = ["auto", "force_daemon", "force_local"]
    
    @classmethod
    def _get_ipadapter_list(cls) -> list:
        """Get list of available IP-Adapter models."""
        if not HAS_COMFY:
            return ["None"]
        
        try:
            # Register ipadapter folder if not already registered
            if "ipadapter" not in folder_paths.folder_names_and_paths:
                import os
                ipadapter_path = os.path.join(folder_paths.models_dir, "ipadapter")
                if os.path.exists(ipadapter_path):
                    folder_paths.folder_names_and_paths["ipadapter"] = (
                        [ipadapter_path], 
                        folder_paths.supported_pt_extensions
                    )
            
            ipadapter_list = folder_paths.get_filename_list("ipadapter")
            return ["None"] + ipadapter_list
        except:
            return ["None"]
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
        
        # CLIP list - now includes models from clip_gguf, clip_vision, and LLM folders
        clip_list = _get_clip_model_list()
        
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
                # Now searches: models/clip, models/clip_gguf, models/clip_vision, models/LLM
                "clip_1": (clip_list, {
                    "default": "None",
                    "tooltip": "Primary CLIP encoder (CLIP-L for most, Qwen3 for Z-IMAGE). Searches: clip/, clip_gguf/, clip_vision/, LLM/"
                }),
                "clip_2": (clip_list, {
                    "default": "None",
                    "tooltip": "Secondary CLIP encoder (CLIP-G for SDXL/SD3). Searches: clip/, clip_gguf/, clip_vision/, LLM/"
                }),
                "clip_3": (clip_list, {
                    "default": "None",
                    "tooltip": "Tertiary CLIP encoder (T5-XXL for Flux/SD3). Searches: clip/, clip_gguf/, clip_vision/, LLM/"
                }),
                "clip_4": (clip_list, {
                    "default": "None",
                    "tooltip": "Vision encoder (SigLIP/CLIP-H for vision models). Searches: clip/, clip_gguf/, clip_vision/, LLM/"
                }),
                
                # === VAE Selection ===
                "vae_name": (vae_list, {
                    "default": "None",
                    "tooltip": "VAE for encoding/decoding. 'None' uses VAE from checkpoint."
                }),
                
                # === IP-Adapter for LSD structural anchoring ===
                "ip_adapter_name": (cls._get_ipadapter_list(), {
                    "default": "None",
                    "tooltip": "IP-Adapter model for structural anchoring. Used by LSD detailing pipeline."
                }),
                
                # === Daemon Mode ===
                "daemon_mode": (cls.DAEMON_MODES, {
                    "default": "auto",
                    "tooltip": "auto: use daemon if running | force_daemon: require daemon | force_local: never use daemon"
                }),
            },
            "optional": {
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
        ip_adapter_name: str,
        daemon_mode: str,
        dynprompt=None,
        unique_id=None
    ) -> Tuple[Any, Any, Any, Any, Any, Any, str, str]:
        """
        Load model with explicit configuration and runtime CLIP validation.
        
        Returns:
            model: UNet/Diffusion model
            clip: Combined CLIP for text conditioning
            vae: VAE for encode/decode
            llm: Full LLM for prompt generation (Z-IMAGE only, None for others)
            clip_vision: Vision encoder for image→embedding (if vision model type)
            ip_adapter: IP-Adapter model for LSD structural anchoring
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
        
        # Check if daemon is running by pinging the work port directly
        daemon_running = self._is_daemon_running_direct()
        
        if require_daemon and not daemon_running:
            error_msg = (
                "Daemon mode is 'force_daemon' but Luna Daemon is not running!\n"
                f"  Expected daemon at {DAEMON_HOST}:{DAEMON_PORT}\n"
                "  Start the daemon or change mode to 'auto' or 'force_local'."
            )
            raise RuntimeError(error_msg)
        
        # === STEP 3: Load MODEL ===
        output_model = None
        output_model_name = ""
        model_path_to_register = None
        
        if model_name and model_name != "None":
            output_model, output_model_name, model_path_to_register = self._load_model(
                model_source, model_name, dynamic_precision
            )
            precision_str = f" → {dynamic_precision}" if dynamic_precision != "None" else ""
            status_parts.append(f"MODEL: {model_source}/{os.path.basename(model_name)}{precision_str}")
            
            # Clean up any cached tensors from model loading
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Register checkpoint with daemon for tracking (if daemon is running)
            if daemon_running and use_daemon:
                self._register_checkpoint_with_daemon(
                    output_model, model_name, dynamic_precision
                )
                
                # WRAP in DaemonModel proxy for centralized inference
                # Use the registered path (could be converted path if conversion happened)
                output_model = self._wrap_model_as_daemon_proxy(output_model, model_type, model_path_to_register)
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
        
        # === STEP 5: Load CLIP_VISION (auto-enabled when clip_4 is set) ===
        output_clip_vision = None
        has_clip_4 = clip_config.get("clip_4") is not None
        is_zimage = model_type == "Z-IMAGE"
        
        if has_clip_4 or is_zimage:
            output_clip_vision = self._load_clip_vision(
                model_type, clip_config, daemon_running, use_daemon
            )
            if output_clip_vision is not None:
                if is_zimage:
                    status_parts.append("VISION: mmproj loaded")
                else:
                    status_parts.append("VISION: loaded (for IP-Adapter)")
        
        # === STEP 6: Load IP-Adapter (for LSD structural anchoring) ===
        output_ip_adapter = None
        
        if ip_adapter_name and ip_adapter_name != "None":
            output_ip_adapter = self._load_ip_adapter(ip_adapter_name, model_type)
            if output_ip_adapter is not None:
                status_parts.append(f"IP-Adapter: {os.path.basename(ip_adapter_name)}")
        
        # === STEP 7: Load VAE ===
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
        daemon_status = "daemon [OK]" if daemon_running and use_daemon else "local"
        status = f"[{model_type}] {daemon_status} | " + " | ".join(status_parts)
        
        print(f"[LunaModelRouter] {status}")
        
        return (output_model, output_clip, output_vae, output_llm, output_clip_vision, output_ip_adapter, output_model_name, status)
    
    def _is_daemon_running_direct(self) -> bool:
        """
        Check if daemon is running by directly pinging the work port.
        No imports or DAEMON_AVAILABLE check needed - just test connectivity.
        
        Returns True if daemon responds, False otherwise.
        """
        import socket
        import time
        
        host = "127.0.0.1"
        port = 19283  # Daemon work port
        
        # Try 3 times with 0.1s delays
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)  # 2 second timeout
                sock.connect((host, port))
                sock.close()
                print(f"[Luna.ModelRouter] [OK] Daemon is running (attempt {attempt + 1})")
                return True
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)
                elif attempt == 2:
                    print(f"[Luna.ModelRouter] ✗ Daemon not responding on {host}:{port}")
        
        return False
    
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
        precision: str
    ) -> Tuple[Any, str, str]:
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
        
        # Detect actual precision in the model weights
        actual_precision = self._detect_model_precision(model_path)
        
        # Check if dynamic precision conversion is needed
        # Only convert if:
        # 1. Conversion requested (precision != "None")
        # 2. Model is in a convertible format (fp16, bf16)
        # 3. Requested precision is different from actual
        should_convert = (
            precision != "None" and 
            actual_precision in ["fp16", "bf16"] and 
            precision != actual_precision
        )
        
        if should_convert:
            print(f"[LunaModelRouter] Converting model from {actual_precision} to {precision}")
            model, converted_path = self._load_with_conversion(model_path, precision)
            return model, output_name, converted_path
        elif actual_precision not in ["fp16", "bf16"] and precision != "None":
            # Model is already in a quantized/converted format
            print(f"[LunaModelRouter] Model is already in {actual_precision} format. Skipping conversion to {precision}")
        
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
        
        # Return model and the path to use for daemon registration
        return model, output_name, model_path
    
    def _detect_model_precision(self, model_path: str) -> str:
        """
        Detect the actual precision of model weights.
        
        Loads minimal metadata from safetensors to determine dtype without
        loading the entire model into memory.
        
        Returns:
            Precision string: "fp16", "bf16", "fp8", "int8", "nf4", or "unknown"
        """
        try:
            from safetensors import safe_open
            import torch
            
            # For .gguf files, assume fp8 or quantized
            if model_path.endswith('.gguf'):
                return "fp8"  # GGUF models are typically quantized
            
            # For safetensors, peek at metadata
            if model_path.endswith('.safetensors'):
                with safe_open(model_path, framework="pt") as f:
                    # Get first tensor to detect dtype
                    for key in list(f.keys())[:1]:  # Just check first tensor
                        # get_slice returns PySafeSlice which doesn't have dtype attribute
                        # Use get_tensor instead to get actual dtype
                        tensor_slice = f.get_tensor(key)
                        if hasattr(tensor_slice, 'dtype'):
                            dtype = tensor_slice.dtype
                        else:
                            # Fallback: check metadata
                            metadata = f.metadata()
                            if metadata and 'format' in metadata:
                                format_str = metadata['format'].lower()
                                if 'fp16' in format_str or 'float16' in format_str:
                                    return 'fp16'
                                elif 'bf16' in format_str or 'bfloat16' in format_str:
                                    return 'bf16'
                                elif 'fp8' in format_str:
                                    return 'fp8'
                            return "unknown"
                        
                        # Map torch dtypes to our precision strings
                        dtype_map = {
                            'torch.float16': 'fp16',
                            'torch.bfloat16': 'bf16',
                            'torch.float32': 'fp32',
                            'torch.float8_e4m3fn': 'fp8',
                            'torch.int8': 'int8',
                            'torch.float8_e5m2': 'fp8',
                        }
                        
                        dtype_str = str(dtype)
                        for torch_type, precision_type in dtype_map.items():
                            if torch_type in dtype_str:
                                return precision_type
                        
                        # If we detect float32, model is unquantized (original)
                        if 'float32' in dtype_str.lower():
                            return "fp32"
                        
                        return "unknown"
            
            # For .pth or .pt files, would need to load - skip for now
            return "unknown"
            
        except Exception as e:
            print(f"[LunaModelRouter] Warning: Could not detect model precision: {e}")
            return "unknown"
    
    def _load_with_conversion(
        self,
        model_path: str,
        precision: str
    ) -> Tuple[Any, str]:
        """
        Load model with precision conversion using smart caching.
        
        Uses conversion_cache + smart_converter to:
        1. Check if converted model already exists
        2. Convert only if needed
        3. Return path to converted model
        
        Output locations (standardized):
        - Precision (fp16/bf16/fp8): models/diffusion_models/converted/{basename}_{precision}.safetensors
        - BitsAndBytes (nf4/int8): models/diffusion_models/converted/{basename}_{precision}.safetensors
        - GGUF (Q4/Q8): models/diffusion_models/gguf/{basename}_{precision}.gguf
        
        Args:
            model_path: Path to source model
            precision: Target precision (fp16, bf16, fp8_e4m3fn, nf4, int8, Q4_K_M, etc.)
        
        Returns:
            Tuple of (loaded_model, converted_path)
        """
        import sys
        import importlib.util
        from pathlib import Path
        
        try:
            # Load smart_converter dynamically
            converter_path = Path(__file__).parent.parent.parent / "utils" / "smart_converter.py"
            
            if not converter_path.exists():
                raise FileNotFoundError(f"smart_converter.py not found at {converter_path}")
            
            spec = importlib.util.spec_from_file_location("smart_converter", converter_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to create module spec for smart_converter")
            
            smart_converter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smart_converter_module)
            smart_convert = smart_converter_module.smart_convert
            
            # Use smart converter - checks cache, converts if needed
            converted_path, was_newly_converted = smart_convert(model_path, precision)
            
            if was_newly_converted:
                print(f"[LunaModelRouter] ✓ Conversion complete: {converted_path}")
            else:
                print(f"[LunaModelRouter] ✓ Using cached conversion: {converted_path}")
            
            # Load the converted model
            if converted_path.endswith('.gguf'):
                model = self._load_gguf_model(converted_path)
            else:
                # Safetensors (fp16, bf16, fp8, nf4, int8)
                model = self._load_safetensors_model(converted_path)
            
            return model, converted_path
        
        except Exception as e:
            print(f"[LunaModelRouter] ✗ Conversion error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_safetensors_model(self, path: str) -> Any:
        """Load safetensors model file (UNet only).
        
        Routes to specialized loaders for quantized formats:
        - INT8: Uses LunaINT8Loader for dequantization
        - NF4: Uses LunaNF4Loader for dequantization
        - Other: Uses standard comfy.sd.load_unet()
        """
        import os
        from pathlib import Path
        import json
        
        print(f"[LunaModelRouter] Loading safetensors: {path}")
        
        # Check for Luna metadata to identify quantization type
        dtype_hint = None
        try:
            with open(path, 'rb') as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) == 8:
                    header_len = int.from_bytes(header_len_bytes, 'little')
                    header_json = f.read(header_len).decode('utf-8')
                    header = json.loads(header_json)
                    if '__metadata__' in header and isinstance(header['__metadata__'], dict):
                        dtype_hint = header['__metadata__'].get('luna_dtype')
        except Exception:
            pass
        
        # Route to appropriate loader
        if dtype_hint == 'int8':
            print(f"[LunaModelRouter] Detected INT8 format, using specialized loader")
            from .luna_quantized_loader import LunaINT8Loader
            loader = LunaINT8Loader()
            result = loader.load_int8_unet(path, target_dtype="auto")
            return result[0]
        
        elif dtype_hint == 'nf4':
            print(f"[LunaModelRouter] Detected NF4 format, using specialized loader")
            from .luna_quantized_loader import LunaNF4Loader
            loader = LunaNF4Loader()
            result = loader.load_nf4_unet(path)
            return result[0]
        
        else:
            # Standard float format (fp16, bf16, fp8, etc.)
            print(f"[LunaModelRouter] Using standard UNet loader")
            model = comfy.sd.load_unet(path)
            return model
    
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
        model = comfy.sd.load_diffusion_model_state_dict(  # type: ignore
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
        
        # Collect CLIP paths with component mapping
        clip_components = {}
        clip_paths = [] # Keep for local fallback
        
        # Map slot to encoder type
        slot_to_type = {
            "clip_1": "clip_l",
            "clip_2": "clip_g",
            "clip_3": "t5xxl",
            "clip_4": "vision",
        }
        
        for slot, path in clip_config.items():
            if path is not None:
                # Use the new resolver that handles custom folders
                full_path = _resolve_clip_path(path)
                if full_path and os.path.exists(full_path):
                    clip_paths.append(full_path)
                    component_type = slot_to_type.get(slot, "clip_l")
                    clip_components[component_type] = full_path
        
        if not clip_paths:
            return None
        
        # Get clip_type string for daemon/ComfyUI
        clip_type_str = CLIP_TYPE_MAP.get(model_type, "stable_diffusion")
        
        # Route through daemon if available (path-based for efficiency)
        if daemon_running and use_daemon and DaemonCLIP is not None and daemon_client is not None:
            try:
                # Register CLIP by path (daemon loads from disk)
                # Send dictionary of components for granular reuse
                print(f"[LunaModelRouter] Registering CLIP with daemon: {clip_components}")
                result = daemon_client.register_clip_by_path(clip_components, model_type, clip_type_str)  # type: ignore
                print(f"[LunaModelRouter] Registration result: {result}")
                
                if result.get("success"):
                    print(f"[LunaModelRouter] [OK] Registered CLIP with daemon: {model_type} -> {clip_type_str}")
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
                    print(f"[LunaModelRouter] ✗ Daemon CLIP registration failed: {result.get('error')}")
            except Exception as e:
                print(f"[LunaModelRouter] ✗ Daemon CLIP exception, using local: {e}")
        
        # Load locally as fallback
        clip_type_map = {
            "SD1.5": getattr(comfy.sd, 'CLIPType', type(None)).STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
            "SDXL": getattr(comfy.sd, 'CLIPType', type(None)).STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
            "SDXL + Vision": getattr(comfy.sd, 'CLIPType', type(None)).STABLE_DIFFUSION if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
            "Flux": getattr(comfy.sd, 'CLIPType', type(None)).FLUX if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
            "Flux + Vision": getattr(comfy.sd, 'CLIPType', type(None)).FLUX if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
            "SD3": getattr(comfy.sd, 'CLIPType', type(None)).SD3 if hasattr(comfy.sd, 'CLIPType') else None,  # type: ignore
        }
        
        try:
            clip = comfy.sd.load_clip(  # type: ignore
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_map.get(model_type)
            )
        except Exception as e:
            print(f"[LunaModelRouter] Multi-CLIP load failed, trying single: {e}")
            clip = comfy.sd.load_clip(  # type: ignore
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
        
        # Get full path to Qwen3 model using the custom resolver
        full_path = _resolve_clip_path(clip_1_path)
        if not full_path or not os.path.exists(full_path):
            raise FileNotFoundError(f"Qwen3 model not found: {clip_1_path}")
        
        # For Z-IMAGE, we prefer daemon's Qwen3-VL if available
        if daemon_running and use_daemon and DaemonZImageCLIP is not None:
            try:
                # Load with Z-IMAGE text encoder (same as local path)
                print(f"[LunaModelRouter] Loading Qwen3-VL for daemon (Z-IMAGE 2560-dim encoder)")
                
                import comfy.text_encoders.z_image
                import comfy.text_encoders.hunyuan_video
                import comfy.utils  # type: ignore
                from comfy.sd1_clip import CLIP
                
                clip_data = comfy.utils.load_torch_file(full_path, safe_load=True)  # type: ignore
                
                # For multi-shard models, merge all shards into single state_dict
                # Check if this is a sharded model by looking for shard info in the path
                if "model-" in os.path.basename(full_path):
                    # Multi-shard model - need to load and merge all shards
                    model_dir = os.path.dirname(full_path)
                    shard_files = sorted([f for f in os.listdir(model_dir) if f.startswith("model-") and f.endswith(".safetensors")])
                    
                    # Merge all shards into single state_dict
                    merged_state_dict = {}
                    for shard_file in shard_files:
                        shard_data = comfy.utils.load_torch_file(os.path.join(model_dir, shard_file), safe_load=True)  # type: ignore
                        merged_state_dict.update(shard_data)
                    clip_data = merged_state_dict
                
                hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect([clip_data], "model.")
                
                tokenizer = comfy.text_encoders.z_image.ZImageTokenizer(
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
                text_encoder_model = comfy.text_encoders.z_image.te(**hunyuan_detect)()
                text_encoder_model.load_sd(clip_data)
                
                local_clip = CLIP(tokenizer=tokenizer, text_encoder=text_encoder_model, name="qwen3_4b")
                
                # Create daemon proxy that uses the same weights for both CLIP and LLM
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
            # Handle GGUF loading via ComfyUI-GGUF loader
            if full_path.lower().endswith(".gguf"):
                if not HAS_GGUF or CLIPLoaderGGUF is None or gguf_clip_loader is None:
                    raise RuntimeError(
                        "GGUF CLIP loading requires ComfyUI-GGUF extension. "
                        "Install from: https://github.com/city96/ComfyUI-GGUF"
                    )
                
                print(f"[LunaModelRouter] Loading GGUF CLIP: {full_path}")
                
                # CLIPLoaderGGUF expects to resolve paths via folder_paths.get_full_path(),
                # but our custom LLM/ folder isn't registered. So we call the underlying
                # loader functions directly with the full path.
                
                # Use the imported gguf_clip_loader from module-level imports
                # Load the GGUF clip data
                clip_data = [gguf_clip_loader(full_path)]
                
                # Create the CLIP patcher using the same method as CLIPLoaderGGUF
                loader = CLIPLoaderGGUF()
                clip_type = getattr(comfy.sd, 'CLIPType', type(None)).LUMINA2 if hasattr(comfy.sd, 'CLIPType') else None  # type: ignore
                clip = loader.load_patcher([full_path], clip_type, clip_data)
                
            else:
                # Standard loading for safetensors/other formats
                # User selected Z-IMAGE model type, so load with Z-IMAGE text encoder (Qwen3_4B - 2560 dims)
                print(f"[LunaModelRouter] Loading Qwen3-VL for Z-IMAGE (2560-dim text encoder)")
                
                # Load with Z-IMAGE text encoder configuration
                import comfy.text_encoders.z_image
                import comfy.text_encoders.hunyuan_video
                
                # Load state dict
                clip_data = comfy.utils.load_torch_file(full_path, safe_load=True)  # type: ignore
                
                # For multi-shard models, merge all shards into single state_dict
                # Check if this is a sharded model by looking for shard info in the path
                if "model-" in os.path.basename(full_path):
                    # Multi-shard model - need to load and merge all shards
                    model_dir = os.path.dirname(full_path)
                    shard_files = sorted([f for f in os.listdir(model_dir) if f.startswith("model-") and f.endswith(".safetensors")])
                    
                    # Merge all shards into single state_dict
                    merged_state_dict = {}
                    for shard_file in shard_files:
                        shard_data = comfy.utils.load_torch_file(os.path.join(model_dir, shard_file), safe_load=True)  # type: ignore
                        merged_state_dict.update(shard_data)
                    clip_data = merged_state_dict
                
                # Detect llama architecture params from the Qwen3 model
                # Keys in standalone Qwen3-VL are "model.*" not "text_encoders.qwen3_4b.transformer.*"
                hunyuan_detect = comfy.text_encoders.hunyuan_video.llama_detect([clip_data], "model.")
                
                # Build the CLIP object with Z-IMAGE tokenizer and Qwen3_4B model
                from comfy.sd1_clip import CLIP
                
                tokenizer = comfy.text_encoders.z_image.ZImageTokenizer(
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
                
                text_encoder_model = comfy.text_encoders.z_image.te(**hunyuan_detect)()
                text_encoder_model.load_sd(clip_data)
                
                clip = CLIP(tokenizer=tokenizer, text_encoder=text_encoder_model, name="qwen3_4b")
            
            # Create LLM reference that shares the CLIP model
            # The GGUF file contains lm_head weights - we can use them for generation
            llm = self._create_llm_reference(full_path, daemon_running=False, shared_clip=clip)
            
            # Attach paths to CLIP object for downstream nodes (LunaZImageEncoder)
            clip.model_path = full_path
            clip.mmproj_path = llm.get("mmproj_path")
            
            return clip, llm
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3 model: {e}")
    
    def _create_llm_reference(self, model_path: str, daemon_running: bool, shared_clip: Any = None) -> Dict[str, Any]:
        """
        Create an LLM reference object that can be passed to VLM nodes.
        
        If shared_clip is provided, the LLM reference will use the same loaded
        model weights for text generation, avoiding loading the model twice.
        
        The GGUF file contains lm_head weights - ComfyUI-GGUF loads them but
        ComfyUI's llama.py doesn't use them. We can access the state_dict and
        create a generation-capable wrapper that shares the transformer weights.
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
        
        llm_ref = {
            "type": "qwen3_vl",
            "model_path": model_path,
            "model_name": model_name,
            "mmproj_path": mmproj_path,
            "use_daemon": daemon_running,
            "loaded": shared_clip is not None,
        }
        
        # If we have a shared CLIP, attach it for weight sharing
        # The VLM Prompt Generator can use this instead of loading separately
        if shared_clip is not None:
            llm_ref["shared_clip"] = shared_clip
            llm_ref["generation_mode"] = "shared"  # Use shared weights
            print(f"[LunaModelRouter] LLM will share weights with CLIP encoder")
        else:
            llm_ref["generation_mode"] = "separate"  # Load via llama-cpp
        
        return llm_ref
    
    def _load_clip_vision(
        self,
        model_type: str,
        clip_config: Dict[str, Optional[str]],
        daemon_running: bool,
        use_daemon: bool
    ) -> Any:
        """
        Load vision encoder for image→embedding conversion.
        
        Automatically loads when clip_4 is set (for IP-Adapter anchoring).
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
                try:
                    from nodes import CLIPVisionLoader  # type: ignore
                except ImportError:
                    CLIPVisionLoader = None  # type: ignore
                
                if CLIPVisionLoader:
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
        """Load VAE with optional daemon routing (CUDA IPC weight sharing)."""
        
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE not found: {vae_name}")
        
        # NOTE: VAE is loaded locally (not via daemon)
        # VAE is only 168MB and loads in 1-2 seconds locally
        # The bottleneck is CLIP (1.6GB), which is handled by daemon
        # For multi-instance workflows, power users can use managed spawning
        # with torch.multiprocessing IPC (planned feature)
        
        # Load VAE locally
        print(f"[LunaModelRouter] Loading VAE locally: {vae_name}")
        try:
            try:
                from nodes import VAELoader  # type: ignore
            except ImportError:
                VAELoader = None  # type: ignore
            
            if VAELoader:
                loader = VAELoader()
                vae = loader.load_vae(vae_name)[0]
            else:
                raise ImportError("VAELoader not available")
        except ImportError:
            # Fallback
            sd = comfy.utils.load_torch_file(vae_path)  # type: ignore
            vae = comfy.sd.VAE(sd=sd)  # type: ignore
        
        return vae
    
    def _load_ip_adapter(
        self,
        ip_adapter_name: str,
        model_type: str
    ) -> Any:
        """
        Load IP-Adapter model for LSD structural anchoring.
        
        Args:
            ip_adapter_name: Filename of IP-Adapter model
            model_type: Model architecture for compatibility detection
        
        Returns:
            Loaded IP-Adapter state dict, or None if loading fails
        """
        # Get the full path
        try:
            ip_adapter_path = folder_paths.get_full_path("ipadapter", ip_adapter_name)
        except:
            # Fallback if ipadapter folder not registered
            ip_adapter_path = os.path.join(folder_paths.models_dir, "ipadapter", ip_adapter_name)
        
        if not ip_adapter_path or not os.path.exists(ip_adapter_path):
            print(f"[LunaModelRouter] ⚠ IP-Adapter not found: {ip_adapter_name}")
            return None
        
        print(f"[LunaModelRouter] Loading IP-Adapter: {ip_adapter_name}")
        
        try:
            # Try to use IPAdapterPlus loader if available
            try:
                from custom_nodes.comfyui_ipadapter_plus.utils import ipadapter_model_loader
                ip_adapter = ipadapter_model_loader(ip_adapter_path)
            except ImportError:
                # Fallback to direct loading
                ip_adapter = comfy.utils.load_torch_file(ip_adapter_path)  # type: ignore
            
            # Validate compatibility with model type
            is_sdxl = "SDXL" in model_type or "Flux" in model_type
            if "ip_adapter" in ip_adapter and "1.to_k_ip.weight" in ip_adapter["ip_adapter"]:
                output_dim = ip_adapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
                model_is_sdxl = output_dim == 2048
                
                if is_sdxl != model_is_sdxl:
                    print(f"[LunaModelRouter] ⚠ IP-Adapter mismatch: model is {'SDXL' if is_sdxl else 'SD1.5'}, "
                          f"IP-Adapter is {'SDXL' if model_is_sdxl else 'SD1.5'}")
            
            # Add metadata for downstream use
            ip_adapter["_luna_path"] = ip_adapter_path
            ip_adapter["_luna_is_sdxl"] = is_sdxl
            
            print(f"[LunaModelRouter] ✓ IP-Adapter loaded: {ip_adapter_name}")
            return ip_adapter
            
        except Exception as e:
            print(f"[LunaModelRouter] ✗ Failed to load IP-Adapter: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def _register_checkpoint_with_daemon(self, model: Any, model_name: str, precision: str) -> None:
        """Register checkpoint with daemon for tracking across instances."""
        try:
            from server import PromptServer
            from luna_daemon import client as daemon_client
            import torch
            
            # Generate instance ID from server port
            port = getattr(PromptServer.instance, 'port', 8188)
            instance_id = f"comfyui:{port}"
            
            # Get model size - use estimated size instead of state_dict to avoid VRAM spike
            size_mb = 0
            if hasattr(model, 'model'):
                # Try to get size from model parameters without creating state_dict copy
                try:
                    for param in model.model.parameters():
                        if hasattr(param, 'numel') and hasattr(param, 'element_size'):
                            size_mb += param.numel() * param.element_size()
                    size_mb = size_mb / (1024 * 1024)  # Convert to MB
                except:
                    # Fallback: estimate from file size or use default
                    size_mb = 2500  # Default estimate
            
            # Get device
            device = "cuda:0"  # Default
            if hasattr(model, 'load_device'):
                device = str(model.load_device)
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = str(model.model.device)
            
            # Determine dtype
            dtype = precision if precision != "None" else "unknown"
            if hasattr(model, 'model') and hasattr(model.model, 'dtype'):
                dtype_obj = model.model.dtype
                if dtype == "None" or dtype == "unknown":
                    dtype = str(dtype_obj).replace("torch.", "")
            
            # Register with daemon
            result = daemon_client.register_checkpoint(
                instance_id=instance_id,
                name=os.path.basename(model_name),
                path=model_name,
                size_mb=round(size_mb, 1),
                device=device,
                dtype=dtype
            )
            
            if result.get('success'):
                print(f"[LunaModelRouter] [OK] Registered checkpoint with daemon: {os.path.basename(model_name)} ({size_mb:.1f} MB) on {device}")
            else:
                print(f"[LunaModelRouter] ✗ Checkpoint registration failed: {result.get('message', 'unknown error')}")
            
        except ImportError as e:
            print(f"[LunaModelRouter] Cannot register checkpoint - daemon client not available: {e}")
        except Exception as e:
            import traceback
            print(f"[LunaModelRouter] ✗ Error registering checkpoint with daemon: {e}")
            print(traceback.format_exc())
    
    def _wrap_model_as_daemon_proxy(self, model: Any, model_type: str, model_path: str) -> Any:
        """
        Wrap loaded model with InferenceModeWrapper for VRAM optimization.
        
        This wraps the model to force inference_mode() on all forward passes,
        which disables gradient tracking and significantly reduces VRAM usage.
        
        The model stays local - no daemon communication for inference.
        VAE/CLIP can still use the daemon for shared encoding.
        
        Args:
            model: The loaded ModelPatcher object
            model_type: Model type string (flux, sdxl, etc.)
            model_path: Path to the model file
        
        Returns:
            InferenceModeWrapper wrapped model
        """
        try:
            from luna_daemon.inference_wrapper import wrap_model_for_inference
            
            # Wrap model with inference_mode
            wrapped = wrap_model_for_inference(model)
            
            print(f"[LunaModelRouter] ✓ Model wrapped with InferenceModeWrapper for VRAM optimization")
            return wrapped
        
        except ImportError:
            print(f"[LunaModelRouter] ⚠ InferenceModeWrapper not available, using local model")
            return model
        except Exception as e:
            print(f"[LunaModelRouter] ⚠ Error wrapping model: {e}")
            return model



# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaModelRouter": LunaModelRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaModelRouter": "Luna Model Router ⚡",
}
