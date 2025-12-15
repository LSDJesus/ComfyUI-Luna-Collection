"""
Luna Dynamic Model Loader - Smart Precision Loader

The ultimate checkpoint loading solution for multi-GPU setups:

1. **Source Library** (HDD): Your master FP16/FP32 checkpoints
2. **Local Optimized Weights** (NVMe): PC-specific UNet precision
3. **Smart Hybrid Loading**: CLIP/VAE from source + optimized UNet

Supported UNet precisions:
- bf16: Native bfloat16 safetensors
- fp8_e4m3fn: Native FP8 (Ada/Blackwell optimized)
- nf4: 4-bit NormalFloat (BitsAndBytes, QLoRA standard)
- int8: 8-bit integer (BitsAndBytes, 50% VRAM reduction)
- gguf_Q8_0: 8-bit GGUF (Ampere INT8 tensor cores)
- gguf_Q4_K_M: 4-bit GGUF (Blackwell INT4 tensor cores!)

Workflow:
1. Select any checkpoint from your FP16 library
2. Pick target precision for your GPU
3. First run: extracts UNet, converts, saves to NVMe
4. All runs: loads CLIP/VAE from source + optimized UNet

This means you can delete all 358 checkpoints from NVMe, keep them on
the HDD, and only store the optimized UNets locally (~2-4GB each vs 6.5GB).
"""

import os
from pathlib import Path
from typing import Tuple, Any

import torch

# Import centralized path constants from Luna Collection
try:
    from __init__ import COMFY_PATH, LUNA_PATH
except (ImportError, ModuleNotFoundError, AttributeError):
    # Fallback: construct paths if Luna constants aren't available
    COMFY_PATH = None
    LUNA_PATH = None

try:
    import folder_paths
    import comfy.sd
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

# Check for GGUF support
try:
    from gguf import GGUFReader
    HAS_GGUF_LIB = True
except ImportError:
    HAS_GGUF_LIB = False

# Try to import ComfyUI-GGUF loader
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
    HAS_GGUF_NODE = UnetLoaderGGUF is not None
except:
    HAS_GGUF_NODE = False
    UnetLoaderGGUF = None

# Check for BitsAndBytes support
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# =============================================================================
# Main Node
# =============================================================================

class LunaDynamicModelLoader:
    """
    Smart Precision Loader - Hybrid checkpoint loading with optimized UNets.
    
    Automatically detects which outputs are connected:
    - MODEL always loads from optimized UNet
    - CLIP/VAE only load from source FP16 if their outputs are connected
    
    First use: Extracts UNet, converts to target precision, saves locally
    All uses: Loads from local optimized weights + source CLIP/VAE as needed
    
    This lets you keep master FP16 checkpoints on HDD while only storing
    small optimized UNets (~2-4GB) on fast NVMe per-PC.
    """
    
    CATEGORY = "Luna/Loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "unet_path")
    FUNCTION = "load_smart"
    
    # Precision options with hardware notes
    # bf16: Best default - fp32 range, native on Ampere+, stable gradients
    # fp16: Slightly more precision but limited range, legacy compat
    # fp8: 75% VRAM reduction, Ada/Blackwell native
    # BnB: QLoRA-compatible quantization, widely used for fine-tuning
    # GGUF: Integer quantization, GPU-specific tensor core optimization
    PRECISION_OPTIONS = [
        "bf16 (recommended, fp32 range)",
        "fp16 (legacy, more precision)",
        "fp8_e4m3fn (Ada/Blackwell, 75% smaller)",
        "nf4 (4-bit BitsAndBytes, QLoRA standard)",
        "int8 (8-bit BitsAndBytes, 50% VRAM)",
        "gguf_Q8_0 (Ampere INT8 tensor cores)",
        "gguf_Q4_K_M (Blackwell INT4 tensor cores)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        if not HAS_COMFY:
            return {"required": {}}
        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Source checkpoint (FP16/FP32 on HDD). UNet will be extracted and optimized."
                }),
                "precision": (cls.PRECISION_OPTIONS, {
                    "default": cls.PRECISION_OPTIONS[0],
                    "tooltip": "Target UNet precision. Choose based on your GPU architecture."
                }),
            },
            "optional": {
                "local_weights_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Local NVMe directory for optimized UNets. Default: models/unet/optimized"
                }),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
    
    def check_lazy_status(self, ckpt_name, precision, local_weights_dir="", dynprompt=None, unique_id=None):
        """
        ComfyUI lazy evaluation hook - determine which outputs need computation.
        Returns list of output indices that need to be computed.
        """
        # MODEL (0) and unet_path (3) are always needed
        needed = [0, 3]
        
        if dynprompt is not None and unique_id is not None:
            try:
                # Check if CLIP (1) or VAE (2) outputs are connected
                # DynamicPrompt API: use all_node_ids() and get_node() directly
                for output_idx in [1, 2]:  # CLIP=1, VAE=2
                    if self._is_output_connected(dynprompt, unique_id, output_idx):
                        needed.append(output_idx)
            except Exception:
                # If we can't determine, assume all needed
                needed = [0, 1, 2, 3]
        else:
            # No graph info - assume all needed
            needed = [0, 1, 2, 3]
        
        return needed
    
    def _is_output_connected(self, dynprompt, node_id, output_idx):
        """Check if a specific output slot is connected to any downstream node."""
        try:
            # Iterate all nodes to find connections from this output
            for other_id in dynprompt.all_node_ids():
                try:
                    other_node = dynprompt.get_node(other_id)
                except Exception:
                    continue
                
                inputs = other_node.get("inputs", {})
                for input_name, input_val in inputs.items():
                    # Input connections are [node_id, output_index]
                    if isinstance(input_val, list) and len(input_val) >= 2:
                        if str(input_val[0]) == str(node_id) and input_val[1] == output_idx:
                            return True
            return False
        except Exception:
            # If we can't determine, assume connected
            return True
    
    def load_smart(
        self, 
        ckpt_name: str, 
        precision: str,
        local_weights_dir: str = "",
        dynprompt=None,
        unique_id=None
    ) -> Tuple[Any, Any, Any, str]:
        """Load checkpoint with smart lazy evaluation."""
        
        # Determine what's needed
        needed_outputs = self.check_lazy_status(
            ckpt_name, precision, local_weights_dir, dynprompt, unique_id
        )
        need_clip = 1 in needed_outputs
        need_vae = 2 in needed_outputs
        
        # 1. Parse options
        precision_key = precision.split()[0]  # "bf16 (universal...)" -> "bf16"
        is_gguf = "gguf" in precision_key
        is_bnb = precision_key in ["nf4", "int8"]
        
        # 2. Resolve source checkpoint path (HDD)
        src_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not src_path or not os.path.exists(src_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")
        
        # 3. Local weights directory (NVMe)
        if local_weights_dir and os.path.isdir(local_weights_dir):
            weights_root = local_weights_dir
        else:
            # Use COMFY_PATH if available, fallback to folder_paths.models_dir
            if COMFY_PATH:
                models_base = os.path.join(COMFY_PATH, "models")
            else:
                models_base = folder_paths.models_dir
            weights_root = os.path.join(models_base, "unet", "optimized")
        os.makedirs(weights_root, exist_ok=True)
        
        # 4. Build target UNet filename
        base_name = os.path.splitext(os.path.basename(ckpt_name))[0]
        
        if is_gguf:
            quant_type = precision_key.replace("gguf_", "")
            unet_filename = f"{base_name}_{quant_type}.gguf"
        elif is_bnb:
            unet_filename = f"{base_name}_{precision_key}_unet.safetensors"
        else:
            unet_filename = f"{base_name}_{precision_key}_unet.safetensors"
        
        unet_path = os.path.join(weights_root, unet_filename)
        
        # 5. Convert UNet if not already done
        if not os.path.exists(unet_path):
            # Import conversion utilities from utils
            # ComfyUI's dynamic loading doesn't always preserve package context,
            # so we use importlib to load it dynamically
            import importlib.util
            
            # Calculate path to checkpoint_converter.py using centralized LUNA_PATH
            if LUNA_PATH:
                converter_path = Path(LUNA_PATH) / "utils" / "checkpoint_converter.py"
            else:
                # Fallback: luna_dynamic_loader.py is at nodes/loaders/, go up 3 levels
                converter_path = Path(__file__).parent.parent.parent / "utils" / "checkpoint_converter.py"
            
            if not converter_path.exists():
                raise FileNotFoundError(f"checkpoint_converter.py not found at {converter_path}")
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("checkpoint_converter", converter_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create module spec for {converter_path}")
            converter_module = importlib.util.module_from_spec(spec)
            # spec.loader is guaranteed to be not None here
            spec.loader.exec_module(converter_module)
            
            convert_to_precision = converter_module.convert_to_precision
            convert_to_gguf = converter_module.convert_to_gguf
            convert_to_bnb = converter_module.convert_to_bnb
            
            print(f"[Luna] First use - extracting and converting UNet...")
            print(f"[Luna] Source: {src_path}")
            print(f"[Luna] Target: {unet_path}")
            
            if is_gguf:
                quant_type = precision_key.replace("gguf_", "")
                convert_to_gguf(src_path, unet_path, quant_type)
            elif is_bnb:
                convert_to_bnb(src_path, unet_path, precision_key)
            else:
                convert_to_precision(src_path, unet_path, precision_key, strip_components=True)
            
            print(f"[Luna] UNet saved to local weights directory")
        else:
            print(f"[Luna] Using local optimized UNet: {unet_filename}")
        
        # 6. Smart loading based on what's connected
        clip = None
        vae = None
        
        if need_clip or need_vae:
            # Load source checkpoint for CLIP/VAE
            components = "CLIP" if need_clip else ""
            components += ("+" if components and need_vae else "") + ("VAE" if need_vae else "")
            print(f"[Luna] Loading {components} from source: {os.path.basename(src_path)}")
            
            out = comfy.sd.load_checkpoint_guess_config(
                src_path,
                output_vae=need_vae,
                output_clip=need_clip,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            # out = (model, clip, vae, ...)
            if need_clip:
                clip = out[1]
            if need_vae:
                vae = out[2]
            
            # Discard source model immediately
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[Luna] UNet-only mode (CLIP/VAE outputs not connected)")
        
        # 7. Load optimized UNet
        print(f"[Luna] Loading optimized UNet: {unet_filename}")
        if is_gguf:
            model = self._load_gguf_unet(unet_path)
        elif is_bnb:
            model = self._load_bnb_unet(unet_path)
        else:
            model = self._load_safetensors_unet(unet_path)
        
        return (model, clip, vae, unet_path)
    
    def _load_safetensors_unet(self, path: str) -> Any:
        """Load UNet from safetensors file."""
        try:
            from comfy.sd import load_unet
            return load_unet(path)
        except Exception as e:
            # Fallback: try loading as checkpoint
            print(f"[Luna] Direct UNet load failed, trying checkpoint loader: {e}")
            out = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=False,
                output_clip=False
            )
            return out[0]
    
    def _load_gguf_unet(self, path: str) -> Any:
        """Load UNet from GGUF file."""
        
        if not HAS_GGUF_NODE or UnetLoaderGGUF is None:
            raise ImportError(
                "ComfyUI-GGUF is required to load .gguf files.\n"
                "Install from: https://github.com/city96/ComfyUI-GGUF"
            )
        
        loader = UnetLoaderGGUF()
        gguf_filename = os.path.basename(path)
        
        try:
            model_tuple = loader.load_unet(unet_name=gguf_filename)
            return model_tuple[0]
        except Exception as e:
            # Try direct load
            try:
                from comfy.sd import load_unet
                return load_unet(path)
            except:
                raise RuntimeError(f"Failed to load GGUF: {e}")
    
    def _load_bnb_unet(self, path: str) -> Any:
        """Load UNet from BitsAndBytes quantized file."""
        
        if not HAS_BNB:
            raise ImportError(
                "bitsandbytes is required to load quantized models.\n"
                "Install with: pip install bitsandbytes"
            )
        
        # BitsAndBytes models are saved as safetensors with quantized tensors
        # ComfyUI should handle them transparently
        try:
            from comfy.sd import load_unet
            return load_unet(path)
        except Exception as e:
            # Fallback: try loading as checkpoint
            print(f"[Luna] Direct UNet load failed, trying checkpoint loader: {e}")
            out = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=False,
                output_clip=False
            )
            return out[0]


# =============================================================================
# Utility Node: Local Weights Manager
# =============================================================================

class LunaOptimizedWeightsManager:
    """
    Manage your local optimized UNet weights.
    
    View stored models, clear old entries, or purge weights to re-convert.
    These are your PC-specific optimized UNets stored on NVMe.
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "manage_weights"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list", "stats", "clear_old", "purge_all"], {
                    "default": "list",
                    "tooltip": "Action: list files, show stats, clear old, or purge all"
                }),
            },
            "optional": {
                "weights_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Override location. Default: models/unet/optimized"
                }),
                "days_old": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 365,
                    "tooltip": "For clear_old: delete files older than this many days"
                }),
            }
        }
    
    def manage_weights(
        self, 
        action: str,
        weights_directory: str = "",
        days_old: int = 30
    ) -> Tuple[str]:
        """Manage local optimized weights."""
        
        if weights_directory and os.path.isdir(weights_directory):
            weights_root = weights_directory
        else:
            # Use COMFY_PATH if available, fallback to folder_paths.models_dir
            if COMFY_PATH:
                models_base = os.path.join(COMFY_PATH, "models")
            else:
                models_base = folder_paths.models_dir
            weights_root = os.path.join(models_base, "unet", "optimized")
        
        if not os.path.exists(weights_root):
            return (f"Weights directory not found: {weights_root}",)
        
        import time
        now = time.time()
        cutoff = now - (days_old * 24 * 60 * 60)
        
        lines = [f"Luna Optimized Weights Manager", f"Location: {weights_root}", "=" * 50]
        
        files = []
        total_size = 0
        
        for f in os.listdir(weights_root):
            path = os.path.join(weights_root, f)
            if os.path.isfile(path):
                stat = os.stat(path)
                files.append({
                    "name": f,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "mtime": stat.st_mtime,
                    "age_days": (now - stat.st_mtime) / (24 * 60 * 60)
                })
                total_size += stat.st_size
        
        if action == "list":
            for f in sorted(files, key=lambda x: -x["mtime"]):
                lines.append(f"{f['name']}: {f['size_mb']:.1f}MB ({f['age_days']:.0f}d old)")
        
        elif action == "stats":
            lines.append(f"Total files: {len(files)}")
            lines.append(f"Total size: {total_size / (1024**3):.2f}GB")
            
            by_type = {}
            for f in files:
                ext = os.path.splitext(f["name"])[1]
                by_type[ext] = by_type.get(ext, 0) + 1
            for ext, count in by_type.items():
                lines.append(f"  {ext}: {count} files")
        
        elif action == "clear_old":
            deleted = 0
            freed = 0
            for f in files:
                if f["mtime"] < cutoff:
                    path = os.path.join(weights_root, f["name"])
                    os.remove(path)
                    deleted += 1
                    freed += f["size_mb"]
            lines.append(f"Deleted {deleted} files older than {days_old} days")
            lines.append(f"Freed {freed:.1f}MB")
        
        elif action == "purge_all":
            deleted = 0
            for f in files:
                os.remove(os.path.join(weights_root, f["name"]))
                deleted += 1
            lines.append(f"Purged {deleted} files ({total_size / (1024**3):.2f}GB)")
        
        return ("\n".join(lines),)


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaDynamicModelLoader": LunaDynamicModelLoader,
    "LunaOptimizedWeightsManager": LunaOptimizedWeightsManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaDynamicModelLoader": "Luna Dynamic Model Loader",
    "LunaOptimizedWeightsManager": "Luna Optimized Weights Manager",
}
