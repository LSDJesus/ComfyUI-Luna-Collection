"""
Luna Dynamic Model Loader - Smart Precision Loader

The ultimate checkpoint loading solution for multi-GPU setups:

1. **Source Library** (HDD): Your master FP16/FP32 checkpoints
2. **Local Optimized Weights** (NVMe): PC-specific UNet precision
3. **Smart Hybrid Loading**: CLIP/VAE from source + optimized UNet

Supported UNet precisions:
- bf16: Native bfloat16 safetensors
- fp8_e4m3fn: Native FP8 (Ada/Blackwell optimized)
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
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from safetensors.torch import load_file, save_file

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


# =============================================================================
# Conversion Utilities
# =============================================================================

def get_unet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract only UNet-related keys from a checkpoint state dict.
    Filters out VAE, CLIP, and other non-UNet components.
    """
    unet_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "unet.",
        "model.model.",
    ]
    
    exclude_prefixes = [
        "first_stage_model.",  # VAE
        "cond_stage_model.",   # CLIP
        "conditioner.",        # CLIP (SDXL format)
        "vae.",
        "text_encoder.",
        "text_model.",
        "clip.",
    ]
    
    unet_tensors = {}
    
    for key, tensor in state_dict.items():
        if any(key.startswith(exc) for exc in exclude_prefixes):
            continue
        
        if any(key.startswith(pre) for pre in unet_prefixes):
            unet_tensors[key] = tensor
        elif "diffusion" in key.lower() or "unet" in key.lower():
            unet_tensors[key] = tensor
    
    return unet_tensors


def convert_to_precision(
    src_path: str, 
    dst_path: str, 
    precision: str, 
    strip_components: bool
) -> Tuple[float, float]:
    """
    Convert checkpoint to target precision safetensors.
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
    }
    target_dtype = dtype_map.get(precision, torch.bfloat16)
    
    print(f"[Luna] Loading {src_path}...")
    state_dict = load_file(src_path)
    original_size = os.path.getsize(src_path) / (1024 * 1024)
    
    # Filter to UNet only if stripping
    if strip_components:
        print("[Luna] Extracting UNet weights only...")
        state_dict = get_unet_keys(state_dict)
        if not state_dict:
            raise ValueError("No UNet keys found in checkpoint. Is this a valid model?")
        print(f"[Luna] Extracted {len(state_dict)} UNet tensors")
    
    # Convert precision
    print(f"[Luna] Converting to {precision}...")
    new_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            new_dict[key] = tensor.to(target_dtype)
        else:
            new_dict[key] = tensor
    
    # Save
    print(f"[Luna] Saving to {dst_path}...")
    save_file(new_dict, dst_path)
    
    converted_size = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"[Luna] Conversion complete: {original_size:.1f}MB -> {converted_size:.1f}MB")
    
    return (original_size, converted_size)


def convert_to_gguf(
    src_path: str, 
    dst_path: str, 
    quant_type: str
) -> Tuple[float, float]:
    """
    Convert checkpoint to GGUF format using Luna's internal converter
    or external ComfyUI-GGUF script.
    
    Returns:
        Tuple of (original_size_mb, converted_size_mb)
    """
    original_size = os.path.getsize(src_path) / (1024 * 1024)
    
    # Try using Luna's internal GGUF converter first
    try:
        from .luna_gguf_converter import LunaGGUFConverter
        
        output_dir = os.path.dirname(dst_path)
        output_name = os.path.splitext(os.path.basename(dst_path))[0]
        
        # Map quant type to converter format
        quant_map = {
            "Q8_0": "Q8_0 (recommended for Ampere/Ada)",
            "Q4_0": "Q4_0 (smaller, Blackwell optimized)",
            "Q4_K_M": "Q4_K_M (best quality Q4, Blackwell)",
        }
        quant_option = quant_map.get(quant_type, quant_map["Q8_0"])
        
        converter = LunaGGUFConverter()
        result = converter.convert(
            src_path, output_dir, quant_option,
            output_filename=output_name,
            extract_unet_only=True
        )
        
        converted_size = os.path.getsize(dst_path) / (1024 * 1024)
        return (original_size, converted_size)
        
    except ImportError:
        pass
    
    # Fall back to external ComfyUI-GGUF script
    converter_paths = [
        os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-GGUF", "tools", "convert.py"),
        os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI_GGUF", "tools", "convert.py"),
    ]
    
    converter_script = None
    for path in converter_paths:
        if os.path.exists(path):
            converter_script = path
            break
    
    if converter_script is None:
        raise FileNotFoundError(
            "GGUF converter not found. Install ComfyUI-GGUF or ensure Luna GGUF converter is available."
        )
    
    import subprocess
    
    cmd = [
        sys.executable, converter_script,
        "--in", src_path,
        "--out", dst_path,
        "--out-type", quant_type.lower(),
        "--unet-only"
    ]
    
    print(f"[Luna] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    converted_size = os.path.getsize(dst_path) / (1024 * 1024)
    return (original_size, converted_size)


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
    PRECISION_OPTIONS = [
        "bf16 (universal, fast)",
        "fp8_e4m3fn (Ada/Blackwell native)",
        "gguf_Q8_0 (Ampere INT8 native)",
        "gguf_Q4_K_M (Blackwell INT4 native)",
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
            # Check if CLIP (1) or VAE (2) outputs are connected
            graph = dynprompt.get_graph()
            node = graph.get_node(unique_id)
            
            if node is not None:
                # Check downstream connections for each output
                for output_idx in [1, 2]:  # CLIP=1, VAE=2
                    if self._is_output_connected(graph, unique_id, output_idx):
                        needed.append(output_idx)
        else:
            # No graph info - assume all needed
            needed = [0, 1, 2, 3]
        
        return needed
    
    def _is_output_connected(self, graph, node_id, output_idx):
        """Check if a specific output slot is connected to any downstream node."""
        try:
            # Iterate all nodes to find connections from this output
            for other_id in graph.get_nodes():
                other_node = graph.get_node(other_id)
                if other_node is None:
                    continue
                
                inputs = other_node.get("inputs", {})
                for input_name, input_val in inputs.items():
                    # Input connections are [node_id, output_index]
                    if isinstance(input_val, list) and len(input_val) >= 2:
                        if str(input_val[0]) == str(node_id) and input_val[1] == output_idx:
                            return True
            return False
        except:
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
        
        # 2. Resolve source checkpoint path (HDD)
        src_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not src_path or not os.path.exists(src_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")
        
        # 3. Local weights directory (NVMe)
        if local_weights_dir and os.path.isdir(local_weights_dir):
            weights_root = local_weights_dir
        else:
            weights_root = os.path.join(folder_paths.models_dir, "unet", "optimized")
        os.makedirs(weights_root, exist_ok=True)
        
        # 4. Build target UNet filename
        base_name = os.path.splitext(os.path.basename(ckpt_name))[0]
        
        if is_gguf:
            quant_type = precision_key.replace("gguf_", "")
            unet_filename = f"{base_name}_{quant_type}.gguf"
        else:
            unet_filename = f"{base_name}_{precision_key}_unet.safetensors"
        
        unet_path = os.path.join(weights_root, unet_filename)
        
        # 5. Convert UNet if not already done
        if not os.path.exists(unet_path):
            print(f"[Luna] First use - extracting and converting UNet...")
            print(f"[Luna] Source: {src_path}")
            print(f"[Luna] Target: {unet_path}")
            
            if is_gguf:
                quant_type = precision_key.replace("gguf_", "")
                convert_to_gguf(src_path, unet_path, quant_type)
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
            weights_root = os.path.join(folder_paths.models_dir, "unet", "optimized")
        
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
