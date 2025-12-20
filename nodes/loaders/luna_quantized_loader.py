"""
Luna Quantized Model Loaders

Provides loaders for INT8 and NF4 quantized UNet weights.
These are created by the Luna converters and need special dequantization on load.

Supported formats:
- INT8: 8-bit integer quantization (stored as torch.int8)
- NF4: 4-bit NormalFloat quantization (via BitsAndBytes)
"""

import os
from typing import Tuple
import torch
from safetensors.torch import load_file

# Try to import bitsandbytes for NF4 support
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import comfy.sd
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class LunaINT8Loader:
    """
    Load INT8 quantized UNet weights and dequantize to float.
    
    INT8 is stored as torch.int8 (-128 to 127 range).
    Scale factors are computed during quantization and need to be recomputed
    from the min/max of each tensor or provided externally.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "target_dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_int8_unet"
    CATEGORY = "loaders/Luna"
    
    def load_int8_unet(self, unet_path: str, target_dtype: str = "auto") -> Tuple:
        """Load INT8 quantized UNet and dequantize."""
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet not found: {unet_path}")
        
        if not unet_path.endswith('.safetensors'):
            raise ValueError(f"Expected .safetensors file, got: {unet_path}")
        
        print(f"[LunaINT8Loader] Loading INT8 UNet: {os.path.basename(unet_path)}")
        
        # Load quantized state dict
        state_dict = load_file(unet_path)
        
        # Dequantize INT8 tensors
        dequant_dict = {}
        total_tensors = len(state_dict)
        
        for idx, (key, tensor) in enumerate(state_dict.items(), 1):
            if idx % 100 == 0:
                print(f"[LunaINT8Loader] Dequantizing {idx}/{total_tensors}...")
            
            if tensor.dtype == torch.int8:
                # Dequantize: for INT8, find the original scale by getting abs_max
                # This is approximate - ideally the scale would be stored in metadata
                tensor_float = tensor.float()
                
                # Find approximate scale from max value
                # Original quantization: tensor_scaled = tensor / scale, quantized = (tensor_scaled * 127).int8
                # So: tensor ≈ (quantized.float() / 127) * original_max
                abs_max_quantized = tensor.abs().float().max()
                if abs_max_quantized > 0:
                    # Assume the original range was roughly [-1, 1] normalized
                    # Use 127 as the max int8 value
                    scale = abs_max_quantized / 127.0
                    tensor_float = tensor_float / 127.0
                else:
                    tensor_float = torch.zeros_like(tensor_float)
            else:
                tensor_float = tensor.to(torch.float32)
            
            # Convert to target dtype if needed
            if target_dtype == "auto":
                # Keep as-is or use float32
                dequant_dict[key] = tensor_float
            else:
                target_torch_dtype = getattr(torch, target_dtype)
                dequant_dict[key] = tensor_float.to(target_torch_dtype)
        
        # Load as UNet via ComfyUI
        print(f"[LunaINT8Loader] Loading dequantized UNet into ComfyUI...")
        if not HAS_COMFY:
            raise ImportError("ComfyUI required for UNet loading")
        
        model = comfy.sd.load_diffusion_model_state_dict(dequant_dict)
        if model is None:
            raise RuntimeError("Could not load UNet model")
        
        print(f"[LunaINT8Loader] ✓ INT8 UNet loaded and dequantized")
        return (model,)


class LunaNF4Loader:
    """
    Load NF4 quantized UNet weights via BitsAndBytes.
    
    NF4 is BitsAndBytes' 4-bit NormalFloat format, optimized for neural network weights.
    Requires bitsandbytes library.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_nf4_unet"
    CATEGORY = "loaders/Luna"
    
    def load_nf4_unet(self, unet_path: str) -> Tuple:
        """Load NF4 quantized UNet."""
        if not HAS_BNB:
            raise ImportError(
                "BitsAndBytes required for NF4 loading.\n"
                "Install with: pip install bitsandbytes"
            )
        
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet not found: {unet_path}")
        
        if not unet_path.endswith('.safetensors'):
            raise ValueError(f"Expected .safetensors file, got: {unet_path}")
        
        print(f"[LunaNF4Loader] Loading NF4 UNet: {os.path.basename(unet_path)}")
        
        # Load NF4 state dict
        state_dict = load_file(unet_path)
        
        # Dequantize NF4 tensors
        dequant_dict = {}
        total_tensors = len(state_dict)
        
        # NF4 quantization bins
        nf4_code = torch.tensor([
            -1.0, -0.6961928, -0.5250730, -0.39625454, -0.28530699, -0.18396355,
            -0.09618758, 0.0, 0.09618758, 0.18396355, 0.28530699, 0.39625454,
            0.5250730, 0.6961928, 1.0, 1.0
        ], dtype=torch.float32)
        
        for idx, (key, tensor) in enumerate(state_dict.items(), 1):
            if idx % 100 == 0:
                print(f"[LunaNF4Loader] Dequantizing {idx}/{total_tensors}...")
            
            if tensor.dtype == torch.uint8:
                # Dequantize from uint8 (packed 4-bit values)
                tensor_flat = tensor.view(-1)
                
                # Unpack 4-bit indices from bytes
                unpacked = []
                for byte_val in tensor_flat:
                    # Lower 4 bits
                    unpacked.append(int(byte_val & 0x0F))
                    # Upper 4 bits
                    unpacked.append(int((byte_val >> 4) & 0x0F))
                
                # Convert indices to NF4 values
                indices = torch.tensor(unpacked[:len(tensor_flat) * 2], dtype=torch.long)
                dequant = nf4_code[indices].to(torch.float32)
                dequant_dict[key] = dequant
            else:
                # Already float (shouldn't happen with proper NF4 conversion)
                dequant_dict[key] = tensor.to(torch.float32)
        
        # Load as UNet via ComfyUI
        print(f"[LunaNF4Loader] Loading dequantized UNet into ComfyUI...")
        if not HAS_COMFY:
            raise ImportError("ComfyUI required for UNet loading")
        
        model = comfy.sd.load_diffusion_model_state_dict(dequant_dict)
        if model is None:
            raise RuntimeError("Could not load UNet model")
        
        print(f"[LunaNF4Loader] ✓ NF4 UNet loaded and dequantized")
        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaINT8Loader": LunaINT8Loader,
    "LunaNF4Loader": LunaNF4Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaINT8Loader": "Luna INT8 Loader",
    "LunaNF4Loader": "Luna NF4 Loader",
}
