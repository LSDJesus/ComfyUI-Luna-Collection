import torch
import comfy.utils
import comfy.model_management
import folder_paths
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from spandrel import ImageModelDescriptor
import sys

# Add the parent directory to sys.path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils modules directly
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from trt_engine import Engine  # type: ignore
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

class Luna_SimpleUpscaler:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["bicubic", "bilinear"],), # Limited to modes supported by F.interpolate
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }
        
        # Add TensorRT engine path if available
        if TENSORRT_AVAILABLE:
            inputs["optional"] = {
                "tensorrt_engine_path": ("STRING", {"default": "", "tooltip": "Path to TensorRT .engine file for faster upscaling. If provided, upscale_model is ignored."}),
            }
            inputs["required"]["upscale_model"] = ("UPSCALE_MODEL",)
        else:
            inputs["required"]["upscale_model"] = ("UPSCALE_MODEL",)
            
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Luna Collection"
    
    def upscale(self, image: torch.Tensor, scale_by: float, resampling: str, show_preview: bool, upscale_model=None, tensorrt_engine_path=None):
        self.OUTPUT_NODE = show_preview
        
        device = comfy.model_management.get_torch_device()
        
        # Determine whether to use TensorRT or spandrel model
        use_tensorrt = TENSORRT_AVAILABLE and tensorrt_engine_path and tensorrt_engine_path.strip()
        
        if use_tensorrt:
            # Use TensorRT engine
            if tensorrt_engine_path and not os.path.isabs(tensorrt_engine_path):
                # Try common TensorRT upscale model locations
                possible_paths = [
                    os.path.join("models", "tensorrt", "upscale_models", tensorrt_engine_path),
                    os.path.join("models", "upscale_models", tensorrt_engine_path),
                    os.path.join("models", "tensorrt", tensorrt_engine_path),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        tensorrt_engine_path = path
                        break
            
            # Load and use TensorRT engine
            trt_engine = Engine(tensorrt_engine_path)  # type: ignore
            trt_engine.load()
            trt_engine.activate()
            
            in_img = image.permute(0, 3, 1, 2).to(device)
            feed_dict = {'input_image': in_img}
            with torch.cuda.stream(torch.cuda.current_stream()):
                result = trt_engine.infer(feed_dict, torch.cuda.current_stream())
            s = result['output_image'].cpu()
            
        else:
            # Use spandrel model (original logic)
            if upscale_model is None:
                raise ValueError("Either upscale_model or tensorrt_engine_path must be provided")
                
            upscale_model.to(device)
            in_img = image.permute(0, 3, 1, 2).to(device)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=512, tile_y=512, overlap=32, upscale_amount=upscale_model.scale)
            upscale_model.to("cpu")
        
        target_width = round(image.shape[2] * scale_by)
        target_height = round(image.shape[1] * scale_by)
        
        s_resized = s
        if s.shape[2] != target_height or s.shape[3] != target_width:
            # We use the GPU-native resize, the true path.
            s_resized = F.interpolate(s, size=(target_height, target_width), mode=resampling, antialias=True)
        
        # And we return the tensor in the form the gods demand.
        s_final_for_comfy = s_resized.permute(0, 2, 3, 1)

        return (s_final_for_comfy,)

NODE_CLASS_MAPPINGS = {"Luna_SimpleUpscaler": Luna_SimpleUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna_SimpleUpscaler": "Luna Simple Upscaler"}