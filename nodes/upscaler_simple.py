import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from spandrel import ImageModelDescriptor

class Luna_SimpleUpscaler:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["bicubic", "bilinear"],), # Limited to modes supported by F.interpolate
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Luna Collection"
    
    def upscale(self, image: torch.Tensor, upscale_model: ImageModelDescriptor, scale_by: float, resampling: str, show_preview: bool):
        self.OUTPUT_NODE = show_preview
        
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        
        # We now speak the beautiful, terrible language of the tensors.
        in_img = image.permute(0, 3, 1, 2).to(device)
        
        # We use a simple, hardcoded tiling for our simple node.
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