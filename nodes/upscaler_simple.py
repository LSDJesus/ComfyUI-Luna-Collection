import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import os

class ImageModelDescriptor: pass

class Pyrite_SimpleUpscaler:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",),"upscale_model": ("UPSCALE_MODEL",),"scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),"resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),"show_preview": ("BOOLEAN", {"default": True}),}}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("upscaled_image",); FUNCTION = "upscale"; CATEGORY = "Pyrite Core"
    
    def upscale(self, image: torch.Tensor, upscale_model: ImageModelDescriptor, scale_by: float, resampling: str, show_preview: bool):
        device = comfy.model_management.get_torch_device(); upscale_model.to(device); in_img = image.movedim(-1, -3).to(device)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=512, tile_y=512, overlap=32, upscale_amount=upscale_model.scale)
        upscale_model.to("cpu"); target_width = round(image.shape[3] * scale_by); target_height = round(image.shape[2] * scale_by)
        if s.shape[3] != target_width or s.shape[2] != target_height: s = comfy.utils.common_upscale(s, target_width, target_height, resampling, "disabled")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0); s = s.to(comfy.model_management.intermediate_device())
        if show_preview:
            previews = []
            for i in range(s.shape[0]):
                preview_image = s[i].cpu().numpy(); preview_image = (preview_image * 255).astype(np.uint8); img = Image.fromarray(preview_image)
                output_dir = folder_paths.get_temp_directory(); filename = f"PyritePreview_{torch.randint(0, 2**32, (1,)).item()}.jpg"
                img.save(os.path.join(output_dir, filename), "JPEG"); previews.append({'filename': filename, 'type': 'temp'})
            return {"ui": {"images": previews}, "result": (s,)}
        else: return (s,)

NODE_CLASS_MAPPINGS = {"Pyrite_SimpleUpscaler": Pyrite_SimpleUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler"}