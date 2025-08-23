import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import math
import os
# --- FIX: The arrogant relative import is dead. Long live the confident absolute import.
from utils.tiling import pyrite_tiling_orchestrator

class ImageModelDescriptor: pass

class Pyrite_AdvancedUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),
                "supersample": ("BOOLEAN", {"default": False}), "rescale_after_model": ("BOOLEAN", {"default": True}),
                "tile_strategy": (["linear", "chess", "none"],), "tile_mode": (["default", "auto"],),
                "tile_resolution": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("upscaled_image",); FUNCTION = "upscale"; CATEGORY = "Pyrite Core"

    def _calculate_auto_tile_size(self, image_width: int, image_height: int, target_resolution: int):
        num_tiles_h = math.ceil(image_height / target_resolution); tile_height = math.ceil(image_height / num_tiles_h)
        num_tiles_w = math.ceil(image_width / target_resolution); tile_width = math.ceil(image_width / num_tiles_w)
        tile_width = math.ceil(tile_width / 8) * 8; tile_height = math.ceil(tile_height / 8) * 8
        return tile_width, tile_height

    def upscale(self, image: torch.Tensor, upscale_model: ImageModelDescriptor, scale_by: float, resampling: str, tile_strategy: str, tile_mode: str, tile_resolution: int, tile_overlap: int, supersample: bool, rescale_after_model: bool, show_preview: bool):
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        
        if tile_strategy == "none": s = upscale_model(in_img)
        else:
            if tile_mode == 'default': tile_x, tile_y = 512, 512
            else: tile_x, tile_y = self._calculate_auto_tile_size(in_img.shape[3], in_img.shape[2], tile_resolution)
            s = pyrite_tiling_orchestrator(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
        
        upscale_model.to("cpu")

        target_width = round(image.shape[3] * scale_by); target_height = round(image.shape[2] * scale_by)
        if supersample: s = comfy.utils.common_upscale(s, target_width, target_height, "lanczos", "disabled")
        else:
            if rescale_after_model and (s.shape[3] != target_width or s.shape[2] != target_height):
                s = comfy.utils.common_upscale(s, target_width, target_height, resampling, "disabled")

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0); s = s.to(comfy.model_management.intermediate_device())
        if show_preview:
            previews = []
            for i in range(s.shape[0]):
                preview_image = s[i].cpu().numpy(); preview_image = (preview_image * 255).astype(np.uint8); img = Image.fromarray(preview_image)
                output_dir = folder_paths.get_temp_directory(); filename = f"PyritePreview_{torch.randint(0, 2**32, (1,)).item()}.jpg"
                img.save(os.path.join(output_dir, filename), "JPEG"); previews.append({'filename': filename, 'type': 'temp'})
            return {"ui": {"images": previews}, "result": (s,)}
        else: return (s,)

NODE_CLASS_MAPPINGS = {"Pyrite_AdvancedUpscaler": Pyrite_AdvancedUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Pyrite_AdvancedUpscaler": "Pyrite Advanced Upscaler"}