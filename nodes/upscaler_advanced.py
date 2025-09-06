import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import math
import os
import torch.nn.functional as F
import logging
from spandrel import ImageModelDescriptor # THE TRUE AND FINAL SUMMONING RITE
from .utils.tiling import luna_tiling_orchestrator

class Luna_Advanced_Upscaler:
    luna_tiling_orchestrator = None
    OUTPUT_NODE = True # Our default state, for the City Planner.

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["bicubic", "bilinear", "lanczos", "nearest-exact", "area"],),
                "supersample": ("BOOLEAN", {"default": True}),
                "rescale_after_model": ("BOOLEAN", {"default": True}),
                "tile_strategy": (["linear", "chess", "none"],),
                "tile_mode": (["default", "auto"],),
                "tile_resolution": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("upscaled_image",); FUNCTION = "upscale"; CATEGORY = "Luna Collection"

    def _calculate_auto_tile_size(self, image_width: int, image_height: int, target_resolution: int):
        num_tiles_h = math.ceil(image_height / target_resolution); tile_height = math.ceil(image_height / num_tiles_h)
        num_tiles_w = math.ceil(image_width / target_resolution); tile_width = math.ceil(image_width / num_tiles_w)
        tile_width = math.ceil(tile_width / 8) * 8; tile_height = math.ceil(tile_height / 8) * 8
        return tile_width, tile_height

    def _adaptive_resize(self, tensor, target_height, target_width, resampling, use_antialias):
        if resampling == 'lanczos': resampling_mode = 'bicubic'
        elif resampling == 'nearest-exact': resampling_mode = 'nearest'
        else: resampling_mode = resampling

        try:
            return F.interpolate(tensor, size=(target_height, target_width), mode=resampling_mode, antialias=use_antialias if use_antialias else None)
        except RuntimeError as e:
            if "Too much shared memory required" in str(e):
                original_height, original_width = tensor.shape[2], tensor.shape[3]
                if original_height <= 1 or original_width <= 1:
                    logging.warning(f"[Lunacollection] GPU resize failed on irreducible tensor {original_height}x{original_width}. Offloading to CPU.")
                    tensor_for_cpu = tensor.permute(0, 2, 3, 1).squeeze(0)
                    pil_img = Image.fromarray((tensor_for_cpu.cpu().numpy() * 255).astype(np.uint8))
                    resample_filters = {'bicubic': Image.Resampling.BICUBIC, 'bilinear': Image.Resampling.BILINEAR, 'lanczos': Image.Resampling.LANCZOS, 'nearest-exact': Image.Resampling.NEAREST}
                    cpu_resample_method = resample_filters.get(resampling, Image.Resampling.BICUBIC)
                    resized_pil = pil_img.resize((target_width, target_height), resample=cpu_resample_method)
                    resized_tensor = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0).unsqueeze(0)
                    return resized_tensor.permute(0, 3, 1, 2).to(tensor.device)
                mid_h, mid_w = original_height // 2, original_width // 2
                target_mid_h = math.floor(mid_h * (target_height / original_height))
                target_rem_h = target_height - target_mid_h
                target_mid_w = math.floor(mid_w * (target_width / original_width))
                target_rem_w = target_width - target_mid_w
                top_left     = self._adaptive_resize(tensor[:, :, :mid_h, :mid_w], target_mid_h, target_mid_w, resampling, use_antialias)
                top_right    = self._adaptive_resize(tensor[:, :, :mid_h, mid_w:], target_mid_h, target_rem_w, resampling, use_antialias)
                bottom_left  = self._adaptive_resize(tensor[:, :, mid_h:, :mid_w], target_rem_h, target_mid_w, resampling, use_antialias)
                bottom_right = self._adaptive_resize(tensor[:, :, mid_h:, mid_w:], target_rem_h, target_rem_w, resampling, use_antialias)
                top_half = torch.cat((top_left, top_right), dim=3)
                bottom_half = torch.cat((bottom_left, bottom_right), dim=3)
                return torch.cat((top_half, bottom_half), dim=2)
            else:
                raise e

    def upscale(self, image: torch.Tensor, upscale_model: ImageModelDescriptor, scale_by: float, resampling: str, tile_strategy: str, tile_mode: str, tile_resolution: int, tile_overlap: int, supersample: bool, rescale_after_model: bool, show_preview: bool):
        self.OUTPUT_NODE = show_preview
        
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        
        in_img = image.permute(0, 3, 1, 2).to(device)
        
        if tile_strategy == "none":
            s = upscale_model(in_img)
        else:
            if tile_mode == 'auto': tile_x, tile_y = self._calculate_auto_tile_size(in_img.shape[3], in_img.shape[2], tile_resolution)
            else: tile_x, tile_y = tile_resolution, tile_resolution
            s = self.luna_tiling_orchestrator(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
        
        upscale_model.to("cpu")
        s_for_resizing = s

        target_height = round(image.shape[1] * scale_by)
        target_width = round(image.shape[2] * scale_by)

        s_resized = s_for_resizing
        if rescale_after_model and (s_for_resizing.shape[2] != target_height or s_for_resizing.shape[3] != target_width):
            use_antialias = resampling in ['bicubic', 'bilinear']

            if supersample:
                s_resized = self._adaptive_resize(s_for_resizing, target_height, target_width, resampling, use_antialias)
            else:
                if resampling == 'lanczos': resampling_mode = 'bicubic'
                elif resampling == 'nearest-exact': resampling_mode = 'nearest'
                else: resampling_mode = resampling
                s_resized = F.interpolate(s_for_resizing, size=(target_height, target_width), mode=resampling_mode, antialias=use_antialias if use_antialias else None)
        
        s_final_for_comfy = s_resized.permute(0, 2, 3, 1)

        return (s_final_for_comfy,)

NODE_CLASS_MAPPINGS = {"Luna_Advanced_Upscaler": Luna_Advanced_Upscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna_Advanced_Upscaler": "Luna Advanced Upscaler"}