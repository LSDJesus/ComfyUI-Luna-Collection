import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import math
import os

class ImageModelDescriptor:
    pass

class Pyrite_SimpleUpscaler:
    # ... (This class remains completely unchanged) ...
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
            previews = [];
            for i in range(s.shape[0]):
                preview_image = s[i].cpu().numpy(); preview_image = (preview_image * 255).astype(np.uint8); img = Image.fromarray(preview_image)
                output_dir = folder_paths.get_temp_directory(); filename = f"PyritePreview_{torch.randint(0, 2**32, (1,)).item()}.jpg"
                img.save(os.path.join(output_dir, filename), "JPEG"); previews.append({'filename': filename, 'type': 'temp'})
            return {"ui": {"images": previews}, "result": (s,)}
        else: return (s,)


class Pyrite_AdvancedUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),
                "supersample": ("BOOLEAN", {"default": False}),
                "rescale_after_model": ("BOOLEAN", {"default": True}),
                "tile_strategy": (["linear", "chess", "none"],),
                "tile_mode": (["default", "auto"],),
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
        
        # --- THE NEW ENGINE ---
        if tile_strategy == "none":
            s = upscale_model(in_img)
        else:
            if tile_mode == 'default':
                tile_x, tile_y = 512, 512
            else:
                tile_x, tile_y = self._calculate_auto_tile_size(in_img.shape[3], in_img.shape[2], tile_resolution)
            
            # We now use our new custom tiling orchestrator, stolen from the giants.
            s = self._pyrite_tiled_upscale(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
        
        upscale_model.to("cpu")

        target_width = round(image.shape[3] * scale_by); target_height = round(image.shape[2] * scale_by)
        if supersample: s = comfy.utils.common_upscale(s, target_width, target_height, "lanczos", "disabled")
        else:
            if rescale_after_model and (s.shape[3] != target_width or s.shape[2] != target_height):
                s = comfy.utils.common_upscale(s, target_width, target_height, resampling, "disabled")

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0); s = s.to(comfy.model_management.intermediate_device())
        if show_preview:
            previews = [];
            for i in range(s.shape[0]):
                preview_image = s[i].cpu().numpy(); preview_image = (preview_image * 255).astype(np.uint8); img = Image.fromarray(preview_image)
                output_dir = folder_paths.get_temp_directory(); filename = f"PyritePreview_{torch.randint(0, 2**32, (1,)).item()}.jpg"
                img.save(os.path.join(output_dir, filename), "JPEG"); previews.append({'filename': filename, 'type': 'temp'})
            return {"ui": {"images": previews}, "result": (s,)}
        else: return (s,)

    def _pyrite_tiled_upscale(self, image_tensor, model, tile_x, tile_y, overlap, strategy):
        # A simplified, ruthlessly effective implementation of the giant's logic.
        
        # 1. Prepare the canvas.
        height, width = image_tensor.shape[2], image_tensor.shape[3]
        scale = model.scale
        new_height, new_width = int(height * scale), int(width * scale)
        
        # Use the device of the input tensor.
        device = image_tensor.device
        
        # Create a blank tensor on the specified device.
        canvas = torch.zeros((image_tensor.shape[0], image_tensor.shape[1], new_height, new_width), device=device)
        
        # 2. Generate the tiling grids.
        rows = math.ceil(height / tile_y)
        cols = math.ceil(width / tile_x)
        
        pass_one_tiles = []
        pass_two_tiles = []

        for r in range(rows):
            for c in range(cols):
                tile = (c * tile_x, r * tile_y, tile_x, tile_y)
                if strategy == 'chess' and (r + c) % 2 == 1:
                    pass_two_tiles.append(tile)
                else:
                    pass_one_tiles.append(tile)
        
        # 3. Process the passes.
        for tile_list in [pass_one_tiles, pass_two_tiles]:
            if not tile_list: continue # Skip if a pass has no tiles (e.g., linear strategy)
            
            for x, y, tile_w, tile_h in tile_list:
                # Add overlap for processing, clamping to image boundaries.
                y_start = max(0, y - overlap)
                x_start = max(0, x - overlap)
                y_end = min(height, y + tile_h + overlap)
                x_end = min(width, x + tile_w + overlap)

                # Crop the input tensor.
                tile_input = image_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # Upscale the tile.
                upscaled_tile = model(tile_input)

                # Paste the upscaled tile onto the canvas.
                # We calculate the paste coordinates, accounting for the overlap we added.
                paste_y = y * scale
                paste_x = x * scale
                
                # Calculate the crop region from the upscaled tile to remove the overlap.
                crop_y_start = (y - y_start) * scale
                crop_x_start = (x - x_start) * scale
                crop_y_end = crop_y_start + (tile_h * scale)
                crop_x_end = crop_x_start + (tile_w * scale)
                
                # Ensure the cropped region does not exceed the canvas size.
                paste_h = min(new_height - paste_y, tile_h * scale)
                paste_w = min(new_width - paste_x, tile_w * scale)
                
                canvas[:, :, paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = upscaled_tile[:, :, crop_y_start:crop_y_start+paste_h, crop_x_start:crop_x_start+paste_w]
        
        return canvas

NODE_CLASS_MAPPINGS = { "Pyrite_SimpleUpscaler": Pyrite_SimpleUpscaler, "Pyrite_AdvancedUpscaler": Pyrite_AdvancedUpscaler }
NODE_DISPLAY_NAME_MAPPINGS = { "Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler", "Pyrite_AdvancedUpscaler": "Pyrite Advanced Upscaler" }