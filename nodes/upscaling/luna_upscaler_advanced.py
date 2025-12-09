import torch
import comfy.utils
import comfy.model_management
import folder_paths
import numpy as np
from PIL import Image
import math
import os
import torch.nn.functional as F
import logging
from spandrel import ImageModelDescriptor # THE TRUE AND FINAL SUMMONING RITE
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

class Luna_Advanced_Upscaler:
    luna_tiling_orchestrator = None
    OUTPUT_NODE = True # Our default state, for the City Planner.

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
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
        
        # Add TensorRT engine path if available
        if TENSORRT_AVAILABLE:
            inputs["optional"] = {
                "tensorrt_engine_path": ("STRING", {"default": "", "tooltip": "Path to TensorRT .engine file for faster upscaling. If provided, upscale_model is ignored."}),
            }
            inputs["required"]["upscale_model"] = ("UPSCALE_MODEL",)
        else:
            inputs["required"]["upscale_model"] = ("UPSCALE_MODEL",)
            
        return inputs
    
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("upscaled_image",); FUNCTION = "upscale"; CATEGORY = "Luna/Upscaling"

    def _calculate_auto_tile_size(self, image_width: int, image_height: int, target_resolution: int):
        num_tiles_h = math.ceil(image_height / target_resolution); tile_height = math.ceil(image_height / num_tiles_h)
        num_tiles_w = math.ceil(image_width / target_resolution); tile_width = math.ceil(image_width / num_tiles_w)
        tile_width = math.ceil(tile_width / 8) * 8; tile_height = math.ceil(tile_height / 8) * 8
        return tile_width, tile_height

    def _basic_tiling_upscale(self, in_img: torch.Tensor, upscale_model, tile_x: int, tile_y: int, tile_overlap: int):
        """Basic tiling implementation as fallback when luna_tiling_orchestrator is not available"""
        with torch.inference_mode():  # Disable gradient tracking
            batch_size, channels, height, width = in_img.shape
            
            # Calculate number of tiles
            num_tiles_x = math.ceil(width / (tile_x - tile_overlap))
            num_tiles_y = math.ceil(height / (tile_y - tile_overlap))
            
            # Estimate output size and use CPU canvas only for massive outputs (>8GB)
            output_bytes = batch_size * channels * height * width * upscale_model.scale * upscale_model.scale * 4
            output_gb = output_bytes / (1024**3)
            canvas_device = 'cpu' if output_gb > 8.0 else in_img.device
            
            # Prepare output tensor
            output = torch.zeros((batch_size, channels, height * upscale_model.scale, width * upscale_model.scale), 
                               dtype=in_img.dtype, device=canvas_device)
        
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate tile coordinates with overlap
                x_start = x * (tile_x - tile_overlap)
                y_start = y * (tile_y - tile_overlap)
                x_end = min(x_start + tile_x, width)
                y_end = min(y_start + tile_y, height)
                
                # Extract tile
                tile = in_img[:, :, y_start:y_end, x_start:x_end]
                
                # Pad tile if necessary to match tile size
                pad_x = tile_x - (x_end - x_start)
                pad_y = tile_y - (y_end - y_start)
                if pad_x > 0 or pad_y > 0:
                    tile = F.pad(tile, (0, pad_x, 0, pad_y), mode='replicate')
                
                # Upscale tile
                upscaled_tile = upscale_model(tile)
                
                # Remove padding from upscaled tile
                if pad_x > 0 or pad_y > 0:
                    upscaled_tile = upscaled_tile[:, :, :upscaled_tile.shape[2]-pad_y*upscale_model.scale, 
                                                :upscaled_tile.shape[3]-pad_x*upscale_model.scale]
                
                # Calculate output coordinates
                out_x_start = x_start * upscale_model.scale
                out_y_start = y_start * upscale_model.scale
                out_x_end = out_x_start + upscaled_tile.shape[3]
                out_y_end = out_y_start + upscaled_tile.shape[2]
                
                # Move tile to canvas device and blend with existing content if there's overlap
                upscaled_tile_canvas = upscaled_tile.to(canvas_device)
                
                if x > 0 or y > 0:
                    # Simple average blending for overlapping regions
                    overlap_region = output[:, :, out_y_start:out_y_end, out_x_start:out_x_end]
                    if overlap_region.shape[2] > 0 and overlap_region.shape[3] > 0:
                        blend_mask = torch.ones_like(upscaled_tile_canvas)
                        # Create gradient blend mask for overlap
                        if tile_overlap > 0:
                            blend_width = tile_overlap * upscale_model.scale
                            if x > 0:  # Left overlap
                                blend_mask[:, :, :, :blend_width] = torch.linspace(0.5, 1.0, blend_width, device=canvas_device).unsqueeze(0).unsqueeze(0).unsqueeze(2)
                            if y > 0:  # Top overlap
                                blend_mask[:, :, :blend_width, :] = torch.linspace(0.5, 1.0, blend_width, device=canvas_device).unsqueeze(0).unsqueeze(0).unsqueeze(3)
                        
                        output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                            overlap_region * (1 - blend_mask) + upscaled_tile_canvas * blend_mask
                        del blend_mask
                    else:
                        output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = upscaled_tile_canvas
                else:
                    output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = upscaled_tile_canvas
                
                # Clean up GPU memory after each tile
                del tile, upscaled_tile, upscaled_tile_canvas
                if in_img.device != 'cpu':
                    torch.cuda.empty_cache()
            
            # Move final canvas back to original device if needed
            if canvas_device != in_img.device:
                output = output.to(in_img.device)
            
            return output

    def _adaptive_resize(self, tensor, target_height, target_width, resampling, use_antialias):
        if resampling == 'lanczos': resampling_mode = 'bicubic'
        elif resampling == 'nearest-exact': resampling_mode = 'nearest'
        else: resampling_mode = resampling

        try:
            return F.interpolate(tensor, size=(target_height, target_width), mode=resampling_mode, antialias=bool(use_antialias))
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

    def upscale(self, image: torch.Tensor, scale_by: float, resampling: str, supersample: bool, rescale_after_model: bool, tile_strategy: str, tile_mode: str, tile_resolution: int, tile_overlap: int, show_preview: bool, upscale_model=None, tensorrt_engine_path=None):
        with torch.inference_mode():  # Disable gradient tracking for entire upscale
            self.OUTPUT_NODE = show_preview
            
            device = comfy.model_management.get_torch_device()
            
            # Determine whether to use TensorRT or spandrel model
            use_tensorrt = TENSORRT_AVAILABLE and tensorrt_engine_path and tensorrt_engine_path.strip()
            
            in_img = image.permute(0, 3, 1, 2).to(device)
        
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
            if TENSORRT_AVAILABLE:
                trt_engine = Engine(tensorrt_engine_path)  # type: ignore
                trt_engine.load()
                trt_engine.activate()
                
                feed_dict = {'input_image': in_img}
                with torch.cuda.stream(torch.cuda.current_stream()):
                    result = trt_engine.infer(feed_dict, torch.cuda.current_stream())
                s = result['output_image'].cpu()
            else:
                raise ValueError("TensorRT is not available but tensorrt_engine_path was provided")
            
        else:
            # Use spandrel model (original logic)
            if upscale_model is None:
                raise ValueError("Either upscale_model or tensorrt_engine_path must be provided")
            
            # Estimate input size to determine if we need aggressive memory management
            input_bytes = in_img.numel() * in_img.element_size()
            input_mb = input_bytes / (1024**2)
            large_image = input_mb > 500  # >500MB input (~2K images)
            
            upscale_model.to(device)
            
            if tile_strategy == "none":
                s = upscale_model(in_img)
            else:
                if tile_mode == 'auto': tile_x, tile_y = self._calculate_auto_tile_size(in_img.shape[3], in_img.shape[2], tile_resolution)
                else: tile_x, tile_y = tile_resolution, tile_resolution
                
                # For very large images, offload model between tiles to maximize VRAM for canvas
                if large_image:
                    logging.info(f"[Luna Upscaler] Large image detected ({input_mb:.1f}MB), using memory-efficient tiling")
                
                # Use luna_tiling_orchestrator if available, otherwise fallback to basic tiling
                if self.luna_tiling_orchestrator is not None:
                    s = self.luna_tiling_orchestrator(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
                else:
                    # Basic tiling fallback
                    s = self._basic_tiling_upscale(in_img, upscale_model, tile_x, tile_y, tile_overlap)
            
            upscale_model.to("cpu")
            if device != 'cpu':
                torch.cuda.empty_cache()
        
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
                s_resized = F.interpolate(s_for_resizing, size=(target_height, target_width), mode=resampling_mode, antialias=bool(use_antialias))
        
            s_final_for_comfy = s_resized.permute(0, 2, 3, 1)

            return (s_final_for_comfy,)

NODE_CLASS_MAPPINGS = {"Luna_Advanced_Upscaler": Luna_Advanced_Upscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna_Advanced_Upscaler": "Luna Advanced Upscaler"}