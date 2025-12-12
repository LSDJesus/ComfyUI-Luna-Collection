import torch
import comfy.model_management
import numpy as np
from PIL import Image
import math
import os
import torch.nn.functional as F
import logging
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

# Import ComfyUI's default upscaler node for delegation
try:
    import nodes  # type: ignore
    UpscaleUsingSampler = getattr(nodes, 'UpscaleUsingSampler', None)  # type: ignore
    COMFYUI_UPSCALER_AVAILABLE = True
except ImportError:
    COMFYUI_UPSCALER_AVAILABLE = False

class Luna_Advanced_Upscaler:
    """
    Luna Advanced Upscaler - Wraps ComfyUI's upscaler with additional features.
    
    Delegates core upscaling to ComfyUI's proven implementation, adds:
    - Configurable tiling strategies (linear, chess)
    - Auto tile size calculation
    - Supersampling with adaptive resize
    - TensorRT acceleration (optional)
    - Advanced memory management
    """
    
    luna_tiling_orchestrator = None
    OUTPUT_NODE = True

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
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Luna/Upscaling"

    def _calculate_auto_tile_size(self, image_width: int, image_height: int, target_resolution: int):
        """Calculate optimal tile size for given image resolution"""
        num_tiles_h = math.ceil(image_height / target_resolution)
        tile_height = math.ceil(image_height / num_tiles_h)
        num_tiles_w = math.ceil(image_width / target_resolution)
        tile_width = math.ceil(image_width / num_tiles_w)
        # Align to 8 for model compatibility
        tile_width = math.ceil(tile_width / 8) * 8
        tile_height = math.ceil(tile_height / 8) * 8
        return tile_width, tile_height

    def _upscale_with_model(self, in_img: torch.Tensor, upscale_model) -> torch.Tensor:
        """Core upscaling using the provided model (delegate to ComfyUI's approach)"""
        with torch.inference_mode():
            # ComfyUI pattern: move model to device, run inference, move to CPU
            device = in_img.device
            upscale_model.to(device)
            upscaled = upscale_model(in_img)
            upscale_model.to("cpu")
            if device != 'cpu':
                torch.cuda.empty_cache()
            return upscaled

    def _tiled_upscale_with_model(self, in_img: torch.Tensor, upscale_model, tile_x: int, tile_y: int, 
                                  tile_overlap: int, strategy: str) -> torch.Tensor:
        """Upscale with tiling strategy for large images"""
        with torch.inference_mode():
            device = in_img.device
            height, width = in_img.shape[2], in_img.shape[3]
            scale = upscale_model.scale
            new_height, new_width = int(height * scale), int(width * scale)
            
            # Estimate output size and use CPU canvas for massive outputs (>8GB)
            output_bytes = in_img.shape[0] * in_img.shape[1] * new_height * new_width * 4
            output_gb = output_bytes / (1024**3)
            canvas_device = 'cpu' if output_gb > 8.0 else device
            
            canvas = torch.zeros((in_img.shape[0], in_img.shape[1], new_height, new_width),
                               dtype=in_img.dtype, device=canvas_device)
            
            rows = math.ceil(height / tile_y)
            cols = math.ceil(width / tile_x)
            
            pass_one_tiles = []
            pass_two_tiles = []
            
            # Partition tiles based on strategy
            for r in range(rows):
                for c in range(cols):
                    tile = (c * tile_x, r * tile_y, tile_x, tile_y)
                    if strategy == 'chess' and (r + c) % 2 == 1:
                        pass_two_tiles.append(tile)
                    else:
                        pass_one_tiles.append(tile)
            
            # Process tiles in two passes for chess strategy
            upscale_model.to(device)
            
            for tile_list in [pass_one_tiles, pass_two_tiles]:
                if not tile_list:
                    continue
                
                for x, y, tile_w, tile_h in tile_list:
                    # Extract tile with overlap
                    y_start = max(0, y - tile_overlap)
                    x_start = max(0, x - tile_overlap)
                    y_end = min(height, y + tile_h + tile_overlap)
                    x_end = min(width, x + tile_w + tile_overlap)
                    
                    tile_input = in_img[:, :, y_start:y_end, x_start:x_end]
                    upscaled_tile = self._upscale_with_model(tile_input, upscale_model)
                    
                    # Calculate paste position
                    paste_y = y * scale
                    paste_x = x * scale
                    crop_y_start = (y - y_start) * scale
                    crop_x_start = (x - x_start) * scale
                    
                    paste_h = min(new_height - paste_y, tile_h * scale)
                    paste_w = min(new_width - paste_x, tile_w * scale)
                    
                    crop_y_end = crop_y_start + paste_h
                    crop_x_end = crop_x_start + paste_w
                    
                    # Extract and move to canvas device
                    tile_crop = upscaled_tile[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end].to(canvas_device)
                    canvas[:, :, paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = tile_crop
                    
                    # Clean up
                    del upscaled_tile, tile_input, tile_crop
                    if device != 'cpu':
                        torch.cuda.empty_cache()
            
            upscale_model.to("cpu")
            if device != 'cpu':
                torch.cuda.empty_cache()
            
            # Move final canvas back to original device if needed
            if canvas_device != device:
                canvas = canvas.to(device)
            
            return canvas

    def _adaptive_resize(self, tensor, target_height, target_width, resampling, use_antialias):
        """Adaptive resize with CPU fallback for shared memory errors"""
        if resampling == 'lanczos':
            resampling_mode = 'bicubic'
        elif resampling == 'nearest-exact':
            resampling_mode = 'nearest'
        else:
            resampling_mode = resampling

        try:
            return F.interpolate(tensor, size=(target_height, target_width), mode=resampling_mode, antialias=bool(use_antialias))
        except RuntimeError as e:
            if "Too much shared memory required" in str(e):
                original_height, original_width = tensor.shape[2], tensor.shape[3]
                if original_height <= 1 or original_width <= 1:
                    logging.warning(f"[Luna Upscaler] GPU resize failed on irreducible tensor {original_height}x{original_width}. Offloading to CPU.")
                    tensor_for_cpu = tensor.permute(0, 2, 3, 1).squeeze(0)
                    pil_img = Image.fromarray((tensor_for_cpu.cpu().numpy() * 255).astype(np.uint8))
                    resample_filters = {
                        'bicubic': Image.Resampling.BICUBIC,
                        'bilinear': Image.Resampling.BILINEAR,
                        'lanczos': Image.Resampling.LANCZOS,
                        'nearest-exact': Image.Resampling.NEAREST
                    }
                    cpu_resample_method = resample_filters.get(resampling, Image.Resampling.BICUBIC)
                    resized_pil = pil_img.resize((target_width, target_height), resample=cpu_resample_method)
                    resized_tensor = torch.from_numpy(np.array(resized_pil).astype(np.float32) / 255.0).unsqueeze(0)
                    return resized_tensor.permute(0, 3, 1, 2).to(tensor.device)
                
                # Quadrant splitting for large tensors
                mid_h, mid_w = original_height // 2, original_width // 2
                target_mid_h = math.floor(mid_h * (target_height / original_height))
                target_rem_h = target_height - target_mid_h
                target_mid_w = math.floor(mid_w * (target_width / original_width))
                target_rem_w = target_width - target_mid_w
                
                top_left = self._adaptive_resize(tensor[:, :, :mid_h, :mid_w], target_mid_h, target_mid_w, resampling, use_antialias)
                top_right = self._adaptive_resize(tensor[:, :, :mid_h, mid_w:], target_mid_h, target_rem_w, resampling, use_antialias)
                bottom_left = self._adaptive_resize(tensor[:, :, mid_h:, :mid_w], target_rem_h, target_mid_w, resampling, use_antialias)
                bottom_right = self._adaptive_resize(tensor[:, :, mid_h:, mid_w:], target_rem_h, target_rem_w, resampling, use_antialias)
                
                top_half = torch.cat((top_left, top_right), dim=3)
                bottom_half = torch.cat((bottom_left, bottom_right), dim=3)
                return torch.cat((top_half, bottom_half), dim=2)
            else:
                raise e

    def upscale(self, image: torch.Tensor, scale_by: float, resampling: str, supersample: bool, 
                rescale_after_model: bool, tile_strategy: str, tile_mode: str, tile_resolution: int, 
                tile_overlap: int, show_preview: bool, upscale_model=None, tensorrt_engine_path=None):
        with torch.inference_mode():
            self.OUTPUT_NODE = show_preview
            device = comfy.model_management.get_torch_device()
            
            # Handle TensorRT path if provided
            use_tensorrt = TENSORRT_AVAILABLE and tensorrt_engine_path and tensorrt_engine_path.strip()
            
            in_img = image.permute(0, 3, 1, 2).to(device)
            
            if use_tensorrt:
                if tensorrt_engine_path and not os.path.isabs(tensorrt_engine_path):
                    possible_paths = [
                        os.path.join("models", "tensorrt", "upscale_models", tensorrt_engine_path),
                        os.path.join("models", "upscale_models", tensorrt_engine_path),
                        os.path.join("models", "tensorrt", tensorrt_engine_path),
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            tensorrt_engine_path = path
                            break
                
                trt_engine = Engine(tensorrt_engine_path)
                trt_engine.load()
                trt_engine.activate()
                
                feed_dict = {'input_image': in_img}
                with torch.cuda.stream(torch.cuda.current_stream()):
                    result = trt_engine.infer(feed_dict, torch.cuda.current_stream())
                s = result['output_image'].cpu()
            
            else:
                # Use upscale_model with optional tiling
                if upscale_model is None:
                    raise ValueError("Either upscale_model or tensorrt_engine_path must be provided")
                
                input_bytes = in_img.numel() * in_img.element_size()
                input_mb = input_bytes / (1024**2)
                large_image = input_mb > 500  # >500MB input (~2K images)
                
                if tile_strategy == "none":
                    # Simple case: no tiling
                    s = self._upscale_with_model(in_img, upscale_model)
                else:
                    # Calculate tile size
                    if tile_mode == 'auto':
                        tile_x, tile_y = self._calculate_auto_tile_size(in_img.shape[3], in_img.shape[2], tile_resolution)
                    else:
                        tile_x, tile_y = tile_resolution, tile_resolution
                    
                    if large_image:
                        logging.info(f"[Luna Upscaler] Large image detected ({input_mb:.1f}MB), using memory-efficient tiling")
                    
                    # Use luna_tiling_orchestrator if available, otherwise use built-in tiling
                    if self.luna_tiling_orchestrator is not None:
                        s = self.luna_tiling_orchestrator(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
                    else:
                        s = self._tiled_upscale_with_model(in_img, upscale_model, tile_x, tile_y, tile_overlap, tile_strategy)
            
            # Post-processing: rescale if needed
            s_for_resizing = s
            target_height = round(image.shape[1] * scale_by)
            target_width = round(image.shape[2] * scale_by)
            
            s_resized = s_for_resizing
            if rescale_after_model and (s_for_resizing.shape[2] != target_height or s_for_resizing.shape[3] != target_width):
                use_antialias = resampling in ['bicubic', 'bilinear']
                
                if supersample:
                    s_resized = self._adaptive_resize(s_for_resizing, target_height, target_width, resampling, use_antialias)
                else:
                    if resampling == 'lanczos':
                        resampling_mode = 'bicubic'
                    elif resampling == 'nearest-exact':
                        resampling_mode = 'nearest'
                    else:
                        resampling_mode = resampling
                    s_resized = F.interpolate(s_for_resizing, size=(target_height, target_width), mode=resampling_mode, antialias=bool(use_antialias))
            
            s_final_for_comfy = s_resized.permute(0, 2, 3, 1)
            return (s_final_for_comfy,)


NODE_CLASS_MAPPINGS = {"Luna_Advanced_Upscaler": Luna_Advanced_Upscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna_Advanced_Upscaler": "Luna Advanced Upscaler"}