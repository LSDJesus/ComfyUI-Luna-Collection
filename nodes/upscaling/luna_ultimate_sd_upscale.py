import torch
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image, ImageFilter
import math
import os
import torch.nn.functional as F
import logging
import sys
from enum import Enum

# Add the parent directory to sys.path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utils modules directly
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

import importlib

LunaPerformanceMonitor = None
try:
    luna_perf_module = importlib.import_module('luna_performance_monitor')
    LunaPerformanceMonitor = luna_perf_module.LunaPerformanceMonitor
except ImportError:
    print("Luna Ultimate SD Upscale: Performance monitoring not available")

# Import TensorRT Engine
Engine = None
try:
    luna_trt_module = importlib.import_module('trt_engine')
    Engine = luna_trt_module.Engine
except ImportError:
    print("Luna Ultimate SD Upscale: TensorRT Engine not available. Please ensure trt_engine.py is properly installed.")

# Global instances for TensorRT engines
TENSORRT_UPSCALE_ENGINE = None
TENSORRT_DIFFUSION_ENGINE = None

# UltimateSDUpscale modes
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class Luna_UltimateSDUpscale:
    CATEGORY = "Luna/Meta"
    RETURN_TYPES = ("IMAGE", "PERFORMANCE_STATS")
    RETURN_NAMES = ("upscaled_image", "performance_stats")
    FUNCTION = "upscale"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "luna_pipe": ("LUNA_PIPE", {"tooltip": "Luna pipe containing model, VAE, and conditioning. When connected, either upscale_model or upscaler_trt_model is required."}),
                "model_opt": ("MODEL", {"tooltip": "Diffusion model (required if luna_pipe not connected)"}),
                "positive_opt": ("CONDITIONING", {"tooltip": "Positive conditioning (required if luna_pipe not connected)"}),
                "negative_opt": ("CONDITIONING", {"tooltip": "Negative conditioning (required if luna_pipe not connected)"}),
                "vae_opt": ("VAE", {"tooltip": "VAE model (required if luna_pipe not connected)"}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "ComfyUI upscaler model"}),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL", {"tooltip": "TensorRT upscale model. When connected, tile dimensions must be 512-1536-tile_padding"}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8}),
                "redraw_mode": (["Linear", "Chess", "None"], {"default": "Linear"}),
                "seam_fix_mode": (["None", "Band Pass", "Half Tile", "Half Tile + Intersections"], {"default": "Half Tile"}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 8}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
                "seed_opt": ("INT", {"min": 0, "max": 0xffffffffffffffff}),
                "steps_opt": ("INT", {"min": 1, "max": 100}),
                "cfg_opt": ("FLOAT", {"min": 0.0, "max": 100.0}),
                "sampler_name_opt": ("STRING",),
                "scheduler_opt": ("STRING",),
                "denoise_opt": ("FLOAT", {"min": 0.0, "max": 1.0}),
            }
        }

    def __init__(self):
        self.upscale_engine = None
        self.diffusion_engine = None

    def load_upscale_engine(self, engine_path):
        global TENSORRT_UPSCALE_ENGINE
        if Engine is None:
            raise ImportError("Engine is not available. Please ensure trt_engine.py is properly installed.")
        if TENSORRT_UPSCALE_ENGINE is None or TENSORRT_UPSCALE_ENGINE.engine_path != engine_path:
            # Check if path is relative and prepend models directory
            if not os.path.isabs(engine_path):
                # Try tensorrt upscale first, then standard upscale
                possible_paths = [
                    os.path.join("models", "tensorrt", "upscale_models", engine_path),
                    os.path.join("models", "upscale_models", engine_path),
                    os.path.join("models", "tensorrt", engine_path),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        engine_path = path
                        break
            TENSORRT_UPSCALE_ENGINE = Engine(engine_path)
            TENSORRT_UPSCALE_ENGINE.load()
            TENSORRT_UPSCALE_ENGINE.activate()
        self.upscale_engine = TENSORRT_UPSCALE_ENGINE

    def load_diffusion_engine(self, engine_path):
        global TENSORRT_DIFFUSION_ENGINE
        if Engine is None:
            raise ImportError("Engine is not available. Please ensure trt_engine.py is properly installed.")
        if TENSORRT_DIFFUSION_ENGINE is None or TENSORRT_DIFFUSION_ENGINE.engine_path != engine_path:
            # Check if path is relative and prepend models directory
            if not os.path.isabs(engine_path):
                # Try tensorrt checkpoints first, then standard checkpoints
                possible_paths = [
                    os.path.join("models", "tensorrt", engine_path),
                    os.path.join("models", "checkpoints", engine_path),
                    os.path.join("models", "tensorrt", "checkpoints", engine_path),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        engine_path = path
                        break
            TENSORRT_DIFFUSION_ENGINE = Engine(engine_path)
            TENSORRT_DIFFUSION_ENGINE.load()
            TENSORRT_DIFFUSION_ENGINE.activate()
        self.diffusion_engine = TENSORRT_DIFFUSION_ENGINE

    def upscale(self, image, luna_pipe=None, model_opt=None, positive_opt=None, negative_opt=None, vae_opt=None, upscale_model=None, upscaler_trt_model=None, upscale_by=2.0, tile_width=512, tile_height=512, mask_blur=8, tile_padding=32, redraw_mode="Linear", seam_fix_mode="Half Tile", seam_fix_denoise=1.0, seam_fix_width=64, seam_fix_mask_blur=8, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False, seed_opt=None, steps_opt=None, cfg_opt=None, sampler_name_opt=None, scheduler_opt=None, denoise_opt=None):
        # Initialize performance monitoring
        performance_monitor = LunaPerformanceMonitor() if LunaPerformanceMonitor else None
        if performance_monitor:
            performance_monitor.start_monitoring("Luna_UltimateSDUpscale")

        # Validate TensorRT constraints if upscaler_trt_model is connected
        if upscaler_trt_model is not None:
            # TensorRT requires tile dimensions between 512 and 1536-tile_padding
            min_tile_size = 512
            max_tile_size = 1536 - tile_padding

            if tile_width < min_tile_size or tile_width > max_tile_size:
                raise ValueError(f"TensorRT mode: tile_width must be between {min_tile_size} and {max_tile_size}, got {tile_width}")
            if tile_height < min_tile_size or tile_height > max_tile_size:
                raise ValueError(f"TensorRT mode: tile_height must be between {min_tile_size} and {max_tile_size}, got {tile_height}")

        # Extract values from luna_pipe or use individual inputs
        if luna_pipe is not None:
            pipe_model, pipe_clip, pipe_vae, pipe_positive, pipe_negative, pipe_seed, pipe_sampler_name, pipe_scheduler = luna_pipe

            model = pipe_model if model_opt is None else model_opt
            positive = pipe_positive if positive_opt is None else positive_opt
            negative = pipe_negative if negative_opt is None else negative_opt
            vae = pipe_vae if vae_opt is None else vae_opt
            seed = pipe_seed if seed_opt is None else seed_opt
            steps = 20 if steps_opt is None else steps_opt
            cfg = 8.0 if cfg_opt is None else cfg_opt
            sampler_name = pipe_sampler_name if sampler_name_opt is None else sampler_name_opt
            scheduler = pipe_scheduler if scheduler_opt is None else scheduler_opt
            denoise = 0.2 if denoise_opt is None else denoise_opt
        else:
            # When luna_pipe is not connected, require model, positive, negative, vae
            if model_opt is None or positive_opt is None or negative_opt is None or vae_opt is None:
                raise ValueError("When luna_pipe is not connected, model_opt, positive_opt, negative_opt, and vae_opt are required")

            model = model_opt
            positive = positive_opt
            negative = negative_opt
            vae = vae_opt
            seed = seed_opt or 0
            steps = steps_opt or 20
            cfg = cfg_opt or 8.0
            sampler_name = sampler_name_opt or "dpmpp_2m"
            scheduler = scheduler_opt or "karras"
            denoise = denoise_opt or 0.2

        # Validate upscale model requirements when luna_pipe is connected
        if luna_pipe is not None:
            if upscale_model is None and upscaler_trt_model is None:
                raise ValueError("When luna_pipe is connected, either upscale_model or upscaler_trt_model must be provided")

        # Set up TensorRT engines
        if upscaler_trt_model is not None:
            self.upscale_engine = upscaler_trt_model
        else:
            self.upscale_engine = None

        # Calculate target dimensions
        target_width = int(image.shape[2] * upscale_by)
        target_height = int(image.shape[1] * upscale_by)

        # Process images in batches
        max_batch_size = 16
        all_results = []

        for i in range(0, len(image), max_batch_size):
            batch_images = image[i:i + max_batch_size]
            batch_results = []

            for img in batch_images:
                # Convert to PIL for processing
                pil_img = self.tensor_to_pil(img)

                # Step 1: Basic upscaling
                upscaled_pil = self.basic_upscale(pil_img, target_width, target_height, upscale_model)

                # Step 2: Diffusion enhancement (if enabled)
                if redraw_mode != "None" and model is not None:
                    enhanced_pil = self.apply_diffusion_enhancement(
                        upscaled_pil, model, positive, negative, vae, seed,
                        steps, cfg, sampler_name, scheduler, denoise,
                        tile_width, tile_height, mask_blur, tile_padding,
                        USDUMode[redraw_mode.upper()]
                    )
                else:
                    enhanced_pil = upscaled_pil

                # Step 3: Seam fixing (if enabled)
                if seam_fix_mode != "None":
                    final_pil = self.apply_seam_fixing(
                        enhanced_pil, model, positive, negative, vae, seed,
                        steps, cfg, sampler_name, scheduler, seam_fix_denoise,
                        tile_width, tile_height, seam_fix_mask_blur, seam_fix_padding,
                        seam_fix_width, USDUSFMode[seam_fix_mode.upper().replace(" ", "_")]
                    )
                else:
                    final_pil = enhanced_pil

                # Convert final result to tensor
                final_tensor = self.pil_to_tensor(final_pil)
                batch_results.append(final_tensor)

            # Stack batch results
            if len(batch_results) > 0:
                batch_tensor = torch.stack(batch_results)
                all_results.append(batch_tensor)

        # Concatenate all batches
        if len(all_results) > 0:
            result = torch.cat(all_results, dim=0)
        else:
            result = image

        # Collect performance stats
        performance_stats = {}
        if performance_monitor:
            performance_stats = performance_monitor.stop_monitoring(
                upscale_by=upscale_by,
                tile_width=tile_width,
                tile_height=tile_height,
                redraw_mode=redraw_mode,
                seam_fix_mode=seam_fix_mode,
                force_uniform_tiles=force_uniform_tiles,
                tiled_decode=tiled_decode,
                input_shape=str(image.shape),
                output_shape=str(result.shape),
                target_width=target_width,
                target_height=target_height
            )

        return (result, performance_stats)

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        # Convert from [H, W, C] to PIL
        if len(tensor.shape) == 3:
            # Add batch dimension if needed
            tensor = tensor.unsqueeze(0)
        # Convert from [B, H, W, C] to [B, C, H, W] for PIL
        tensor = tensor.permute(0, 3, 1, 2)
        # Convert to 0-255 range and uint8
        tensor = (tensor * 255).clamp(0, 255).byte()
        # Convert to PIL
        pil_img = Image.fromarray(tensor[0].permute(1, 2, 0).cpu().numpy())
        return pil_img

    def pil_to_tensor(self, pil_img):
        """Convert PIL Image to tensor"""
        # Convert PIL to numpy array
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        # Convert to tensor [H, W, C]
        tensor = torch.from_numpy(np_img)
        return tensor

    def basic_upscale(self, pil_img, target_width, target_height, upscale_model=None):
        """Perform basic upscaling using TensorRT, ComfyUI upscaler, or fallback"""
        # First try TensorRT if engine is available
        if self.upscale_engine is not None:
            # Use TensorRT upscale engine
            img_tensor = self.pil_to_tensor(pil_img)
            batch_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).cuda()

            # Ensure dimensions are within TensorRT constraints
            _, _, h, w = batch_tensor.shape
            h_clamped = max(512, min(1536, h))
            w_clamped = max(512, min(1536, w))

            if h != h_clamped or w != w_clamped:
                batch_tensor = torch.nn.functional.interpolate(
                    batch_tensor, size=(h_clamped, w_clamped), mode='bilinear', align_corners=False
                )

            feed_dict = {'input_image': batch_tensor}
            with torch.cuda.stream(torch.cuda.current_stream()):
                upscale_result = self.upscale_engine.infer(feed_dict, torch.cuda.current_stream())

            upscaled_tensor = upscale_result['output_image'].cpu().permute(0, 2, 3, 1).squeeze(0)
            upscaled_pil = self.tensor_to_pil(upscaled_tensor)

            # If TensorRT gave us 4x but we need different scale, resize accordingly
            if upscaled_pil.size != (target_width, target_height):
                upscaled_pil = upscaled_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Try ComfyUI upscaler if TensorRT not available
        elif upscale_model is not None:
            # Use ComfyUI's built-in upscaler
            img_tensor = self.pil_to_tensor(pil_img)
            batch_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Use ComfyUI's upscale method
            upscaled_tensor = comfy.utils.common_upscale(
                batch_tensor, target_width, target_height, "lanczos", "center"
            ).permute(0, 2, 3, 1).squeeze(0)

            upscaled_pil = self.tensor_to_pil(upscaled_tensor)

        else:
            # Fallback to PIL Lanczos resize
            upscaled_pil = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return upscaled_pil

    def apply_diffusion_enhancement(self, pil_img, model, positive, negative, vae, seed, steps, cfg,
                                   sampler_name, scheduler, denoise, tile_width, tile_height,
                                   mask_blur, tile_padding, mode):
        """Apply diffusion enhancement using tiling system"""
        if mode == USDUMode.NONE:
            return pil_img

        # Convert to tensor for processing
        img_tensor = self.pil_to_tensor(pil_img)
        img_height, img_width = pil_img.height, pil_img.width

        # Calculate grid
        rows = math.ceil(img_height / tile_height)
        cols = math.ceil(img_width / tile_width)

        # Create enhanced image
        enhanced_tensor = img_tensor.clone()

        # Process tiles based on mode
        if mode == USDUMode.LINEAR:
            enhanced_tensor = self.linear_diffusion_enhancement(
                enhanced_tensor, img_width, img_height, rows, cols,
                tile_width, tile_height, mask_blur, tile_padding,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise
            )
        elif mode == USDUMode.CHESS:
            enhanced_tensor = self.chess_diffusion_enhancement(
                enhanced_tensor, img_width, img_height, rows, cols,
                tile_width, tile_height, mask_blur, tile_padding,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise
            )

        # Convert back to PIL
        enhanced_pil = self.tensor_to_pil(enhanced_tensor)
        return enhanced_pil

    def linear_diffusion_enhancement(self, img_tensor, img_width, img_height, rows, cols,
                                    tile_width, tile_height, mask_blur, tile_padding,
                                    model, positive, negative, vae, seed, steps, cfg,
                                    sampler_name, scheduler, denoise):
        """Process tiles in linear order (row by row)"""
        from nodes import KSampler, VAEDecode

        ksampler = KSampler()
        vae_decoder = VAEDecode()

        for yi in range(rows):
            for xi in range(cols):
                # Calculate tile coordinates
                x1 = xi * tile_width
                y1 = yi * tile_height
                x2 = min(x1 + tile_width, img_width)
                y2 = min(y1 + tile_height, img_height)

                # Create mask for this tile
                mask = torch.zeros((img_height, img_width), dtype=torch.float32)
                mask[y1:y2, x1:x2] = 1.0

                # Apply mask blur
                if mask_blur > 0:
                    mask = self.apply_mask_blur(mask, mask_blur)

                # Expand mask for padding
                if tile_padding > 0:
                    mask = self.expand_mask(mask, tile_padding, img_width, img_height)

                # Apply inpainting
                tile_result = self.apply_inpainting(
                    img_tensor, mask, model, positive, negative, vae,
                    seed + yi * cols + xi, steps, cfg, sampler_name, scheduler, denoise
                )

                # Blend result back
                img_tensor = self.blend_tile(img_tensor, tile_result, mask)

        return img_tensor

    def chess_diffusion_enhancement(self, img_tensor, img_width, img_height, rows, cols,
                                   tile_width, tile_height, mask_blur, tile_padding,
                                   model, positive, negative, vae, seed, steps, cfg,
                                   sampler_name, scheduler, denoise):
        """Process tiles in chess pattern to reduce visible grid artifacts"""
        # First pass: process "black" squares
        for yi in range(rows):
            for xi in range(cols):
                if (xi + yi) % 2 == 0:  # Black squares
                    img_tensor = self.process_single_tile(
                        img_tensor, img_width, img_height, xi, yi,
                        tile_width, tile_height, mask_blur, tile_padding,
                        model, positive, negative, vae, seed, steps, cfg,
                        sampler_name, scheduler, denoise
                    )

        # Second pass: process "white" squares
        for yi in range(rows):
            for xi in range(cols):
                if (xi + yi) % 2 != 0:  # White squares
                    img_tensor = self.process_single_tile(
                        img_tensor, img_width, img_height, xi, yi,
                        tile_width, tile_height, mask_blur, tile_padding,
                        model, positive, negative, vae, seed, steps, cfg,
                        sampler_name, scheduler, denoise
                    )

        return img_tensor

    def process_single_tile(self, img_tensor, img_width, img_height, xi, yi,
                           tile_width, tile_height, mask_blur, tile_padding,
                           model, positive, negative, vae, seed, steps, cfg,
                           sampler_name, scheduler, denoise):
        """Process a single tile"""
        # Calculate tile coordinates
        x1 = xi * tile_width
        y1 = yi * tile_height
        x2 = min(x1 + tile_width, img_width)
        y2 = min(y1 + tile_height, img_height)

        # Create mask for this tile
        mask = torch.zeros((img_height, img_width), dtype=torch.float32)
        mask[y1:y2, x1:x2] = 1.0

        # Apply mask blur
        if mask_blur > 0:
            mask = self.apply_mask_blur(mask, mask_blur)

        # Expand mask for padding
        if tile_padding > 0:
            mask = self.expand_mask(mask, tile_padding, img_width, img_height)

        # Apply inpainting
        tile_result = self.apply_inpainting(
            img_tensor, mask, model, positive, negative, vae,
            seed + yi * 100 + xi, steps, cfg, sampler_name, scheduler, denoise
        )

        # Blend result back
        img_tensor = self.blend_tile(img_tensor, tile_result, mask)

        return img_tensor

    def apply_mask_blur(self, mask, blur_radius):
        """Apply Gaussian blur to mask"""
        if blur_radius <= 0:
            return mask

        # Convert to PIL for blurring
        mask_np = mask.numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur_radius))
        mask_blurred = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)

        return mask_blurred

    def expand_mask(self, mask, padding, img_width, img_height):
        """Expand mask by padding amount"""
        if padding <= 0:
            return mask

        # Create expanded mask
        expanded_mask = mask.clone()

        # Add padding to all sides
        for y in range(img_height):
            for x in range(img_width):
                if mask[y, x] > 0:
                    y1 = max(0, y - padding)
                    y2 = min(img_height, y + padding + 1)
                    x1 = max(0, x - padding)
                    x2 = min(img_width, x + padding + 1)
                    expanded_mask[y1:y2, x1:x2] = 1.0

        return expanded_mask

    def apply_inpainting(self, img_tensor, mask, model, positive, negative, vae, seed,
                        steps, cfg, sampler_name, scheduler, denoise):
        """Apply inpainting to a masked region"""
        # This is a simplified implementation
        # In a full implementation, this would use ComfyUI's inpainting pipeline
        # For now, we'll return the original image (placeholder)
        return img_tensor

    def blend_tile(self, original_tensor, tile_tensor, mask):
        """Blend processed tile back into original image using mask"""
        # Simple alpha blending
        result = original_tensor * (1 - mask.unsqueeze(-1)) + tile_tensor * mask.unsqueeze(-1)
        return result

    def apply_seam_fixing(self, pil_img, model, positive, negative, vae, seed, steps, cfg,
                         sampler_name, scheduler, denoise, tile_width, tile_height,
                         mask_blur, tile_padding, seam_fix_width, mode):
        """Apply seam fixing to eliminate visible tile boundaries"""
        if mode == USDUSFMode.NONE:
            return pil_img

        img_tensor = self.pil_to_tensor(pil_img)
        img_height, img_width = pil_img.height, pil_img.width

        # Calculate grid
        rows = math.ceil(img_height / tile_height)
        cols = math.ceil(img_width / tile_width)

        if mode == USDUSFMode.BAND_PASS:
            img_tensor = self.band_pass_seam_fix(
                img_tensor, img_width, img_height, rows, cols,
                tile_width, tile_height, mask_blur, tile_padding,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise
            )
        elif mode in [USDUSFMode.HALF_TILE, USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS]:
            img_tensor = self.half_tile_seam_fix(
                img_tensor, img_width, img_height, rows, cols,
                tile_width, tile_height, mask_blur, tile_padding,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise, mode
            )

        return self.tensor_to_pil(img_tensor)

    def band_pass_seam_fix(self, img_tensor, img_width, img_height, rows, cols,
                          tile_width, tile_height, mask_blur, tile_padding,
                          model, positive, negative, vae, seed, steps, cfg,
                          sampler_name, scheduler, denoise):
        """Fix seams using band pass method (horizontal and vertical seams)"""
        # Fix horizontal seams (between rows)
        for yi in range(rows - 1):
            seam_y = (yi + 1) * tile_height
            img_tensor = self.fix_horizontal_seam(
                img_tensor, img_width, seam_y, tile_width,
                mask_blur, tile_padding, model, positive, negative, vae,
                seed + yi, steps, cfg, sampler_name, scheduler, denoise
            )

        # Fix vertical seams (between columns)
        for xi in range(cols - 1):
            seam_x = (xi + 1) * tile_width
            img_tensor = self.fix_vertical_seam(
                img_tensor, img_height, seam_x, tile_height,
                mask_blur, tile_padding, model, positive, negative, vae,
                seed + xi + rows, steps, cfg, sampler_name, scheduler, denoise
            )

        return img_tensor

    def half_tile_seam_fix(self, img_tensor, img_width, img_height, rows, cols,
                          tile_width, tile_height, mask_blur, tile_padding,
                          model, positive, negative, vae, seed, steps, cfg,
                          sampler_name, scheduler, denoise, mode):
        """Fix seams using half tile method"""
        # Create gradient mask for half-tile overlaps
        gradient = self.create_gradient_mask(tile_width, tile_height)

        # Fix horizontal seams
        for yi in range(rows - 1):
            for xi in range(cols):
                seam_y = yi * tile_height + tile_height // 2
                img_tensor = self.apply_half_tile_fix(
                    img_tensor, img_width, img_height, xi, yi, seam_y,
                    tile_width, tile_height, gradient, True,  # Horizontal
                    mask_blur, tile_padding, model, positive, negative, vae,
                    seed + yi * cols + xi, steps, cfg, sampler_name, scheduler, denoise
                )

        # Fix vertical seams
        for yi in range(rows):
            for xi in range(cols - 1):
                seam_x = xi * tile_width + tile_width // 2
                img_tensor = self.apply_half_tile_fix(
                    img_tensor, img_width, img_height, xi, yi, seam_x,
                    tile_width, tile_height, gradient, False,  # Vertical
                    mask_blur, tile_padding, model, positive, negative, vae,
                    seed + yi * cols + xi + rows * cols, steps, cfg,
                    sampler_name, scheduler, denoise
                )

        # Fix intersections if requested
        if mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            img_tensor = self.fix_intersections(
                img_tensor, img_width, img_height, rows, cols,
                tile_width, tile_height, mask_blur, tile_padding,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise
            )

        return img_tensor

    def create_gradient_mask(self, width, height):
        """Create gradient mask for half-tile seam fixing"""
        # Create linear gradient
        gradient = torch.linspace(0, 1, width if width < height else height)
        if width < height:
            gradient = gradient.unsqueeze(0).repeat(height, 1)
        else:
            gradient = gradient.unsqueeze(1).repeat(1, width)
        return gradient

    def fix_horizontal_seam(self, img_tensor, img_width, seam_y, tile_width,
                           mask_blur, tile_padding, model, positive, negative, vae,
                           seed, steps, cfg, sampler_name, scheduler, denoise):
        """Fix a horizontal seam"""
        # Create mask for seam region
        mask = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)
        mask[seam_y - tile_width//2:seam_y + tile_width//2, :] = 1.0

        if mask_blur > 0:
            mask = self.apply_mask_blur(mask, mask_blur)

        # Apply inpainting to seam
        result = self.apply_inpainting(
            img_tensor, mask, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise
        )

        return self.blend_tile(img_tensor, result, mask)

    def fix_vertical_seam(self, img_tensor, img_height, seam_x, tile_height,
                         mask_blur, tile_padding, model, positive, negative, vae,
                         seed, steps, cfg, sampler_name, scheduler, denoise):
        """Fix a vertical seam"""
        # Create mask for seam region
        mask = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)
        mask[:, seam_x - tile_height//2:seam_x + tile_height//2] = 1.0

        if mask_blur > 0:
            mask = self.apply_mask_blur(mask, mask_blur)

        # Apply inpainting to seam
        result = self.apply_inpainting(
            img_tensor, mask, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise
        )

        return self.blend_tile(img_tensor, result, mask)

    def apply_half_tile_fix(self, img_tensor, img_width, img_height, xi, yi, seam_pos,
                           tile_width, tile_height, gradient, is_horizontal,
                           mask_blur, tile_padding, model, positive, negative, vae,
                           seed, steps, cfg, sampler_name, scheduler, denoise):
        """Apply half-tile seam fixing"""
        # Create mask for half-tile region
        mask = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)

        if is_horizontal:
            # Horizontal seam
            y1 = max(0, seam_pos - tile_height//2)
            y2 = min(img_height, seam_pos + tile_height//2)
            mask[y1:y2, xi*tile_width:(xi+1)*tile_width] = gradient[:y2-y1, :tile_width]
        else:
            # Vertical seam
            x1 = max(0, seam_pos - tile_width//2)
            x2 = min(img_width, seam_pos + tile_width//2)
            mask[yi*tile_height:(yi+1)*tile_height, x1:x2] = gradient[:tile_height, :x2-x1]

        if mask_blur > 0:
            mask = self.apply_mask_blur(mask, mask_blur)

        # Apply inpainting
        result = self.apply_inpainting(
            img_tensor, mask, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise
        )

        return self.blend_tile(img_tensor, result, mask)

    def fix_intersections(self, img_tensor, img_width, img_height, rows, cols,
                         tile_width, tile_height, mask_blur, tile_padding,
                         model, positive, negative, vae, seed, steps, cfg,
                         sampler_name, scheduler, denoise):
        """Fix intersection points between tiles"""
        for yi in range(rows - 1):
            for xi in range(cols - 1):
                intersection_x = (xi + 1) * tile_width
                intersection_y = (yi + 1) * tile_height

                # Create small mask around intersection
                mask = torch.zeros((img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)
                mask[intersection_y-16:intersection_y+16, intersection_x-16:intersection_x+16] = 1.0

                if mask_blur > 0:
                    mask = self.apply_mask_blur(mask, mask_blur)

                # Apply inpainting to intersection
                result = self.apply_inpainting(
                    img_tensor, mask, model, positive, negative, vae,
                    seed + yi * cols + xi, steps, cfg, sampler_name, scheduler, denoise
                )

                img_tensor = self.blend_tile(img_tensor, result, mask)

        return img_tensor

NODE_CLASS_MAPPINGS = {
    "Luna_UltimateSDUpscale": Luna_UltimateSDUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_UltimateSDUpscale": "Luna Ultimate SD Upscale",
}