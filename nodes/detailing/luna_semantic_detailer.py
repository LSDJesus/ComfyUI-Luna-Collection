"""
Luna Semantic Detailer - Surgical 1:1 Refinement

The "Finishing Carpenter" - performs targeted refinement on detected objects
using 1:1 noise density from the master scaffold. Processes crops in batches
with per-detection conditioning for maximum quality and efficiency.

Architecture:
- Coordinate mapping from normalized (0-1) to full scaffold pixels
- 1:1 square expansion with padding (1.15x, snapped to 256px)
- Variance-corrected noise slicing from master scaffold
- Batched sampling with individual prompts per detection
- Smoothstep alpha blending for seamless compositing

Mathematical Foundation:
- All crops refined at 1024×1024 (optimal for SDXL/Flux)
- Variance preservation during resize: multiply by scale_factor
- Smoothstep blending: t²(3-2t) for C¹ continuity
"""

import torch
import torch.nn.functional as F
import numpy as np
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management


def get_scheduler_names():
    """Get current scheduler names dynamically to avoid type mismatches."""
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except (AttributeError, TypeError):
        return list(comfy.samplers.SCHEDULER_NAMES)


class LunaSemanticDetailer:
    """
    Surgical refinement of detected objects using batched 1:1 sampling.
    
    Takes detection data from SAM3 Detector and performs targeted refinement
    on each detected region. Uses the master noise scaffold to maintain
    perfect noise density across all scales.
    
    Key Features:
    - Batched processing of all detections (efficient)
    - Per-detection conditioning (individual prompts)
    - Variance-corrected noise slicing
    - Smoothstep blending for seamless integration
    - Hierarchical layer support (0=structural, 1+=details)
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE", "LATENT", "LUNA_DETECTION_PIPE", "MASK")
    RETURN_NAMES = ("refined_image", "refined_latent", "detection_pipe", "refinement_mask")
    FUNCTION = "refine"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Upscaled image from Scaffold Upscaler"
                }),
                "full_latent": ("LATENT", {
                    "tooltip": "Encoded latent of upscaled image (for latent compositing)"
                }),
                "full_scaffold": ("LATENT", {
                    "tooltip": "Full-resolution noise from Pyramid Generator"
                }),
                "detection_pipe": ("LUNA_DETECTION_PIPE", {
                    "tooltip": "Detection pipe from SAM3 Detector (includes pre-encoded conditioning)"
                }),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "target_layers": ("STRING", {
                    "default": "0",
                    "tooltip": "Comma-separated layer IDs to refine (e.g., '0,1')"
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5
                }),
                "denoise": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (get_scheduler_names(),),
                "tile_batch_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Process detections in batches (VRAM safety)"
                }),
                "enlarge_crops": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Paste refined crops at 1024px (enables upscaling for small inputs)"
                }),
            },
            "optional": {
                "previous_refinement_mask": ("MASK", {
                    "tooltip": "Refinement mask from previous detailer (for chaining layers)"
                }),
            }
        }
    
    def refine(
        self,
        image: torch.Tensor,
        full_latent: dict,
        full_scaffold: dict,
        detection_pipe: dict,
        model,
        vae,
        positive,
        negative,
        target_layers: str,
        steps: int,
        cfg: float,
        denoise: float,
        seed: int,
        sampler: str,
        scheduler: str,
        tile_batch_size: int,
        enlarge_crops: bool,
        previous_refinement_mask = None
    ) -> tuple:
        """
        Perform surgical refinement on detected objects.
        
        Args:
            image: Upscaled pixels [B, H, W, C]
            full_scaffold: Full-res noise {"samples": tensor}
            detection_data: Detection dict from SAM3
            ... (standard diffusion params)
            
        Returns:
            Tuple of (refined_image, refinement_mask)
        """
        device = comfy.model_management.get_torch_device()
        
        # Parse target layers
        try:
            target_layer_ids = [int(x.strip()) for x in target_layers.split(",")]
        except:
            print("[LunaSemanticDetailer] Invalid target_layers, using layer 0")
            target_layer_ids = [0]
        
        # Extract detections and conditioning from pipe
        all_detections = detection_pipe.get("detections", [])
        all_positive = detection_pipe.get("positive_batch", [])
        
        # Filter detections by target layers
        filtered_data = [
            (det, pos) for det, pos in zip(all_detections, all_positive)
            if det.get("layer", 0) in target_layer_ids
        ]
        
        if len(filtered_data) == 0:
            print(f"[LunaSemanticDetailer] No detections for layers {target_layer_ids}")
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            empty_mask = previous_refinement_mask if previous_refinement_mask is not None else torch.zeros(b, h, w, device=device)
            return (image, full_latent, detection_pipe, empty_mask)
        
        detections, positive_list = zip(*filtered_data)
        
        if len(detections) == 0:
            print(f"[LunaSemanticDetailer] No detections for layers {target_layer_ids}")
            # Return unchanged image + empty mask
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            return (image, torch.zeros(b, h, w, device=device))
        
        print(f"[LunaSemanticDetailer] Refining {len(detections)} detections across layers {target_layer_ids}")
        
        # Get image dimensions
        b, h, w, c = image.shape
        
        # Prepare crops and metadata (no prompts needed, using pre-encoded conditioning)
        crop_data = self._prepare_crops(
            image, full_scaffold, list(detections), h, w
        )
        
        if len(crop_data) == 0:
            print("[LunaSemanticDetailer] No valid crops generated")
            empty_mask = previous_refinement_mask if previous_refinement_mask is not None else torch.zeros(b, h, w, device=device)
            return (image, full_latent, detection_pipe, empty_mask)
        
        # Initialize sampler
        sampler_obj = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=device,
            sampler=sampler,
            scheduler=scheduler,
            denoise=1.0,
            model_options=model.model_options
        )
        
        # Calculate start step based on denoise
        total_steps = len(sampler_obj.sigmas) - 1
        start_step = int(total_steps * (1.0 - denoise))
        sigmas = sampler_obj.sigmas[start_step:]
        initial_sigma = sampler_obj.sigmas[start_step]
        
        # Process crops in batches
        refined_crops = []
        
        for i in range(0, len(crop_data), tile_batch_size):
            batch = crop_data[i:i + tile_batch_size]
            
            print(f"[LunaSemanticDetailer] Processing batch {i//tile_batch_size + 1}/{(len(crop_data) + tile_batch_size - 1)//tile_batch_size}")
            
            # Stack crops into batch
            batch_pixels = torch.cat([c["pixel_crop"] for c in batch], dim=0)
            batch_noise = torch.cat([c["noise_crop"] for c in batch], dim=0)
            
            # Encode pixels to latent
            batch_latents = vae.encode(batch_pixels.permute(0, 3, 1, 2))
            
            # Get pre-encoded conditioning for this batch
            batch_indices = range(i, min(i + tile_batch_size, len(crop_data)))
            batch_positive = [positive_list[idx] for idx in batch_indices]
            
            # Replicate negative for batch
            batch_negative = self._replicate_conditioning(negative, len(batch))
            
            # Ensure all tensors are on the same device
            device = batch_latents.device
            batch_noise = batch_noise.to(device)
            
            # Pre-inject scaffold noise at starting sigma
            initial_sigma_val = initial_sigma.item() if isinstance(initial_sigma, torch.Tensor) else initial_sigma
            noised_latents = batch_latents + batch_noise * initial_sigma_val
            
            # Sample with pre-noised latent (like KSampler Advanced with add_noise="disable")
            with torch.inference_mode():
                refined_latent = comfy.sample.sample(
                    model,
                    noise=torch.zeros_like(noised_latents),  # Zero noise (already pre-noised)
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=batch_positive,
                    negative=batch_negative,
                    latent_image=noised_latents,   # Pre-noised with scaffold
                    denoise=denoise,
                    disable_noise=True,            # Don't add more noise!
                    start_step=start_step,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed + i
                )
            
            # Ensure refined latent stays on GPU
            device = batch_latents.device
            if refined_latent.device != device:
                refined_latent = refined_latent.to(device)
            
            # Decode to pixels (pixels can go to CPU later if needed)
            refined_pixels = vae.decode(refined_latent).permute(0, 2, 3, 1)
            
            # Move pixels to CPU to save VRAM (they won't go back to GPU)
            refined_pixels = refined_pixels.cpu()
            
            # Store both (latents stay on GPU, pixels on CPU)
            refined_crops.extend([
                {
                    "pixels": refined_pixels[j:j+1],
                    "latent": refined_latent[j:j+1]  # Stays on GPU
                }
                for j in range(refined_pixels.shape[0])
            ])
            
            # Clear VRAM
            torch.cuda.empty_cache()
            
            # Allow interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        # Composite back onto canvas (both pixel and latent)
        working_canvas_pixels = image.clone()
        working_canvas_latent = full_latent["samples"].clone()
        
        # Start with previous refinement mask or create new
        if previous_refinement_mask is not None:
            refinement_mask = previous_refinement_mask.clone()
        else:
            refinement_mask = torch.zeros(b, h, w, device=device)
        
        for crop_info, refined_crop in zip(crop_data, refined_crops):
            self._composite_crop(
                working_canvas_pixels,
                working_canvas_latent,
                refinement_mask,
                refined_crop,
                crop_info,
                h, w,
                enlarge_crops
            )
        
        print(f"[LunaSemanticDetailer] Refinement complete")
        
        # Wrap latent back into dict
        refined_latent_dict = {"samples": working_canvas_latent}
        
        return (working_canvas_pixels, refined_latent_dict, detection_pipe, refinement_mask)
    
    def _prepare_crops(
        self,
        image: torch.Tensor,
        full_scaffold: dict,
        detections: list,
        img_h: int,
        img_w: int
    ) -> list:
        """
        Prepare all crops for batched processing.
        
        Returns list of dicts with:
        - pixel_crop: [1, 1024, 1024, 3]
        - noise_crop: [1, 4, 128, 128]
        - mask_1024: [1024, 1024]
        - original_box: (x1, y1, x2, y2) in full image        - original_box_latent: (x1, y1, x2, y2) in latent space        - resize_info: dict with scaling info
        - prompt: str
        """
        crop_data = []
        
        for det in detections:
            # Scale normalized coords to full image
            x1_norm, y1_norm, x2_norm, y2_norm = det["bbox_norm"]
            x1 = int(x1_norm * img_w)
            y1 = int(y1_norm * img_h)
            x2 = int(x2_norm * img_w)
            y2 = int(y2_norm * img_h)
            
            # Calculate object size
            obj_w = x2 - x1
            obj_h = y2 - y1
            obj_size = max(obj_w, obj_h)
            
            # Apply 15% padding and snap to 256
            padded_size = int(obj_size * 1.15)
            padded_size = int(np.ceil(padded_size / 256)) * 256
            padded_size = min(padded_size, 2048)  # Hard cap
            
            # Calculate centered crop bounds
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            crop_x1 = max(0, cx - padded_size // 2)
            crop_y1 = max(0, cy - padded_size // 2)
            crop_x2 = min(img_w, crop_x1 + padded_size)
            crop_y2 = min(img_h, crop_y1 + padded_size)
            
            # Snap to VAE grid
            crop_x1, crop_y1, crop_x2, crop_y2 = self._snap_box_to_grid(
                crop_x1, crop_y1, crop_x2, crop_y2
            )
            
            actual_w = crop_x2 - crop_x1
            actual_h = crop_y2 - crop_y1
            
            # Extract pixel crop
            pixel_crop = image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            
            # Extract noise crop from scaffold
            lat_x1, lat_y1 = crop_x1 // 8, crop_y1 // 8
            lat_x2, lat_y2 = crop_x2 // 8, crop_y2 // 8
            
            noise_crop = full_scaffold["samples"][:, :, lat_y1:lat_y2, lat_x1:lat_x2]
            
            # Always resize to 1024×1024 (upscale or downscale as needed)
            # No padding - bicubic interpolation preserves actual image content
            if actual_w == 1024 and actual_h == 1024:
                # Perfect size, no resize needed
                pixel_1024 = pixel_crop
                noise_1024 = noise_crop
                scale_factor = 1.0
            elif actual_w < 1024 or actual_h < 1024:
                # Upscale smaller crops to 1024
                pixel_1024 = F.interpolate(
                    pixel_crop.permute(0, 3, 1, 2),
                    size=(1024, 1024),
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                
                # Upscale noise with area (maintains variance better)
                noise_1024 = F.interpolate(
                    noise_crop,
                    size=(128, 128),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Variance correction for upscaling
                # When upscaling, variance increases - we need to reduce
                scale_factor = max(actual_w, actual_h) / 1024
                noise_1024 = noise_1024 * scale_factor
            else:
                # Downscale larger crops to 1024
                pixel_1024 = F.interpolate(
                    pixel_crop.permute(0, 3, 1, 2),
                    size=(1024, 1024),
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                
                noise_1024 = F.interpolate(
                    noise_crop,
                    size=(128, 128),
                    mode='area'
                )
                
                # Variance correction for downscaling
                scale_factor = max(actual_w, actual_h) / 1024
                noise_1024 = noise_1024 * scale_factor
            
            # Prepare mask at 1024×1024
            mask_resized = F.interpolate(
                det["mask"].unsqueeze(0).unsqueeze(0).float(),
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            crop_data.append({
                "pixel_crop": pixel_1024,
                "noise_crop": noise_1024,
                "mask_1024": mask_resized,
                "original_box": (crop_x1, crop_y1, crop_x2, crop_y2),
                "original_box_latent": (lat_x1, lat_y1, lat_x2, lat_y2),
                "original_size": (actual_w, actual_h),
                "scale_factor": scale_factor,
            })
        
        return crop_data
    
    def _composite_crop(
        self,
        pixel_canvas: torch.Tensor,
        latent_canvas: torch.Tensor,
        mask_canvas: torch.Tensor,
        refined_data: dict,
        crop_info: dict,
        img_h: int,
        img_w: int,
        enlarge_crops: bool
    ):
        """
        Composite refined crop back onto both pixel and latent canvases with smoothstep blending.
        
        If enlarge_crops=True: Paste refined crop at 1024×1024 (allows upscaling small regions)
        If enlarge_crops=False: Resize back to original crop size (default, for 4K inputs)
        """
        x1, y1, x2, y2 = crop_info["original_box"]
        lat_x1, lat_y1, lat_x2, lat_y2 = crop_info["original_box_latent"]
        orig_w, orig_h = crop_info["original_size"]
        
        refined_pixels = refined_data["pixels"]
        refined_latent = refined_data["latent"]
        
        # Determine target size for pasting
        if enlarge_crops:
            # Paste at refined size (1024×1024) - allows upscaling
            target_w, target_h = 1024, 1024
            
            # Recalculate paste coordinates centered on original region
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            paste_x1 = max(0, cx - 512)
            paste_y1 = max(0, cy - 512)
            paste_x2 = min(img_w, paste_x1 + 1024)
            paste_y2 = min(img_h, paste_y1 + 1024)
            
            # Adjust if near edges
            if paste_x2 - paste_x1 < 1024:
                paste_x1 = max(0, paste_x2 - 1024)
            if paste_y2 - paste_y1 < 1024:
                paste_y1 = max(0, paste_y2 - 1024)
            
            actual_paste_w = paste_x2 - paste_x1
            actual_paste_h = paste_y2 - paste_y1
            
            # Crop refined to fit if near edge
            if actual_paste_w < 1024 or actual_paste_h < 1024:
                refined_resized = refined_pixels[:, :actual_paste_h, :actual_paste_w, :]
            else:
                refined_resized = refined_pixels
            
            # Update coordinates for latent
            paste_lat_x1 = paste_x1 // 8
            paste_lat_y1 = paste_y1 // 8
            paste_lat_x2 = paste_x2 // 8
            paste_lat_y2 = paste_y2 // 8
            
            refined_latent_resized = refined_latent[:, :, :paste_lat_y2-paste_lat_y1, :paste_lat_x2-paste_lat_x1]
            
            # Use full mask (no resizing needed)
            mask_resized = crop_info["mask_1024"][:actual_paste_h, :actual_paste_w]
            
        else:
            # Default behavior: resize back to original crop size
            paste_x1, paste_y1, paste_x2, paste_y2 = x1, y1, x2, y2
            paste_lat_x1, paste_lat_y1, paste_lat_x2, paste_lat_y2 = lat_x1, lat_y1, lat_x2, lat_y2
            
            # Always resize refined output back to original crop size
            # (we scaled up or down to 1024, now scale back)
            refined_resized = F.interpolate(
                refined_pixels.permute(0, 3, 1, 2),
                size=(orig_h, orig_w),
                mode='bicubic',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Resize refined latent back to original latent size
            orig_lat_h = (lat_y2 - lat_y1)
            orig_lat_w = (lat_x2 - lat_x1)
            refined_latent_resized = F.interpolate(
                refined_latent,
                size=(orig_lat_h, orig_lat_w),
                mode='bicubic',
                align_corners=False
            )
            
            # Resize mask back to original size
            mask_resized = F.interpolate(
                crop_info["mask_1024"].unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )[0, 0]
        
        # Apply smoothstep to mask
        mask_smooth = self._smoothstep(mask_resized)
        
        # Blend pixels onto canvas
        current_region = pixel_canvas[:, paste_y1:paste_y2, paste_x1:paste_x2, :]
        mask_3d = mask_smooth.unsqueeze(0).unsqueeze(-1)
        blended_pixels = current_region * (1 - mask_3d) + refined_resized * mask_3d
        pixel_canvas[:, paste_y1:paste_y2, paste_x1:paste_x2, :] = blended_pixels
        
        # Blend latent onto latent canvas
        current_latent_region = latent_canvas[:, :, paste_lat_y1:paste_lat_y2, paste_lat_x1:paste_lat_x2]
        
        # Resize mask for latent space
        lat_h = paste_lat_y2 - paste_lat_y1
        lat_w = paste_lat_x2 - paste_lat_x1
        mask_latent = F.interpolate(
            mask_smooth.unsqueeze(0).unsqueeze(0),
            size=(lat_h, lat_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply smoothstep to latent mask
        mask_latent_smooth = self._smoothstep(mask_latent[0, 0])
        mask_latent_4d = mask_latent_smooth.unsqueeze(0).unsqueeze(0)
        
        blended_latent = current_latent_region * (1 - mask_latent_4d) + refined_latent_resized * mask_latent_4d
        latent_canvas[:, :, paste_lat_y1:paste_lat_y2, paste_lat_x1:paste_lat_x2] = blended_latent
        
        # Update refinement mask (pixel space)
        mask_canvas[:, paste_y1:paste_y2, paste_x1:paste_x2] = torch.maximum(
            mask_canvas[:, paste_y1:paste_y2, paste_x1:paste_x2],
            mask_smooth
        )
    
    def _smoothstep(self, t: torch.Tensor) -> torch.Tensor:
        """Polynomial smoothstep: t²(3-2t)"""
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def _snap_box_to_grid(self, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Snap box to VAE grid (multiples of 8)"""
        x1 = (x1 // 8) * 8
        y1 = (y1 // 8) * 8
        x2 = ((x2 + 7) // 8) * 8
        y2 = ((y2 + 7) // 8) * 8
        return x1, y1, x2, y2
    
    def _replicate_conditioning(self, cond, count: int):
        """Replicate conditioning for batch"""
        if count == 1:
            return cond
        
        # ComfyUI conditioning format: list of [tensor, dict]
        replicated = []
        for c in cond:
            tensor, opts = c
            # Repeat tensor along batch dimension
            batched_tensor = tensor.repeat(count, 1, 1)
            replicated.append([batched_tensor, opts])
        
        return replicated


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSemanticDetailer": LunaSemanticDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSemanticDetailer": "🌙 Luna: Semantic Detailer",
}
