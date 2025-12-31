"""
Luna Semantic Detailer - Surgical 1:1 Refinement

The "Finishing Carpenter" - performs targeted refinement on detected objects
using 1:1 noise density from the master scaffold. Processes crops in batches
with per-detection conditioning for maximum quality and efficiency.

Architecture:
- Coordinate mapping from normalized (0-1) to full scaffold pixels
- 1:1 square expansion with padding (1.15x, snapped to 256px)
- Fresh VAE encoding of pixel crops (proper convolutional context)
- IP-Adapter structural anchoring with TRUE BATCHING
- Batched sampling with individual prompts per detection
- Smoothstep alpha blending for seamless compositing

Mathematical Foundation:
- All crops refined at 1024×1024 (optimal for SDXL/Flux)
- IP-Adapter injects vision features per-crop (no averaging!)
- Smoothstep blending: t²(3-2t) for C¹ continuity

IP-Adapter Batching:
- Latent[i] only "sees" Adapter Embed[i] (PyTorch attention physics)
- No broadcasting if batch dims match: [N, 4, H, W] + [N, 16, 2048]
- One patch, one sample call, N distinct results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management

# Dynamic vision routing (for CLIP-ViT encoding)
try:
    from .vision_routing import VisionRouter, get_vision_router
except ImportError:
    # Fallback for ComfyUI's dynamic loading
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from vision_routing import VisionRouter, get_vision_router

# IP-Adapter for proper attention injection (import directly from IPAdapterPlus)
try:
    from custom_nodes.comfyui_ipadapter_plus.IPAdapterPlus import IPAdapter
    from custom_nodes.comfyui_ipadapter_plus.CrossAttentionPatch import Attn2Replace, ipadapter_attention
    HAS_IPADAPTER = True
except ImportError:
    HAS_IPADAPTER = False
    IPAdapter = None


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
    on each detected region. Uses fresh VAE encoding per crop and optional
    CLIP-ViT structural anchoring for maximum quality.
    
    Key Features:
    - Fresh VAE encoding of pixel crops (proper context)
    - Optional per-crop CLIP-ViT structural anchoring
    - Per-detection conditioning (individual prompts)
    - Scaffold noise slicing for texture consistency
    - Smoothstep blending for seamless integration
    - Hierarchical layer support (0=structural, 1+=details)
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE", "LUNA_DETECTION_PIPE", "MASK")
    RETURN_NAMES = ("refined_image", "detection_pipe", "refinement_mask")
    FUNCTION = "refine"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Upscaled 4K image from Prep Upscaler"
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
                "clip_vision": ("CLIP_VISION", {
                    "tooltip": "CLIP-ViT model for structural anchoring (encodes each crop)"
                }),
                "ip_adapter": ("IPADAPTER", {
                    "tooltip": "IP-Adapter model for proper vision→attention injection"
                }),
                "use_structural_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable per-crop CLIP-ViT + IP-Adapter for structural preservation"
                }),
                "ip_adapter_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IP-Adapter strength (0=text only, 1=full image guidance)"
                }),
            }
        }
    
    def refine(
        self,
        image: torch.Tensor,
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
        previous_refinement_mask = None,
        clip_vision = None,
        ip_adapter = None,
        use_structural_anchor: bool = True,
        ip_adapter_weight: float = 0.5
    ) -> tuple:
        """
        Perform surgical refinement on detected objects.
        
        All work happens in pixel space - crops from 4K canvas, encodes fresh,
        refines, decodes, and pastes back to 4K pixel canvas.
        
        Args:
            image: 4K pixel canvas [B, H, W, C]
            full_scaffold: Full-res noise {"samples": tensor}
            detection_pipe: Detection dict from SAM3
            ... (standard diffusion params)
            
        Returns:
            Tuple of (refined_image, detection_pipe, refinement_mask)
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
            return (image, detection_pipe, empty_mask)
        
        detections, positive_list = zip(*filtered_data)
        
        if len(detections) == 0:
            print(f"[LunaSemanticDetailer] No detections for layers {target_layer_ids}")
            # Return unchanged image + empty mask
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            return (image, torch.zeros(b, h, w, device=device))
        
        print(f"[LunaSemanticDetailer] Refining {len(detections)} detections across layers {target_layer_ids}")
        
        # Get image dimensions
        b, h, w, c = image.shape
        
        # Prepare crops and metadata
        crop_data = self._prepare_crops(
            image, full_scaffold, list(detections), h, w
        )
        
        if len(crop_data) == 0:
            print("[LunaSemanticDetailer] No valid crops generated")
            empty_mask = previous_refinement_mask if previous_refinement_mask is not None else torch.zeros(b, h, w, device=device)
            return (image, detection_pipe, empty_mask)
        
        # Crop global conditioning for each detection region
        # Import the helper from chess refiner
        from .luna_chess_refiner import crop_conditioning_for_tile
        
        canvas_size_px = (w, h)
        cropped_global_positive = []
        
        for crop_info in crop_data:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_info["original_box"]
            tile_region_px = (crop_x1, crop_y1, crop_x2, crop_y2)
            
            # Crop global conditioning to this detection region (no scaling needed - same resolution)
            cropped_pos = crop_conditioning_for_tile(positive, tile_region_px, canvas_size_px, scale_from=1.0)
            cropped_global_positive.append(cropped_pos)
        
        # Process crops in batches
        refined_crops = []
        
        # Initialize vision router for CLIP-ViT encoding (daemon or local)
        vision_router = None
        use_anchor = use_structural_anchor and (clip_vision is not None or ip_adapter is not None)
        
        if use_structural_anchor:
            if ip_adapter is not None and HAS_IPADAPTER:
                print(f"[LunaSemanticDetailer] ✓ IP-Adapter structural anchoring ENABLED (weight={ip_adapter_weight})")
                # Vision router for encoding, IP-Adapter for injection
                vision_router = get_vision_router(clip_vision)
            elif clip_vision is not None:
                vision_router = get_vision_router(clip_vision)
                if vision_router.available:
                    if vision_router.using_daemon:
                        print(f"[LunaSemanticDetailer] ✓ CLIP-ViT routing: DAEMON (GPU offload)")
                    else:
                        print(f"[LunaSemanticDetailer] ✓ CLIP-ViT routing: LOCAL")
                    # Note: Without IP-Adapter, vision embeds won't be properly injected
                    print(f"[LunaSemanticDetailer] ⚠ No IP-Adapter provided - vision encoding won't affect output!")
                    use_anchor = False
                else:
                    print(f"[LunaSemanticDetailer] ⚠ No vision encoder available, using text-only")
                    use_anchor = False
            else:
                print(f"[LunaSemanticDetailer] ⚠ No clip_vision or ip_adapter provided")
                use_anchor = False
        else:
            print(f"[LunaSemanticDetailer] Using text-only conditioning")
        
        for i in range(0, len(crop_data), tile_batch_size):
            batch = crop_data[i:i + tile_batch_size]
            batch_size = len(batch)
            
            print(f"[LunaSemanticDetailer] Processing batch {i//tile_batch_size + 1}/{(len(crop_data) + tile_batch_size - 1)//tile_batch_size}")
            
            # Stack crops into batch
            batch_pixels = torch.cat([c["pixel_crop"] for c in batch], dim=0)
            batch_noise = torch.cat([c["noise_crop"] for c in batch], dim=0)
            
            # Encode pixels to latent (FRESH encoding with proper context!)
            batch_latents = vae.encode(batch_pixels.permute(0, 3, 1, 2))
            
            # === IP-ADAPTER STRUCTURAL ANCHORING (TRUE BATCHING) ===
            # Key insight: Latent[i] only sees Embed[i] - no averaging!
            work_model = model
            
            if use_anchor and ip_adapter is not None and vision_router is not None:
                # Get crop coordinates for this batch
                batch_crop_coords = [
                    (c["original_box"][0], c["original_box"][1], c["original_box"][2], c["original_box"][3])
                    for c in batch
                ]
                
                # Encode crops with CLIP-ViT (daemon or local)
                vision_embeds_list = vision_router.encode_crops(
                    full_image=image,
                    crop_coords=batch_crop_coords,
                    tile_size=1024
                )
                
                if vision_embeds_list and len(vision_embeds_list) == batch_size:
                    # Stack into batch: [N, seq_len, embed_dim]
                    vision_batch = torch.cat(vision_embeds_list, dim=0)
                    
                    # Create uncond (zeros) with matching batch size
                    uncond_batch = torch.zeros_like(vision_batch)
                    
                    # Apply IP-Adapter patch to model clone
                    # The patch preserves batch dimension - Latent[i] sees Embed[i]
                    work_model = self._apply_ip_adapter_batch(
                        model, ip_adapter, vision_batch, uncond_batch, 
                        weight=ip_adapter_weight
                    )
                    print(f"[LunaSemanticDetailer] ✓ IP-Adapter patch applied: {batch_size} crops → {vision_batch.shape}")
            
            # Build conditioning for this batch
            batch_indices = range(i, min(i + tile_batch_size, len(crop_data)))
            batch_positive = []
            
            for j, idx in enumerate(batch_indices):
                # Start with cropped global conditioning
                combined_cond = cropped_global_positive[idx]
                
                # If concept override exists, concatenate it
                if idx < len(positive_list) and positive_list[idx] is not None:
                    concept_cond = positive_list[idx]
                    # Concatenate: cropped global + concept override
                    combined_cond = combined_cond + concept_cond
                
                batch_positive.append(combined_cond)
            
            # Replicate negative for batch
            batch_negative = self._replicate_conditioning(negative, len(batch))
            
            # Ensure all tensors are on the same device
            device = batch_latents.device
            batch_noise = batch_noise.to(device)
            
            # Pass scaffold noise to sampler - it will scale by appropriate sigma based on denoise
            # Use work_model (patched with IP-Adapter if available)
            with torch.inference_mode():
                refined_latent = comfy.sample.sample(
                    work_model,                     # Patched model (or original if no IP-Adapter)
                    noise=batch_noise,              # Scaffold noise - sampler scales it
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=batch_positive,
                    negative=batch_negative,
                    latent_image=batch_latents,    # Clean latent
                    denoise=denoise,
                    disable_noise=False,           # Let sampler inject scaled noise
                    start_step=None,               # Let denoise handle the schedule
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed + i
                )
            
            # Decode to pixels
            refined_pixels = vae.decode(refined_latent).permute(0, 2, 3, 1)
            
            # Move pixels to CPU to save VRAM
            refined_pixels = refined_pixels.cpu()
            
            # Store pixels only (no need for latent anymore)
            refined_crops.extend([
                {"pixels": refined_pixels[j:j+1]}
                for j in range(refined_pixels.shape[0])
            ])
            
            # Clear VRAM
            del refined_latent, batch_latents
            torch.cuda.empty_cache()
            
            # Allow interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        # Composite back onto 4K pixel canvas
        working_canvas_pixels = image.clone()
        
        # Start with previous refinement mask or create new
        if previous_refinement_mask is not None:
            refinement_mask = previous_refinement_mask.clone()
        else:
            refinement_mask = torch.zeros(b, h, w, device=device)
        
        for crop_info, refined_crop in zip(crop_data, refined_crops):
            self._composite_crop(
                working_canvas_pixels,
                refinement_mask,
                refined_crop,
                crop_info,
                h, w,
                enlarge_crops
            )
        
        print(f"[LunaSemanticDetailer] Refinement complete")
        
        return (working_canvas_pixels, detection_pipe, refinement_mask)
    
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
        - original_box: (x1, y1, x2, y2) in full image
        - resize_info: dict with scaling info
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
        mask_canvas: torch.Tensor,
        refined_data: dict,
        crop_info: dict,
        img_h: int,
        img_w: int,
        enlarge_crops: bool
    ):
        """
        Composite refined crop back onto 4K pixel canvas with smoothstep blending.
        
        If enlarge_crops=True: Paste refined crop at 1024×1024 (allows upscaling small regions)
        If enlarge_crops=False: Resize back to original crop size (default, for 4K inputs)
        """
        x1, y1, x2, y2 = crop_info["original_box"]
        orig_w, orig_h = crop_info["original_size"]
        
        refined_pixels = refined_data["pixels"]
        
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
            
            # Use full mask (no resizing needed)
            mask_resized = crop_info["mask_1024"][:actual_paste_h, :actual_paste_w]
            
        else:
            # Default behavior: resize back to original crop size
            paste_x1, paste_y1, paste_x2, paste_y2 = x1, y1, x2, y2
            
            # Always resize refined output back to original crop size
            # (we scaled up or down to 1024, now scale back)
            refined_resized = F.interpolate(
                refined_pixels.permute(0, 3, 1, 2),
                size=(orig_h, orig_w),
                mode='bicubic',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
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
    
    def _apply_ip_adapter_batch(
        self,
        model,
        ip_adapter,
        vision_batch: torch.Tensor,
        uncond_batch: torch.Tensor,
        weight: float = 0.5
    ):
        """
        Apply IP-Adapter patch to model with batched vision embeddings.
        
        KEY INSIGHT: PyTorch attention maps Latent[i] → Embed[i] when batch dims match.
        This enables TRUE BATCHING - 9 crops get 9 distinct vision anchors in ONE pass.
        
        Args:
            model: ComfyUI model to patch
            ip_adapter: IP-Adapter state dict
            vision_batch: Batched vision embeddings [N, seq_len, embed_dim]
            uncond_batch: Batched uncond embeddings [N, seq_len, embed_dim]
            weight: IP-Adapter strength
        
        Returns:
            Patched model clone
        """
        import comfy.model_management as model_management
        
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        
        # Clone model to avoid modifying original
        work_model = model.clone()
        
        # Move embeddings to correct device/dtype
        vision_batch = vision_batch.to(device=device, dtype=dtype)
        uncond_batch = uncond_batch.to(device=device, dtype=dtype)
        
        # Detect IP-Adapter configuration
        is_plus = (
            "proj.3.weight" in ip_adapter.get("image_proj", {}) or
            "latents" in ip_adapter.get("image_proj", {}) or
            "perceiver_resampler.proj_in.weight" in ip_adapter.get("image_proj", {})
        )
        
        # Get output cross-attention dim
        if "ip_adapter" in ip_adapter and "1.to_k_ip.weight" in ip_adapter["ip_adapter"]:
            output_cross_attention_dim = ip_adapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        else:
            output_cross_attention_dim = 2048  # SDXL default
        
        is_sdxl = output_cross_attention_dim == 2048
        
        # Import IP-Adapter components
        try:
            from custom_nodes.comfyui_ipadapter_plus.IPAdapterPlus import IPAdapter
            from custom_nodes.comfyui_ipadapter_plus.CrossAttentionPatch import Attn2Replace, ipadapter_attention
        except ImportError:
            print("[LunaSemanticDetailer] ⚠ IPAdapterPlus not available, skipping vision injection")
            return model
        
        # Create IPAdapter module
        cross_attention_dim = 1280 if is_plus and is_sdxl else output_cross_attention_dim
        clip_embeddings_dim = vision_batch.shape[-1]
        clip_extra_context_tokens = 16 if is_plus else 4
        
        ipa = IPAdapter(
            ip_adapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=False,
            is_faceid=False,
            is_portrait_unnorm=False,
        ).to(device, dtype=dtype)
        
        # Project embeddings through IP-Adapter
        # Batch dimension is preserved: [N, seq, dim] → [N, tokens, cross_dim]
        cond_embeds, uncond_embeds = ipa.get_image_embeds(vision_batch, uncond_batch, batch_size=0)
        
        # Set up patch kwargs
        # The critical part: cond_embeds has batch dim N, so Latent[i] sees Embed[i]
        patch_kwargs = {
            "ipadapter": ipa,
            "weight": weight,
            "cond": cond_embeds,
            "uncond": uncond_embeds,
            "weight_type": "linear",
            "mask": None,
            "sigma_start": 0.0,
            "sigma_end": 1.0,
            "unfold_batch": False,  # CRITICAL: Keep False to preserve per-item mapping
            "embeds_scaling": "V only",
        }
        
        # Apply attention patch
        work_model.set_model_attn2_replace(
            Attn2Replace(ipadapter_attention, **patch_kwargs),
            patch_kwargs
        )
        
        return work_model
    
    def _fuse_vision_conditioning(self, text_cond, vision_embed: torch.Tensor):
        """
        DEPRECATED: Use _apply_ip_adapter_batch instead.
        
        This naive fusion doesn't properly inject vision features into attention.
        Kept for backward compatibility but IP-Adapter is the correct approach.
        
        Args:
            text_cond: ComfyUI conditioning [(tensor [1, 77, dim], dict), ...]
            vision_embed: CLIP-ViT output [1, 257, vision_dim] (CLS + 256 patches)
        
        Returns:
            Fused conditioning with vision embedding appended
        """
        fused = []
        
        for emb, cond_dict in text_cond:
            # Get dimensions
            text_dim = emb.shape[-1]
            vision_dim = vision_embed.shape[-1]
            
            # Create modified dict with vision embedding
            new_dict = cond_dict.copy()
            
            # Store vision embedding in conditioning dict for cross-attention
            # This follows IPAdapter's approach of using 'pooled_output' or similar
            if "pooled_output" in new_dict:
                # If pooled output exists, we can enhance it
                existing_pooled = new_dict["pooled_output"]
                
                # Use CLS token from vision embed as additional pooled signal
                vision_cls = vision_embed[:, 0, :]  # [1, vision_dim]
                
                # Project vision to match text dim if needed
                if vision_dim != existing_pooled.shape[-1]:
                    # Simple linear projection (could be learned, but mean works)
                    # For now, just use mean pooling of vision features
                    vision_pooled = vision_embed.mean(dim=1)  # [1, vision_dim]
                    # Pad or truncate to match
                    if vision_dim > existing_pooled.shape[-1]:
                        vision_pooled = vision_pooled[:, :existing_pooled.shape[-1]]
                    else:
                        padding = torch.zeros(1, existing_pooled.shape[-1] - vision_dim, device=vision_pooled.device)
                        vision_pooled = torch.cat([vision_pooled, padding], dim=-1)
                    
                    # Blend with existing pooled output (50/50)
                    new_dict["pooled_output"] = existing_pooled * 0.5 + vision_pooled * 0.5
                else:
                    # Direct blend with CLS token
                    new_dict["pooled_output"] = existing_pooled * 0.5 + vision_cls * 0.5
            else:
                # No pooled output - add vision embedding as cross-attention target
                # Store in a key that samplers can use
                vision_cls = vision_embed[:, 0, :]  # [1, vision_dim]
                new_dict["structural_anchor"] = vision_cls
            
            fused.append([emb, new_dict])
        
        return fused


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaSemanticDetailer": LunaSemanticDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSemanticDetailer": "🌙 Luna: Semantic Detailer",
}
