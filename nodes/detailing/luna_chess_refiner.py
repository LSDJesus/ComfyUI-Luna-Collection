"""
Luna Chess Refiner - Global Tiled Refinement

The final quality pass - performs chess-pattern tiled refinement across the
entire image with optional supersampling. Uses the same noise scaffold for
perfect variance preservation.

Architecture:
- Chess pattern (even/odd passes) for seamless blending
- Uses master noise scaffold (1:1 noise density)
- Dynamic CLIP-ViT routing (daemon or local) for structural anchoring
- Optional supersampling via Lanczos downscale
- Respects refinement mask (lighter touch on semantic-refined areas)
- Batched tile processing for VRAM efficiency

Workflow Integration:
    Semantic Detailer → Chess Refiner → Final Output (2K supersampled)
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.sampler_helpers

# Dynamic vision routing
from .vision_routing import VisionRouter, get_vision_router


def crop_conditioning_for_tile(conditioning, tile_region_px, canvas_size_px, scale_from=1.0):
    """
    Crop conditioning to match a specific tile region.
    
    Critical for tiled refinement - tells the model which spatial region
    it's working on instead of trying to generate the entire image in each tile.
    
    Args:
        conditioning: ComfyUI conditioning tuple
        tile_region_px: (x0, y0, x1, y1) in pixel space
        canvas_size_px: (width, height) of full canvas in pixels
        scale_from: Scale factor if conditioning was created at different resolution
                   (e.g., 4.0 if conditioning is from 1K but canvas is 4K)
    
    Returns:
        Cropped conditioning for the tile
    """
    x0, y0, x1, y1 = tile_region_px
    canvas_w, canvas_h = canvas_size_px
    
    cropped = []
    for emb, cond_dict in conditioning:
        cond_dict = cond_dict.copy()
        
        # Crop area conditioning if present
        if "area" in cond_dict:
            # Area format: (h, w, y, x) in latent space (1/8 pixels)
            h, w, y, x = cond_dict["area"]
            
            # Scale up area coords if conditioning was created at different resolution
            if scale_from != 1.0:
                h = int(h * scale_from)
                w = int(w * scale_from)
                y = int(y * scale_from)
                x = int(x * scale_from)
            
            # Convert to pixel space
            area_x0, area_y0 = x * 8, y * 8
            area_x1, area_y1 = area_x0 + w * 8, area_y0 + h * 8
            
            # Check intersection with tile
            intersect_x0 = max(area_x0, x0)
            intersect_y0 = max(area_y0, y0)
            intersect_x1 = min(area_x1, x1)
            intersect_y1 = min(area_y1, y1)
            
            if intersect_x0 < intersect_x1 and intersect_y0 < intersect_y1:
                # There's an intersection - adjust area to tile-relative coords
                rel_x0 = intersect_x0 - x0
                rel_y0 = intersect_y0 - y0
                rel_x1 = intersect_x1 - x0
                rel_y1 = intersect_y1 - y0
                
                # Convert back to latent space
                cond_dict["area"] = (
                    (rel_y1 - rel_y0) // 8,  # h
                    (rel_x1 - rel_x0) // 8,  # w
                    rel_y0 // 8,              # y
                    rel_x0 // 8               # x
                )
            else:
                # No intersection - remove area conditioning
                if "area" in cond_dict:
                    del cond_dict["area"]
                if "strength" in cond_dict:
                    del cond_dict["strength"]
        
        cropped.append([emb, cond_dict])
    
    return cropped


def get_scheduler_names():
    """Get current scheduler names dynamically to avoid type mismatches."""
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except (AttributeError, TypeError):
        return list(comfy.samplers.SCHEDULER_NAMES)


class LunaChessRefiner:
    """
    Global tiled refinement with chess pattern and optional supersampling.
    
    Performs final quality pass across entire image using batched tile
    processing with fresh VAE encoding and optional CLIP-ViT structural anchoring.
    
    Key Features:
    - Chess pattern (even/odd) for seamless blending
    - Fresh VAE encoding per tile (proper context)
    - Optional per-tile CLIP-ViT structural anchoring
    - Smoothstep blending for invisible seams
    - Optional supersampling (Lanczos downscale)
    - Refinement mask awareness
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "refine"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image to refine (from Semantic Detailer or Scaffold Upscaler)"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent of image (with semantic refinements if chained)"
                }),
                "full_scaffold": ("LATENT", {
                    "tooltip": "Full-resolution noise from Pyramid Generator"
                }),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
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
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoise strength for tiles"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sampler": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (get_scheduler_names(),),
                "tile_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size (1024 optimal for SDXL/Flux)"
                }),
                "tile_batch_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Process tiles in batches (VRAM safety)"
                }),
                "scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Output scale (0.5 = supersample to 50%, 1.0 = keep full size)"
                }),
                "feathering": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Edge softness (0.0 = hard, 1.0 = full smoothstep)"
                }),
            },
            "optional": {
                "refinement_mask": ("MASK", {
                    "tooltip": "Mask from Semantic Detailer (reduces noise in refined areas)"
                }),
                "mask_noise_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Noise multiplier for masked (already refined) areas. 0.5 = half noise in those regions"
                }),
                "clip_vision": ("CLIP_VISION", {
                    "tooltip": "CLIP-ViT model for structural anchoring (encodes each tile)"
                }),
                "use_structural_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable per-tile CLIP-ViT encoding for structural preservation"
                }),
            }
        }
    
    def refine(
        self,
        image: torch.Tensor,
        latent: dict,
        full_scaffold: dict,
        model,
        vae,
        positive,
        negative,
        steps: int,
        cfg: float,
        denoise: float,
        seed: int,
        sampler: str,
        scheduler: str,
        tile_size: int,
        tile_batch_size: int,
        scale: float,
        feathering: float,
        refinement_mask = None,
        mask_noise_factor: float = 0.5,
        clip_vision = None,
        use_structural_anchor: bool = True
    ) -> tuple:
        """
        Perform global tiled refinement with chess pattern.
        
        Returns:
            Tuple of (final_image,)
        """
        device = comfy.model_management.get_torch_device()
        
        # Get dimensions
        b, h, w, c = image.shape
        lat_samples = latent["samples"]
        lat_h, lat_w = lat_samples.shape[2], lat_samples.shape[3]
        
        print(f"[LunaChessRefiner] Starting refinement:")
        print(f"  Image: {w}×{h}, Latent: {lat_w}×{lat_h}")
        print(f"  Tile: {tile_size}px, Denoise: {denoise}, Scale: {scale}")
        
        # Calculate grid
        tile_lat = tile_size // 8
        rows, cols, overlap_h, overlap_w = self._calc_grid(lat_h, lat_w, tile_lat)
        
        print(f"  Grid: {rows}×{cols} tiles, Overlap: {overlap_h}×{overlap_w} latent")
        
        # Warn if only 1 tile (chess pattern ineffective)
        if rows == 1 and cols == 1:
            print(f"  ⚠️  Warning: Only 1 tile detected. Chess refinement ineffective.")
            print(f"     Suggest reducing tile_size to {max(32, tile_lat // 2)}px for multiple tiles.")
        
        # Extract noise slices from master scaffold (matching latent dimensions)
        scaff_samples = full_scaffold["samples"]
        noise_slices = scaff_samples[:, :, :lat_h, :lat_w]
        
        # Create working canvases
        latent_original = lat_samples.clone()  # Keep original for tile extraction
        latent_canvas = lat_samples.clone()    # GPU - for compositing results
        pixel_canvas = image.clone().cpu()     # CPU
        
        # Generate blend mask for tiles
        blend_mask = self._create_blend_mask(tile_lat, tile_lat, overlap_h, overlap_w, feathering, device)
        
        # Prepare refinement mask in latent space if provided
        latent_refinement_mask = None
        if refinement_mask is not None:
            # Refinement mask is in pixel space [B, H, W], resize to latent space
            mask_tensor = refinement_mask
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dim
            latent_refinement_mask = F.interpolate(
                mask_tensor.unsqueeze(1).float(),  # [B, 1, H, W]
                size=(lat_h, lat_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).to(device)  # [B, H, W]
            print(f"  Refinement mask provided: noise in masked areas scaled by {mask_noise_factor}")
        
        # Progress tracking
        total_tiles = rows * cols
        pbar = comfy.utils.ProgressBar(total_tiles)
        
        # Determine if we're using structural anchoring
        # Initialize vision router for dynamic daemon/local routing
        vision_router = None
        use_anchor = use_structural_anchor
        
        if use_structural_anchor:
            vision_router = get_vision_router(clip_vision)
            if vision_router.available:
                if vision_router.using_daemon:
                    print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: DAEMON (GPU offload)")
                else:
                    print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: LOCAL")
            else:
                print(f"[LunaChessRefiner] ⚠ No vision encoder available, using text-only")
                use_anchor = False
        else:
            print(f"[LunaChessRefiner] Using text-only conditioning")
        
        # Chess Pass 1: EVEN tiles
        latent_canvas, pixel_canvas = self._process_chess_pass(
            latent_canvas, latent_original, pixel_canvas, noise_slices, "even", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed, sampler, scheduler,
            tile_batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
            vision_router, use_anchor, image
        )
        
        # Clear GPU cache between passes
        torch.cuda.empty_cache()
        
        # Chess Pass 2: ODD tiles
        latent_canvas, pixel_canvas = self._process_chess_pass(
            latent_canvas, latent_original, pixel_canvas, noise_slices, "odd", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed + 1, sampler, scheduler,
            tile_batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
            vision_router, use_anchor, image
        )
        
        # Clear GPU cache (pixel canvas is already on CPU, fully built)
        torch.cuda.empty_cache()
        
        print("[LunaChessRefiner] Refinement complete!")
        
        # Optional supersampling on pixel canvas (already on CPU)
        if scale < 1.0:
            target_h = int(h * scale)
            target_w = int(w * scale)
            
            print(f"[LunaChessRefiner] Supersampling: {w}×{h} → {target_w}×{target_h}")
            pixel_canvas = self._lanczos_downscale(pixel_canvas, target_h, target_w)
        
        print("[LunaChessRefiner] Complete!")
        
        return (pixel_canvas,)
    
    def _calc_grid(self, lat_h: int, lat_w: int, tile_size: int) -> tuple:
        """
        Simple grid calculation: grid_size = round(dimension / tile_size) + 1
        Automatically provides good overlap (typically 256px / 32 latent).
        """
        # Calculate grid size
        rows = int(round(lat_h / tile_size)) + 1
        cols = int(round(lat_w / tile_size)) + 1
        
        # Handle edge case: if image is smaller than tile
        rows = max(1, rows)
        cols = max(1, cols)
        
        # Calculate overlap
        if rows > 1:
            total_coverage_h = rows * tile_size
            excess_h = total_coverage_h - lat_h
            overlap_h = excess_h // (rows - 1)
        else:
            overlap_h = 0
        
        if cols > 1:
            total_coverage_w = cols * tile_size
            excess_w = total_coverage_w - lat_w
            overlap_w = excess_w // (cols - 1)
        else:
            overlap_w = 0
        
        return rows, cols, overlap_h, overlap_w
    
    def _process_chess_pass(
        self, latent_canvas, latent_original, pixel_canvas, noise, parity, rows, cols, tile_size, ov_h, ov_w,
        model, pos, neg, vae, steps, cfg, seed, sampler_name, scheduler_name,
        batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
        vision_router=None, use_anchor=False, full_image=None
    ):
        """
        Process one chess pass (even or odd tiles).
        
        CRITICAL FIX: Crop PIXELS and encode them fresh (like USDU does).
        Slicing pre-encoded latents loses VAE's tile-local encoding context.
        
        Flow per batch:
        1. Crop pixel tiles from pixel_canvas
        2. Batch encode all tiles → fresh latents with proper VAE context
        3. CLIP-ViT encode via dynamic router (daemon or local)
        4. Slice scaffold noise (this is fine - no encoding issues)
        5. Refine with fresh latents + scaffold noise + anchored conditioning
        6. Composite refined latents and pixels back
        
        Returns both updated latent_canvas (GPU) and pixel_canvas (CPU).
        """
        stride_h = tile_size - ov_h
        stride_w = tile_size - ov_w
        
        H, W = latent_canvas.shape[2], latent_canvas.shape[3]
        device = latent_canvas.device
        
        # Pixel dimensions (latent * 8)
        pixel_H = H * 8
        pixel_W = W * 8
        tile_size_px = tile_size * 8
        
        # Extract tiles for this parity
        tiles_data = []
        target_mod = 0 if parity == "even" else 1
        
        for yi in range(rows):
            for xi in range(cols):
                if (yi + xi) % 2 == target_mod:
                    y0 = yi * stride_h
                    x0 = xi * stride_w
                    
                    # Edge snapping
                    if yi == rows - 1:
                        y0 = H - tile_size
                    if xi == cols - 1:
                        x0 = W - tile_size
                    
                    y1, x1 = y0 + tile_size, x0 + tile_size
                    
                    # Calculate pixel coordinates
                    y0_px = y0 * 8
                    x0_px = x0 * 8
                    y1_px = y1 * 8
                    x1_px = x1 * 8
                    
                    # Store tile info (no latent crop yet!)
                    tiles_data.append({
                        "coords_lat": (y0, x0, y1, x1),
                        "coords_px": (y0_px, x0_px, y1_px, x1_px),
                        "grid_pos": (yi, xi),
                        "is_edge": {
                            "top": yi == 0,
                            "bottom": yi == rows - 1,
                            "left": xi == 0,
                            "right": xi == cols - 1
                        }
                    })
        
        # Process in batches
        for i in range(0, len(tiles_data), batch_size):
            chunk = tiles_data[i:i + batch_size]
            
            # STEP 1: Crop PIXELS from canvas (BHWC format)
            pixel_crops = []
            for t in chunk:
                y0_px, x0_px, y1_px, x1_px = t["coords_px"]
                pixel_crop = pixel_canvas[:, y0_px:y1_px, x0_px:x1_px, :].clone()
                pixel_crops.append(pixel_crop)
            
            # Stack into batch [N, H, W, C]
            pixel_batch = torch.cat(pixel_crops, dim=0)
            
            # STEP 2: Encode pixels → fresh latents (SINGLE VAE CALL!)
            # Convert BHWC → BCHW for VAE
            pixel_batch_vae = pixel_batch.permute(0, 3, 1, 2).to(device)
            
            with torch.no_grad():
                batch_latents = vae.encode(pixel_batch_vae)  # Fresh encoding with tile context! ✨
            
            # STEP 3: Slice scaffold noise (this is fine - noise has no encoding context)
            noise_crops = []
            for t in chunk:
                y0, x0, y1, x1 = t["coords_lat"]
                noise_crop = noise[:, :, y0:y1, x0:x1]
                noise_crops.append(noise_crop)
            
            batch_scaffold_noise = torch.cat(noise_crops, dim=0)
            
            # STEP 4: CLIP-ViT structural anchoring via dynamic router
            anchored_positive = pos
            if use_anchor and vision_router is not None and full_image is not None:
                # Get crop coordinates in pixel space
                tile_crop_coords = [(t["coords_px"][1], t["coords_px"][0], t["coords_px"][3], t["coords_px"][2]) 
                                   for t in chunk]  # (x1, y1, x2, y2)
                
                # Actually we need (x1, y1, x2, y2) format - let me fix
                tile_crop_coords = []
                for t in chunk:
                    y0_px, x0_px, y1_px, x1_px = t["coords_px"]
                    tile_crop_coords.append((x0_px, y0_px, x1_px, y1_px))
                
                # Use router - handles daemon vs local automatically
                vision_embeds_list = vision_router.encode_crops(
                    full_image=full_image,
                    crop_coords=tile_crop_coords,
                    tile_size=tile_size * 8  # Pixel size
                )
                
                # Stack embeddings for batch fusion
                if vision_embeds_list:
                    vision_embeds = torch.cat(vision_embeds_list, dim=0)
                    anchored_positive = self._fuse_vision_conditioning_batch(
                        pos, vision_embeds, len(chunk)
                    )
            
            # STEP 5: Extract refinement mask crops if available
            batch_noise_mask = None
            if latent_refinement_mask is not None:
                mask_crops = []
                for t in chunk:
                    y0, x0, y1, x1 = t["coords_lat"]
                    mask_crop = latent_refinement_mask[:, y0:y1, x0:x1]
                    mask_crops.append(mask_crop)
                
                mask_batch = torch.cat(mask_crops, dim=0)  # [B, H, W]
                # Convert to noise scaling: unrefined=1.0, refined=mask_noise_factor
                batch_noise_mask = 1.0 - mask_batch * (1.0 - mask_noise_factor)
                batch_noise_mask = batch_noise_mask.unsqueeze(1).to(device)  # [B, 1, H, W]
            
            # Ensure same device
            batch_scaffold_noise = batch_scaffold_noise.to(device)
            
            # Apply noise mask to scaffold noise if available
            if batch_noise_mask is not None:
                batch_scaffold_noise = batch_scaffold_noise * batch_noise_mask
                print(f"[LunaChessRefiner] {parity.upper()} Batch {i//batch_size + 1}: {len(chunk)} tiles, "
                      f"fresh VAE encode, denoise={denoise}, mask_factor={mask_noise_factor}")
            else:
                print(f"[LunaChessRefiner] {parity.upper()} Batch {i//batch_size + 1}: {len(chunk)} tiles, "
                      f"fresh VAE encode, denoise={denoise}")
            
            # STEP 6: Refine with fresh latents + scaffold noise + anchored conditioning
            # Pass scaffold noise to sampler - it will scale by appropriate sigma based on denoise
            with torch.inference_mode():
                refined_batch = comfy.sample.sample(
                    model,
                    noise=batch_scaffold_noise,     # Scaffold noise (optionally masked)
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler_name,
                    positive=anchored_positive,     # With CLIP-ViT anchor if enabled
                    negative=neg,
                    latent_image=batch_latents,     # Fresh tile-encoded latents! ✨
                    denoise=denoise,
                    disable_noise=False,            # Let sampler inject scaled noise
                    start_step=None,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,                # We pre-scaled the noise
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed + i
                )
            
            # Ensure refined batch stays on GPU for latent compositing
            if refined_batch.device != device:
                refined_batch = refined_batch.to(device)
            
            # STEP 6: Decode refined tiles to pixels (in sub-batches to save VRAM)
            refined_pixels_list = []
            for j in range(0, refined_batch.shape[0], 4):
                sub_batch = refined_batch[j:j+4]
                with torch.no_grad():
                    decoded = vae.decode(sub_batch)
                    if decoded.dim() == 4 and decoded.shape[1] == 3:  # [B, C, H, W]
                        decoded = decoded.permute(0, 2, 3, 1)  # → [B, H, W, C]
                    decoded = decoded.cpu()
                refined_pixels_list.append(decoded)
                del decoded
                torch.cuda.empty_cache()
            
            refined_pixels = torch.cat(refined_pixels_list, dim=0)
            del refined_pixels_list
            
            # STEP 7: Composite latent tiles onto latent canvas (GPU)
            self._composite_latent_tiles(latent_canvas, refined_batch, chunk, blend_mask, parity)
            
            # Free GPU memory before pixel compositing
            del refined_batch, batch_latents, batch_scaffold_noise, pixel_batch, pixel_batch_vae
            if batch_noise_mask is not None:
                del batch_noise_mask
            torch.cuda.empty_cache()
            
            # STEP 8: Composite pixel tiles onto pixel canvas (CPU)
            self._composite_pixel_tiles(pixel_canvas, refined_pixels, chunk, tile_size, ov_h, ov_w, feathering, parity)
            
            # Free pixel batch
            del refined_pixels
            
            # Update progress
            pbar.update(len(chunk))
            
            # Allow interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        return latent_canvas, pixel_canvas
    
    def _composite_latent_tiles(self, latent_canvas, refined_batch, chunk_info, mask, parity):
        """Composite refined latent tiles onto latent canvas (GPU operation)."""
        # Even pass: no blending (first tiles placed)
        # Odd pass: blend with mask
        use_blending = (parity == "odd")
        
        for i, info in enumerate(chunk_info):
            refined_tile = refined_batch[i]
            y0, x0, y1, x1 = info["coords_lat"]  # Use latent coords
            
            if use_blending:
                current_bg = latent_canvas[:, :, y0:y1, x0:x1]
                
                # Ensure same device
                device = latent_canvas.device
                refined_tile = refined_tile.to(device)
                mask_tile = mask.to(device)
                
                # Blend with mask
                diff = refined_tile - current_bg
                diff = diff * mask_tile
                current_bg = current_bg + diff
                
                latent_canvas[:, :, y0:y1, x0:x1] = current_bg
            else:
                # Even pass: direct paste, no blending
                latent_canvas[:, :, y0:y1, x0:x1] = refined_tile.to(latent_canvas.device)
    
    def _composite_pixel_tiles(self, pixel_canvas, refined_pixels, chunk_info, tile_size, ov_h, ov_w, feathering, parity):
        """Composite refined pixel tiles onto pixel canvas (CPU operation)."""
        # Even pass: no feathering (first tiles)
        # Odd pass: feather only edges that overlap with even tiles
        use_feathering = (parity == "odd")
        
        pixel_tile_h = tile_size * 8
        pixel_tile_w = tile_size * 8
        pixel_ov_h = ov_h * 8
        pixel_ov_w = ov_w * 8
        
        for i, info in enumerate(chunk_info):
            refined_tile = refined_pixels[i:i+1]  # Keep batch dim [1, H, W, C]
            y0_lat, x0_lat, y1_lat, x1_lat = info["coords_lat"]  # Use latent coords
            
            # Convert latent coords to pixel coords
            y0 = y0_lat * 8
            x0 = x0_lat * 8
            y1 = y1_lat * 8
            x1 = x1_lat * 8
            
            if use_feathering:
                # Create edge-specific mask for odd tiles
                # Only feather edges that aren't on the canvas boundary
                is_edge = info["is_edge"]
                pixel_mask = self._create_edge_aware_blend_mask(
                    pixel_tile_h, pixel_tile_w, 
                    pixel_ov_h, pixel_ov_w, 
                    feathering,
                    feather_top=not is_edge["top"],
                    feather_bottom=not is_edge["bottom"],
                    feather_left=not is_edge["left"],
                    feather_right=not is_edge["right"]
                )
                
                current_bg = pixel_canvas[:, y0:y1, x0:x1, :]
                
                # Blend with mask (all CPU)
                diff = refined_tile - current_bg
                diff = diff * pixel_mask
                current_bg = current_bg + diff
                
                pixel_canvas[:, y0:y1, x0:x1, :] = current_bg
            else:
                # Even pass: direct paste, no feathering
                pixel_canvas[:, y0:y1, x0:x1, :] = refined_tile
    
    def _create_edge_aware_blend_mask(self, h, w, ov_h, ov_w, feather, 
                                       feather_top=True, feather_bottom=True, 
                                       feather_left=True, feather_right=True):
        """Create blend mask with selective edge feathering."""
        mask = torch.ones((1, h, w, 1))  # BHWC format for pixels
        
        def smoothstep(length):
            t = torch.linspace(0, 1, length)
            if feather < 1.0:
                scale = 1.0 / max(0.01, feather)
                t = (t - 0.5) * scale + 0.5
                t = torch.clamp(t, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        
        # Only apply feathering to edges that should be blended
        if ov_h > 0:
            curve = smoothstep(ov_h).view(1, -1, 1, 1)
            if feather_top:
                mask[:, :ov_h, :, :] *= curve
            if feather_bottom:
                mask[:, -ov_h:, :, :] *= curve.flip(1)
        
        if ov_w > 0:
            curve = smoothstep(ov_w).view(1, 1, -1, 1)
            if feather_left:
                mask[:, :, :ov_w, :] *= curve
            if feather_right:
                mask[:, :, -ov_w:, :] *= curve.flip(2)
        
        return mask
    
    def _create_pixel_blend_mask(self, h, w, ov_h, ov_w, feather):
        """Create blend mask for pixel tiles (CPU). DEPRECATED - use _create_edge_aware_blend_mask"""
        return self._create_edge_aware_blend_mask(h, w, ov_h, ov_w, feather, 
                                                   True, True, True, True)
    
    def _composite_tiles(self, canvas, refined_batch, chunk_info, mask):
        """Composite refined tiles onto canvas with blending."""
        for i, info in enumerate(chunk_info):
            refined_tile = refined_batch[i]
            y0, x0, y1, x1 = info["coords"]
            
            current_bg = canvas[:, :, y0:y1, x0:x1]
            
            # Ensure same device
            device = canvas.device
            refined_tile = refined_tile.to(device)
            mask_tile = mask.to(device)
            
            # Blend with mask
            diff = refined_tile - current_bg
            diff = diff * mask_tile
            current_bg = current_bg + diff
            
            canvas[:, :, y0:y1, x0:x1] = current_bg
    
    def _create_blend_mask(self, h, w, ov_h, ov_w, feather, device):
        """Create smoothstep blend mask for tiles."""
        mask = torch.ones((1, 1, h, w), device=device)
        
        def smoothstep_curve(length):
            t = torch.linspace(0, 1, length, device=device)
            
            # Apply feathering
            if feather < 1.0:
                scale = 1.0 / max(0.01, feather)
                t = (t - 0.5) * scale + 0.5
                t = torch.clamp(t, 0.0, 1.0)
            
            # Smoothstep polynomial
            return t * t * (3.0 - 2.0 * t)
        
        # Apply to edges
        if ov_h > 0:
            curve = smoothstep_curve(ov_h).view(1, 1, -1, 1)
            mask[:, :, :ov_h, :] *= curve
            mask[:, :, -ov_h:, :] *= curve.flip(2)
        
        if ov_w > 0:
            curve = smoothstep_curve(ov_w).view(1, 1, 1, -1)
            mask[:, :, :, :ov_w] *= curve
            mask[:, :, :, -ov_w:] *= curve.flip(3)
        
        return mask
    
    def _lanczos_downscale(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """GPU-accelerated bicubic downscale for supersampling."""
        # Convert BHWC → BCHW
        img_chw = img.permute(0, 3, 1, 2)
        
        # Bicubic resize (fully GPU-accelerated)
        downscaled_chw = F.interpolate(
            img_chw,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        # Convert back BCHW → BHWC
        return downscaled_chw.permute(0, 2, 3, 1)
    
    def _fuse_vision_conditioning_batch(self, text_cond, vision_embeds: torch.Tensor, batch_size: int):
        """
        Fuse CLIP-ViT vision embeddings with text conditioning for a batch of tiles.
        
        Creates structural anchors by blending vision features with text embeddings.
        This preserves the spatial structure of each tile during refinement.
        
        Args:
            text_cond: ComfyUI conditioning [(tensor [1, 77, dim], dict), ...]
            vision_embeds: CLIP-ViT outputs [N, 257, vision_dim] (CLS + 256 patches per tile)
            batch_size: Number of tiles in this batch
        
        Returns:
            Fused conditioning with vision embeddings incorporated
        """
        fused = []
        
        for emb, cond_dict in text_cond:
            # Get dimensions
            text_dim = emb.shape[-1]
            vision_dim = vision_embeds.shape[-1]
            
            # Create modified dict with vision embedding
            new_dict = cond_dict.copy()
            
            # Use mean of CLS tokens across batch as pooled anchor
            # This provides a batch-averaged structural reference
            vision_cls_batch = vision_embeds[:, 0, :]  # [N, vision_dim]
            vision_pooled = vision_cls_batch.mean(dim=0, keepdim=True)  # [1, vision_dim]
            
            if "pooled_output" in new_dict:
                existing_pooled = new_dict["pooled_output"]
                
                # Match dimensions if needed
                if vision_dim != existing_pooled.shape[-1]:
                    if vision_dim > existing_pooled.shape[-1]:
                        vision_pooled = vision_pooled[:, :existing_pooled.shape[-1]]
                    else:
                        padding = torch.zeros(1, existing_pooled.shape[-1] - vision_dim, 
                                            device=vision_pooled.device)
                        vision_pooled = torch.cat([vision_pooled, padding], dim=-1)
                
                # Blend 50/50 with existing pooled output
                new_dict["pooled_output"] = existing_pooled * 0.5 + vision_pooled * 0.5
            else:
                # Store as structural anchor reference
                new_dict["structural_anchor"] = vision_pooled
            
            # Replicate embedding for batch
            batched_emb = emb.repeat(batch_size, 1, 1)
            fused.append([batched_emb, new_dict])
        
        return fused


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaChessRefiner": LunaChessRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaChessRefiner": "🌙 Luna: Chess Refiner",
}
