"""
Luna Chess Refiner - Global Tiled Refinement

The final quality pass - performs chess-pattern tiled refinement across the
entire image with optional supersampling. Uses the same noise scaffold for
perfect variance preservation.

Architecture:
- Chess pattern (even/odd passes) for seamless blending
- Uses master noise scaffold (1:1 noise density)
- IP-Adapter structural anchoring with TRUE BATCHING
- Optional supersampling via Lanczos downscale
- Respects refinement mask (lighter touch on semantic-refined areas)
- Batched tile processing for VRAM efficiency

IP-Adapter Batching:
- Latent[i] only "sees" Adapter Embed[i] (PyTorch attention physics)
- No broadcasting if batch dims match: [N, 4, H, W] + [N, 16, 2048]
- One patch, one sample call, N distinct results

Workflow Integration:
    Semantic Detailer → Chess Refiner → Final Output (2K supersampled)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.sampler_helpers

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

# IP-Adapter for proper attention injection
try:
    from custom_nodes.comfyui_ipadapter_plus.IPAdapterPlus import IPAdapter
    from custom_nodes.comfyui_ipadapter_plus.CrossAttentionPatch import Attn2Replace, ipadapter_attention
    HAS_IPADAPTER = True
except ImportError:
    HAS_IPADAPTER = False
    IPAdapter = None


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
                    "tooltip": "4K pixel image (from Semantic Detailer or Prep Upscaler)"
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
                "ip_adapter": ("IPADAPTER", {
                    "tooltip": "IP-Adapter model for proper vision→attention injection"
                }),
                "use_structural_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable per-tile CLIP-ViT + IP-Adapter for structural preservation"
                }),
                "ip_adapter_weight": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IP-Adapter strength for tile refinement (lower than detailer for global coherence)"
                }),
            }
        }
    
    def refine(
        self,
        image: torch.Tensor,
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
        ip_adapter = None,
        use_structural_anchor: bool = True,
        ip_adapter_weight: float = 0.4
    ) -> tuple:
        """
        Perform global tiled refinement with chess pattern.
        
        All work happens in pixel space - crops from 4K canvas, encodes fresh,
        refines, decodes, and pastes back to 4K pixel canvas.
        
        Returns:
            Tuple of (final_image,)
        """
        device = comfy.model_management.get_torch_device()
        
        # Get dimensions
        b, h, w, c = image.shape
        
        # Extract noise from master scaffold (needed for refinement)
        scaff_samples = full_scaffold["samples"]
        lat_h, lat_w = scaff_samples.shape[2], scaff_samples.shape[3]
        noise_slices = scaff_samples[:, :, :lat_h, :lat_w]
        
        print(f"[LunaChessRefiner] Starting refinement:")
        print(f"  Image: {w}×{h} pixels, Scaffold: {lat_w}×{lat_h} latent")
        print(f"  Tile: {tile_size}px, Denoise: {denoise}, Scale: {scale}")
        
        # Calculate grid
        tile_lat = tile_size // 8
        rows, cols, overlap_h, overlap_w = self._calc_grid(lat_h, lat_w, tile_lat)
        
        print(f"  Grid: {rows}×{cols} tiles, Overlap: {overlap_h}×{overlap_w} latent")
        
        # Warn if only 1 tile (chess pattern ineffective)
        if rows == 1 and cols == 1:
            print(f"  ⚠️  Warning: Only 1 tile detected. Chess refinement ineffective.")
            print(f"     Suggest reducing tile_size to {max(32, tile_lat // 2)}px for multiple tiles.")
        
        # Create working pixel canvas (CPU)
        pixel_canvas = image.clone().cpu()
        
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
        
        # Initialize vision router for CLIP-ViT encoding (daemon or local)
        vision_router = None
        use_anchor = use_structural_anchor and (clip_vision is not None or ip_adapter is not None)
        
        if use_structural_anchor:
            if ip_adapter is not None and HAS_IPADAPTER:
                vision_router = get_vision_router(clip_vision)
                if vision_router.available:
                    print(f"[LunaChessRefiner] ✓ IP-Adapter structural anchoring ENABLED (weight={ip_adapter_weight})")
                    if vision_router.using_daemon:
                        print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: DAEMON")
                    else:
                        print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: LOCAL")
                else:
                    print(f"[LunaChessRefiner] ⚠ IP-Adapter provided but no CLIP-ViT available!")
                    print(f"[LunaChessRefiner] ⚠ Provide clip_vision input OR ensure daemon has vision loaded")
                    use_anchor = False
                    vision_router = None
            elif clip_vision is not None:
                vision_router = get_vision_router(clip_vision)
                if vision_router.available:
                    if vision_router.using_daemon:
                        print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: DAEMON")
                    else:
                        print(f"[LunaChessRefiner] ✓ CLIP-ViT routing: LOCAL")
                    # Without IP-Adapter, vision won't be properly injected
                    print(f"[LunaChessRefiner] ⚠ No IP-Adapter provided - using text-only!")
                    use_anchor = False
                else:
                    print(f"[LunaChessRefiner] ⚠ No vision encoder available, using text-only")
                    use_anchor = False
            else:
                print(f"[LunaChessRefiner] ⚠ No clip_vision or ip_adapter provided")
                use_anchor = False
        else:
            print(f"[LunaChessRefiner] Using text-only conditioning")
        
        # Chess Pass 1: EVEN tiles
        pixel_canvas = self._process_chess_pass(
            pixel_canvas, noise_slices, "even", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed, sampler, scheduler,
            tile_batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
            vision_router, use_anchor, image, ip_adapter, ip_adapter_weight
        )
        
        # Clear GPU cache between passes
        torch.cuda.empty_cache()
        
        # Chess Pass 2: ODD tiles
        pixel_canvas = self._process_chess_pass(
            pixel_canvas, noise_slices, "odd", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed + 1, sampler, scheduler,
            tile_batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
            vision_router, use_anchor, image, ip_adapter, ip_adapter_weight
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
        self, pixel_canvas, noise, parity, rows, cols, tile_size, ov_h, ov_w,
        model, pos, neg, vae, steps, cfg, seed, sampler_name, scheduler_name,
        batch_size, blend_mask, latent_refinement_mask, mask_noise_factor, denoise, feathering, pbar,
        vision_router=None, use_anchor=False, full_image=None, ip_adapter=None, ip_adapter_weight=0.4
    ):
        """
        Process one chess pass (even or odd tiles).
        
        CRITICAL FIX: Crop PIXELS and encode them fresh (like USDU does).
        Slicing pre-encoded latents loses VAE's tile-local encoding context.
        
        Flow per batch:
        1. Crop pixel tiles from pixel_canvas
        2. Batch encode all tiles → fresh latents with proper VAE context
        3. Encode via CLIP-ViT (daemon or local), apply via IP-Adapter
        4. Slice scaffold noise (this is fine - no encoding issues)
        5. Refine with patched model + scaffold noise + text conditioning
        6. Decode and composite refined pixels back to pixel_canvas
        
        IP-Adapter: Batch dimension preserved - Latent[i] sees Embed[i]
        
        Returns updated pixel_canvas (CPU).
        """
        stride_h = tile_size - ov_h
        stride_w = tile_size - ov_w
        
        # Get dimensions from noise scaffold
        H, W = noise.shape[2], noise.shape[3]
        device = noise.device
        
        # Pixel dimensions (latent * 8)
        pixel_H = H * 8
        pixel_W = W * 8
        tile_size_px = tile_size * 8
        
        # Debug: Verify pixel_canvas matches expected dimensions
        actual_canvas_h, actual_canvas_w = pixel_canvas.shape[1], pixel_canvas.shape[2]
        if actual_canvas_h != pixel_H or actual_canvas_w != pixel_W:
            print(f"[LunaChessRefiner] ⚠ Canvas size mismatch!")
            print(f"  Expected (from latent): {pixel_H}×{pixel_W}")
            print(f"  Actual canvas: {actual_canvas_h}×{actual_canvas_w}")
            print(f"  Latent scaffold: {H}×{W}")
            # Adjust pixel dimensions to match actual canvas
            pixel_H = actual_canvas_h
            pixel_W = actual_canvas_w
        
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
            valid_tiles = []  # Track which tiles produced valid crops
            for t in chunk:
                y0_px, x0_px, y1_px, x1_px = t["coords_px"]
                
                # Validate coordinates to prevent zero-size crops
                if y0_px >= y1_px or x0_px >= x1_px:
                    print(f"[LunaChessRefiner] ✗ Invalid crop coords: y={y0_px}:{y1_px}, x={x0_px}:{x1_px}")
                    continue
                    
                pixel_crop = pixel_canvas[:, y0_px:y1_px, x0_px:x1_px, :].clone()
                
                # Validate crop shape
                if pixel_crop.shape[1] == 0 or pixel_crop.shape[2] == 0:
                    print(f"[LunaChessRefiner] ✗ Zero-size crop: {pixel_crop.shape} from coords y={y0_px}:{y1_px}, x={x0_px}:{x1_px}, canvas={pixel_canvas.shape}")
                    continue
                    
                pixel_crops.append(pixel_crop)
                valid_tiles.append(t)  # Only add tile if crop is valid
            
            if len(pixel_crops) == 0:
                print(f"[LunaChessRefiner] ✗ No valid pixel crops in batch, skipping")
                continue
            
            # Use only valid tiles from here on
            chunk = valid_tiles
            
            # Stack into batch [N, H, W, C]
            pixel_batch = torch.cat(pixel_crops, dim=0)
            
            # STEP 2: Encode pixels → fresh latents (SINGLE VAE CALL!)
            # ComfyUI's vae.encode() expects BHWC format and handles conversion internally
            pixel_batch_for_vae = pixel_batch.to(device)
            
            with torch.no_grad():
                batch_latents = vae.encode(pixel_batch_for_vae)  # Fresh encoding with tile context! ✨
            
            # STEP 3: Slice scaffold noise (this is fine - noise has no encoding context)
            noise_crops = []
            for t in chunk:
                y0, x0, y1, x1 = t["coords_lat"]
                noise_crop = noise[:, :, y0:y1, x0:x1]
                noise_crops.append(noise_crop)
            
            batch_scaffold_noise = torch.cat(noise_crops, dim=0)
            
            # STEP 4: IP-ADAPTER STRUCTURAL ANCHORING (TRUE BATCHING)
            # Key insight: Latent[i] only sees Embed[i] - no averaging!
            work_model = model
            chunk_size = len(chunk)
            
            if use_anchor and ip_adapter is not None and vision_router is not None and full_image is not None:
                # Get crop coordinates in pixel space
                tile_crop_coords = []
                for t in chunk:
                    y0_px, x0_px, y1_px, x1_px = t["coords_px"]
                    tile_crop_coords.append((x0_px, y0_px, x1_px, y1_px))
                
                # Encode via CLIP-ViT (daemon or local)
                vision_embeds_list = vision_router.encode_crops(
                    full_image=full_image,
                    crop_coords=tile_crop_coords,
                    tile_size=tile_size * 8  # Pixel size
                )
                
                if vision_embeds_list and len(vision_embeds_list) == chunk_size:
                    # Stack into batch: [N, seq_len, embed_dim]
                    vision_batch = torch.cat(vision_embeds_list, dim=0)
                    uncond_batch = torch.zeros_like(vision_batch)
                    
                    # Apply IP-Adapter patch - batch dimension preserved!
                    work_model = self._apply_ip_adapter_batch(
                        model, ip_adapter, vision_batch, uncond_batch,
                        weight=ip_adapter_weight
                    )
            
            # Replicate positive conditioning for batch
            anchored_positive = self._replicate_conditioning(pos, chunk_size)
            
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
            
            # STEP 6: Refine with patched model + scaffold noise + text conditioning
            # Use work_model (patched with IP-Adapter if available)
            with torch.inference_mode():
                refined_batch = comfy.sample.sample(
                    work_model,                     # Patched model (or original if no IP-Adapter)
                    noise=batch_scaffold_noise,     # Scaffold noise (optionally masked)
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler_name,
                    positive=anchored_positive,     # Replicated for batch
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
            
            # STEP 7: Composite pixel tiles onto pixel canvas (CPU)
            self._composite_pixel_tiles(pixel_canvas, refined_pixels, chunk, tile_size, ov_h, ov_w, feathering, parity)
            
            # Free memory
            del refined_batch, batch_latents, batch_scaffold_noise, pixel_batch, pixel_batch_for_vae, refined_pixels
            if batch_noise_mask is not None:
                del batch_noise_mask
            torch.cuda.empty_cache()
            
            # Update progress
            pbar.update(len(chunk))
            
            # Allow interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        return pixel_canvas
    
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
        weight: float = 0.4
    ):
        """
        Apply IP-Adapter patch to model with batched vision embeddings.
        
        KEY INSIGHT: PyTorch attention maps Latent[i] → Embed[i] when batch dims match.
        This enables TRUE BATCHING - N tiles get N distinct vision anchors in ONE pass.
        
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
        if not HAS_IPADAPTER:
            print("[LunaChessRefiner] ⚠ IPAdapterPlus not available, skipping vision injection")
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
        
        # Set up patch kwargs - batch dimension preserved!
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
    
    def _fuse_vision_conditioning_batch(self, text_cond, vision_embeds: torch.Tensor, batch_size: int):
        """
        DEPRECATED: Use _apply_ip_adapter_batch instead.
        
        This naive fusion doesn't properly inject vision features into attention.
        Kept for backward compatibility but IP-Adapter is the correct approach.
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
