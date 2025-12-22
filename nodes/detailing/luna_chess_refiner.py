"""
Luna Chess Refiner - Global Tiled Refinement

The final quality pass - performs chess-pattern tiled refinement across the
entire image with optional supersampling. Uses the same noise scaffold for
perfect variance preservation.

Architecture:
- Chess pattern (even/odd passes) for seamless blending
- Uses master noise scaffold (1:1 noise density)
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
    processing. Can optionally reduce denoise in areas already refined
    by semantic detailer.
    
    Key Features:
    - Chess pattern (even/odd) for seamless blending
    - Variance-preserved noise slicing from master scaffold
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
                    "tooltip": "Mask from Semantic Detailer (reduces denoise in refined areas)"
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
        refinement_mask = None
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
        
        # Initialize sampler for sigma calculation only
        sampler_obj = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=device,
            sampler=sampler,
            scheduler=scheduler,
            denoise=denoise,  # Use actual denoise for sigma calculation
            model_options=model.model_options
        )
        
        # Get the initial sigma for pre-noising (first sigma in the denoised schedule)
        initial_sigma = sampler_obj.sigmas[0]
        
        print(f"  Denoise: {denoise}, Initial sigma: {initial_sigma:.4f}, Steps: {len(sampler_obj.sigmas)-1}")
        
        # Extract noise slices from master scaffold (matching latent dimensions)
        scaff_samples = full_scaffold["samples"]
        noise_slices = scaff_samples[:, :, :lat_h, :lat_w]
        
        # Create working canvases
        latent_original = lat_samples.clone()  # Keep original for tile extraction
        latent_canvas = lat_samples.clone()    # GPU - for compositing results
        pixel_canvas = image.clone().cpu()     # CPU
        
        # Generate blend mask for tiles
        blend_mask = self._create_blend_mask(tile_lat, tile_lat, overlap_h, overlap_w, feathering, device)
        
        # Progress tracking
        total_tiles = rows * cols
        pbar = comfy.utils.ProgressBar(total_tiles)
        
        # Chess Pass 1: EVEN tiles
        latent_canvas, pixel_canvas = self._process_chess_pass(
            latent_canvas, pixel_canvas, noise_slices, "even", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed, sampler, scheduler,
            initial_sigma, tile_batch_size, blend_mask, refinement_mask, denoise, feathering, pbar
        )
        
        # Clear GPU cache between passes
        torch.cuda.empty_cache()
        
        # Chess Pass 2: ODD tiles
        latent_canvas, pixel_canvas = self._process_chess_pass(
            latent_canvas, pixel_canvas, noise_slices, "odd", rows, cols, tile_lat, overlap_h, overlap_w,
            model, positive, negative, vae, steps, cfg, seed + 1, sampler, scheduler,
            initial_sigma, tile_batch_size, blend_mask, refinement_mask, denoise, feathering, pbar
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
        self, latent_canvas, pixel_canvas, noise, parity, rows, cols, tile_size, ov_h, ov_w,
        model, pos, neg, vae, steps, cfg, seed, sampler_name, scheduler_name,
        initial_sigma, batch_size, blend_mask, refinement_mask, denoise, feathering, pbar
    ):
        """
        Process one chess pass (even or odd tiles).
        
        Key fix: Extract tiles from latent_original (unchanged), composite onto latent_canvas.
        This prevents artifacts from refining partially-refined latents.
        
        Returns both updated latent_canvas (GPU) and pixel_canvas (CPU).
        """
        stride_h = tile_size - ov_h
        stride_w = tile_size - ov_w
        
        H, W = latent_canvas.shape[2], latent_canvas.shape[3]
        device = latent_canvas.device
        
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
                    
                    # Extract crops FROM ORIGINAL (key fix!)
                    lat_crop = latent_original[:, :, y0:y1, x0:x1]
                    noise_crop = noise[:, :, y0:y1, x0:x1]
                    
                    tiles_data.append({
                        "latent": lat_crop,
                        "noise": noise_crop,
                        "coords": (y0, x0, y1, x1)
                    })
        
        # Process in batches
        for i in range(0, len(tiles_data), batch_size):
            chunk = tiles_data[i:i + batch_size]
            
            # Stack batch (clean latents from canvas)
            batch_latents = torch.cat([t["latent"] for t in chunk], dim=0)
            # Stack scaffold noise slices (1:1 matching, no resize needed)
            batch_scaffold_noise = torch.cat([t["noise"] for t in chunk], dim=0)
            
            # Ensure same device
            batch_scaffold_noise = batch_scaffold_noise.to(device)
            
            # Pre-inject scaffold noise at the starting sigma level
            initial_sigma_val = initial_sigma.item() if isinstance(initial_sigma, torch.Tensor) else initial_sigma
            noised_latents = batch_latents + batch_scaffold_noise * initial_sigma_val
            
            print(f"[LunaChessRefiner] {parity.upper()} Batch {i//batch_size + 1}: {len(chunk)} tiles, σ={initial_sigma_val:.4f}")
            
            # Sample with disable_noise=True since we pre-noised
            # This is like KSampler Advanced with add_noise="disable"
            with torch.inference_mode():
                refined_batch = comfy.sample.sample(
                    model,
                    noise=torch.zeros_like(noised_latents),  # Zero noise (already pre-noised)
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler_name,
                    positive=pos,
                    negative=neg,
                    latent_image=noised_latents,    # Pre-noised with scaffold
                    denoise=denoise,                # Let sampler handle the schedule!
                    disable_noise=True,             # Don't add more noise!
                    start_step=None,                # Let denoise handle it
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    sigmas=None,                    # Let sampler calculate from denoise
                    callback=None,
                    disable_pbar=False,
                    seed=seed + i
                )
            
            # Ensure refined batch stays on GPU for latent compositing
            if refined_batch.device != device:
                refined_batch = refined_batch.to(device)
            
            # DECODE tiles directly to CPU (in smaller sub-batches if needed)
            refined_pixels_list = []
            for j in range(0, refined_batch.shape[0], 4):  # Decode 4 at a time max
                sub_batch = refined_batch[j:j+4]
                with torch.no_grad():
                    decoded = vae.decode(sub_batch)
                    # VAE decode returns BCHW, convert to BHWC
                    if decoded.dim() == 4 and decoded.shape[1] == 3:  # [B, C, H, W]
                        decoded = decoded.permute(0, 2, 3, 1)  # → [B, H, W, C]
                    decoded = decoded.cpu()
                refined_pixels_list.append(decoded)
                del decoded
                torch.cuda.empty_cache()
            
            refined_pixels = torch.cat(refined_pixels_list, dim=0)
            del refined_pixels_list
            
            # Composite latent tiles onto latent canvas (GPU)
            self._composite_latent_tiles(latent_canvas, refined_batch, chunk, blend_mask)
            
            # Free GPU memory before pixel compositing
            del refined_batch, batch_latents, batch_scaffold_noise, noised_latents
            torch.cuda.empty_cache()
            
            # Composite pixel tiles onto pixel canvas (CPU)
            self._composite_pixel_tiles(pixel_canvas, refined_pixels, chunk, tile_size, ov_h, ov_w, feathering)
            
            # Free pixel batch
            del refined_pixels
            
            # Update progress
            pbar.update(len(chunk))
            
            # Allow interruption
            comfy.model_management.throw_exception_if_processing_interrupted()
        
        return latent_canvas, pixel_canvas
    
    def _composite_latent_tiles(self, latent_canvas, refined_batch, chunk_info, mask):
        """Composite refined latent tiles onto latent canvas (GPU operation)."""
        for i, info in enumerate(chunk_info):
            refined_tile = refined_batch[i]
            y0, x0, y1, x1 = info["coords"]
            
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
    
    def _composite_pixel_tiles(self, pixel_canvas, refined_pixels, chunk_info, tile_size, ov_h, ov_w, feathering):
        """Composite refined pixel tiles onto pixel canvas (CPU operation)."""
        # Create pixel-space blend mask
        pixel_tile_h = tile_size * 8
        pixel_tile_w = tile_size * 8
        pixel_ov_h = ov_h * 8
        pixel_ov_w = ov_w * 8
        
        pixel_mask = self._create_pixel_blend_mask(pixel_tile_h, pixel_tile_w, pixel_ov_h, pixel_ov_w, feathering)
        
        for i, info in enumerate(chunk_info):
            refined_tile = refined_pixels[i:i+1]  # Keep batch dim [1, H, W, C]
            y0_lat, x0_lat, y1_lat, x1_lat = info["coords"]
            
            # Convert latent coords to pixel coords
            y0 = y0_lat * 8
            x0 = x0_lat * 8
            y1 = y1_lat * 8
            x1 = x1_lat * 8
            
            current_bg = pixel_canvas[:, y0:y1, x0:x1, :]
            
            # Blend with mask (all CPU)
            diff = refined_tile - current_bg
            diff = diff * pixel_mask
            current_bg = current_bg + diff
            
            pixel_canvas[:, y0:y1, x0:x1, :] = current_bg
    
    def _create_pixel_blend_mask(self, h, w, ov_h, ov_w, feather):
        """Create blend mask for pixel tiles (CPU)."""
        mask = torch.ones((1, h, w, 1))  # BHWC format for pixels
        
        def smoothstep(length):
            t = torch.linspace(0, 1, length)
            if feather < 1.0:
                scale = 1.0 / max(0.01, feather)
                t = (t - 0.5) * scale + 0.5
                t = torch.clamp(t, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        
        if ov_h > 0:
            curve = smoothstep(ov_h).view(1, -1, 1, 1)
            mask[:, :ov_h, :, :] *= curve
            mask[:, -ov_h:, :, :] *= curve.flip(1)
        
        if ov_w > 0:
            curve = smoothstep(ov_w).view(1, 1, -1, 1)
            mask[:, :, :ov_w, :] *= curve
            mask[:, :, -ov_w:, :] *= curve.flip(2)
        
        return mask
    
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaChessRefiner": LunaChessRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaChessRefiner": "🌙 Luna: Chess Refiner",
}
