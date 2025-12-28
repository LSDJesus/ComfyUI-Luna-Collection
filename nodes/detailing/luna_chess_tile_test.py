"""
Luna Chess Tile Test - Single Tile Refinement Debug Node

Extracts a single tile from the 4K canvas and refines it using the canvas
latent and noise scaffold, then composites the refined tile back onto the
original canvas. This isolates tile-level issues (color shifts, artifacts).

Key for debugging:
- Tile extraction and replacement
- Latent vs pixel space blending
- Color consistency
- Artifact sources

Once working on a single tile, we can expand to full chess pattern.
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management


def get_scheduler_names():
    """Get current scheduler names dynamically."""
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except (AttributeError, TypeError):
        return list(comfy.samplers.SCHEDULER_NAMES)


class LunaChessTileTest:
    """
    Single tile extraction, refinement, and compositing test.
    
    Inputs:
    - canvas_image: Original 4K canvas pixel image (BHWC)
    - canvas_latent: Upscaled 4K canvas latent (BCHW)
    - noise_scaffold: Original 4K noise latent (BCHW)
    
    Process:
    1. Extract tile at (row, col) from all three
    2. Refine tile latent using canvas latent + noise scaffold slices
    3. Decode refined latent to pixels
    4. Paste refined tile back onto canvas image
    
    Outputs:
    - canvas_with_tile: Canvas with single refined tile composited
    - refined_tile_image: Just the refined tile (for inspection)
    - refined_tile_latent: Just the refined latent (for inspection)
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("canvas_with_tile", "tile_image", "tile_latent")
    FUNCTION = "test_tile"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_image": ("IMAGE", {
                    "tooltip": "Upscaled 4K canvas image (BHWC)"
                }),
                "canvas_latent": ("LATENT", {
                    "tooltip": "Upscaled 4K canvas latent (BCHW)"
                }),
                "noise_scaffold": ("LATENT", {
                    "tooltip": "Original 4K noise scaffold (BCHW)"
                }),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "tile_row": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "tooltip": "Grid row (0 = top)"
                }),
                "tile_col": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "tooltip": "Grid column (0 = left)"
                }),
                "tile_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size in pixels (latent = pixel / 8)"
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
                    "default": 0.35,
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
                "feathering": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend edge softness"
                }),
            }
        }
    
    def test_tile(
        self,
        canvas_image: torch.Tensor,
        canvas_latent: dict,
        noise_scaffold: dict,
        model,
        vae,
        positive,
        negative,
        tile_row: int,
        tile_col: int,
        tile_size: int,
        steps: int,
        cfg: float,
        denoise: float,
        seed: int,
        sampler: str,
        scheduler: str,
        feathering: float
    ) -> tuple:
        """
        Extract single tile, refine, and composite back.
        """
        device = comfy.model_management.get_torch_device()
        
        # Get dimensions
        b, h, w, c = canvas_image.shape
        lat_samples = canvas_latent["samples"]
        lat_h, lat_w = lat_samples.shape[2], lat_samples.shape[3]
        tile_lat = tile_size // 8
        
        print(f"[LunaChessTileTest] Starting single tile test:")
        print(f"  Canvas image: {w}×{h} pixels")
        print(f"  Canvas latent: {lat_w}×{lat_h}")
        print(f"  Tile size: {tile_size}px ({tile_lat} latent)")
        print(f"  Testing tile: row={tile_row}, col={tile_col}")
        
        # Calculate grid using SAME logic as chess refiner
        rows = int(round(lat_h / tile_lat)) + 1
        cols = int(round(lat_w / tile_lat)) + 1
        rows = max(1, rows)
        cols = max(1, cols)
        
        # Calculate overlap
        if rows > 1:
            total_coverage_h = rows * tile_lat
            excess_h = total_coverage_h - lat_h
            overlap_h = excess_h // (rows - 1)
        else:
            overlap_h = 0
        
        if cols > 1:
            total_coverage_w = cols * tile_lat
            excess_w = total_coverage_w - lat_w
            overlap_w = excess_w // (cols - 1)
        else:
            overlap_w = 0
        
        stride_h = tile_lat - overlap_h
        stride_w = tile_lat - overlap_w
        
        print(f"  Grid: {rows}×{cols} tiles (total {rows * cols})")
        print(f"  Overlap: {overlap_h}×{overlap_w} latent")
        print(f"  Stride: {stride_h}×{stride_w} latent")
        
        # Validate tile indices
        if tile_row >= rows or tile_col >= cols:
            raise ValueError(f"Tile ({tile_row}, {tile_col}) out of bounds for {rows}×{cols} grid")
        
        # Calculate tile coordinates using stride (SAME as chess refiner)
        y0_lat = tile_row * stride_h
        x0_lat = tile_col * stride_w
        
        # Edge snapping (SAME as chess refiner)
        if tile_row == rows - 1:
            y0_lat = lat_h - tile_lat
        if tile_col == cols - 1:
            x0_lat = lat_w - tile_lat
        
        y1_lat = min(y0_lat + tile_lat, lat_h)
        x1_lat = min(x0_lat + tile_lat, lat_w)
        
        # Pixel coordinates
        y0_pix = y0_lat * 8
        x0_pix = x0_lat * 8
        y1_pix = y1_lat * 8
        x1_pix = x1_lat * 8
        
        print(f"  Latent coords: [{y0_lat}:{y1_lat}, {x0_lat}:{x1_lat}]")
        print(f"  Pixel coords: [{y0_pix}:{y1_pix}, {x0_pix}:{x1_pix}]")
        
        # Extract tile crops
        canvas_lat_crop = lat_samples[:, :, y0_lat:y1_lat, x0_lat:x1_lat].clone()
        noise_crop = noise_scaffold["samples"][:, :, y0_lat:y1_lat, x0_lat:x1_lat].clone()
        canvas_pix_crop = canvas_image[:, y0_pix:y1_pix, x0_pix:x1_pix, :].clone()
        
        print(f"  Extracted latent shape: {canvas_lat_crop.shape}")
        print(f"  Extracted noise shape: {noise_crop.shape}")
        print(f"  Extracted pixel shape: {canvas_pix_crop.shape}")
        
        # Initialize sampler to get sigma
        sampler_obj = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=device,
            sampler=sampler,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options
        )
        
        initial_sigma = sampler_obj.sigmas[0]
        print(f"  Denoise: {denoise}, Initial sigma: {initial_sigma:.4f}, Steps: {len(sampler_obj.sigmas)-1}")
        
        # Ensure noise is on correct device
        noise_crop = noise_crop.to(device)
        
        print(f"  Canvas latent stats: min={canvas_lat_crop.min():.4f}, max={canvas_lat_crop.max():.4f}, mean={canvas_lat_crop.mean():.4f}")
        print(f"  Scaffold noise stats: min={noise_crop.min():.4f}, max={noise_crop.max():.4f}, std={noise_crop.std():.4f}")
        
        # Refine tile - pass scaffold noise to sampler, it will scale by appropriate sigma
        print(f"[LunaChessTileTest] Refining tile...")
        with torch.inference_mode():
            refined_latent = comfy.sample.sample(
                model,
                noise=noise_crop,                   # Scaffold noise - sampler scales it
                steps=steps,
                cfg=cfg,
                sampler_name=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=canvas_lat_crop,       # Clean latent
                denoise=denoise,
                disable_noise=False,                # Let sampler inject scaled noise
                start_step=None,
                last_step=None,
                force_full_denoise=True,
                noise_mask=None,
                sigmas=None,
                callback=None,
                disable_pbar=False,
                seed=seed
            )
        
        print(f"  Refined latent stats: min={refined_latent.min():.4f}, max={refined_latent.max():.4f}, mean={refined_latent.mean():.4f}")
        
        # Decode refined latent to pixels
        print(f"[LunaChessTileTest] Decoding refined tile...")
        with torch.no_grad():
            refined_pixels = vae.decode(refined_latent)
            # VAE returns BCHW, convert to BHWC
            if refined_pixels.dim() == 4 and refined_pixels.shape[1] == 3:
                refined_pixels = refined_pixels.permute(0, 2, 3, 1).cpu()
            else:
                refined_pixels = refined_pixels.cpu()
        
        print(f"  Refined pixels shape: {refined_pixels.shape}")
        print(f"  Refined pixels stats: min={refined_pixels.min():.4f}, max={refined_pixels.max():.4f}, mean={refined_pixels.mean():.4f}")
        
        # Composite refined tile onto canvas (with feathering)
        print(f"[LunaChessTileTest] Compositing refined tile back...")
        canvas_composite = canvas_image.clone()
        
        # Create feathered blend mask
        tile_h_pix = y1_pix - y0_pix
        tile_w_pix = x1_pix - x0_pix
        blend_mask = self._create_pixel_blend_mask(
            tile_h_pix, tile_w_pix, 
            int(tile_lat * 0.25 * 8),  # 25% overlap in pixels
            int(tile_lat * 0.25 * 8),
            feathering
        )
        
        # Composite
        diff = refined_pixels - canvas_pix_crop
        diff = diff * blend_mask
        canvas_composite[:, y0_pix:y1_pix, x0_pix:x1_pix, :] = canvas_pix_crop + diff
        
        print(f"  Composite stats: min={canvas_composite.min():.4f}, max={canvas_composite.max():.4f}, mean={canvas_composite.mean():.4f}")
        
        print(f"[LunaChessTileTest] Complete!")
        
        # Return canvas with tile, just the tile, and the refined latent
        return (canvas_composite, refined_pixels, {"samples": refined_latent})
    
    def _create_pixel_blend_mask(self, h, w, ov_h, ov_w, feather):
        """Create blend mask for pixel tiles (CPU)."""
        mask = torch.ones((1, h, w, 1))
        
        def smoothstep(length):
            t = torch.linspace(0, 1, length)
            if feather < 1.0:
                scale = 1.0 / max(0.01, feather)
                t = (t - 0.5) * scale + 0.5
                t = torch.clamp(t, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        
        # Apply to vertical edges
        if ov_h > 0 and ov_h < h:
            curve = smoothstep(ov_h).view(-1, 1, 1)
            # Top edge
            mask[:, :ov_h, :, :] *= curve.view(1, -1, 1, 1)
            # Bottom edge
            mask[:, -ov_h:, :, :] *= curve.flip(0).view(1, -1, 1, 1)
        
        # Apply to horizontal edges
        if ov_w > 0 and ov_w < w:
            curve = smoothstep(ov_w).view(-1, 1, 1)
            # Left edge
            mask[:, :, :ov_w, :] *= curve.view(1, 1, -1, 1)
            # Right edge
            mask[:, :, -ov_w:, :] *= curve.flip(0).view(1, 1, -1, 1)
        
        return mask


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaChessTileTest": LunaChessTileTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaChessTileTest": "🌙 Luna: Chess Tile Test",
}
