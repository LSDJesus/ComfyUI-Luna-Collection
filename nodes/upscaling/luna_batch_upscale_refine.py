"""
Luna Batch Upscale Refine - "The Scaffolding Method"
Production Build v1.2.1

Changelog:
- Fixed Progress Bar initialization and updating.
- Added 'blending_mode' (Linear, Sigmoid) for smoother seams.
- Added 'feathering' parameter to control edge softness.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import nodes

class LunaBatchUpscaleRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (determines tile size)"}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Model determines grid size (Factor + 1)"}),
                "scale": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25,
                    "tooltip": "Final output scale (post-refinement downscale target)"
                }),
                
                # BATCH SAFETY
                "tile_batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "VRAM Safety: Process tiles in chunks"}),
                
                # REFINEMENT CONTROLS
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                
                # BLENDING CONTROLS
                "blending_mode": (["Sigmoid", "Linear"], {"default": "Sigmoid"}),
                "feathering": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "0.0=Hard Edge, 1.0=Full Softness"}),
                
                "use_tiled_vae": ("BOOLEAN", {"default": True, "tooltip": "Use VAE Tiled Decode to prevent seams/OOM"}),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "luna_pipe": ("LUNA_PIPE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "Luna/Upscaling"
    
    def refine(self, image, upscale_model, scale, tile_batch_size, denoise, steps, cfg, seed, sampler, scheduler, 
               blending_mode, feathering, use_tiled_vae, model=None, positive=None, negative=None, vae=None, luna_pipe=None):
        
        # 0. INPUT VALIDATION & UNPACKING
        pipe_latent = None
        if luna_pipe is not None:
            try:
                if model is None: model = luna_pipe[0]
                if positive is None: positive = luna_pipe[3]
                if negative is None: negative = luna_pipe[4]
                if vae is None: vae = luna_pipe[2]
                pipe_latent = luna_pipe[5]
            except IndexError:
                pass
        
        if not all([model, positive, negative, vae]):
            raise ValueError("[Luna] Missing model, positive, negative, or vae inputs.")

        device = comfy.model_management.get_torch_device()
        
        # 1. PROGRESS BAR PRE-CALCULATION
        # We need to know the total number of tiles to initialize the progress bar correctly.
        input_batch_size = image.shape[0]
        input_h, input_w = image.shape[1], image.shape[2]
        
        # Run a "dry" upscale on the first frame to get output dimensions and calculate grid.
        # This is very fast compared to the sampling steps.
        upscale_model.to(device)
        prototype = upscale_model(image[0:1])
        upscale_model.to("cpu")
        proto_h, proto_w = prototype.shape[1], prototype.shape[2]
        proto_factor = max(1, int(round(proto_h / input_h)))
        
        rows, cols, _, _, _, _ = self._calc_auto_grid(
            input_h, input_w, proto_h, proto_w, proto_factor
        )
        
        # Total steps = (Total Tiles in Grid) * (Number of Images in Batch)
        total_steps = (rows * cols) * input_batch_size
        pbar = comfy.utils.ProgressBar(total_steps)
        
        # 2. BATCH PROCESSING LOOP
        output_images = []
        
        for batch_idx in range(input_batch_size):
            print(f"[Luna] Processing Image {batch_idx + 1}/{input_batch_size}")
            
            # Extract single image from batch (preserves [1, H, W, 3])
            curr_image = image[batch_idx:batch_idx + 1]
            
            # Extract corresponding scaffolding latent if available
            curr_pipe_latent = None
            if pipe_latent is not None:
                p_samples = pipe_latent['samples']
                # Safety: reuse first latent if batch doesn't match
                idx = batch_idx if p_samples.shape[0] > batch_idx else 0
                curr_pipe_latent = {'samples': p_samples[idx:idx + 1]}
            
            # Process single image through refinement pipeline
            refined_single = self._refine_single_image(
                curr_image, upscale_model, scale, tile_batch_size,
                denoise, steps, cfg, seed + batch_idx, sampler, scheduler,
                blending_mode, feathering, use_tiled_vae,
                model, positive, negative, vae, curr_pipe_latent, device,
                pbar  # Pass pbar down
            )
            
            output_images.append(refined_single)
        
        # Merge back into single batch tensor [Batch, H, W, 3]
        result = torch.cat(output_images, dim=0)
        return (result,)
    
    def _refine_single_image(self, image, upscale_model, scale, tile_batch_size,
                            denoise, steps, cfg, seed, sampler, scheduler,
                            blending_mode, feathering, use_tiled_vae,
                            model, positive, negative, vae, pipe_latent, device, pbar):
        """
        Refine a single image through the full pipeline.
        """
        
        # 1. PIXEL UPSCALING (The Canvas)
        upscale_model.to(device)
        upscaled = upscale_model(image)
        upscale_model.to("cpu")
        
        input_h, input_w = image.shape[1], image.shape[2]
        output_h, output_w = upscaled.shape[1], upscaled.shape[2]
        upscale_factor = max(1, int(round(output_h / input_h)))
        
        # 2. AUTO-GRID CALCULATION
        rows, cols, tile_h, tile_w, ov_h, ov_w = self._calc_auto_grid(
            input_h, input_w, output_h, output_w, upscale_factor
        )
        
        print(f"[Luna] {upscale_factor}x Upscale | Grid: {rows}x{cols} | Tile: {tile_h}x{tile_w} | Overlap: {ov_h}x{ov_w}")

        # 3. ENCODE TO LATENT
        with torch.no_grad():
            upscaled_lat = vae.encode(upscaled)
            
        # 4. SCAFFOLDING NOISE GENERATION
        target_lat_h, target_lat_w = upscaled_lat['samples'].shape[2], upscaled_lat['samples'].shape[3]
        
        if pipe_latent is not None and 'samples' in pipe_latent:
            base_noise = self._gen_noise_shape(pipe_latent['samples'].shape, seed, upscaled_lat['samples'].device)
        else:
            lat_h, lat_w = input_h // 8, input_w // 8
            base_noise = self._gen_noise_shape((1, 4, lat_h, lat_w), seed, upscaled_lat['samples'].device)
            
        scaffolding_noise = self._upscale_noise(base_noise, target_lat_h, target_lat_w)

        # 5. MANUAL NOISE INJECTION
        sampler_obj = comfy.samplers.KSampler(
            model, steps=steps, device=device, sampler=sampler, scheduler=scheduler, 
            denoise=1.0, model_options=model.model_options
        )
        
        total_steps = len(sampler_obj.sigmas) - 1
        start_step = int(total_steps * (1.0 - denoise))
        initial_sigma = sampler_obj.sigmas[start_step]
        
        # Create the Noisy Canvas (In-Place to save VRAM)
        canvas_samples = upscaled_lat['samples'].clone()
        canvas_samples.add_(scaffolding_noise * initial_sigma)
        canvas = {'samples': canvas_samples}
        
        # 6. SEQUENTIAL CHESS REFINEMENT
        # Pass 1: EVENS
        canvas = self._process_batch_pass(
            canvas, "even", rows, cols, tile_h, tile_w, ov_h, ov_w, tile_batch_size,
            model, positive, negative, start_step, steps, cfg, seed, sampler, scheduler, sampler_obj,
            blending_mode, feathering, pbar
        )

        # Pass 2: ODDS
        canvas = self._process_batch_pass(
            canvas, "odd", rows, cols, tile_h, tile_w, ov_h, ov_w, tile_batch_size,
            model, positive, negative, start_step, steps, cfg, seed+1, sampler, scheduler, sampler_obj,
            blending_mode, feathering, pbar
        )
        
        # 7. DECODE
        if use_tiled_vae:
            refined_px = vae.decode_tiled(
                canvas['samples'], 
                tile_x=512, tile_y=512, 
                overlap=64
            )
        else:
            refined_px = vae.decode(canvas)
            
        # 8. SUPERSAMPLING (LANCZOS DOWNSCALE)
        target_final_h = int(input_h * scale)
        target_final_w = int(input_w * scale)
        
        if (target_final_h != output_h) or (target_final_w != output_w):
            print(f"[Luna] Lanczos: {output_h}x{output_w} → {target_final_h}x{target_final_w}")
            refined_px = self._lanczos(refined_px, target_final_h, target_final_w)
            
        return refined_px

    def _calc_auto_grid(self, input_h, input_w, output_h, output_w, factor):
        """
        Calculates grid based on upscale factor.
        Includes a 'Safety Floor' to prevent overlap from becoming too small on extreme upscales.
        """
        grid_size = factor + 1
        
        # Start with Tile Size = Original Input Size
        tile_h = input_h
        tile_w = input_w
        
        # SAFETY: Minimum 64px overlap to prevent hard seams
        MIN_OVERLAP = 64
        
        # Check Vertical Overlap
        # If natural overlap is too small, expand tile size
        current_overlap_h = (tile_h * grid_size - output_h) / max(1, grid_size - 1)
        if current_overlap_h < MIN_OVERLAP:
            required_excess = MIN_OVERLAP * (grid_size - 1)
            tile_h = (output_h + required_excess) / grid_size
            
        # Check Horizontal Overlap
        current_overlap_w = (tile_w * grid_size - output_w) / max(1, grid_size - 1)
        if current_overlap_w < MIN_OVERLAP:
            required_excess = MIN_OVERLAP * (grid_size - 1)
            tile_w = (output_w + required_excess) / grid_size
            
        # Snap to Multiples of 8 (Rounding UP for safety)
        tile_h = int((tile_h + 7) // 8) * 8
        tile_w = int((tile_w + 7) // 8) * 8
        
        # Calculate Final Overlaps
        if grid_size > 1:
            total_cov_h = tile_h * grid_size
            excess_h = total_cov_h - output_h
            overlap_h = excess_h // (grid_size - 1)
            
            total_cov_w = tile_w * grid_size
            excess_w = total_cov_w - output_w
            overlap_w = excess_w // (grid_size - 1)
        else:
            overlap_h = 0
            overlap_w = 0
        
        # Floor overlaps to 8
        overlap_h = (overlap_h // 8) * 8
        overlap_w = (overlap_w // 8) * 8
        
        return grid_size, grid_size, tile_h, tile_w, overlap_h, overlap_w

    def _process_batch_pass(self, canvas, parity, rows, cols, tile_h, tile_w, ov_h, ov_w, batch_size,
                            model, pos, neg, start_step, steps, cfg, seed, sampler_name, scheduler_name, sampler_obj,
                            blending_mode, feathering, pbar=None):
        
        tile_lat_h, tile_lat_w = tile_h // 8, tile_w // 8
        ov_lat_h, ov_lat_w = ov_h // 8, ov_w // 8
        
        tiles_data = self._extract_specific_tiles(canvas, parity, rows, cols, tile_lat_h, tile_lat_w, ov_lat_h, ov_lat_w)
        
        if not tiles_data:
            return canvas
            
        sigmas = sampler_obj.sigmas[start_step:]
        total_tiles = len(tiles_data)
        
        # Generate Blend Mask (Linear or Sigmoid)
        mask = self._blend_mask(tile_lat_h, tile_lat_w, ov_lat_h, ov_lat_w, blending_mode, feathering, canvas['samples'].device)

        # Process in Sub-Batches
        for i in range(0, total_tiles, batch_size):
            chunk = tiles_data[i:i + batch_size]
            batch_samples = torch.cat([t['samples'] for t in chunk], dim=0)
            
            # Generate Random Noise for Sampler
            torch.manual_seed(seed)
            sampler_noise = torch.randn_like(batch_samples)
            
            print(f"[Luna] {parity.upper()} Batch {i//batch_size + 1}: Processing {len(chunk)} tiles")
            
            with torch.inference_mode():
                refined_chunk = comfy.sample.sample_custom(
                    model,
                    noise=sampler_noise,
                    cfg=cfg,
                    sampler=sampler_name,
                    sigmas=sigmas,
                    positive=pos,
                    negative=neg,
                    latent_image=batch_samples,
                    noise_mask=None,
                    seed=seed,
                    disable_noise=True
                )
            
            self._composite_chunk(canvas, refined_chunk, chunk, mask)
            
            # Update Progress Bar
            if pbar is not None:
                pbar.update(len(chunk))
                
            # Allow user interruption
            comfy.model_management.throw_exception_if_processing_interrupted() # type: ignore
            
        return canvas

    def _blend_mask(self, h, w, ov_h, ov_w, mode, feather, device):
        """
        Generates a blend mask using Linear or Smoothstep (Sigmoid-like) interpolation.
        Optimized to use polynomial Smoothstep instead of trigonometric functions.
        """
        mask = torch.ones((1, 1, h, w), device=device)
        
        def get_curve(length):
            t = torch.linspace(0, 1, length, device=device)
            
            # Apply Feathering
            if feather < 1.0:
                scale = 1.0 / max(0.01, feather)
                t = (t - 0.5) * scale + 0.5
                t = torch.clamp(t, 0.0, 1.0)
            
            if mode == "Sigmoid":
                # Optimization: Use Smoothstep (Hermite interpolation)
                return t * t * (3.0 - 2.0 * t)
            else:
                return t # Linear

        if ov_h > 0:
            curve = get_curve(ov_h).view(1, 1, -1, 1)
            mask[:, :, :ov_h, :] *= curve
            mask[:, :, -ov_h:, :] *= curve.flip(2)
            
        if ov_w > 0:
            curve = get_curve(ov_w).view(1, 1, 1, -1)
            mask[:, :, :, :ov_w] *= curve
            mask[:, :, :, -ov_w:] *= curve.flip(3)
            
        return mask

    def _extract_specific_tiles(self, lat, parity, rows, cols, th, tw, oh, ow):
        s = lat['samples']
        H, W = s.shape[2], s.shape[3]
        stride_h, stride_w = th - oh, tw - ow
        extracted = []

        target_mod = 0 if parity == "even" else 1

        for yi in range(rows):
            for xi in range(cols):
                if (yi + xi) % 2 == target_mod:
                    y0 = yi * stride_h
                    x0 = xi * stride_w
                    
                    if yi == rows - 1: y0 = H - th
                    if xi == cols - 1: x0 = W - tw
                    
                    y1, x1 = y0 + th, x0 + tw
                    
                    extracted.append({
                        'samples': s[:, :, y0:y1, x0:x1], 
                        'coords': (y0, x0, y1, x1)
                    })
        return extracted

    def _composite_chunk(self, canvas, refined_batch, chunk_info, mask):
        result = canvas['samples'] 
        for i, info in enumerate(chunk_info):
            refined_tile = refined_batch[i]
            y0, x0, y1, x1 = info['coords']
            current_bg = result[:, :, y0:y1, x0:x1]
            
            # In-Place Linear Interpolation
            # (Refined - BG) * Mask + BG
            diff = refined_tile - current_bg
            diff.mul_(mask)
            current_bg.add_(diff)
            result[:, :, y0:y1, x0:x1] = current_bg

    def _gen_noise_shape(self, shape, seed, device):
        torch.manual_seed(seed)
        return torch.randn(shape, device=device)

    def _upscale_noise(self, noise, th, tw):
        return F.interpolate(noise, size=(th, tw), mode='nearest-exact')

    def _lanczos(self, img, th, tw):
        img_chw = img.permute(0, 3, 1, 2)
        downscaled = TF.resize(img_chw, [th, tw], interpolation=TF.InterpolationMode.LANCZOS)
        return downscaled.permute(0, 2, 3, 1)

NODE_CLASS_MAPPINGS = {
    "LunaBatchUpscaleRefine": LunaBatchUpscaleRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaBatchUpscaleRefine": "Luna Batch Upscale Refine",
}