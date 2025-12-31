"""
Luna Prep Upscaler - 1K→4K Bridge with Quality Options

Bridges the draft generation phase to the LSD refinement phase.
Takes 1K latent output from Luna KSampler and produces 4K pixels
ready for Semantic Detailer and Chess Refiner.

Modes:
- Direct: Fast 4x upscale, single step
- Iterative Quality: 4x upscale → 1.5K supersample → refine → 4K (double supersample)

Supports 2x and 4x upscale models:
- 4x (recommended): Perfect match with 4K scaffold, full supersample benefit
- 2x (supported): Requires Lanczos upscale to 4K, loses final supersample

Workflow Integration:
    Config Gateway (4K scaffold) → Canvas Downscale → Luna KSampler (1K)
    → Prep Upscaler → Semantic Detailer → Chess Refiner → Final
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import folder_paths


def get_scheduler_names():
    """Get current scheduler names dynamically."""
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except (AttributeError, TypeError):
        return list(comfy.samplers.SCHEDULER_NAMES)


class LunaPrepUpscaler:
    """
    Bridge node from 1K draft latent to 4K pixel canvas.
    
    Two quality modes:
    - direct: Fast single upscale pass
    - iterative_quality: Double supersample with intermediate refinement
    
    Output resolution is determined by the scaffold_4k input.
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image_4k", "scaffold_passthrough")
    FUNCTION = "prepare"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_1k": ("LATENT", {
                    "tooltip": "1K latent from Luna KSampler (soft draft)"
                }),
                "scaffold_4k": ("LATENT", {
                    "tooltip": "4K noise scaffold (defines target resolution)"
                }),
                "vae": ("VAE",),
                "upscale_model_name": (folder_paths.get_filename_list("upscale_models"), {
                    "default": "4x-UltraSharp.pth",
                    "tooltip": "2x or 4x upscale model (4x recommended for supersample benefit)"
                }),
                "prep_mode": (["direct", "iterative_quality"], {
                    "default": "direct",
                    "tooltip": "direct=fast, iterative_quality=double supersample with refinement"
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Required for iterative_quality mode"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Required for iterative_quality mode"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Required for iterative_quality mode"
                }),
                "refine_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Steps for intermediate refinement (iterative mode)"
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for intermediate refinement"
                }),
                "refine_cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "CFG scale for intermediate refinement"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for intermediate refinement"
                }),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {
                    "default": "dpmpp_2m",
                    "tooltip": "Sampler for intermediate refinement"
                }),
                "scheduler": (get_scheduler_names(), {
                    "default": "karras",
                    "tooltip": "Scheduler for intermediate refinement"
                }),
            }
        }
    
    def prepare(
        self,
        latent_1k: dict,
        scaffold_4k: dict,
        vae,
        upscale_model_name: str,
        prep_mode: str,
        model=None,
        positive=None,
        negative=None,
        refine_steps: int = 12,
        refine_denoise: float = 0.3,
        refine_cfg: float = 7.0,
        seed: int = 0,
        sampler_name: str = "dpmpp_2m",
        scheduler: str = "karras"
    ) -> tuple:
        """
        Prepare 4K pixel canvas from 1K draft latent.
        
        Returns:
            (image_4k, scaffold_passthrough)
        """
        device = comfy.model_management.get_torch_device()
        
        # Load upscale model
        upscale_model = self._load_upscale_model(upscale_model_name)
        if upscale_model is None:
            raise ValueError(f"Failed to load upscale model: {upscale_model_name}")
        
        # Extract target resolution from scaffold
        scaffold_samples = scaffold_4k["samples"]
        target_h_lat = scaffold_samples.shape[2]
        target_w_lat = scaffold_samples.shape[3]
        target_h_px = target_h_lat * 8
        target_w_px = target_w_lat * 8
        
        print(f"[LunaPrepUpscaler] Target resolution: {target_w_px}×{target_h_px}px (from scaffold)")
        print(f"[LunaPrepUpscaler] Mode: {prep_mode}")
        
        # Decode 1K latent to pixels
        with torch.no_grad():
            pixels_1k = vae.decode(latent_1k["samples"])
        
        # Get 1K dimensions
        if pixels_1k.dim() == 4 and pixels_1k.shape[1] == 3:  # BCHW
            pixels_1k = pixels_1k.permute(0, 2, 3, 1)  # → BHWC
        
        src_h, src_w = pixels_1k.shape[1], pixels_1k.shape[2]
        print(f"[LunaPrepUpscaler] Source: {src_w}×{src_h}px")
        
        # Apply upscale model
        pixels_upscaled = self._apply_upscale_model(pixels_1k, upscale_model)
        up_h, up_w = pixels_upscaled.shape[1], pixels_upscaled.shape[2]
        
        # Detect scale factor
        scale_factor = up_h / src_h
        print(f"[LunaPrepUpscaler] Upscale model: {scale_factor:.1f}x ({src_w}×{src_h} → {up_w}×{up_h})")
        
        # Validate scale factor
        if scale_factor not in [2.0, 4.0]:
            print(f"[LunaPrepUpscaler] ⚠ Warning: {scale_factor}x upscale model detected. "
                  f"Only 2x and 4x are officially supported.")
        
        if prep_mode == "direct":
            pixels_4k = self._direct_mode(pixels_upscaled, target_h_px, target_w_px, scale_factor)
        else:  # iterative_quality
            if model is None or positive is None or negative is None:
                raise ValueError("iterative_quality mode requires model, positive, and negative inputs")
            
            pixels_4k = self._iterative_mode(
                pixels_upscaled, target_h_px, target_w_px, scale_factor,
                scaffold_4k, vae, model, positive, negative,
                refine_steps, refine_denoise, refine_cfg, seed, sampler_name, scheduler, device
            )
        
        print(f"[LunaPrepUpscaler] ✓ Output: {pixels_4k.shape[2]}×{pixels_4k.shape[1]}px")
        
        return (pixels_4k, scaffold_4k)
    
    def _apply_upscale_model(self, pixels: torch.Tensor, upscale_model) -> torch.Tensor:
        """Apply upscale model to pixels."""
        device = comfy.model_management.get_torch_device()
        
        # Convert BHWC → BCHW for upscale model
        pixels_bchw = pixels.permute(0, 3, 1, 2).to(device)
        
        # Apply upscale model - expects BCHW, returns BCHW
        with torch.no_grad():
            # ComfyUI upscale models are callable and return the upscaled tensor
            upscaled = upscale_model(pixels_bchw)
            
            # Some models return tuple, some return tensor directly
            if isinstance(upscaled, tuple):
                upscaled = upscaled[0]
        
        # Convert back BCHW → BHWC and move to CPU
        return upscaled.permute(0, 2, 3, 1).cpu()
    
    def _direct_mode(
        self,
        pixels_upscaled: torch.Tensor,
        target_h: int,
        target_w: int,
        scale_factor: float
    ) -> torch.Tensor:
        """
        Direct mode: Just resize to target if needed.
        """
        up_h, up_w = pixels_upscaled.shape[1], pixels_upscaled.shape[2]
        
        if up_h == target_h and up_w == target_w:
            print(f"[LunaPrepUpscaler] Direct: Perfect match, no resize needed")
            return pixels_upscaled
        
        if up_h > target_h:
            # Supersample down (good!)
            print(f"[LunaPrepUpscaler] Direct: Supersampling {up_w}×{up_h} → {target_w}×{target_h}")
        else:
            # Need to upscale (less ideal with 2x model)
            print(f"[LunaPrepUpscaler] Direct: ⚠ Lanczos upscale {up_w}×{up_h} → {target_w}×{target_h}")
        
        return self._resize_pixels(pixels_upscaled, target_h, target_w)
    
    def _iterative_mode(
        self,
        pixels_upscaled: torch.Tensor,
        target_h: int,
        target_w: int,
        scale_factor: float,
        scaffold_4k: dict,
        vae,
        model,
        positive,
        negative,
        steps: int,
        denoise: float,
        cfg: float,
        seed: int,
        sampler_name: str,
        scheduler: str,
        device
    ) -> torch.Tensor:
        """
        Iterative quality mode:
        1. Supersample to 1.5K (37.5% of target)
        2. Refine with downscaled scaffold
        3. Upscale back to target
        """
        up_h, up_w = pixels_upscaled.shape[1], pixels_upscaled.shape[2]
        
        # Calculate intermediate resolution (1.5K for 4K target)
        intermediate_h = int(target_h * 0.375)
        intermediate_w = int(target_w * 0.375)
        
        # Ensure multiple of 8 for VAE
        intermediate_h = (intermediate_h // 8) * 8
        intermediate_w = (intermediate_w // 8) * 8
        
        print(f"[LunaPrepUpscaler] Iterative: {up_w}×{up_h} → {intermediate_w}×{intermediate_h} (1.5K)")
        
        # STEP 1: Supersample to intermediate (first supersample!)
        pixels_intermediate = self._resize_pixels(pixels_upscaled, intermediate_h, intermediate_w)
        print(f"[LunaPrepUpscaler] ✓ First supersample: {up_w}×{up_h} → {intermediate_w}×{intermediate_h}")
        
        # STEP 2: Downscale scaffold to intermediate resolution
        scaffold_samples = scaffold_4k["samples"]  # [B, C, H, W]
        intermediate_h_lat = intermediate_h // 8
        intermediate_w_lat = intermediate_w // 8
        
        scaffold_intermediate = F.interpolate(
            scaffold_samples,
            size=(intermediate_h_lat, intermediate_w_lat),
            mode='area'
        )
        
        # Variance correction for downscaled scaffold
        scale_down = target_h / intermediate_h
        scaffold_intermediate = scaffold_intermediate * scale_down
        
        print(f"[LunaPrepUpscaler] ✓ Scaffold downscaled with variance correction (×{scale_down:.2f})")
        
        # STEP 3: Encode intermediate pixels
        pixels_for_vae = pixels_intermediate.permute(0, 3, 1, 2).to(device)  # BHWC → BCHW
        with torch.no_grad():
            latent_intermediate = vae.encode(pixels_for_vae)
        
        # STEP 4: Refine at intermediate resolution
        print(f"[LunaPrepUpscaler] Refining at {intermediate_w}×{intermediate_h}px "
              f"(steps={steps}, cfg={cfg}, denoise={denoise})")
        
        scaffold_intermediate = scaffold_intermediate.to(device)
        
        with torch.inference_mode():
            refined_latent = comfy.sample.sample(
                model,
                noise=scaffold_intermediate,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_intermediate,
                denoise=denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=True,
                noise_mask=None,
                sigmas=None,
                callback=None,
                disable_pbar=False,
                seed=seed
            )
        
        # Clear VRAM after sampling
        torch.cuda.empty_cache()
        
        # STEP 5: Decode refined latent
        with torch.no_grad():
            pixels_refined = vae.decode(refined_latent)
        
        if pixels_refined.dim() == 4 and pixels_refined.shape[1] == 3:  # BCHW
            pixels_refined = pixels_refined.permute(0, 2, 3, 1)  # → BHWC
        
        pixels_refined = pixels_refined.cpu()
        
        # Clear VRAM after decode
        del refined_latent, latent_intermediate, scaffold_intermediate
        torch.cuda.empty_cache()
        
        print(f"[LunaPrepUpscaler] ✓ Refinement complete")
        
        # STEP 6: Upscale to target resolution
        # If we had a 4x model and target is 4K, this is a second supersample opportunity
        # But since we're going 1.5K → 4K, we need to upscale
        pixels_4k = self._resize_pixels(pixels_refined, target_h, target_w)
        print(f"[LunaPrepUpscaler] ✓ Final resize: {intermediate_w}×{intermediate_h} → {target_w}×{target_h}")
        
        return pixels_4k
    
    def _resize_pixels(self, pixels: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize pixels using Lanczos interpolation."""
        # Convert BHWC → BCHW for interpolation
        pixels_bchw = pixels.permute(0, 3, 1, 2)
        
        resized = F.interpolate(
            pixels_bchw,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        
        # Clamp to valid range
        resized = torch.clamp(resized, 0.0, 1.0)
        
        # Convert back BCHW → BHWC
        return resized.permute(0, 2, 3, 1)
    
    def _load_upscale_model(self, model_name: str):
        """
        Load upscale model from models/upscale_models/.
        
        Args:
            model_name: Filename of the upscale model
        
        Returns:
            Loaded upscale model, or None if loading fails
        """
        try:
            model_path = folder_paths.get_full_path("upscale_models", model_name)
            if not model_path:
                print(f"[LunaPrepUpscaler] ⚠ Upscale model not found: {model_name}")
                return None
            
            print(f"[LunaPrepUpscaler] Loading upscale model: {model_name}")
            
            # Use ComfyUI's upscale model loader
            from comfy_extras.chainner_models import model_loading
            upscale_model = model_loading.load_state_dict(model_path)
            
            # Wrap in ComfyUI's upscale model format
            from comfy import model_management
            
            class UpscaleModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def upscale(self, image):
                    """Upscale image tensor [B, H, W, C] → [B, H*scale, W*scale, C]"""
                    device = model_management.get_torch_device()
                    
                    # Convert BHWC → BCHW
                    image_bchw = image.permute(0, 3, 1, 2).to(device)
                    
                    # Run upscale model
                    with torch.no_grad():
                        upscaled = self.model(image_bchw)
                    
                    # Convert back BCHW → BHWC and move to CPU
                    return upscaled.permute(0, 2, 3, 1).cpu()
            
            return UpscaleModelWrapper(upscale_model)
            
        except Exception as e:
            print(f"[LunaPrepUpscaler] ✗ Failed to load upscale model: {e}")
            import traceback
            traceback.print_exc()
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaPrepUpscaler": LunaPrepUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPrepUpscaler": "🌙 Luna: Prep Upscaler",
}
