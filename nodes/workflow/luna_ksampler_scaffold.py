"""
Luna KSampler with Scaffold Noise - Testing Noise Reuse

Test node to examine the effect of reusing the same noise scaffold across multiple
sampling passes. Helps debug whether scaffold reuse causes smoothing/plasticy artifacts.

Test scenario:
1. Generate 1K image with random noise
2. Use same noise to refine that image
3. Compare to normal img2img refinement with fresh noise
"""

import torch
import comfy.samplers
import comfy.sample

# Import common_ksampler from ComfyUI's nodes.py file
try:
    import nodes
    common_ksampler = nodes.common_ksampler # type: ignore
except (ImportError, AttributeError):
    raise ImportError(
        "Could not import common_ksampler from ComfyUI nodes.py. "
        "Ensure ComfyUI root is on sys.path when loading custom nodes."
    )


def get_scheduler_names():
    """Get current scheduler names dynamically to avoid type mismatches."""
    try:
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except (AttributeError, TypeError):
        return list(comfy.samplers.SCHEDULER_NAMES)


class LunaKSamplerScaffold:
    """
    KSampler with optional noise scaffold input for testing noise reuse.
    
    If noise_scaffold is provided, uses it instead of generating random noise.
    This lets you test whether reusing the same noise causes artifacts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (get_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "noise_scaffold": ("LATENT", {
                    "tooltip": "Optional: Pre-generated noise to use instead of random noise. For testing scaffold reuse."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Luna/Workflow/Debug"
    
    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, 
               sampler_name, scheduler, denoise=1.0, noise_scaffold=None):
        """
        Execute sampling with optional noise scaffold override.
        
        If noise_scaffold is provided:
        - Uses scaffold["samples"] as the noise tensor
        - Ignores seed (scaffold determines noise)
        
        If noise_scaffold is None:
        - Normal behavior: generates random noise from seed
        
        This lets you test whether reusing the same noise causes artifacts.
        """
        
        if noise_scaffold is not None:
            # Use scaffold noise instead of random noise
            scaffold_samples = noise_scaffold["samples"]
            latent_samples = latent_image["samples"]
            
            print(f"[LunaKSamplerScaffold] Using scaffold noise")
            print(f"  Scaffold shape: {scaffold_samples.shape}")
            print(f"  Latent shape: {latent_samples.shape}")
            
            # Validate shapes match
            if scaffold_samples.shape != latent_samples.shape:
                raise ValueError(
                    f"[LunaKSamplerScaffold] Scaffold shape {scaffold_samples.shape} "
                    f"doesn't match latent shape {latent_samples.shape}"
                )
            
            # Use inference_mode for memory optimization
            with torch.inference_mode():
                # Call the sampler with custom noise
                samples = comfy.sample.sample(
                    model,
                    noise=scaffold_samples,  # Use scaffold instead of random
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent_samples,
                    denoise=denoise,
                    disable_noise=False,  # Let sampler scale the noise
                    start_step=None,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed  # Still used for any internal randomness
                )
                
                # comfy.sample.sample returns just the tensor, wrap in dict
                result_latent = {"samples": samples}
        else:
            # Normal path: use random noise from seed
            print(f"[LunaKSamplerScaffold] Using random noise (seed={seed})")
            
            with torch.inference_mode():
                result_tuple = common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image,
                    denoise=denoise
                )
                # common_ksampler returns (latent_dict,) - unpack it
                result_latent = result_tuple[0]
        
        # Clear GPU memory cache
        torch.cuda.empty_cache()
        
        return (result_latent,)


NODE_CLASS_MAPPINGS = {
    "LunaKSamplerScaffold": LunaKSamplerScaffold,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaKSamplerScaffold": "Luna KSampler (Scaffold Test)",
}
