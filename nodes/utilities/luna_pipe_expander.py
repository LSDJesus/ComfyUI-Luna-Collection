"""
Luna Pipe Expander - Splits LUNA_PIPE back into individual components.

This node works with Luna Config Gateway to provide clean workflows:
- Config Gateway outputs a single LUNA_PIPE
- Pipe Expander splits it into individual components
- Connect the expander to downstream nodes instead of running 14 noodles

This pattern eliminates the need for set/get nodes or anything_anywhere nodes.
"""

import comfy.samplers

# Define scheduler list explicitly to ensure consistency with receiving nodes
SCHEDULER_LIST = comfy.samplers.KSampler.SCHEDULERS

class LunaPipeExpander:
    """
    Expands a LUNA_PIPE back into individual generation components.

    Takes the unified pipe from Luna Config Gateway and splits it into:
    model, clip, vae, positive, negative, latent, width, height, seed,
    steps, cfg, denoise, sampler_name, scheduler

    Use this to connect the pipe to downstream nodes (samplers, etc.)
    """
    CATEGORY = "Luna"
    RETURN_TYPES = (
        "MODEL", "CLIP", "VAE",
        "CONDITIONING", "CONDITIONING",
        "LATENT",
        "INT", "INT", "INT", "INT",
        "FLOAT", "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        SCHEDULER_LIST
    )
    RETURN_NAMES = (
        "model", "clip", "vae",
        "positive", "negative",
        "latent",
        "width", "height", "seed", "steps",
        "cfg", "denoise",
        "sampler_name", "scheduler"
    )
    FUNCTION = "expand"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luna_pipe": ("LUNA_PIPE",),
            }
        }

    def expand(self, luna_pipe):
        """
        Expand LUNA_PIPE tuple back into individual components.

        Args:
            luna_pipe: Tuple from Luna Config Gateway containing all generation data

        Returns:
            Tuple of 14 individual components (model, clip, vae, positive, negative,
            latent, width, height, seed, steps, cfg, denoise, sampler_name, scheduler)
        """
        if not isinstance(luna_pipe, (tuple, list)) or len(luna_pipe) != 14:
            raise ValueError(
                f"Expected LUNA_PIPE tuple with 14 elements, got {type(luna_pipe)} "
                f"with {len(luna_pipe) if isinstance(luna_pipe, (tuple, list)) else 'unknown'} elements"
            )

        (
            model, clip, vae,
            positive_cond, negative_cond,
            latent,
            width, height, seed, steps, cfg, denoise,
            sampler, scheduler
        ) = luna_pipe

        return (
            model, clip, vae,
            positive_cond, negative_cond,
            latent,
            width, height, seed, steps,
            cfg, denoise,
            sampler, scheduler
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaPipeExpander": LunaPipeExpander,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPipeExpander": "Luna Pipe Expander",
}
