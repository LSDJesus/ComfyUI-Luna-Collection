import os
import torch
import comfy.sd
from comfy.cli_args import args
import folder_paths
import nodes


class LunaParametersBridge:
    """
    Advanced parameters bridge that combines multiple sources of conditionings and parameters
    """
    CATEGORY = "Luna/Meta"
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "INT", "VAE", "METADATA", "LUNA_PIPE", "LATENT", "PARAMETERS_PIPE")
    RETURN_NAMES = ("MODEL", "CLIP", "positive", "negative", "model_name", "positive_text", "width", "height", "cfg", "seed", "steps", "denoise", "sampler_name", "scheduler", "clip_skip", "batch_size", "vae", "metadata", "luna_pipe", "latent_image", "parameters_pipe")
    FUNCTION = "bridge_parameters"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to use for generation"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model for text encoding"}),
                "vae": ("VAE", {"tooltip": "The VAE for latent decoding"}),
            },
            "optional": {
                # Conditioning sources
                "positive_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed positive conditioning from Luna nodes"}),
                "negative_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed negative conditioning from Luna nodes"}),
                "positive_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw positive prompt text"}),
                "negative_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw negative prompt text"}),

                # Parameters sources
                "parameters_pipe": ("PARAMETERS_PIPE", {"tooltip": "Parameters from LunaLoadParameters or other sources"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "ddim", "uni_pc", "uni_pc_bh2"], {"default": "dpmpp_2m"}),
                "scheduler_name": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"], {"default": "karras"}),

                # Advanced options
                "conditioning_blend_mode": (["replace", "add", "multiply", "average"], {"default": "replace", "tooltip": "How to combine multiple conditioning sources"}),
                "conditioning_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Strength multiplier for conditioning combination"}),
            }
        }

    def bridge_parameters(self, model, clip, vae,
                         positive_conditioning=None, negative_conditioning=None,
                         positive_text="", negative_text="",
                         parameters_pipe=None,
                         width=1024, height=1024, cfg=7.0, seed=0, steps=25, denoise=1.0,
                         clip_skip=-2, batch_size=1, sampler_name="dpmpp_2m", scheduler_name="karras",
                         conditioning_blend_mode="replace", conditioning_strength=1.0):

        # Apply CLIP skip
        clip_skip_node = nodes.CLIPSetLastLayer()
        clip_skipped = clip_skip_node.set_last_layer(clip, clip_skip)[0]

        # Handle parameters - use parameters_pipe if provided, otherwise use individual parameters
        if parameters_pipe is not None:
            param_steps, param_cfg, param_denoise, param_sampler, param_scheduler, param_seed = parameters_pipe
            # Override with provided parameters if specified
            steps = steps if steps != 25 else param_steps  # Only override if not explicitly set
            cfg = cfg if cfg != 7.0 else param_cfg
            denoise = denoise if denoise != 1.0 else param_denoise
            sampler_name = param_sampler
            scheduler_name = param_scheduler
            seed = seed if seed != 0 else param_seed

        # Handle positive conditioning
        if positive_conditioning is not None:
            if positive_text and conditioning_blend_mode != "replace":
                # Blend with text conditioning
                text_encode_node = nodes.CLIPTextEncode()
                text_positive = text_encode_node.encode(clip_skipped, positive_text)[0]
                positive = self._blend_conditionings(positive_conditioning, text_positive,
                                                   conditioning_blend_mode, conditioning_strength)
                positive_text_display = f"Blended: {positive_text}"
            else:
                positive = positive_conditioning
                positive_text_display = "Preprocessed conditioning"
        else:
            # Encode text prompt
            text_encode_node = nodes.CLIPTextEncode()
            positive = text_encode_node.encode(clip_skipped, positive_text)[0]
            positive_text_display = positive_text

        # Handle negative conditioning
        if negative_conditioning is not None:
            if negative_text and conditioning_blend_mode != "replace":
                # Blend with text conditioning
                text_encode_node = nodes.CLIPTextEncode()
                text_negative = text_encode_node.encode(clip_skipped, negative_text)[0]
                negative = self._blend_conditionings(negative_conditioning, text_negative,
                                                   conditioning_blend_mode, conditioning_strength)
                negative_text_display = f"Blended: {negative_text}"
            else:
                negative = negative_conditioning
                negative_text_display = "Preprocessed conditioning"
        else:
            # Encode text prompt
            text_encode_node = nodes.CLIPTextEncode()
            negative = text_encode_node.encode(clip_skipped, negative_text)[0]
            negative_text_display = negative_text

        # Create empty latent image
        empty_latent_node = nodes.EmptyLatentImage()
        latent_image = empty_latent_node.generate(width, height, batch_size)[0]

        # Extract model name
        model_name = "unknown"  # Placeholder

        # Create metadata
        metadata = {
            "modelname": model_name,
            "positive": positive_text_display,
            "negative": negative_text_display,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler_name,
            "denoise": denoise,
            "clip_skip": clip_skip,
            "batch_size": batch_size,
            "conditioning_source": "preprocessed" if (positive_conditioning is not None or negative_conditioning is not None) else "raw_text",
            "blend_mode": conditioning_blend_mode,
            "conditioning_strength": conditioning_strength,
        }

        # Create luna_pipe and parameters_pipe
        luna_pipe = (model, clip_skipped, vae, positive, negative, seed, sampler_name, scheduler_name)
        parameters_pipe_out = (steps, cfg, denoise, sampler_name, scheduler_name, seed)

        return (model, clip_skipped, positive, negative, model_name, positive_text_display,
                width, height, cfg, seed, steps, denoise, sampler_name, scheduler_name,
                clip_skip, batch_size, vae, metadata, luna_pipe, latent_image, parameters_pipe_out)

    def _blend_conditionings(self, cond1, cond2, blend_mode, strength):
        """
        Blend two conditionings using different modes.
        
        ComfyUI conditioning format: [[tensor, {"pooled_output": pooled, ...}], ...]
        We need to blend the actual tensors, not the list wrappers.
        """
        if blend_mode == "replace":
            return cond1
        
        # Extract tensors from conditioning format
        # Each conditioning is a list of [tensor, metadata_dict] pairs
        blended = []
        for c1 in cond1:
            tensor1 = c1[0]
            metadata1 = c1[1].copy() if len(c1) > 1 else {}
            
            # Find corresponding tensor in cond2 (or use first one)
            tensor2 = cond2[0][0] if cond2 else tensor1
            metadata2 = cond2[0][1] if cond2 and len(cond2[0]) > 1 else {}
            
            # Blend the main conditioning tensors
            if blend_mode == "add":
                blended_tensor = tensor1 + (tensor2 * strength)
            elif blend_mode == "multiply":
                blended_tensor = tensor1 * (1.0 + (tensor2 - 1.0) * strength)
            elif blend_mode == "average":
                blended_tensor = (tensor1 + tensor2 * strength) / (1.0 + strength)
            else:
                blended_tensor = tensor1
            
            # Blend pooled_output if both have it
            if "pooled_output" in metadata1 and "pooled_output" in metadata2:
                pooled1 = metadata1["pooled_output"]
                pooled2 = metadata2["pooled_output"]
                if blend_mode == "add":
                    metadata1["pooled_output"] = pooled1 + (pooled2 * strength)
                elif blend_mode == "multiply":
                    metadata1["pooled_output"] = pooled1 * (1.0 + (pooled2 - 1.0) * strength)
                elif blend_mode == "average":
                    metadata1["pooled_output"] = (pooled1 + pooled2 * strength) / (1.0 + strength)
            
            blended.append([blended_tensor, metadata1])
        
        return blended


NODE_CLASS_MAPPINGS = {
    "LunaParametersBridge": LunaParametersBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaParametersBridge": "Luna Parameters Bridge (Advanced)",
}