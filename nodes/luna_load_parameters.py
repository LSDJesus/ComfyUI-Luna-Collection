import os
import torch
import comfy.sd
from comfy.cli_args import args
import folder_paths
import nodes

class LunaLoadParameters:
    CATEGORY = "Luna/Meta"
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "INT", "VAE", "METADATA", "LUNA_PIPE", "LATENT", "PARAMETERS_PIPE")
    RETURN_NAMES = ("MODEL", "CLIP", "positive", "negative", "model_name", "positive_text", "width", "height", "cfg", "seed", "steps", "denoise", "sampler_name", "scheduler", "clip_skip", "batch_size", "vae", "metadata", "luna_pipe", "latent_image", "parameters_pipe")
    FUNCTION = "load_parameters"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
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
            },
            "optional": {
                "positive_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw positive prompt text (ignored if positive_conditioning provided)"}),
                "negative_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Raw negative prompt text (ignored if negative_conditioning provided)"}),
                "positive_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed positive conditioning (takes precedence over positive_text)"}),
                "negative_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed negative conditioning (takes precedence over negative_text)"}),
            }
        }

    def load_parameters(self, model, clip, vae, width, height, cfg, seed, steps, denoise, clip_skip, batch_size, sampler_name, scheduler_name, positive_text="", negative_text="", positive_conditioning=None, negative_conditioning=None):
        # Apply CLIP skip
        clip_skip_node = nodes.CLIPSetLastLayer()
        clip_skipped = clip_skip_node.set_last_layer(clip, clip_skip)[0]

        # Handle conditioning inputs - use preprocessed if provided, otherwise encode text
        if positive_conditioning is not None:
            positive = positive_conditioning
            positive_text = "Preprocessed conditioning"  # Placeholder for metadata
        else:
            # Encode positive prompt
            text_encode_node = nodes.CLIPTextEncode()
            positive = text_encode_node.encode(clip_skipped, positive_text)[0]

        if negative_conditioning is not None:
            negative = negative_conditioning
            negative_text = "Preprocessed conditioning"  # Placeholder for metadata
        else:
            # Encode negative prompt
            text_encode_node = nodes.CLIPTextEncode()
            negative = text_encode_node.encode(clip_skipped, negative_text)[0]

        # Create empty latent image
        empty_latent_node = nodes.EmptyLatentImage()
        latent_image = empty_latent_node.generate(width, height, batch_size)[0]

        # Extract model name (this is a simplified extraction, may need adjustment based on how model is loaded)
        model_name = "unknown"  # Placeholder, ideally extract from model path or metadata

        # Create metadata dict compatible with Image Saver Simple/Metadata
        metadata = {
            "modelname": model_name,
            "positive": positive_text if positive_conditioning is None else "Preprocessed conditioning",
            "negative": negative_text if negative_conditioning is None else "Preprocessed conditioning",
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
        }

        # Create luna_pipe as tuple: (model, clip, vae, positive, negative, seed, sampler_name, scheduler)
        luna_pipe = (model, clip_skipped, vae, positive, negative, seed, sampler_name, scheduler_name)

        # Create parameters_pipe for LunaSampler: (steps, cfg, denoise, sampler_name, scheduler_name, seed)
        parameters_pipe = (steps, cfg, denoise, sampler_name, scheduler_name, seed)

        return (model, clip_skipped, positive, negative, model_name, positive_text, width, height, cfg, seed, steps, denoise, sampler_name, scheduler_name, clip_skip, batch_size, vae, metadata, luna_pipe, latent_image, parameters_pipe)

NODE_CLASS_MAPPINGS = {
    "LunaLoadParameters": LunaLoadParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaLoadParameters": "Luna Load Parameters",
}