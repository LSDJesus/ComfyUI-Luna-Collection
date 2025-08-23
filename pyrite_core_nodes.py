import torch
import comfy.utils
# --- FIX: We no longer need folder_paths for this, as we are not doing file I/O.
import comfy.upscale_models

# Our first node for the Pyrite Core pack. V2.3 Corrected.
# A simple, clean upscaler with a proper, non-blocking preview toggle.

class Pyrite_SimpleUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Pyrite Core"

    def upscale(self, image, upscale_model, scale_by, resampling, show_preview):
        # --- FIX: Instantiate the upscaler directly from the loaded upscale_model data.
        upscaler = comfy.upscale_models.PrefixedUpscaler(upscale_model)
        
        s = upscaler.upscale(image)

        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)
        
        if s.shape[3] != target_width or s.shape[2] != target_height:
            s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, resampling, "disabled").movedim(-1, 1)
        
        s = s.to(comfy.model_management.intermediate_device())

        if show_preview:
            return {"ui": {"images": s[0:1]}, "result": (s,)}
        else:
            return (s,)

# Our second node for the Pyrite Core pack. V1.3 Corrected.
# An advanced, precision upscaler with professional-grade controls.

class Pyrite_AdvancedUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.01}),
                "resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),
                "supersample": ("BOOLEAN", {"default": False}),
                "rounding_modulus": ([1, 2, 4, 8, 16, 32, 64], {"default": 8}), # Rounding modulus is an interesting case we will ignore for now.
                "rescale_after_model": ("BOOLEAN", {"default": True}),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Pyrite Core"

    def upscale(self, image, upscale_model, scale_by, resampling, supersample, rounding_modulus, rescale_after_model, show_preview):
        # --- FIX: Same instantiation logic as the simple upscaler.
        upscaler = comfy.upscale_models.PrefixedUpscaler(upscale_model)
        
        s = upscaler.upscale(image)

        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)

        if supersample:
            s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, "lanczos", "disabled").movedim(-1, 1)
        else:
            if rescale_after_model and (s.shape[3] != target_width or s.shape[2] != target_height):
                s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, resampling, "disabled").movedim(-1, 1)

        s = s.to(comfy.model_management.intermediate_device())

        if show_preview:
            return {"ui": {"images": s[0:1]}, "result": (s,)}
        else:
            return (s,)

# This boilerplate remains the same.
NODE_CLASS_MAPPINGS = {
    "Pyrite_SimpleUpscaler": Pyrite_SimpleUpscaler,
    "Pyrite_AdvancedUpscaler": Pyrite_AdvancedUpscaler
}

# This dictionary defines the user-friendly name that will appear in the node menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler",
    "Pyrite_AdvancedUpscaler": "Pyrite Advanced Upscaler"
}