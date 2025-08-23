import torch
import comfy.utils

# Our first node for the Pyrite Core pack. V2.1 Corrected.
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
        device = comfy.model_management.get_torch_device()
        image = image.to(device)

        # --- FIX: Actually use the upscale_model ---
        # The model does the primary, intelligent upscaling.
        s = upscale_model.upscale(image)

        # --- FIX: Post-model rescale to match user's exact 'scale_by' ---
        # This ensures the final output size always respects the user's input, even if the model is not an exact 2x/4x.
        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)
        
        # We only perform this final resize if the model's output doesn't already match the target.
        if s.shape[3] != target_width or s.shape[2] != target_height:
            s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, resampling, "disabled").movedim(-1, 1)
        
        s = s.to(comfy.model_management.intermediate_device())

        if show_preview:
            return {"ui": {"images": s[0:1]}, "result": (s,)}
        else:
            return (s,)

# Our second node for the Pyrite Core pack. V1.1 Corrected.
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
                "rounding_modulus": ([1, 2, 4, 8, 16, 32, 64], {"default": 8}),
                "rescale_after_model": ("BOOLEAN", {"default": True}),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Pyrite Core"

    def upscale(self, image, upscale_model, scale_by, resampling, supersample, rounding_modulus, rescale_after_model, show_preview):
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        
        # --- FIX: The upscale_model is now the star of the show ---
        s = upscale_model.upscale(image)

        # Calculate the final target dimensions based on the original image
        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)

        # --- Supersample and Rescale logic now applies *after* the model ---
        if supersample:
            # Downscale smoothly from the model's large output to the final target size
            s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, "lanczos", "disabled").movedim(-1, 1)
        else:
            # --- Rescale After Model Logic (now more direct) ---
            if rescale_after_model and (s.shape[3] != target_width or s.shape[2] != target_height):
                # The rounding modulus is implicitly handled by the model, so we just need to correct to the final size.
                s = comfy.utils.common_upscale(s.movedim(1, -1), target_width, target_height, resampling, "disabled").movedim(-1, 1)

        s = s.to(comfy.model_management.intermediate_device())

        if show_preview:
            return {"ui": {"images": s[0:1]}, "result": (s,)}
        else:
            return (s,)

# This is the standard boilerplate that tells ComfyUI how to register our new node.
NODE_CLASS_MAPPINGS = {
    "Pyrite_SimpleUpscaler": Pyrite_SimpleUpscaler,
    "Pyrite_AdvancedUpscaler": Pyrite_AdvancedUpscaler
}

# This dictionary defines the user-friendly name that will appear in the node menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler",
    "Pyrite_AdvancedUpscaler": "Pyrite Advanced Upscaler"
}