import torch
import comfy.utils
# We don't need any special imports beyond the basics,
# as we are receiving the already-loaded model object.

# Our first node for the Pyrite Core pack. V2.4 Corrected.
# A simple, clean upscaler built to the correct ComfyUI specifications.

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
        
        # --- FIX: Move the model to the GPU before use. ---
        upscale_model.to(device)
        
        # Prepare the image tensor for processing
        in_img = image.movedim(-1, -3).to(device)

        # --- FIX: Use the robust tiled_scale utility, treating the model as a callable function. ---
        # We use the model's inherent scale factor for this primary pass.
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=512, tile_y=512, overlap=32, upscale_amount=upscale_model.scale)

        # --- FIX: Clean up by moving the model back to the CPU. ---
        upscale_model.to("cpu")
        
        # Post-model rescale to match the user's exact 'scale_by' target.
        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)
        
        if s.shape[3] != target_width or s.shape[2] != target_height:
            s = comfy.utils.common_upscale(s, target_width, target_height, resampling, "disabled")
        
        # Final formatting and cleanup.
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        s = s.to(comfy.model_management.intermediate_device())

        if show_preview:
            return {"ui": {"images": s[0:1]}, "result": (s,)}
        else:
            return (s,)

# Our second node for the Pyrite Core pack. V1.4 Corrected.
# An advanced, precision upscaler built to the same correct specifications.

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
                "rescale_after_model": ("BOOLEAN", {"default": True}),
                "show_preview": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Pyrite Core"

    def upscale(self, image, upscale_model, scale_by, resampling, supersample, rescale_after_model, show_preview):
        device = comfy.model_management.get_torch_device()
        
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        # --- FIX: The core logic is the same, using the callable model with tiling.
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=512, tile_y=512, overlap=32, upscale_amount=upscale_model.scale)
        
        upscale_model.to("cpu")

        target_width = round(image.shape[3] * scale_by)
        target_height = round(image.shape[2] * scale_by)

        if supersample:
            s = comfy.utils.common_upscale(s, target_width, target_height, "lanczos", "disabled")
        else:
            if rescale_after_model and (s.shape[3] != target_width or s.shape[2] != target_height):
                s = comfy.utils.common_upscale(s, target_width, target_height, resampling, "disabled")

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
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

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler",
    "Pyrite_AdvancedUpscaler": "Pyrite Advanced Upscaler"
}