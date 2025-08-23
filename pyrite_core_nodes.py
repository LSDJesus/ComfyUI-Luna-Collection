import torch
import comfy.utils

# Our first node for the Pyrite Core pack.
# A simple, clean upscaler based on the user's specifications.

class Pyrite_SimpleUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        # This defines the inputs and widgets for our node.
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_by": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.0, 
                    "max": 8.0, 
                    "step": 0.01
                }),
                "resampling": (["lanczos", "bicubic", "bilinear", "nearest-exact"],),
                "show_preview": ("BOOLEAN", {"default": True}), # This is our new toggle
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Pyrite Core" # Our very own category in the node menu!

    def upscale(self, image, upscale_model, scale_by, resampling, show_preview):
        # If the user has toggled off the preview/functionality, we'll just return the original image.
        # This makes the toggle act like a "bypass" switch for this node.
        if not show_preview:
            return (image,)

        # The core logic for upscaling the image.
        # We get the device, move the image tensor to it, and then upscale.
        device = comfy.model_management.get_torch_device()
        image = image.to(device)

        width = round(image.shape[3] * scale_by)
        height = round(image.shape[2] * scale_by)

        # Using ComfyUI's built-in, efficient upscaling utility.
        s = comfy.utils.common_upscale(image.movedim(1, -1), width, height, resampling, "disabled").movedim(-1, 1)
        
        s = s.to(comfy.model_management.intermediate_device())

        return (s,)

# This is the standard boilerplate that tells ComfyUI how to register our new node.
NODE_CLASS_MAPPINGS = {
    "Pyrite_SimpleUpscaler": Pyrite_SimpleUpscaler
}

# This dictionary defines the user-friendly name that will appear in the node menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_SimpleUpscaler": "Pyrite Simple Upscaler"
}