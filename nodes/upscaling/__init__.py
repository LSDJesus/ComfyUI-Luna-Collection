from .luna_ultimate_sd_upscale import Luna_UltimateSDUpscale
from .luna_upscaler_advanced import Luna_Advanced_Upscaler
from .luna_upscaler_simple import Luna_SimpleUpscaler
from .luna_super_upscaler import LunaSuperUpscaler, LunaSuperUpscalerSimple

NODE_CLASS_MAPPINGS = {
    "Luna_UltimateSDUpscale": Luna_UltimateSDUpscale,
    "Luna_Advanced_Upscaler": Luna_Advanced_Upscaler,
    "Luna_SimpleUpscaler": Luna_SimpleUpscaler,
    "LunaSuperUpscaler": LunaSuperUpscaler,
    "LunaSuperUpscalerSimple": LunaSuperUpscalerSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_UltimateSDUpscale": "Luna Ultimate SD Upscale",
    "Luna_Advanced_Upscaler": "Luna Advanced Upscaler",
    "Luna_SimpleUpscaler": "Luna Simple Upscaler",
    "LunaSuperUpscaler": "Luna Super Upscaler ⚡",
    "LunaSuperUpscalerSimple": "Luna Super Upscaler (Simple)",
}