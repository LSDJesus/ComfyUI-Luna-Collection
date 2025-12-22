"""
Luna Pyramid Noise Generator - 4K Noise Scaffold Creator

Creates the master high-resolution noise scaffold that serves as the blueprint
for the entire detailing workflow. This noise is sliced and variance-corrected
at different resolutions to maintain perfect statistical properties.

Architecture:
- Generates noise at target resolution based on model type + aspect ratio + scale
- Outputs both full-size scaffold AND model-native draft scaffold
- Noise is in latent space (H/8, W/8 dimensions)
- Provides dimension outputs for downstream nodes
- Seed-reproducible for consistent results

Mathematical Foundation:
- Noise has σ=1.0, μ=0.0 (standard Gaussian)
- Draft scaffold downscaled with variance correction to maintain σ=1.0
- This prevents "dead" or "hot" noise artifacts during refinement
"""

import torch
import torch.nn.functional as F
import comfy.model_management


# Model-specific base resolutions (training buckets)
BASE_RESOLUTIONS = {
    "SDXL": {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "3:2": (1536, 1024),
        "2:3": (1024, 1536),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    },
    "SD1.5": {
        "1:1": (512, 512),
        "16:9": (768, 432),
        "9:16": (432, 768),
        "3:2": (768, 512),
        "2:3": (512, 768),
        "4:3": (640, 480),
        "3:4": (480, 640),
    },
    "Flux": {
        # Flux uses same buckets as SDXL
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "3:2": (1536, 1024),
        "2:3": (1024, 1536),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    },
}


class LunaPyramidNoiseGenerator:
    """
    Generate master noise scaffold for pyramid-based detailing.
    
    Creates high-resolution latent noise that will be sliced at different
    scales during the detailing workflow. This eliminates interpolation
    artifacts that occur when upscaling low-res noise.
    
    Uses model-aware resolution selection based on training buckets for
    optimal quality at the draft generation stage.
    
    Workflow Integration:
        Pyramid Noise → Config Gateway → Draft KSampler → Detector → Detailer
    """
    
    CATEGORY = "Luna/Detailing"
    RETURN_TYPES = ("LATENT", "LATENT", "INT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("full_scaffold", "draft_scaffold", "full_width", "full_height", "draft_width", "draft_height", "scale_factor", "seed")
    FUNCTION = "generate"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["SDXL", "SD1.5", "Flux"], {
                    "default": "SDXL",
                    "tooltip": "Model architecture determines base resolution buckets"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9", "9:21"], {
                    "default": "16:9",
                    "tooltip": "Aspect ratio from model's training buckets"
                }),
                "scale_multiplier": ([2, 3, 4, 5, 6, 8], {
                    "default": 4,
                    "tooltip": "Multiply base resolution (4x SDXL 16:9 = 5376x3072)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Number of noise scaffolds to generate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible noise generation"
                }),
            }
        }
    
    def generate(self, model_type: str, aspect_ratio: str, scale_multiplier: int, batch_size: int, seed: int) -> tuple:
        """
        Generate master noise scaffold at target resolution with model-native draft.
        
        Args:
            model_type: Model architecture (SDXL, SD1.5, Flux)
            aspect_ratio: Training bucket aspect ratio
            scale_multiplier: Upscale factor from base resolution
            batch_size: Number of latents to generate
            seed: Random seed
            
        Returns:
            Tuple of:
            - full_scaffold: Full-resolution latent {"samples": tensor}
            - draft_scaffold: Model-native latent with variance correction
            - full_width, full_height: Full dimensions in pixels
            - draft_width, draft_height: Draft dimensions in pixels
            - scale_factor: Ratio between full and draft
        """
        # Get base resolution from lookup table
        if model_type not in BASE_RESOLUTIONS:
            print(f"[LunaPyramidNoise] Unknown model type '{model_type}', defaulting to SDXL")
            model_type = "SDXL"
        
        if aspect_ratio not in BASE_RESOLUTIONS[model_type]:
            print(f"[LunaPyramidNoise] Unknown aspect ratio '{aspect_ratio}' for {model_type}, defaulting to 1:1")
            aspect_ratio = "1:1"
        
        draft_width, draft_height = BASE_RESOLUTIONS[model_type][aspect_ratio]
        
        # Calculate full dimensions
        full_width = draft_width * scale_multiplier
        full_height = draft_height * scale_multiplier
        
        # Validate dimensions (VAE requires multiples of 8)
        full_width = ((full_width + 7) // 8) * 8
        full_height = ((full_height + 7) // 8) * 8
        draft_width = ((draft_width + 7) // 8) * 8
        draft_height = ((draft_height + 7) // 8) * 8
        
        # Calculate latent dimensions (VAE downsamples 8x)
        full_latent_w = full_width // 8
        full_latent_h = full_height // 8
        draft_latent_w = draft_width // 8
        draft_latent_h = draft_height // 8
        
        # Get device
        device = comfy.model_management.get_torch_device()
        
        # Generate noise with seed
        torch.manual_seed(seed)
        
        # SDXL/Flux latent channels = 4
        # Generate full-size scaffold
        full_noise = torch.randn(
            (batch_size, 4, full_latent_h, full_latent_w),
            device=device,
            dtype=torch.float32
        )
        
        # Generate draft scaffold by downscaling with variance correction
        # Use area interpolation for statistical averaging
        draft_noise = F.interpolate(
            full_noise,
            size=(draft_latent_h, draft_latent_w),
            mode='area'
        )
        
        # CRITICAL: Variance correction
        # Area averaging reduces variance by scale_factor
        # Multiply to restore σ=1.0
        draft_noise = draft_noise * scale_multiplier
        
        # Verify statistical properties
        full_std = full_noise.std().item()
        full_mean = full_noise.mean().item()
        draft_std = draft_noise.std().item()
        draft_mean = draft_noise.mean().item()
        
        print(f"[LunaPyramidNoise] Generated scaffold:")
        print(f"  Model: {model_type}, Aspect: {aspect_ratio}, Scale: {scale_multiplier}x")
        print(f"  Full:  {full_width}x{full_height} px ({full_latent_w}x{full_latent_h} latent)")
        print(f"         μ={full_mean:.4f}, σ={full_std:.4f}")
        print(f"  Draft: {draft_width}x{draft_height} px ({draft_latent_w}x{draft_latent_h} latent)")
        print(f"         μ={draft_mean:.4f}, σ={draft_std:.4f} (variance corrected)")
        print(f"  Batch: {batch_size}, Seed: {seed}")
        
        # Wrap in ComfyUI latent format
        full_latent_dict = {"samples": full_noise}
        draft_latent_dict = {"samples": draft_noise}
        
        return (
            full_latent_dict,
            draft_latent_dict,
            full_width,
            full_height,
            draft_width,
            draft_height,
            float(scale_multiplier),
            seed
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaPyramidNoiseGenerator": LunaPyramidNoiseGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPyramidNoiseGenerator": "🌙 Luna: Pyramid Noise",
}
