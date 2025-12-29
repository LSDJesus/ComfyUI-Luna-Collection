"""
Luna Native Canvas Downscale

Downscales 4K master canvas (latent + conditioning) to model-native 1K resolution
for initial generation. The 4K versions remain as the "master blueprint" for
all downstream refinement steps.

Architecture:
- 4K latent → 1K latent (with variance correction)
- 4K area conditioning → 1K area conditioning (coords scaled)
- Text embeddings pass through unchanged (resolution-agnostic)

Workflow Integration:
    Config Gateway (4K) → Native Downscale → KSampler (1K) → Upscale → Refine (4K)
"""

import torch
import torch.nn.functional as F


class LunaNativeCanvasDownscale:
    """
    Downscale 4K master canvas to model-native resolution for initial generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_4k": ("LATENT",),
                "positive_4k": ("CONDITIONING",),
                "negative_4k": ("CONDITIONING",),
                "scale_factor": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Downscale factor (4.0 = 4K→1K, 2.0 = 2K→1K)"
                }),
                "variance_correction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply variance correction after downscaling. Disable for low-variance 'flat' generation."
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent_1k", "positive_1k", "negative_1k")
    FUNCTION = "downscale"
    CATEGORY = "Luna/Preprocessing"
    
    def downscale(
        self,
        latent_4k: dict,
        positive_4k,
        negative_4k,
        scale_factor: float,
        variance_correction: bool = True
    ) -> tuple:
        """
        Downscale latent and conditioning from target resolution to model native.
        
        Returns:
            (latent_1k, positive_1k, negative_1k)
        """
        # Downscale latent noise with variance correction
        latent_samples = latent_4k["samples"]
        
        # Calculate target size
        b, c, h, w = latent_samples.shape
        target_h = int(h / scale_factor)
        target_w = int(w / scale_factor)
        
        print(f"[LunaNativeCanvasDownscale] Downscaling {w*8}×{h*8}px → {target_w*8}×{target_h*8}px")
        
        # Downscale with area mode (maintains variance better than bilinear)
        downscaled_samples = F.interpolate(
            latent_samples,
            size=(target_h, target_w),
            mode='area'
        )
        
        # Variance correction for downscaling (optional)
        if variance_correction:
            # When downscaling, we're averaging pixels - need to scale variance
            # Area mode naturally handles this, but we add a correction
            variance_scale = scale_factor ** 0.5  # Square root for 2D scaling
            downscaled_samples = downscaled_samples * variance_scale
            print(f"[LunaNativeCanvasDownscale] ✓ Variance correction applied (×{variance_scale:.2f})")
        else:
            # Skip correction - creates low-variance "flat" generation
            print(f"[LunaNativeCanvasDownscale] ⚠ Variance correction DISABLED - flat generation mode")
        
        latent_1k = {"samples": downscaled_samples}
        
        # Downscale conditioning area coordinates
        
        if not variance_correction:
            print(f"[LunaNativeCanvasDownscale] → Expect soft/flat 1K generation, detail added by upscale+refinement")
        positive_1k = self._downscale_conditioning(positive_4k, scale_factor)
        negative_1k = self._downscale_conditioning(negative_4k, scale_factor)
        
        print(f"[LunaNativeCanvasDownscale] ✓ Latent: {h}×{w} → {target_h}×{target_w}")
        print(f"[LunaNativeCanvasDownscale] ✓ Conditioning area coords scaled by 1/{scale_factor}")
        
        return (latent_1k, positive_1k, negative_1k)
    
    def _downscale_conditioning(self, conditioning, scale_factor: float):
        """
        Downscale area conditioning coordinates.
        
        Text embeddings pass through unchanged - they're resolution-agnostic.
        Only area coordinates (h, w, y, x) need scaling.
        """
        downscaled = []
        
        for emb, cond_dict in conditioning:
            cond_dict = cond_dict.copy()
            
            # Scale area coordinates if present
            if "area" in cond_dict:
                h, w, y, x = cond_dict["area"]
                
                # Scale down coordinates
                cond_dict["area"] = (
                    int(h / scale_factor),
                    int(w / scale_factor),
                    int(y / scale_factor),
                    int(x / scale_factor)
                )
            
            downscaled.append([emb, cond_dict])
        
        return downscaled


NODE_CLASS_MAPPINGS = {
    "LunaNativeCanvasDownscale": LunaNativeCanvasDownscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaNativeCanvasDownscale": "Luna Native Canvas Downscale",
}
