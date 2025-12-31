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
                "latent_4k": ("LATENT", {
                    "tooltip": "4K noise latent from Config Gateway"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Downscale factor (4.0 = 4K→1K, 2.0 = 2K→1K)"
                }),
                "variance_correction": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Variance restoration amount. 0.0=soft draft (σ≈0.25), 0.5=balanced, 1.0=full correction (σ=1.0)"
                }),
            },
            "optional": {
                "positive_4k": ("CONDITIONING", {
                    "tooltip": "Optional: Only needed if using area/regional conditioning"
                }),
                "negative_4k": ("CONDITIONING", {
                    "tooltip": "Optional: Only needed if using area/regional conditioning"
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
        scale_factor: float,
        variance_correction: float,
        positive_4k = None,
        negative_4k = None
    ) -> tuple:
        """
        Downscale latent and conditioning from target resolution to model native.
        
        Args:
            latent_4k: 4K noise latent from Config Gateway
            scale_factor: Downscale ratio (4.0 = 4K→1K)
            variance_correction: 0.0=soft draft, 1.0=full variance restoration
            positive_4k: Optional area conditioning
            negative_4k: Optional area conditioning
        
        Returns:
            (latent_1k, positive_1k, negative_1k)
        """
        # Downscale latent noise with variable variance correction
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
        
        # Variable variance correction (0.0 to 1.0)
        # Natural downscaling reduces variance by ~scale_factor
        # We restore a percentage of that lost variance
        if variance_correction > 0.0:
            # Calculate the variance multiplier needed for full correction
            full_correction_multiplier = scale_factor
            
            # Apply partial correction based on variance_correction parameter
            # 0.0 = no correction (σ ≈ 0.25 for 4x downscale)
            # 0.5 = half correction (σ ≈ 0.625)
            # 1.0 = full correction (σ = 1.0)
            actual_multiplier = 1.0 + (variance_correction * (full_correction_multiplier - 1.0))
            downscaled_samples = downscaled_samples * actual_multiplier
            
            print(f"[LunaNativeCanvasDownscale] ✓ Variance correction: {variance_correction:.2f} (×{actual_multiplier:.2f})")
            
            if variance_correction < 0.3:
                print(f"[LunaNativeCanvasDownscale] → Soft draft mode - expect smooth, low-detail 1K generation")
            elif variance_correction > 0.7:
                print(f"[LunaNativeCanvasDownscale] → High-variance draft - expect detailed 1K generation")
        else:
            print(f"[LunaNativeCanvasDownscale] ⚠ NO variance correction - very soft/flat generation")
        
        latent_1k = {"samples": downscaled_samples}
        
        # Downscale conditioning area coordinates (if provided)
        if positive_4k is not None:
            positive_1k = self._downscale_conditioning(positive_4k, scale_factor)
            print(f"[LunaNativeCanvasDownscale] ✓ Positive conditioning area coords scaled by 1/{scale_factor}")
        else:
            positive_1k = None
            print(f"[LunaNativeCanvasDownscale] ⊘ No positive conditioning provided (text-only mode)")
        
        if negative_4k is not None:
            negative_1k = self._downscale_conditioning(negative_4k, scale_factor)
            print(f"[LunaNativeCanvasDownscale] ✓ Negative conditioning area coords scaled by 1/{scale_factor}")
        else:
            negative_1k = None
            print(f"[LunaNativeCanvasDownscale] ⊘ No negative conditioning provided (text-only mode)")
        
        print(f"[LunaNativeCanvasDownscale] ✓ Latent: {h}×{w} → {target_h}×{target_w}")
        
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
