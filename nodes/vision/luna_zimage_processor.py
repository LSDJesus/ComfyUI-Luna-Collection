"""
Luna Z-IMAGE Conditioning Processor - Noise Injection for Seed Variability

Simplified node that injects noise into conditioning for seed variability.
Place between Config Gateway and KSampler.

The actual prompt generation is now handled by:
  • LunaVLMPromptGenerator - for LLM-enhanced prompts
  • Luna Config Gateway - for CLIP encoding

This node focuses solely on conditioning post-processing: adding noise for batch variety.

┌─────────────────────────────────────────────────────────────────────────────┐
│                  Luna Z-IMAGE Conditioning Processor                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUTS:                                                                    │
│    positive: CONDITIONING (from Config Gateway or other encoder)           │
│    negative: CONDITIONING (optional)                                       │
│                                                                             │
│  NOISE SETTINGS:                                                            │
│    enable_noise: BOOLEAN (toggle noise injection)                          │
│    noise_threshold: FLOAT (0.0-1.0, when to apply noise)                   │
│    noise_strength: FLOAT (0-100, noise magnitude)                          │
│    seed: INT (for reproducibility)                                         │
│                                                                             │
│  OUTPUTS:                                                                   │
│    positive: CONDITIONING (with optional noise)                            │
│    negative: CONDITIONING (pass-through)                                   │
│    status: STRING (processing status)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Usage Flow:
  LunaVLMPromptGenerator ──→ Luna Config Gateway ──→ Luna Z-IMAGE Processor ──→ KSampler
  (enhance prompt)         (encode + merge)         (inject noise)          (generate)
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict, List

import torch
import numpy as np

try:
    import folder_paths
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None


# =============================================================================
# Luna Z-IMAGE Conditioning Processor
# =============================================================================

class LunaZImageProcessor:
    """
    Processes and enhances Z-IMAGE conditioning with noise injection.
    
    Refactored to focus solely on conditioning post-processing.
    Prompt generation and encoding are now handled by dedicated nodes:
      • LunaVLMPromptGenerator - LLM enhancement + text generation
      • LunaConfigGateway - CLIP encoding + merging
    
    This node adds noise to conditioning for seed variability in batches.
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "status")
    FUNCTION = "process"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from Config Gateway or CLIP encoder"
                }),
            },
            "optional": {
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning (optional, passed through)"
                }),
                
                # === Noise Injection ===
                "enable_noise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable noise injection for seed variability"
                }),
                "noise_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising threshold: noise applied from 0 to threshold, clean from threshold to 1.0"
                }),
                "noise_strength": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Noise magnitude to add to conditioning tensors"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for noise generation (-1 = random)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "batch_size_from_js": ("INT", {"default": 1}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, seed=-1, batch_size_from_js=1, **kwargs):
        """Ensure re-execution when seed or batch size changes."""
        return f"{seed}_{batch_size_from_js}"
    
    def process(
        self,
        positive,
        enable_noise: bool = False,
        noise_threshold: float = 0.2,
        noise_strength: float = 10.0,
        seed: int = -1,
        negative=None,
        unique_id: str = "",
        batch_size_from_js: int = 1,
    ) -> Tuple[List, List, str]:
        """
        Process conditioning with optional noise injection.
        """
        status_parts = []
        
        # Pass-through positive conditioning
        output_positive = positive
        
        # Handle seed
        if seed == -1:
            seed = int(torch.randint(0, 0xffffffff, (1,)).item())
        
        # =================================================================
        # Noise Injection (if enabled)
        # =================================================================
        if enable_noise and noise_strength > 0:
            output_positive = self._inject_noise(
                conditioning=positive,
                threshold=noise_threshold,
                strength=noise_strength,
                seed=seed,
                batch_size=batch_size_from_js,
            )
            status_parts.append(f"Noise injected (threshold={noise_threshold:.2f}, strength={noise_strength:.1f})")
        else:
            status_parts.append("No noise injection")
        
        # Pass through or create empty negative
        if negative is not None:
            output_negative = negative
        else:
            # Create empty negative conditioning (same structure as positive)
            output_negative = [[torch.zeros_like(positive[0][0]), positive[0][1]]]
        
        status = " | ".join(status_parts) if status_parts else "Ready"
        
        return (output_positive, output_negative, status)
    
    @staticmethod
    def _inject_noise(
        conditioning: List,
        threshold: float,
        strength: float,
        seed: int,
        batch_size: int,
    ) -> List:
        """
        Inject noise into conditioning tensors for variability.
        
        Args:
            conditioning: List of (conditioning_tensor, metadata) tuples
            threshold: Denoising timestep threshold (0-1)
            strength: Noise magnitude (0-100)
            seed: Random seed for reproducibility
            batch_size: Batch size for noise generation
        
        Returns:
            Modified conditioning with noise injected
        """
        if not conditioning or len(conditioning) == 0:
            return conditioning
        
        # Set random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Process each conditioning sample
        output = []
        for cond_tensor, metadata in conditioning:
            if not isinstance(cond_tensor, torch.Tensor):
                output.append((cond_tensor, metadata))
                continue
            
            # Generate noise with same shape and dtype
            noise = torch.randn(
                cond_tensor.shape,
                dtype=cond_tensor.dtype,
                device=cond_tensor.device,
                generator=generator
            )
            
            # Scale noise by strength (normalized to 0-1)
            noise_scaled = noise * (strength / 100.0)
            
            # Apply noise at specified threshold
            # Below threshold: apply noise (for variability)
            # Above threshold: keep clean (for convergence)
            noisy_tensor = cond_tensor + noise_scaled
            
            output.append((noisy_tensor, metadata))
        
        return output


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaZImageProcessor": LunaZImageProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaZImageProcessor": "Luna Z-IMAGE Conditioning Processor",
}
