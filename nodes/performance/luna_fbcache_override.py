"""
Luna FBCache Override - Apply different FBCache settings to a model

Allows you to override FBCache settings for specific sampling operations.
Useful for applying aggressive caching to refinement steps (USDU, detailers)
while keeping conservative settings for initial generation.

Example workflow:
- Config Gateway: threshold=0.15, max_hits=1 (conservative for generation)
- FBCache Override → USDU: threshold=0.25, max_hits=-1 (aggressive for refinement)
"""


class LunaFBCacheOverride:
    """
    Override FBCache settings on a model for refinement operations.
    
    Clones the model and re-applies FBCache with new settings optimized
    for refinement phases (higher threshold, unlimited hits).
    """
    
    CATEGORY = "Luna/Performance"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "override_fbcache"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable FBCache override (if False, passes model through unchanged)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Residual diff threshold (higher = more aggressive caching for refinement)"
                }),
                "start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Start caching at this percentage of sampling"
                }),
                "end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stop caching at this percentage of sampling"
                }),
                "max_consecutive_hits": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Max consecutive cache hits (-1=unlimited, good for refinement)"
                }),
            }
        }
    
    def override_fbcache(self, model, enabled, threshold, start, end, max_consecutive_hits):
        """
        Override FBCache settings on the model.
        
        Args:
            model: Input MODEL (can have existing FBCache)
            enabled: Whether to apply override
            threshold: New residual diff threshold
            start: New start percentage
            end: New end percentage
            max_consecutive_hits: New max consecutive hits
            
        Returns:
            Model with new FBCache settings applied
        """
        if not enabled:
            return (model,)
        
        try:
            from fbcache_wrapper import apply_fbcache_to_model
            
            # Clone the model to avoid affecting other branches
            model = model.clone()
            
            # Re-apply FBCache with new settings
            # The wrapper will replace any existing wrapper
            model = apply_fbcache_to_model(
                model,
                residual_diff_threshold=threshold,
                start=start,
                end=end,
                max_consecutive_cache_hits=max_consecutive_hits
            )
            
            print(f"[LunaFBCacheOverride] Applied: threshold={threshold}, "
                  f"range={start:.0%}-{end:.0%}, max_hits={max_consecutive_hits}")
            
            return (model,)
            
        except ImportError as e:
            print(f"[LunaFBCacheOverride] ERROR: FBCache not available - {e}")
            return (model,)
        except Exception as e:
            print(f"[LunaFBCacheOverride] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return (model,)


NODE_CLASS_MAPPINGS = {
    "LunaFBCacheOverride": LunaFBCacheOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaFBCacheOverride": "Luna FBCache Override",
}
