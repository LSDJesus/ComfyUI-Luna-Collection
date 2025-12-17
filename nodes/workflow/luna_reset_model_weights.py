"""
Luna Reset Model Weights Node

Restores model weights to pristine state after LoRA application.
Works in conjunction with Config Gateway's weight caching.

Use at end of workflow to prepare model for next run with different LoRA weights.
"""

from utils.lora_weight_cache import LoRAWeightCache


# Access global weight cache from Config Gateway
from nodes.workflow.luna_config_gateway import _lora_weight_cache


class LunaResetModelWeights:
    """
    Restore model weights to pre-LoRA state.
    
    Purpose:
    - Undo LoRA modifications applied by Config Gateway
    - Prepare model for next workflow run with different LoRA weights
    - Avoids ComfyUI having to reload model from disk
    
    Workflow placement:
    Place at end of generation workflow, after all sampling is complete.
    
    Architecture:
    - Config Gateway caches pristine weights before applying LoRAs
    - This node restores those cached weights
    - No disk I/O, no precision drift, minimal memory overhead
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to restore (must match model used in Config Gateway)"}),
            },
            "optional": {
                "passthrough": ("*", {"tooltip": "Optional input to pass through unchanged"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "*")
    RETURN_NAMES = ("model", "passthrough")
    FUNCTION = "reset_weights"
    CATEGORY = "Luna/workflow"
    
    DESCRIPTION = """
Restores model weights to pristine state after LoRA application.

Use this at the end of your workflow to undo LoRA modifications.
This allows the same model to be reused with different LoRA weights
without reloading from disk.

Works with:
- InferenceModeWrapper models
- Standard ComfyUI models
- DaemonCLIP (CLIP weights handled separately)

Benefits:
- Eliminates disk I/O between workflow runs
- No precision drift (exact restoration)
- Minimal memory overhead (only affected layers cached)
- Enables randomized LoRA weights per run
"""
    
    def reset_weights(self, model, passthrough=None):
        """Restore model to pristine state using cached weights."""
        global _lora_weight_cache
        
        try:
            layers_restored = _lora_weight_cache.restore_weights(model)
            
            if layers_restored > 0:
                print(f"[LunaResetModelWeights] Restored {layers_restored} layer weights to pristine state")
            else:
                print("[LunaResetModelWeights] No cached weights to restore (model may not have LoRAs applied)")
            
            # Clear cache to free memory
            _lora_weight_cache.clear()
            
        except Exception as e:
            print(f"[LunaResetModelWeights] Error restoring weights: {e}")
            # Don't fail workflow - just log the error
        
        return (model, passthrough)


NODE_CLASS_MAPPINGS = {
    "LunaResetModelWeights": LunaResetModelWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaResetModelWeights": "Luna Reset Model Weights",
}
