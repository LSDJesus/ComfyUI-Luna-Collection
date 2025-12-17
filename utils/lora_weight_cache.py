"""
LoRA Weight Cache Utility

Provides efficient weight caching and restoration for transient LoRA application.
Caches only the specific model layers affected by LoRAs, not the entire model.

Architecture:
1. Before LoRA application: identify and cache affected layer weights
2. During workflow: LoRAs modify model weights
3. After workflow: restore cached weights to pristine state

Benefits:
- Minimal memory overhead (only affected layers cached)
- No precision drift (exact restoration from clone)
- No disk I/O between runs (cache in RAM)
- Supports randomized LoRA weights per run
"""

import torch
from typing import Dict, Optional, Any, Set, Callable
import copy


class LoRAWeightCache:
    """
    Caches model weights before LoRA application for later restoration.
    
    Usage:
        cache = LoRAWeightCache()
        cache.cache_weights_for_loras(model, lora_list)
        # ... apply LoRAs, run inference ...
        cache.restore_weights(model)
    """
    
    def __init__(self):
        self.cached_weights: Dict[str, torch.Tensor] = {}
        self.cached_for_model_id: Optional[int] = None
        
    def cache_weights_for_loras(
        self, 
        model: Any, 
        lora_stack: list,
        lora_resolver_fn: Optional[Callable] = None
    ) -> int:
        """
        Cache model weights that will be affected by LoRAs.
        
        Args:
            model: The model whose weights will be cached
            lora_stack: List of (lora_name, model_weight, clip_weight) tuples
            lora_resolver_fn: Optional function to resolve lora_name to full path
        
        Returns:
            Number of layers cached
        """
        if not lora_stack:
            return 0
        
        # Track which model this cache is for
        self.cached_for_model_id = id(model)
        
        # Get affected layers from LoRA stack
        affected_layers = self._identify_affected_layers(
            model, lora_stack, lora_resolver_fn
        )
        
        # Cache weights for affected layers
        layers_cached = 0
        for layer_name in affected_layers:
            if layer_name not in self.cached_weights:
                try:
                    # Get the actual weight tensor
                    weight = self._get_layer_weight(model, layer_name)
                    if weight is not None:
                        # Clone to separate memory
                        self.cached_weights[layer_name] = weight.clone()
                        layers_cached += 1
                except Exception as e:
                    print(f"[LoRAWeightCache] Warning: Failed to cache {layer_name}: {e}")
        
        return layers_cached
    
    def restore_weights(self, model: Any) -> int:
        """
        Restore cached weights to model.
        
        Args:
            model: The model to restore weights to
        
        Returns:
            Number of layers restored
        """
        if not self.cached_weights:
            return 0
        
        # Verify this is the same model we cached for
        if self.cached_for_model_id is not None and id(model) != self.cached_for_model_id:
            print("[LoRAWeightCache] Warning: Restoring to different model instance")
        
        layers_restored = 0
        for layer_name, cached_weight in self.cached_weights.items():
            try:
                # Restore the weight
                self._set_layer_weight(model, layer_name, cached_weight)
                layers_restored += 1
            except Exception as e:
                print(f"[LoRAWeightCache] Warning: Failed to restore {layer_name}: {e}")
        
        return layers_restored
    
    def clear(self):
        """Clear cached weights and free memory."""
        self.cached_weights.clear()
        self.cached_for_model_id = None
        torch.cuda.empty_cache()
    
    def _identify_affected_layers(
        self,
        model: Any,
        lora_stack: list,
        lora_resolver_fn: Optional[Callable] = None
    ) -> Set[str]:
        """
        Identify which model layers will be affected by the LoRA stack.
        
        Strategy: Load each LoRA and check which keys it contains.
        LoRA keys follow patterns like "diffusion_model.input_blocks.0.weight"
        """
        affected_layers = set()
        
        import folder_paths
        import comfy.utils  # type: ignore
        
        for lora_name, model_weight, clip_weight in lora_stack:
            # Skip if model weight is 0 (no model modification)
            if model_weight == 0:
                continue
            
            try:
                # Resolve LoRA path
                if lora_resolver_fn:
                    lora_path = lora_resolver_fn(lora_name)
                else:
                    lora_path = folder_paths.get_full_path("loras", lora_name)
                
                if not lora_path:
                    continue
                
                # Load LoRA to check keys (lightweight - just metadata)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)  # type: ignore
                
                # Extract affected layer names (strip LoRA-specific suffixes)
                for key in lora_data.keys():
                    # Filter out CLIP keys
                    if any(p in key.lower() for p in ['clip_l', 'clip_g', 'te1', 'te2', 'text_encoder', 'lora_te']):
                        continue
                    
                    # Extract base layer name (remove .lora_up, .lora_down, .alpha, etc.)
                    base_key = key.split('.lora_')[0] if '.lora_' in key else key.rsplit('.', 1)[0]
                    affected_layers.add(base_key)
                    
            except Exception as e:
                print(f"[LoRAWeightCache] Warning: Could not analyze LoRA '{lora_name}': {e}")
        
        return affected_layers
    
    def _get_layer_weight(self, model: Any, layer_name: str) -> Optional[torch.Tensor]:
        """Get weight tensor from model by layer name."""
        try:
            # Handle InferenceModeWrapper
            if hasattr(model, 'model'):
                actual_model = model.model
            else:
                actual_model = model
            
            # Navigate nested attributes (e.g., "diffusion_model.input_blocks.0.weight")
            parts = layer_name.split('.')
            obj = actual_model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif hasattr(obj, 'model') and hasattr(obj.model, part):
                    obj = getattr(obj.model, part)
                else:
                    return None
            
            # Return the weight tensor
            if isinstance(obj, torch.Tensor):
                return obj
            elif hasattr(obj, 'weight'):
                return obj.weight
            
        except Exception:
            pass
        
        return None
    
    def _set_layer_weight(self, model: Any, layer_name: str, weight: torch.Tensor):
        """Set weight tensor in model by layer name."""
        # Handle InferenceModeWrapper
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        # Navigate nested attributes
        parts = layer_name.split('.')
        obj = actual_model
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif hasattr(obj, 'model') and hasattr(obj.model, part):
                obj = getattr(obj.model, part)
        
        # Set the weight
        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            target = getattr(obj, final_attr)
            if isinstance(target, torch.Tensor):
                # Direct tensor replacement
                setattr(obj, final_attr, weight.to(target.device, dtype=target.dtype))
            elif hasattr(target, 'weight'):
                # Parameter with .weight attribute
                target.weight.copy_(weight.to(target.weight.device, dtype=target.weight.dtype))
