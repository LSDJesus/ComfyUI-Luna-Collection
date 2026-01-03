"""
FBCache (First Block Cache) Wrapper for Luna Collection

Simplified wrapper around fbcache_utils to apply caching to models
without external dependencies. Based on wavespeed's fbcache_node.py.
"""
import contextlib
import unittest.mock
import torch
from typing import Optional, Callable

import fbcache_utils


def apply_fbcache_to_model(
    model,
    *,
    residual_diff_threshold=0.1,
    start=0.0,
    end=1.0,
    max_consecutive_cache_hits=-1
):
    """
    Apply First Block Cache optimization to a diffusion model.
    
    Args:
        model: ComfyUI MODEL object
        residual_diff_threshold: Similarity threshold (0.0 disables, higher = more caching)
        start: Start percentage of sampling (0.0-1.0)
        end: End percentage of sampling (0.0-1.0)
        max_consecutive_cache_hits: Max consecutive cache uses (-1 = unlimited)
    
    Returns:
        Patched model
    """
    if residual_diff_threshold <= 0.0 or max_consecutive_cache_hits == 0:
        return model
    
    fbcache_utils.patch_get_output_data()
    
    # Validation function for time-based and consecutive cache limiting
    using_validation = max_consecutive_cache_hits >= 0 or start > 0 or end < 1
    if using_validation:
        model_sampling = model.get_model_object("model_sampling")
        start_sigma, end_sigma = (float(
            model_sampling.percent_to_sigma(pct)) for pct in (start, end))
        del model_sampling
        
        @torch.compiler.disable()
        def validate_use_cache_inner(use_cached):
            nonlocal consecutive_cache_hits, current_timestep
            # Only check sigma range if timestep and sigmas are all defined
            if (current_timestep is not None and 
                start_sigma is not None and 
                end_sigma is not None):
                use_cached = use_cached and end_sigma <= current_timestep <= start_sigma
            use_cached = use_cached and (max_consecutive_cache_hits < 0
                                         or consecutive_cache_hits
                                         < max_consecutive_cache_hits)
            consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
            return use_cached
        validate_use_cache = validate_use_cache_inner
    else:
        # Type hint to satisfy Pylance - this will be reassigned if validate_use_cache_fn is provided
        validate_use_cache: Optional[Callable[[bool], bool]] = None
    
    prev_timestep = None
    prev_input_state = None
    current_timestep = None
    consecutive_cache_hits = 0
    
    def reset_cache_state():
        nonlocal prev_input_state, prev_timestep, consecutive_cache_hits
        prev_input_state = prev_timestep = None
        consecutive_cache_hits = 0
        fbcache_utils.set_current_cache_context(
            fbcache_utils.create_cache_context())
    
    def ensure_cache_state(model_input: torch.Tensor, timestep: float):
        nonlocal current_timestep
        input_state = (model_input.shape, model_input.dtype, model_input.device)
        need_reset = (
            prev_timestep is None or
            prev_input_state != input_state or
            fbcache_utils.get_current_cache_context() is None or
            timestep >= prev_timestep
        )
        if need_reset:
            reset_cache_state()
        current_timestep = timestep
    
    def update_cache_state(model_input: torch.Tensor, timestep: float):
        nonlocal prev_timestep, prev_input_state
        prev_timestep = timestep
        prev_input_state = (model_input.shape, model_input.dtype, model_input.device)
    
    model = model.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    
    # Handle UNet and Flux models
    if diffusion_model.__class__.__name__ in ("UNetModel", "Flux"):
        
        if diffusion_model.__class__.__name__ == "UNetModel":
            create_patch_function = fbcache_utils.create_patch_unet_model__forward
        elif diffusion_model.__class__.__name__ == "Flux":
            create_patch_function = fbcache_utils.create_patch_flux_forward_orig
        else:
            raise ValueError(
                f"Unsupported model {diffusion_model.__class__.__name__}")
        
        patch_forward = create_patch_function(
            diffusion_model,
            residual_diff_threshold=residual_diff_threshold,
            validate_can_use_cache_function=validate_use_cache,
        )
        
        def model_unet_function_wrapper(model_function, kwargs):
            try:
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                t = timestep[0].item()
                
                ensure_cache_state(input, t)
                
                with patch_forward():
                    result = model_function(input, timestep, **c)
                    update_cache_state(input, t)
                    return result
            except Exception as exc:
                reset_cache_state()
                raise exc from None
    
    # Handle transformer-based models
    else:
        is_non_native_ltxv = False
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            is_non_native_ltxv = True
            diffusion_model = diffusion_model.transformer
        
        double_blocks_name = None
        single_blocks_name = None
        if hasattr(diffusion_model, "transformer_blocks"):
            double_blocks_name = "transformer_blocks"
        elif hasattr(diffusion_model, "double_blocks"):
            double_blocks_name = "double_blocks"
        elif hasattr(diffusion_model, "joint_blocks"):
            double_blocks_name = "joint_blocks"
        else:
            raise ValueError(
                f"No double blocks found for {diffusion_model.__class__.__name__}"
            )
        
        if hasattr(diffusion_model, "single_blocks"):
            single_blocks_name = "single_blocks"
        
        if is_non_native_ltxv:
            original_create_skip_layer_mask = getattr(
                diffusion_model, "create_skip_layer_mask", None)
            if original_create_skip_layer_mask is not None:
                def new_create_skip_layer_mask(self, *args, **kwargs):
                    raise RuntimeError(
                        "STG is not supported with FBCache yet")
                
                diffusion_model.create_skip_layer_mask = new_create_skip_layer_mask.__get__(
                    diffusion_model)
        
        cached_transformer_blocks = torch.nn.ModuleList([
            fbcache_utils.CachedTransformerBlocks(
                None if double_blocks_name is None else getattr(
                    diffusion_model, double_blocks_name),
                None if single_blocks_name is None else getattr(
                    diffusion_model, single_blocks_name),
                residual_diff_threshold=residual_diff_threshold,
                validate_can_use_cache_function=validate_use_cache,
                cat_hidden_states_first=diffusion_model.__class__.__name__
                == "HunyuanVideo",
                return_hidden_states_only=diffusion_model.__class__.
                __name__ == "LTXVModel" or is_non_native_ltxv,
                clone_original_hidden_states=diffusion_model.__class__.
                __name__ == "LTXVModel",
                return_hidden_states_first=diffusion_model.__class__.
                __name__ != "OpenAISignatureMMDITWrapper",
                accept_hidden_states_first=diffusion_model.__class__.
                __name__ != "OpenAISignatureMMDITWrapper",
            )
        ])
        dummy_single_transformer_blocks = torch.nn.ModuleList()
        
        def model_unet_function_wrapper(model_function, kwargs):
            try:
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                t = timestep[0].item()
                
                ensure_cache_state(input, t)
                
                with unittest.mock.patch.object(
                        diffusion_model,
                        double_blocks_name,
                        cached_transformer_blocks,
                ), unittest.mock.patch.object(
                        diffusion_model,
                        single_blocks_name,
                        dummy_single_transformer_blocks,
                ) if single_blocks_name is not None else contextlib.nullcontext(
                ):
                    result = model_function(input, timestep, **c)
                    update_cache_state(input, t)
                    return result
            except Exception as exc:
                reset_cache_state()
                raise exc from None
    
    model.set_model_unet_function_wrapper(model_unet_function_wrapper)
    return model
