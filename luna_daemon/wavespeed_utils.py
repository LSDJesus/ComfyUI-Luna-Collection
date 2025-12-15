"""
Wavespeed First-Block Cache Integration for Luna Daemon

Integrated FB cache functionality extracted from wavespeed custom node.
Enables 2x speedup on final denoising steps by caching first-block outputs
and reusing them when residuals are similar.

Applied daemon-side at model_forward() time - transparent to ComfyUI clients.
"""

import contextlib
import dataclasses
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Any
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Context Management (extracted from wavespeed)
# =============================================================================

@dataclasses.dataclass
class CacheContext:
    """Manages cache buffers for first-block caching."""
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )

    def get_incremental_name(self, name: Optional[str] = None) -> str:
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable()
    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        return self.buffers.get(name)

    @torch.compiler.disable()
    def set_buffer(self, name: str, buffer: torch.Tensor):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


_current_cache_context: Optional[CacheContext] = None


def create_cache_context() -> CacheContext:
    """Create a new cache context."""
    return CacheContext()


def get_current_cache_context() -> Optional[CacheContext]:
    """Get the current global cache context."""
    return _current_cache_context


def set_current_cache_context(cache_context: Optional[CacheContext] = None):
    """Set the global cache context."""
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(ctx: CacheContext):
    """Context manager for scoped cache context."""
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = ctx
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable()
def get_buffer(name: str) -> Optional[torch.Tensor]:
    """Get a buffer from current cache context."""
    ctx = get_current_cache_context()
    assert ctx is not None, "cache_context must be set before"
    return ctx.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name: str, buffer: torch.Tensor):
    """Set a buffer in current cache context."""
    ctx = get_current_cache_context()
    assert ctx is not None, "cache_context must be set before"
    ctx.set_buffer(name, buffer)


# =============================================================================
# Cache Validation (extracted from wavespeed)
# =============================================================================

@torch.compiler.disable()
def are_two_tensors_similar(t1: torch.Tensor, t2: torch.Tensor, *, threshold: float) -> bool:
    """Check if two tensors are similar within threshold."""
    if t1.shape != t2.shape:
        return False
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item() < threshold


@torch.compiler.disable()
def get_can_use_cache(first_hidden_states_residual: torch.Tensor, threshold: float) -> bool:
    """Check if we can use cached results based on residual similarity."""
    prev_first_hidden_states_residual = get_buffer("first_hidden_states_residual")
    can_use_cache = (
        prev_first_hidden_states_residual is not None
        and are_two_tensors_similar(
            prev_first_hidden_states_residual,
            first_hidden_states_residual,
            threshold=threshold,
        )
    )
    return can_use_cache


# =============================================================================
# FB Cache Configuration
# =============================================================================

class FBCacheConfig:
    """Configuration for first-block caching."""
    
    def __init__(
        self,
        enabled: bool = False,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        residual_diff_threshold: float = 0.1,
        max_consecutive_hits: int = -1,
        object_to_patch: str = "diffusion_model",
    ):
        """
        Args:
            enabled: Enable FB cache for this request
            start_percent: Start caching at this percentage (0.0=start, 1.0=end)
            end_percent: Stop caching at this percentage
            residual_diff_threshold: Tolerance for cache hits (0.0=strict, 1.0=loose)
            max_consecutive_hits: Max consecutive cache hits (-1=unlimited)
            object_to_patch: Model object to patch (typically 'diffusion_model')
        """
        self.enabled = enabled
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.residual_diff_threshold = residual_diff_threshold
        self.max_consecutive_hits = max_consecutive_hits
        self.object_to_patch = object_to_patch
    
    def __bool__(self):
        return self.enabled and self.residual_diff_threshold > 0.0


# =============================================================================
# Block-Level FB Cache Patching (Full Wavespeed Implementation)
# =============================================================================

def create_patch_unet_forward(model, *, residual_diff_threshold: float, validate_can_use_cache_function=None):
    """
    Create UNet forward patch for block-level caching (SD/SDXL models).
    
    Patches the UNet to cache after first 2 input blocks, then reuses
    cached residuals when first block output is similar.
    
    Based on wavespeed's create_patch_unet_model__forward.
    """
    import contextlib
    import unittest.mock
    
    def call_remaining_blocks(self, transformer_options, control, transformer_patches, hs, h, *args, **kwargs):
        """Process remaining blocks and return residual."""
        original_hidden_states = h
        
        # Import these locally to avoid circular imports
        try:
            from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, apply_control  # type: ignore
        except ImportError:
            logger.error("[FBCache] Could not import ComfyUI UNet utilities")
            raise
        
        # Process remaining input blocks (starting from block 2)
        for id, module in enumerate(self.input_blocks):
            if id < 2:
                continue
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, *args, **kwargs)
            h = apply_control(h, control, 'input')
            
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            
            hs.append(h)
            
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)
        
        # Middle block
        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed(self.middle_block, h, *args, **kwargs)
        h = apply_control(h, control, 'middle')
        
        # Output blocks
        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            
            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)
            
            h = torch.cat([h, hsp], dim=1)
            del hsp
            
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, *args, output_shape, **kwargs)
        
        hidden_states_residual = h - original_hidden_states
        return h, hidden_states_residual
    
    def unet_forward_with_cache(
        self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs
    ):
        """Patched UNet forward with first-block caching."""
        try:
            from comfy.ldm.modules.diffusionmodules.openaimodel import (  # type: ignore
                timestep_embedding, forward_timestep_embed, apply_control
            )
        except ImportError:
            logger.error("[FBCache] Could not import ComfyUI UNet utilities")
            raise
        
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        
        num_video_frames = kwargs.get("num_video_frames", getattr(self, 'default_num_video_frames', 1))
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)
        
        # Handle class conditioning gracefully - only use y if model supports it
        # SDXL models have num_classes=None, so y should be ignored
        use_class_conditioning = (self.num_classes is not None) and (y is not None)
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)
        
        if "emb_patch" in transformer_patches:
            patch = transformer_patches["emb_patch"]
            for p in patch:
                emb = p(emb, self.model_channels, transformer_options)
        
        if use_class_conditioning and y is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        
        can_use_cache = False
        h = x
        original_h = h  # Initialize to prevent unbound error
        
        # Process first 2 input blocks
        for id, module in enumerate(self.input_blocks):
            if id >= 2:
                break
            transformer_options["block"] = ("input", id)
            
            if id == 1:
                original_h = h
            
            h = forward_timestep_embed(
                module, h, emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator
            )
            h = apply_control(h, control, 'input')
            
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            
            hs.append(h)
            
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)
            
            # After first block, check if we can use cache
            if id == 1:
                first_hidden_states_residual = h - original_h
                can_use_cache = get_can_use_cache(first_hidden_states_residual, threshold=residual_diff_threshold)
                
                if validate_can_use_cache_function is not None:
                    can_use_cache = validate_can_use_cache_function(can_use_cache)
                
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual", first_hidden_states_residual)
                
                del first_hidden_states_residual
        
        torch._dynamo.graph_break()
        
        if can_use_cache:
            # Use cached residual - skip remaining blocks
            h = apply_prev_hidden_states_residual(h)  # type: ignore
        else:
            # Compute remaining blocks and cache residual
            h, hidden_states_residual = call_remaining_blocks(
                self, transformer_options, control, transformer_patches, hs, h,
                emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator
            )
            set_buffer("hidden_states_residual", hidden_states_residual)
        
        torch._dynamo.graph_break()
        
        h = h.type(x.dtype)  # type: ignore
        
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
    
    new_forward = unet_forward_with_cache.__get__(model)
    
    @contextlib.contextmanager
    def patch_forward():
        with unittest.mock.patch.object(model, "_forward", new_forward):
            yield
    
    return patch_forward


def create_patch_flux_forward(model, *, residual_diff_threshold: float, validate_can_use_cache_function=None):
    """
    Create Flux forward patch for block-level caching.
    
    Patches Flux to cache after first double_block, then reuses
    cached residuals when first block output is similar.
    
    Based on wavespeed's create_patch_flux_forward_orig.
    """
    import contextlib
    import unittest.mock
    
    def call_remaining_blocks(self, blocks_replace, control, img, txt, vec, pe, attn_mask, ca_idx, timesteps, transformer_options):
        """Process remaining blocks and return residual."""
        original_hidden_states = img
        
        extra_block_forward_kwargs = {}
        if attn_mask is not None:
            extra_block_forward_kwargs["attn_mask"] = attn_mask
        
        # Process remaining double blocks (starting from index 1)
        for i, block in enumerate(self.double_blocks):
            if i < 1:
                continue
            
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"],
                        **extra_block_forward_kwargs
                    )
                    return out
                
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_block_forward_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_block_forward_kwargs)
            
            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add
            
            # PuLID attention (if present)
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                    ca_idx += 1
        
        # Single blocks
        img = torch.cat((txt, img), 1)
        
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], **extra_block_forward_kwargs)
                    return out
                
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": vec, "pe": pe, **extra_block_forward_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, **extra_block_forward_kwargs)
            
            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add
            
            # PuLID attention (if present)
            if getattr(self, "pulid_data", {}):
                real_img, txt_split = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                            real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                    ca_idx += 1
                img = torch.cat((txt_split, real_img), 1)
        
        img = img[:, txt.shape[1]:, ...]
        img = img.contiguous()
        
        hidden_states_residual = img - original_hidden_states
        return img, hidden_states_residual
    
    def flux_forward_with_cache(
        self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None, control=None, transformer_options={}, attn_mask=None
    ):
        """Patched Flux forward with first-block caching."""
        try:
            from comfy.ldm.flux.model import timestep_embedding  # type: ignore
        except ImportError:
            logger.error("[FBCache] Could not import Flux utilities")
            raise
        
        patches_replace = transformer_options.get("patches_replace", {})
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        # Prepare embeddings
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        ca_idx = 0
        extra_block_forward_kwargs = {}
        if attn_mask is not None:
            extra_block_forward_kwargs["attn_mask"] = attn_mask
        
        blocks_replace = patches_replace.get("dit", {})
        
        can_use_cache = False  # Initialize before loop
        
        # Process first double block
        for i, block in enumerate(self.double_blocks):
            if i >= 1:
                break
            
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"],
                        **extra_block_forward_kwargs
                    )
                    return out
                
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_block_forward_kwargs},
                    {"original_block": block_wrap, "transformer_options": transformer_options}
                )
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_block_forward_kwargs)
            
            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add
            
            # PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                    ca_idx += 1
            
            # After first block, check cache
            if i == 0:
                first_hidden_states_residual = img
                can_use_cache = get_can_use_cache(first_hidden_states_residual, threshold=residual_diff_threshold)
                
                if validate_can_use_cache_function is not None:
                    can_use_cache = validate_can_use_cache_function(can_use_cache)
                
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual", first_hidden_states_residual)
                
                del first_hidden_states_residual
        
        torch._dynamo.graph_break()
        
        if can_use_cache:
            # Use cached residual
            img = apply_prev_hidden_states_residual(img)  # type: ignore
        else:
            # Compute remaining blocks
            img, hidden_states_residual = call_remaining_blocks(
                self, blocks_replace, control, img, txt, vec, pe, attn_mask, ca_idx, timesteps, transformer_options
            )
            set_buffer("hidden_states_residual", hidden_states_residual)
        
        torch._dynamo.graph_break()
        
        img = self.final_layer(img, vec)
        return img
    
    new_forward_orig = flux_forward_with_cache.__get__(model)
    
    @contextlib.contextmanager
    def patch_forward():
        with unittest.mock.patch.object(model, "forward_orig", new_forward_orig):
            yield
    
    return patch_forward


@torch.compiler.disable()
def apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states=None):
    """Apply cached residual to current hidden states."""
    hidden_states_residual = get_buffer("hidden_states_residual")
    assert hidden_states_residual is not None, "hidden_states_residual must be set before"
    hidden_states = hidden_states_residual + hidden_states
    hidden_states = hidden_states.contiguous()
    
    if encoder_hidden_states is None:
        return hidden_states
    
    encoder_hidden_states_residual = get_buffer("encoder_hidden_states_residual")
    if encoder_hidden_states_residual is None:
        encoder_hidden_states = None
    else:
        encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states
        encoder_hidden_states = encoder_hidden_states.contiguous()
    
    return hidden_states, encoder_hidden_states


# =============================================================================
# FB Cache Application (daemon-side with block-level patching)
# =============================================================================

def try_apply_fb_cache(model: Any, fb_config: FBCacheConfig, model_type: str = "SDXL"):
    """
    Apply block-level FB cache to model (Full Wavespeed implementation).
    
    Patches the diffusion model to cache first-block outputs and reuse
    cached residuals when consecutive steps are similar.
    
    Args:
        model: The model object (RegisteredModel with .model.diffusion_model)
        fb_config: Cache configuration
        model_type: Model type from model_router ("SDXL", "SD", "FLUX", etc.)
    
    Returns:
        Context manager that applies/removes cache patching
    """
    
    @contextlib.contextmanager
    def noop_cache():
        """No-op context when caching is disabled."""
        yield None
    
    # Check if caching is enabled in config
    if not fb_config or not fb_config.enabled:
        return noop_cache()
    
    if fb_config.residual_diff_threshold <= 0.0:
        logger.debug("[FBCache] Threshold <= 0, caching disabled")
        return noop_cache()
    
    # FB cache should be applied CLIENT-SIDE via wavespeed nodes
    # The daemon's job is to run the model efficiently, not implement caching
    logger.info("[FBCache] FB cache is a CLIENT-SIDE feature - use wavespeed ApplyFBCacheOnModel node in your workflow")
    return noop_cache()
    
    # (Rest of code kept for reference but never executed)
    try:
        if hasattr(model, 'model'):
            if hasattr(model.model, 'diffusion_model'):
                diffusion_model = model.model.diffusion_model
            else:
                diffusion_model = model.model
        else:
            diffusion_model = model
    except Exception as e:
        logger.warning(f"[FBCache] Could not access diffusion_model: {e}")
        return noop_cache()
    
    # Determine model class and apply appropriate patch
    model_class_name = diffusion_model.__class__.__name__
    
    # Initialize cache context
    cache_context_obj = create_cache_context()
    set_current_cache_context(cache_context_obj)
    
    # Cache state for timestep/hit tracking
    cache_state = {
        'prev_timestep': None,
        'prev_input_state': None,
        'consecutive_cache_hits': 0,
    }
    
    @torch.compiler.disable()
    def validate_can_use_cache(can_use: bool) -> bool:
        """Validate cache usage based on consecutive hit limits."""
        if not can_use:
            cache_state['consecutive_cache_hits'] = 0
            return False
        
        # Check consecutive hit limit
        if fb_config.max_consecutive_hits >= 0:
            if cache_state['consecutive_cache_hits'] >= fb_config.max_consecutive_hits:
                cache_state['consecutive_cache_hits'] = 0
                return False
        
        cache_state['consecutive_cache_hits'] += 1
        return True
    
    # Apply model-specific patch
    patch_function = None
    
    if model_class_name in ("UNetModel", "UNet2DConditionModel"):
        # SD/SDXL models - use UNet patching
        logger.info(f"[FBCache] Applying UNet block-level cache (model={model_class_name})")
        patch_function = create_patch_unet_forward(
            diffusion_model,
            residual_diff_threshold=fb_config.residual_diff_threshold,
            validate_can_use_cache_function=validate_can_use_cache
        )
    
    elif model_class_name == "Flux":
        # Flux models - use Flux-specific patching
        logger.info(f"[FBCache] Applying Flux block-level cache (model={model_class_name})")
        patch_function = create_patch_flux_forward(
            diffusion_model,
            residual_diff_threshold=fb_config.residual_diff_threshold,
            validate_can_use_cache_function=validate_can_use_cache
        )
    
    else:
        logger.warning(f"[FBCache] Unsupported model class '{model_class_name}', caching disabled")
        return noop_cache()
    
    # Return context that applies the patch
    @contextlib.contextmanager
    def apply_patch():
        try:
            with patch_function():  # Call the function to get context manager
                logger.info(
                    f"[FBCache] Block-level cache active: threshold={fb_config.residual_diff_threshold:.3f}, "
                    f"max_hits={fb_config.max_consecutive_hits}, range={fb_config.start_percent:.0%}-{fb_config.end_percent:.0%}"
                )
                yield
        finally:
            # Clear cache buffers
            cache_context_obj.clear_buffers()
            set_current_cache_context(None)
            logger.debug("[FBCache] Block-level cache removed")
    
    return apply_patch()


def apply_fb_cache_to_model(
    model: Any,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    residual_diff_threshold: float = 0.1,
    object_to_patch: str = "diffusion_model"
) -> Any:
    """
    Apply first-block cache to a model for VRAM/speed optimization.
    
    This wraps the model's forward pass with FB cache logic, caching
    first-block outputs and reusing them when residuals are similar.
    
    Args:
        model: ModelPatcher or InferenceModeWrapper
        start_percent: Start applying cache at this timestep percent
        end_percent: Stop applying cache at this timestep percent
        residual_diff_threshold: Threshold for cache reuse
        object_to_patch: Which object to patch ("diffusion_model" or "transformer")
    
    Returns:
        The same model with FB cache applied
    """
    try:
        # Try to use wavespeed's ApplyFBCacheOnModel if available
        try:
            import sys
            # Check if wavespeed is available
            if 'wavespeed' in sys.modules or any('wavespeed' in p for p in sys.path):
                from wavespeed.fbcache_nodes import ApplyFBCacheOnModel
                node = ApplyFBCacheOnModel()
                result = node.apply(
                    model=model,
                    start_percent=start_percent,
                    end_percent=end_percent,
                    residual_diff_threshold=residual_diff_threshold,
                    max_consecutive_cache_hits=-1,  # Unlimited
                    return_cached_output_when_exceeding_max_hits=True,
                    object_to_patch=object_to_patch
                )
                return result[0]
        except ImportError:
            pass
        
        # Fallback: Apply manually using our integrated code
        # The model's unet function wrapper approach
        if hasattr(model, 'set_model_unet_function_wrapper'):
            config = FBCacheConfig(
                enabled=True,
                start_percent=start_percent,
                end_percent=end_percent,
                residual_diff_threshold=residual_diff_threshold,
                max_consecutive_hits=-1,
                object_to_patch=object_to_patch
            )
            # Store config on model for use during inference
            model._fb_cache_config = config
            logger.info(f"[FBCache] Config stored on model (will apply during inference)")
        
        return model
        
    except Exception as e:
        logger.warning(f"[FBCache] Could not apply FB cache: {e}")
        return model


@contextlib.contextmanager
def apply_fb_cache_transient(model: Any, fb_config: Optional[FBCacheConfig], model_type: str = "SDXL"):
    """
    Context manager for transient block-level FB cache application.
    
    Full wavespeed implementation - caches first-block residuals and reuses
    them when consecutive diffusion steps are similar, providing ~2x speedup.
    
    Args:
        model: The model (RegisteredModel with .model.diffusion_model)
        fb_config: Cache configuration (None to disable)
        model_type: Model type from model_router (for logging)
    
    Yields:
        None
    """
    if fb_config is None:
        yield
        return
    
    cache_ctx = None
    
    try:
        cache_ctx = try_apply_fb_cache(model, fb_config, model_type)
        with cache_ctx:
            yield
    
    except Exception as e:
        logger.error(f"[FBCache] Error during cache application: {e}")
        import traceback
        logger.error(f"[FBCache] Traceback:\n{traceback.format_exc()}")
        yield
    
    finally:
        # Cleanup handled by try_apply_fb_cache's context manager
        pass
