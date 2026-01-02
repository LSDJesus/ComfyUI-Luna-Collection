"""
Luna Refinement Engine - Batched Tile Refinement

This module provides the core batched inference logic shared by:
- LunaChessRefiner (uniform grid tiles)
- LunaSemanticDetailer (detection-based crops)

Key design principles:
1. BATCHED processing - encode/sample/decode entire batches, not per-tile
2. GPU-efficient - minimize kernel launches, maximize parallelism
3. Node-agnostic - doesn't care about tile layout, just processes batches

The calling node handles:
- Tile extraction (grid-based or detection-based)
- Compositing strategy (overlap blending, mask-based)
- Chess pattern scheduling (if applicable)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# ComfyUI imports
try:
    import comfy.samplers  # type: ignore
    import comfy.sample  # type: ignore
    import comfy.utils  # type: ignore
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RefinementBatch:
    """
    A batch of tiles/crops to be refined together.
    
    All tensors have batch dimension N as first axis.
    """
    # Required
    pixels: torch.Tensor          # [N, H, W, C] - pixel tiles (BHWC format)
    
    # Optional - if not provided, fresh noise is generated
    noise: Optional[torch.Tensor] = None  # [N, C_lat, H_lat, W_lat] - scaffold noise
    
    # Optional - per-tile masks for partial refinement
    masks: Optional[torch.Tensor] = None  # [N, H_lat, W_lat] - latent-space masks
    
    # Optional - for IP-Adapter structural anchoring
    vision_embeds: Optional[torch.Tensor] = None  # [N, seq_len, dim]
    uncond_embeds: Optional[torch.Tensor] = None  # [N, seq_len, dim]
    
    # Metadata
    batch_size: int = 0
    
    def __post_init__(self):
        self.batch_size = self.pixels.shape[0]


@dataclass 
class RefinementConfig:
    """Configuration for the refinement process."""
    steps: int = 20
    cfg: float = 7.0
    denoise: float = 0.4
    sampler_name: str = "euler"
    scheduler: str = "normal"
    seed: int = 0
    
    # IP-Adapter
    ip_adapter_weight: float = 0.5
    
    # Mask handling
    mask_noise_factor: float = 0.5  # Reduce noise in already-refined (masked) areas


@dataclass
class RefinementResult:
    """Result of batch refinement."""
    pixels: torch.Tensor  # [N, H, W, C] - refined pixel tiles
    latents: Optional[torch.Tensor] = None  # [N, C, H, W] - refined latents (if needed)


# =============================================================================
# Refinement Engine
# =============================================================================

class RefinementEngine:
    """
    Batched tile refinement engine.
    
    Handles the encode → sample → decode pipeline for arbitrary batches.
    The calling node is responsible for:
    - Extracting tiles from the source image
    - Providing appropriate conditioning
    - Compositing refined tiles back
    """
    
    def __init__(
        self,
        model: Any,
        vae: Any,
        positive: List,
        negative: List,
        config: RefinementConfig,
        ip_adapter: Optional[Any] = None,
    ):
        """
        Initialize the refinement engine.
        
        Args:
            model: ComfyUI model wrapper
            vae: ComfyUI VAE wrapper
            positive: Positive conditioning
            negative: Negative conditioning
            config: Refinement configuration
            ip_adapter: Optional IP-Adapter for structural anchoring
        """
        self.model = model
        self.vae = vae
        self.positive = positive
        self.negative = negative
        self.config = config
        self.ip_adapter = ip_adapter
        
        # Get device from model
        self.device = comfy.model_management.get_torch_device() if HAS_COMFY else torch.device("cuda")
    
    def refine_batch(self, batch: RefinementBatch) -> RefinementResult:
        """
        Refine a batch of tiles.
        
        This is the main entry point. It handles:
        1. VAE encoding of pixel batch
        2. Noise preparation (scaffold or fresh)
        3. Conditioning preparation (with optional IP-Adapter)
        4. KSampler inference
        5. VAE decoding back to pixels
        
        Args:
            batch: RefinementBatch containing tiles and optional extras
        
        Returns:
            RefinementResult with refined pixels
        """
        n = batch.batch_size
        cfg = self.config
        
        # === STEP 1: Encode pixels to latent ===
        # ComfyUI VAE expects BHWC format
        latents = self.vae.encode(batch.pixels)
        
        # === STEP 2: Prepare noise ===
        if batch.noise is not None:
            noise = batch.noise
        else:
            # Generate fresh noise
            noise = torch.randn_like(latents)
        
        # Apply mask-based noise reduction if masks provided
        if batch.masks is not None:
            # Masks indicate already-refined areas (1 = refined, reduce noise there)
            # Expand mask to latent channels
            mask_4d = batch.masks.unsqueeze(1)  # [N, 1, H, W]
            
            # Reduce noise in masked areas
            noise = noise * (1.0 - mask_4d * (1.0 - cfg.mask_noise_factor))
        
        # === STEP 3: Prepare conditioning ===
        positive = self._prepare_conditioning(self.positive, n)
        negative = self._prepare_conditioning(self.negative, n)
        
        # === STEP 4: Apply IP-Adapter if available ===
        work_model = self.model
        if self.ip_adapter is not None and batch.vision_embeds is not None:
            work_model = self._apply_ip_adapter(
                batch.vision_embeds,
                batch.uncond_embeds,
                n
            )
        
        # === STEP 5: Run KSampler ===
        # Prepare latent dict for ComfyUI
        latent_dict = {"samples": latents}
        if batch.masks is not None:
            latent_dict["noise_mask"] = batch.masks
        
        samples = self._sample(
            work_model,
            positive,
            negative,
            latent_dict,
            noise
        )
        
        # === STEP 6: Decode back to pixels ===
        refined_pixels = self.vae.decode(samples)
        
        return RefinementResult(
            pixels=refined_pixels,
            latents=samples
        )
    
    def _prepare_conditioning(self, cond: List, batch_size: int) -> List:
        """
        Replicate conditioning for batch processing.
        
        Args:
            cond: ComfyUI conditioning format - list of [tensor, dict]
            batch_size: Number of tiles in batch
        
        Returns:
            Batched conditioning
        """
        if batch_size == 1:
            return cond
        
        replicated = []
        for c in cond:
            tensor, opts = c
            # Repeat tensor along batch dimension
            batched_tensor = tensor.repeat(batch_size, 1, 1)
            replicated.append([batched_tensor, opts.copy()])
        
        return replicated
    
    def _apply_ip_adapter(
        self,
        vision_embeds: torch.Tensor,
        uncond_embeds: Optional[torch.Tensor],
        batch_size: int
    ) -> Any:
        """
        Apply IP-Adapter to model with batched vision embeddings.
        
        KEY INSIGHT: PyTorch attention maps Latent[i] → Embed[i] when batch dims match.
        This enables TRUE BATCHING - N tiles get N distinct vision anchors in ONE pass.
        
        Args:
            vision_embeds: [N, seq_len, dim] vision embeddings
            uncond_embeds: [N, seq_len, dim] unconditional embeddings (optional)
            batch_size: Number of tiles
        
        Returns:
            Patched model clone
        """
        if self.ip_adapter is None:
            return self.model
        
        try:
            # IPAdapterUnifiedLoader format
            if isinstance(self.ip_adapter, dict) and "ipadapter" in self.ip_adapter:
                ipadapter_model = self.ip_adapter["ipadapter"]["model"]
            else:
                ipadapter_model = self.ip_adapter
            
            # Clone model to avoid modifying original
            work_model = self.model.clone()
            
            # Prepare uncond if not provided
            if uncond_embeds is None:
                uncond_embeds = torch.zeros_like(vision_embeds)
            
            # Create conditioning dict
            cond = {
                "c_crossattn": vision_embeds,
                "uncond": uncond_embeds,
            }
            
            # Apply IP-Adapter patch
            ipadapter_model.set_model_ipadapter_patch(
                work_model.model,
                cond,
                weight=self.config.ip_adapter_weight
            )
            
            return work_model
            
        except Exception as e:
            print(f"[RefinementEngine] IP-Adapter application failed: {e}")
            return self.model
    
    def _sample(
        self,
        model: Any,
        positive: List,
        negative: List,
        latent: Dict,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Run KSampler on latent batch.
        
        Args:
            model: Model to use (possibly with IP-Adapter patch)
            positive: Positive conditioning
            negative: Negative conditioning
            latent: Latent dict with samples and optional noise_mask
            noise: Scaffold noise tensor
        
        Returns:
            Refined latent samples
        """
        cfg = self.config
        
        # Use ComfyUI's common_ksampler
        samples = comfy.sample.sample(
            model,
            noise,
            cfg.steps,
            cfg.cfg,
            cfg.sampler_name,
            cfg.scheduler,
            positive,
            negative,
            latent["samples"],
            denoise=cfg.denoise,
            seed=cfg.seed
        )
        
        return samples


# =============================================================================
# Convenience Functions
# =============================================================================

def create_refinement_batch(
    pixels: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    vision_embeds: Optional[torch.Tensor] = None,
    uncond_embeds: Optional[torch.Tensor] = None
) -> RefinementBatch:
    """
    Create a RefinementBatch from tensors.
    
    Args:
        pixels: [N, H, W, C] pixel tiles
        noise: Optional [N, C, H, W] scaffold noise
        masks: Optional [N, H, W] latent masks
        vision_embeds: Optional [N, seq_len, dim] vision embeddings
        uncond_embeds: Optional [N, seq_len, dim] uncond embeddings
    
    Returns:
        RefinementBatch ready for processing
    """
    return RefinementBatch(
        pixels=pixels,
        noise=noise,
        masks=masks,
        vision_embeds=vision_embeds,
        uncond_embeds=uncond_embeds
    )


def quick_refine(
    model: Any,
    vae: Any,
    positive: List,
    negative: List,
    pixels: torch.Tensor,
    steps: int = 20,
    cfg: float = 7.0,
    denoise: float = 0.4,
    seed: int = 0,
    sampler_name: str = "euler",
    scheduler: str = "normal"
) -> torch.Tensor:
    """
    Quick refinement for simple use cases.
    
    Args:
        model, vae, positive, negative: ComfyUI components
        pixels: [N, H, W, C] pixel batch
        steps, cfg, denoise, seed, sampler_name, scheduler: Sampling params
    
    Returns:
        Refined pixels [N, H, W, C]
    """
    config = RefinementConfig(
        steps=steps,
        cfg=cfg,
        denoise=denoise,
        seed=seed,
        sampler_name=sampler_name,
        scheduler=scheduler
    )
    
    engine = RefinementEngine(model, vae, positive, negative, config)
    batch = create_refinement_batch(pixels)
    result = engine.refine_batch(batch)
    
    return result.pixels


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "RefinementBatch",
    "RefinementConfig", 
    "RefinementResult",
    
    # Engine
    "RefinementEngine",
    
    # Convenience
    "create_refinement_batch",
    "quick_refine",
]
