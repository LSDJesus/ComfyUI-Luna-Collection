"""Type stubs for ComfyUI comfy.sample module."""

from typing import Any, Optional, Callable
import torch

def sample(
    model: Any,
    noise: torch.Tensor,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: torch.Tensor,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step: Optional[int] = None,
    last_step: Optional[int] = None,
    force_full_denoise: bool = False,
    noise_mask: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    callback: Optional[Callable] = None,
    disable_pbar: bool = False,
    seed: Optional[int] = None
) -> dict[str, torch.Tensor]: ...
