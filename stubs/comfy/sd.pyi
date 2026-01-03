"""Type stubs for ComfyUI comfy.sd module."""

from typing import Any, Optional, Tuple, List
import torch

def load_checkpoint_guess_config(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    embedding_directory: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[Any, ...]: ...

def load_unet(unet_path: str, dtype: Optional[torch.dtype] = None) -> Any: ...

def load_diffusion_model_state_dict(state_dict: dict[str, Any], model_options: Optional[dict[str, Any]] = None) -> Any: ...
