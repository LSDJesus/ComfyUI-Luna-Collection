"""Type stubs for ComfyUI comfy.sd module."""

from typing import Any, Optional, Tuple, List

def load_checkpoint_guess_config(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    embedding_directory: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[Any, Any, Any, ...]: ...

def load_unet(unet_path: str) -> Any: ...
