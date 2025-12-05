"""Type stubs for ComfyUI nodes module."""

from typing import Any, Dict, List, Tuple

MAX_RESOLUTION: int

class SaveImage:
    RETURN_TYPES: Tuple[str, ...]
    FUNCTION: str
    OUTPUT_NODE: bool
    CATEGORY: str
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]: ...
    
    def save_images(self, images: Any, **kwargs: Any) -> Dict[str, Any]: ...
