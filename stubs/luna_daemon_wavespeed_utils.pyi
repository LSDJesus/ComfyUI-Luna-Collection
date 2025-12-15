"""Type stubs for luna_daemon.wavespeed_utils module."""

from typing import Any, Optional, ContextManager
from dataclasses import dataclass

@dataclass
class FBCacheConfig:
    enabled: bool = True
    cache_size: Optional[int] = None
    
def apply_fb_cache_transient(model: Any, config: FBCacheConfig) -> ContextManager[Any]: ...

