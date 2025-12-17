"""
Luna LoRA RAM Cache

Simple RAM-based cache for LoRA state dictionaries.
Stores loaded LoRAs in system RAM for fast retrieval across
multiple ComfyUI instances, avoiding redundant disk I/O.

Architecture:
- LoRAs are stored as CPU tensors in a dictionary
- No GPU operations - just RAM storage
- Clients request cached LoRAs via socket, apply them locally
- LRU eviction when cache exceeds max size

Performance:
- Disk read: ~100-500 MB/s
- PCIe transfer: ~32,000 MB/s
- Typical LoRA (100MB): 200ms from disk, ~3ms from cache
"""

import os
import hashlib
import threading
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict

import torch

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    # Define a dummy object to avoid Pylance errors
    class DummySafetensors:
        class torch:
            @staticmethod
            def load_file(*args, **kwargs):
                raise ImportError("safetensors not installed")
    safetensors = DummySafetensors()

# Try relative import first, fallback to direct
try:
    from .core import logger, LoRACacheEntry, LoRANotFoundError
except (ImportError, ValueError):
    # Fallback: load core.py directly
    import importlib.util
    import logging
    
    core_path = os.path.join(os.path.dirname(__file__), "core.py")
    spec = importlib.util.spec_from_file_location("luna_daemon_core", core_path)
    if spec and spec.loader:
        core_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_mod)
        logger = core_mod.logger
        LoRACacheEntry = core_mod.LoRACacheEntry
        LoRANotFoundError = core_mod.LoRANotFoundError
    else:
        logger = logging.getLogger("LunaLoRACache")
        raise RuntimeError("Could not load core.py - LoRA cache initialization failed")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MAX_CACHE_SIZE_GB = 8.0  # Maximum RAM to use for LoRA cache
DEFAULT_MAX_ENTRIES = 100  # Maximum number of LoRAs to cache


# =============================================================================
# LoRA Cache Implementation
# =============================================================================

class LoRACache:
    """
    RAM-based cache for LoRA state dictionaries.
    
    Thread-safe LRU cache that stores LoRA weights in system RAM.
    Provides fast retrieval for frequently-used LoRAs.
    """
    
    def __init__(
        self,
        max_size_gb: float = DEFAULT_MAX_CACHE_SIZE_GB,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        lora_base_path: Optional[str] = None
    ):
        """
        Initialize the LoRA cache.
        
        Args:
            max_size_gb: Maximum cache size in gigabytes
            max_entries: Maximum number of LoRAs to cache
            lora_base_path: Base path for LoRA files (for resolving relative paths)
        """
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.max_entries = max_entries
        self.lora_base_path = lora_base_path
        
        # LRU cache: OrderedDict maintains insertion order
        self._cache: Any = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        logger.info(f"[LoRACache] Initialized: max {max_size_gb:.1f}GB, {max_entries} entries")
    
    def _compute_hash(self, lora_name: str) -> str:
        """Compute a hash for the LoRA name (used as cache key)."""
        return hashlib.md5(lora_name.encode()).hexdigest()[:12]
    
    def _get_full_path(self, lora_name: str) -> str:
        """Resolve full path for a LoRA file."""
        if os.path.isabs(lora_name):
            return lora_name
        if self.lora_base_path:
            return os.path.join(self.lora_base_path, lora_name)
        # Try to use folder_paths if available
        try:
            import folder_paths  # type: ignore
            return folder_paths.get_full_path("loras", lora_name)
        except:
            return lora_name
    
    def _estimate_size(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Estimate memory size of a state dict in bytes."""
        total = 0
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
        return total
    
    def _evict_lru(self, needed_bytes: int = 0) -> int:
        """
        Evict least-recently-used entries to free space.
        
        Args:
            needed_bytes: Minimum bytes to free
            
        Returns:
            Number of entries evicted
        """
        evicted = 0
        while self._cache and (
            self._current_size_bytes + needed_bytes > self.max_size_bytes or
            len(self._cache) >= self.max_entries
        ):
            # Pop oldest (first) item
            name, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= int(entry.size_mb * 1024 * 1024)
            evicted += 1
            logger.debug(f"[LoRACache] Evicted: {name} ({entry.size_mb:.1f}MB)")
        return evicted
    
    def get(self, lora_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a LoRA state dict from cache.
        
        Args:
            lora_name: Name/path of the LoRA
            
        Returns:
            State dict if cached, None otherwise
        """
        with self._lock:
            lora_hash = self._compute_hash(lora_name)
            
            if lora_hash in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(lora_hash)
                entry.access_count += 1
                self._cache[lora_hash] = entry
                self._hits += 1
                logger.debug(f"[LoRACache] HIT: {lora_name}")
                return entry.state_dict
            
            self._misses += 1
            return None
    
    def put(
        self, 
        lora_name: str, 
        state_dict: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Add a LoRA state dict to cache.
        
        Args:
            lora_name: Name/path of the LoRA
            state_dict: The LoRA weights (should be on CPU)
            
        Returns:
            True if cached successfully
        """
        with self._lock:
            lora_hash = self._compute_hash(lora_name)
            
            # Already cached?
            if lora_hash in self._cache:
                return True
            
            # Ensure tensors are on CPU
            cpu_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    cpu_dict[k] = v.cpu() if v.device.type != 'cpu' else v
                else:
                    cpu_dict[k] = v
            
            size_bytes = self._estimate_size(cpu_dict)
            size_mb = size_bytes / (1024 * 1024)
            
            # Evict if needed
            self._evict_lru(size_bytes)
            
            # Store
            entry = LoRACacheEntry(
                name=lora_name,
                state_dict=cpu_dict,
                size_mb=size_mb,
                hash=lora_hash,
                access_count=1
            )
            self._cache[lora_hash] = entry
            self._current_size_bytes += size_bytes
            
            logger.info(f"[LoRACache] Cached: {lora_name} ({size_mb:.1f}MB)")
            return True
    
    def load_and_cache(self, lora_name: str) -> Dict[str, torch.Tensor]:
        """
        Load a LoRA from disk and cache it.
        
        Args:
            lora_name: Name/path of the LoRA
            
        Returns:
            The LoRA state dict
            
        Raises:
            LoRANotFoundError: If file doesn't exist
        """
        # Check cache first
        cached = self.get(lora_name)
        if cached is not None:
            return cached
        
        # Load from disk
        full_path = self._get_full_path(lora_name)
        
        if not os.path.exists(full_path):
            raise LoRANotFoundError(f"LoRA not found: {full_path}")
        
        logger.info(f"[LoRACache] Loading from disk: {lora_name}")
        
        if HAS_SAFETENSORS and full_path.endswith('.safetensors'):
            state_dict = safetensors.torch.load_file(full_path, device='cpu')
        else:
            state_dict = torch.load(full_path, map_location='cpu', weights_only=True)
        
        # Cache it
        self.put(lora_name, state_dict)
        
        return state_dict
    
    def contains(self, lora_name: str) -> bool:
        """Check if a LoRA is in cache."""
        with self._lock:
            lora_hash = self._compute_hash(lora_name)
            return lora_hash in self._cache
    
    def remove(self, lora_name: str) -> bool:
        """Remove a LoRA from cache."""
        with self._lock:
            lora_hash = self._compute_hash(lora_name)
            if lora_hash in self._cache:
                entry = self._cache.pop(lora_hash)
                self._current_size_bytes -= int(entry.size_mb * 1024 * 1024)
                logger.info(f"[LoRACache] Removed: {lora_name}")
                return True
            return False
    
    def clear(self) -> int:
        """Clear the entire cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info(f"[LoRACache] Cleared {count} entries")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = (self._hits / total_accesses * 100) if total_accesses > 0 else 0
            
            return {
                "entries": len(self._cache),
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "cached_loras": [e.name for e in self._cache.values()]
            }
    
    def list_cached(self) -> List[Tuple[str, float, int]]:
        """
        List all cached LoRAs.
        
        Returns:
            List of (name, size_mb, access_count) tuples
        """
        with self._lock:
            return [
                (e.name, e.size_mb, e.access_count) 
                for e in self._cache.values()
            ]


# =============================================================================
# Global Instance
# =============================================================================

_global_cache: Optional[LoRACache] = None


def get_lora_cache() -> LoRACache:
    """Get or create the global LoRA cache instance."""
    global _global_cache
    if _global_cache is None:
        # Try to get lora path from folder_paths
        lora_path = None
        try:
            import folder_paths  # type: ignore
            lora_paths = folder_paths.get_folder_paths("loras")
            if lora_paths:
                lora_path = lora_paths[0]
        except:
            pass
        
        _global_cache = LoRACache(lora_base_path=lora_path)
    return _global_cache


def init_lora_cache(
    max_size_gb: float = DEFAULT_MAX_CACHE_SIZE_GB,
    lora_base_path: Optional[str] = None
) -> LoRACache:
    """Initialize the global LoRA cache with custom settings."""
    global _global_cache
    _global_cache = LoRACache(max_size_gb=max_size_gb, lora_base_path=lora_base_path)
    return _global_cache
