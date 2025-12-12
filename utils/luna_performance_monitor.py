import torch
import psutil
import time
from typing import Dict, Any
import gc

class LunaPerformanceMonitor:
    """
    Shared performance monitoring utility for all Luna Collection nodes
    """

    def __init__(self):
        self.start_time = None
        self.initial_vram = 0
        self.node_name = "Unknown"

    def start_monitoring(self, node_name: str = "Unknown"):
        """Start performance monitoring"""
        self.node_name = node_name
        self.start_time = time.time()
        self.initial_vram = self._get_vram_usage()

        # Force garbage collection before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop_monitoring(self, **kwargs) -> Dict[str, Any]:
        """Stop monitoring and collect performance statistics"""
        if self.start_time is None:
            return {"error": "Monitoring not started"}

        end_time = time.time()
        final_vram = self._get_vram_usage()

        stats = {
            "node_name": self.node_name,
            "processing_time": end_time - self.start_time,
            "vram_usage_mb": final_vram,
            "vram_delta_mb": final_vram - self.initial_vram,
            "timestamp": end_time,
            "system_memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        # Add node-specific metrics
        stats.update(kwargs)

        return stats

    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def get_vram_total(self) -> float:
        """Get total VRAM in MB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return 0

    def optimize_memory(self):
        """Apply memory optimization techniques"""
        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance for shared use
LUNA_PERFORMANCE_MONITOR = LunaPerformanceMonitor()