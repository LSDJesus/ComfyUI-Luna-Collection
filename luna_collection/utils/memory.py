"""
Memory Management Utilities

Tools for monitoring and managing memory usage in Luna Collection nodes.
"""

import psutil
import os
import gc
from typing import Dict, Any, Optional

class MemoryManager:
    """Manage memory usage and provide monitoring capabilities."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_current_memory()

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_delta(self) -> float:
        """Get memory usage delta from initialization."""
        return self.get_current_memory() - self.initial_memory

    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_info = self.process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'delta': self.get_memory_delta(),
            'percent': self.process.memory_percent()
        }

    def is_memory_low(self, threshold_mb: float = 1000) -> bool:
        """Check if memory usage is above threshold."""
        return self.get_current_memory() > threshold_mb