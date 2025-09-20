"""
Luna Collection Utilities

Utility functions and classes for performance monitoring, memory management, and configuration.
"""

import time
import psutil
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager

class PerformanceMonitor:
    """Monitor performance metrics for operations."""

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            self.metrics[operation_name] = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'start_time': start_time,
                'end_time': end_time
            }

    def get_metrics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific operation."""
        return self.metrics.get(operation_name)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()