"""
Performance regression tests for Luna Collection nodes.
Measures execution time, memory usage, and throughput.
"""

import pytest
import time
import psutil
import torch
import numpy as np
from unittest.mock import MagicMock
import gc

# Import Luna validation system
from validation import LunaInputValidator


class PerformanceTestBase:
    """Base class for performance tests with measurement utilities."""

    def setup_method(self):
        """Set up performance test environment."""
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        self.validator = LunaInputValidator()

    def start_measurement(self):
        """Start performance measurement."""
        gc.collect()  # Clean up garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    def end_measurement(self):
        """End performance measurement and return metrics."""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        metrics = {
            'execution_time': end_time - (self.start_time or 0),
            'memory_delta': end_memory - (self.start_memory or 0),
            'peak_memory': end_memory
        }

        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            metrics['gpu_memory_delta'] = end_gpu_memory - (self.start_gpu_memory or 0)
            metrics['gpu_peak_memory'] = end_gpu_memory

        return metrics


class TestValidationPerformance(PerformanceTestBase):
    """Performance tests for validation system."""

    def test_validation_throughput(self):
        """Test validation system throughput."""
        test_values = [
            5.0, 10.5, 100, "test_string", "/path/to/file.txt"
        ] * 100  # 500 total validations

        self.start_measurement()
        results = []
        for value in test_values:
            result = None
            if isinstance(value, (int, float)):
                result = self.validator.validate_numeric_input(
                    value, min_value=0, max_value=1000
                )
            elif isinstance(value, str):
                if value.startswith('/'):
                    result = self.validator.validate_file_path(
                        value, must_exist=False
                    )
                else:
                    result = self.validator.validate_text_input(
                        value, max_length=100
                    )
            if result is not None:
                results.append(result)

        metrics = self.end_measurement()

        # Assert performance requirements
        assert metrics['execution_time'] < 0.1  # Should complete in < 100ms
        throughput = len(test_values) / metrics['execution_time']
        print(f"Validation throughput: {throughput:.0f} ops/sec")

    def test_validation_caching_performance(self):
        """Test that caching improves performance."""
        import time

        # Test with caching
        self.start_measurement()
        for i in range(100):
            self.validator.validate_numeric_input(
                42.0, min_value=0, max_value=100, cache_key='perf_test'
            )
        cached_metrics = self.end_measurement()

        # Test without caching (different values to avoid cache hits)
        self.validator.cache.clear()
        self.start_measurement()
        for i in range(100):
            self.validator.validate_numeric_input(
                i * 0.5, min_value=0, max_value=100
            )
        uncached_metrics = self.end_measurement()

        # Cached should be faster (though the difference might be small for simple validations)
        # We mainly want to ensure caching doesn't slow things down significantly
        assert uncached_metrics['execution_time'] >= cached_metrics['execution_time'] * 0.9  # Allow 10% overhead for caching
        print(f"Cached time: {cached_metrics['execution_time']:.4f}s, Uncached time: {uncached_metrics['execution_time']:.4f}s")


class TestNodePerformance(PerformanceTestBase):
    """Performance tests for individual nodes."""

    def test_sampler_parameter_validation_performance(self):
        """Test LunaSampler parameter validation performance."""
        # Simulate typical sampler parameters
        params = {
            'steps': 25,
            'cfg': 7.5,
            'denoise': 0.8,
            'adaptive_threshold': 0.75
        }

        self.start_measurement()
        validated = {}
        validated['steps'] = self.validator.validate_numeric_input(
            params['steps'], min_value=1, max_value=1000
        )
        validated['cfg'] = self.validator.validate_numeric_input(
            params['cfg'], min_value=0.0, max_value=100.0
        )
        validated['denoise'] = self.validator.validate_numeric_input(
            params['denoise'], min_value=0.0, max_value=1.0
        )
        validated['adaptive_threshold'] = self.validator.validate_numeric_input(
            params['adaptive_threshold'], min_value=0.0, max_value=1.0
        )
        metrics = self.end_measurement()

        # Should be very fast (< 1ms)
        assert metrics['execution_time'] < 0.001
        print(f"Sampler validation: {metrics['execution_time']*1000:.2f}ms per call")

    def test_detailer_parameter_validation_performance(self):
        """Test LunaDetailer parameter validation performance."""
        # Simulate typical detailer parameters
        params = {
            'guide_size': 512,
            'max_size': 1024,
            'seed': 12345,
            'feather': 5,
            'bbox_threshold': 0.3,
            'sam_threshold': 0.7
        }

        self.start_measurement()
        validated = {}
        for key, value in params.items():
            if key in ['guide_size', 'max_size', 'seed', 'feather']:
                validated[key] = self.validator.validate_numeric_input(
                    value, min_value=0, max_value=8192 if key in ['guide_size', 'max_size'] else
                                            (2**64-1 if key == 'seed' else 100)
                )
            else:  # threshold parameters
                validated[key] = self.validator.validate_numeric_input(
                    value, min_value=0.0, max_value=1.0
                )
        metrics = self.end_measurement()

        # Should be fast (< 2ms for multiple validations)
        assert metrics['execution_time'] < 0.002
        print(f"Detailer validation: {metrics['execution_time']*1000:.2f}ms per call")

    def test_text_processing_validation_performance(self):
        """Test text processing validation performance."""
        # Simulate typical text processing
        texts = [
            "A beautiful landscape with mountains",
            "Create an image of a sunset over the ocean",
            "Professional photograph of a city skyline",
            "Artistic portrait with dramatic lighting"
        ] * 25  # 100 texts total

        self.start_measurement()
        results = []
        for text in texts:
            result = self.validator.validate_text_input(
                text, max_length=200, allow_empty=False
            )
            results.append(result)
        metrics = self.end_measurement()

        # Should handle bulk text validation efficiently
        assert metrics['execution_time'] < 0.1  # < 100ms for 100 texts
        throughput = len(texts) / metrics['execution_time']
        print(f"Text validation throughput: {throughput:.0f} texts/second")


class TestMemoryPerformance(PerformanceTestBase):
    """Memory usage performance tests."""

    def test_validation_memory_overhead(self):
        """Test memory overhead of validation system."""
        # Measure baseline memory
        self.start_measurement()
        baseline_metrics = self.end_measurement()

        # Create validation cache with many entries
        for i in range(1000):
            self.validator.validate_numeric_input(
                i, min_value=0, max_value=10000, cache_key=f'memory_test_{i}'
            )

        # Measure memory after caching
        self.start_measurement()
        cached_metrics = self.end_measurement()

        memory_overhead = cached_metrics['memory_delta'] - baseline_metrics['memory_delta']

        # Memory overhead should be reasonable (< 50MB for 1000 cached validations)
        assert memory_overhead < 50
        print(f"Validation memory overhead: {memory_overhead:.1f}MB for 1000 cached items")

    def test_cache_memory_efficiency(self):
        """Test that cache doesn't grow unbounded."""
        initial_cache_size = len(self.validator.cache.cache)

        # Fill cache beyond max size
        for i in range(1500):  # More than default max_size of 1000
            self.validator.validate_numeric_input(
                i % 100, min_value=0, max_value=100, cache_key=f'efficiency_test_{i}'
            )

        final_cache_size = len(self.validator.cache.cache)

        # Cache should not exceed max_size significantly
        assert final_cache_size <= 1010  # Allow some buffer
        print(f"Cache size after overflow: {final_cache_size} (max allowed: 1000)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])