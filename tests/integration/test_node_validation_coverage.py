"""
Integration tests for validation coverage across multiple Luna nodes.
Tests validation on newly validated nodes to ensure comprehensive coverage.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import validation system
from validation import luna_validator


class TestNodeValidationCoverage:
    """Test validation coverage across multiple Luna nodes."""

    def test_luna_sampler_validation(self):
        """Test validation on LunaSampler parameters."""
        # Test valid parameters
        try:
            # This would normally be called through ComfyUI, but we can test the validation logic
            # by directly calling the validation methods
            result = luna_validator.validate_numeric_input(25, min_value=1, max_value=1000)
            assert result == 25

            result = luna_validator.validate_numeric_input(7.0, min_value=0.0, max_value=100.0)
            assert result == 7.0

            result = luna_validator.validate_numeric_input(1.0, min_value=0.0, max_value=1.0)
            assert result == 1.0

            result = luna_validator.validate_numeric_input(0.8, min_value=0.0, max_value=1.0)
            assert result == 0.8

        except Exception as e:
            pytest.fail(f"LunaSampler validation failed: {e}")

    def test_luna_sampler_invalid_parameters(self):
        """Test LunaSampler with invalid parameters."""
        # Test invalid steps
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(0, min_value=1, max_value=1000)

        # Test invalid cfg
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(150.0, min_value=0.0, max_value=100.0)

        # Test invalid denoise
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(1.5, min_value=0.0, max_value=1.0)

        # Test invalid adaptive_threshold
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(-0.1, min_value=0.0, max_value=1.0)

    def test_upscaler_validation(self):
        """Test validation on upscaling nodes."""
        # Test valid scale_by
        result = luna_validator.validate_numeric_input(2.0, min_value=0.0, max_value=8.0)
        assert result == 2.0

        # Test boundary values
        result = luna_validator.validate_numeric_input(0.0, min_value=0.0, max_value=8.0)
        assert result == 0.0

        result = luna_validator.validate_numeric_input(8.0, min_value=0.0, max_value=8.0)
        assert result == 8.0

    def test_upscaler_invalid_parameters(self):
        """Test upscaler with invalid parameters."""
        # Test invalid scale_by (too high)
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(10.0, min_value=0.0, max_value=8.0)

        # Test invalid scale_by (negative)
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(-1.0, min_value=0.0, max_value=8.0)

    def test_checkpoint_loader_validation(self):
        """Test validation on checkpoint loader."""
        # Test valid checkpoint name
        result = luna_validator.validate_text_input("model_v1.safetensors", max_length=255)
        assert result == "model_v1.safetensors"

        # Test valid shorter name
        result = luna_validator.validate_text_input("test", max_length=255)
        assert result == "test"

    def test_checkpoint_loader_invalid_parameters(self):
        """Test checkpoint loader with invalid parameters."""
        # Test checkpoint name too long
        long_name = "a" * 300
        with pytest.raises(ValueError, match="Text too long"):
            luna_validator.validate_text_input(long_name, max_length=255)

    def test_detailer_validation(self):
        """Test validation on detailer nodes."""
        # Test valid guide_size
        result = luna_validator.validate_numeric_input(512, min_value=64, max_value=8192)  # Using reasonable MAX_RESOLUTION
        assert result == 512

        # Test valid max_size
        result = luna_validator.validate_numeric_input(1024, min_value=64, max_value=8192)
        assert result == 1024

        # Test valid seed
        result = luna_validator.validate_numeric_input(12345, min_value=0, max_value=2**64-1)
        assert result == 12345

        # Test valid feather
        result = luna_validator.validate_numeric_input(5, min_value=0, max_value=100)
        assert result == 5

        # Test valid thresholds
        result = luna_validator.validate_numeric_input(0.5, min_value=0.0, max_value=1.0)
        assert result == 0.5

    def test_detailer_invalid_parameters(self):
        """Test detailer with invalid parameters."""
        # Test invalid guide_size (too small)
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(32, min_value=64, max_value=8192)

        # Test invalid max_size (too large)
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(10000, min_value=64, max_value=8192)

        # Test invalid seed (negative)
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(-1, min_value=0, max_value=2**64-1)

        # Test invalid feather (too large)
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(150, min_value=0, max_value=100)

        # Test invalid threshold (above 1.0)
        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(1.5, min_value=0.0, max_value=1.0)

        # Test invalid threshold (below 0.0)
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(-0.1, min_value=0.0, max_value=1.0)

    def test_performance_logger_validation(self):
        """Test validation on performance logger."""
        # Test valid log filename
        result = luna_validator.validate_text_input("performance_log.json", max_length=255)
        assert result == "performance_log.json"

        # Test valid shorter filename
        result = luna_validator.validate_text_input("log.txt", max_length=255)
        assert result == "log.txt"

    def test_performance_logger_invalid_parameters(self):
        """Test performance logger with invalid parameters."""
        # Test filename too long
        long_filename = "a" * 300 + ".json"
        with pytest.raises(ValueError, match="Text too long"):
            luna_validator.validate_text_input(long_filename, max_length=255)

    def test_validation_cache_consistency(self):
        """Test that validation caching works consistently across different nodes."""
        # Clear cache
        luna_validator.clear_cache()

        # Test same validation multiple times
        value = 2.5
        for i in range(3):
            result = luna_validator.validate_numeric_input(value, min_value=0.0, max_value=8.0, cache_key='test_scale')
            assert result == value

        # Verify cache is working (should have only 1 entry despite 3 calls)
        assert len(luna_validator.cache.cache) == 1

    def test_validation_error_messages(self):
        """Test that validation error messages are user-friendly."""
        try:
            luna_validator.validate_numeric_input(-1, min_value=0, max_value=10)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Check that error message contains useful information
            assert "below minimum" in error_msg or "invalid" in error_msg.lower()

    def test_validation_with_none_values(self):
        """Test validation handles None values appropriately."""
        # Test that None min/max values work
        result = luna_validator.validate_numeric_input(5, min_value=None, max_value=None)
        assert result == 5

        # Test partial constraints
        result = luna_validator.validate_numeric_input(5, min_value=0, max_value=None)
        assert result == 5

        result = luna_validator.validate_numeric_input(5, min_value=None, max_value=10)
        assert result == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])