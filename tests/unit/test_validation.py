"""
Unit tests for Luna Collection input validation system.
"""

import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from validation import (
    LunaInputValidator,
    ImageInput,
    NumericInput,
    TextInput,
    FilePathInput,
    luna_validator
)


class TestLunaInputValidator:
    """Test the main input validator class."""

    def test_validate_image_array(self, sample_image_array):
        """Test image array validation."""
        result = luna_validator.validate_image_input(sample_image_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_validate_image_tensor(self, sample_image_tensor):
        """Test image tensor validation."""
        result = luna_validator.validate_image_input(sample_image_tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 64, 64)

    def test_validate_image_pil(self, sample_image_pil):
        """Test PIL image validation."""
        result = luna_validator.validate_image_input(sample_image_pil)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_validate_invalid_image(self):
        """Test invalid image validation raises error."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            luna_validator.validate_image_input("not an image")

    def test_validate_numeric_in_range(self):
        """Test numeric validation within range."""
        result = luna_validator.validate_numeric_input(5, min_value=0, max_value=10)
        assert result == 5

    def test_validate_numeric_out_of_range(self):
        """Test numeric validation outside range raises error."""
        with pytest.raises(ValueError, match="is below minimum"):
            luna_validator.validate_numeric_input(-1, min_value=0, max_value=10)

        with pytest.raises(ValueError, match="is above maximum"):
            luna_validator.validate_numeric_input(15, min_value=0, max_value=10)

    def test_validate_text_input(self):
        """Test text input validation."""
        result = luna_validator.validate_text_input("Hello World", max_length=50)
        assert result == "Hello World"

    def test_validate_empty_text_when_not_allowed(self):
        """Test empty text validation when not allowed."""
        with pytest.raises(ValueError, match="Empty text input not allowed"):
            luna_validator.validate_text_input("", allow_empty=False)

    def test_validate_text_too_long(self):
        """Test text validation for length limits."""
        long_text = "a" * 100
        with pytest.raises(ValueError, match="Text too long"):
            luna_validator.validate_text_input(long_text, max_length=50)

    def test_validate_file_path_exists(self, temp_text_file):
        """Test file path validation for existing file."""
        result = luna_validator.validate_file_path(temp_text_file, must_exist=True)
        assert result == temp_text_file

    def test_validate_file_path_not_exists(self):
        """Test file path validation for non-existing file."""
        with pytest.raises(ValueError, match="File does not exist"):
            luna_validator.validate_file_path("/nonexistent/file.txt", must_exist=True)

    def test_validate_file_extension(self, temp_text_file):
        """Test file extension validation."""
        # Should work with .txt extension
        result = luna_validator.validate_file_path(
            temp_text_file,
            allowed_extensions=['.txt']
        )
        assert result == temp_text_file

        # Should fail with wrong extension
        with pytest.raises(ValueError, match="not allowed"):
            luna_validator.validate_file_path(
                temp_text_file,
                allowed_extensions=['.png']
            )

    def test_validation_caching(self, sample_image_array):
        """Test that validation results are cached."""
        cache_key = "test_image_cache"

        # First call should validate
        result1 = luna_validator.validate_image_input(sample_image_array, cache_key)
        assert isinstance(result1, np.ndarray)

        # Second call should use cache
        result2 = luna_validator.validate_image_input(sample_image_array, cache_key)
        assert result1 is result2  # Same object reference means cached

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        # Add something to cache
        luna_validator.validate_numeric_input(5, cache_key="test_clear")

        # Clear cache
        luna_validator.clear_cache()

        # Cache should be empty now
        assert luna_validator.cache.cache == {}


class TestInputModels:
    """Test individual input model classes."""

    def test_image_input_model(self, sample_image_array):
        """Test ImageInput Pydantic model."""
        img_input = ImageInput(image=sample_image_array)
        assert img_input.image.shape == (64, 64, 3)

    def test_numeric_input_model(self):
        """Test NumericInput Pydantic model."""
        num_input = NumericInput(value=5.5, min_value=0, max_value=10)
        assert num_input.value == 5.5
        assert num_input.min_value == 0
        assert num_input.max_value == 10

    def test_text_input_model(self):
        """Test TextInput Pydantic model."""
        text_input = TextInput(text="Hello World", max_length=50, min_length=1, allow_empty=False)
        assert text_input.text == "Hello World"
        assert text_input.max_length == 50

    def test_file_path_input_model(self, temp_text_file):
        """Test FilePathInput Pydantic model."""
        file_input = FilePathInput(path=temp_text_file, must_exist=True, allowed_extensions=['.txt'])
        assert file_input.path == temp_text_file
        assert file_input.must_exist is True


class TestInputSanitization:
    """Test input sanitization features."""

    def test_text_sanitization(self):
        """Test that text inputs are sanitized."""
        # Test null byte removal
        sanitized = luna_validator.validate_text_input("Hello\x00World")
        assert "\x00" not in sanitized

        # Test control character removal
        sanitized = luna_validator.validate_text_input("Hello\x01World")
        assert "\x01" not in sanitized

        # Test whitespace trimming
        sanitized = luna_validator.validate_text_input("  Hello World  ")
        assert sanitized == "Hello World"

    def test_numeric_edge_cases(self):
        """Test numeric validation edge cases."""
        # Test float precision
        result = luna_validator.validate_numeric_input(5.123456, min_value=0, max_value=10)
        assert result == 5.123456

        # Test integer conversion
        result = luna_validator.validate_numeric_input(5, min_value=0, max_value=10)
        assert result == 5
        assert isinstance(result, int)


if __name__ == "__main__":
    pytest.main([__file__])