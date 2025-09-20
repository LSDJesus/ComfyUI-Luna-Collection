"""
Integration tests for Luna Collection validation system.
Tests validation in real node workflows and error handling.
"""

import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from validation import luna_validator, validate_node_input
from unittest.mock import Mock, patch


class TestValidationIntegration:
    """Integration tests for validation in node workflows."""

    def test_image_validation_workflow(self, sample_image_array):
        """Test complete image validation workflow."""
        # Simulate a node that processes images
        def process_image_node(image_input):
            # Validate input
            validated_image = luna_validator.validate_image_input(image_input)

            # Simulate processing
            if isinstance(validated_image, np.ndarray):
                # Convert to grayscale
                if validated_image.ndim == 3 and validated_image.shape[-1] == 3:
                    return np.mean(validated_image, axis=-1)
            return validated_image

        # Test with different image types
        result_array = process_image_node(sample_image_array)
        assert isinstance(result_array, np.ndarray)

        # Test with PIL image
        pil_image = Image.fromarray((sample_image_array * 255).astype(np.uint8))
        result_pil = process_image_node(pil_image)
        assert isinstance(result_pil, Image.Image)

    def test_numeric_validation_workflow(self):
        """Test numeric validation in parameter processing."""
        def adjust_brightness_node(brightness_factor, min_val=0.0, max_val=2.0):
            # Validate brightness factor
            validated_factor = luna_validator.validate_numeric_input(
                brightness_factor,
                min_value=min_val,
                max_value=max_val
            )

            # Simulate brightness adjustment
            return validated_factor * 100  # Convert to percentage

        # Test valid inputs
        result = adjust_brightness_node(1.5)
        assert result == 150.0

        # Test boundary values
        result_min = adjust_brightness_node(0.0)
        assert result_min == 0.0

        result_max = adjust_brightness_node(2.0)
        assert result_max == 200.0

        # Test invalid inputs
        with pytest.raises(ValueError):
            adjust_brightness_node(-0.1)  # Below minimum

        with pytest.raises(ValueError):
            adjust_brightness_node(2.1)  # Above maximum

    def test_text_validation_workflow(self):
        """Test text validation in text processing nodes."""
        def text_prompt_node(prompt_text, max_words=50):
            # Validate text input
            validated_text = luna_validator.validate_text_input(
                prompt_text,
                max_length=max_words * 10  # Rough estimate
            )

            # Simulate text processing
            word_count = len(validated_text.split())
            if word_count > max_words:
                raise ValueError(f"Too many words: {word_count} > {max_words}")

            return {
                'text': validated_text,
                'word_count': word_count,
                'processed': True
            }

        # Test valid prompt
        result = text_prompt_node("A beautiful landscape with mountains and lakes")
        assert result['processed'] is True
        assert result['word_count'] == 7

        # Test empty text (should fail)
        with pytest.raises(ValueError, match="Empty text input not allowed"):
            text_prompt_node("")

        # Test too long text (should fail at validation level)
        long_text = "a" * 600  # 600 characters, exceeds max_length=500
        with pytest.raises(ValueError, match="Text too long"):
            text_prompt_node(long_text, max_words=50)

    def test_file_path_validation_workflow(self):
        """Test file path validation in file processing nodes."""
        def load_model_node(model_path, allowed_exts=['.safetensors', '.ckpt', '.pth']):
            # Validate file path
            validated_path = luna_validator.validate_file_path(
                model_path,
                must_exist=True,
                allowed_extensions=allowed_exts
            )

            # Simulate model loading
            file_size = os.path.getsize(validated_path)
            return {
                'path': validated_path,
                'size': file_size,
                'loaded': True
            }

        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
            f.write(b'dummy model data')
            temp_model_path = f.name

        try:
            # Test valid model loading
            result = load_model_node(temp_model_path)
            assert result['loaded'] is True
            assert result['path'] == temp_model_path
            assert result['size'] == len(b'dummy model data')

            # Test invalid extension
            with pytest.raises(ValueError, match="not allowed"):
                load_model_node(temp_model_path, allowed_exts=['.bin'])

            # Test non-existent file
            with pytest.raises(ValueError, match="File does not exist"):
                load_model_node("/nonexistent/model.safetensors")

        finally:
            # Clean up
            os.unlink(temp_model_path)

    def test_combined_validation_workflow(self):
        """Test multiple validations working together in a complex node."""
        def advanced_image_processor(
            image,
            brightness=1.0,
            prompt="",
            output_path=None
        ):
            # Validate all inputs
            validated_image = luna_validator.validate_image_input(image)
            validated_brightness = luna_validator.validate_numeric_input(
                brightness, min_value=0.0, max_value=2.0
            )
            validated_prompt = luna_validator.validate_text_input(
                prompt, max_length=200, allow_empty=True
            )

            if output_path:
                validated_output = luna_validator.validate_file_path(
                    output_path, must_exist=False, allowed_extensions=['.png', '.jpg']
                )
            else:
                validated_output = None

            # Simulate processing
            return {
                'image_shape': validated_image.shape if hasattr(validated_image, 'shape') else 'PIL',
                'brightness': validated_brightness,
                'prompt': validated_prompt,
                'output_path': validated_output,
                'processed': True
            }

        # Test with all valid inputs
        result = advanced_image_processor(
            image=np.random.rand(64, 64, 3),
            brightness=1.2,
            prompt="Bright and colorful image",
            output_path="/tmp/output.png"
        )
        assert result['processed'] is True
        assert result['brightness'] == 1.2
        assert result['prompt'] == "Bright and colorful image"

    def test_validation_error_handling(self):
        """Test that validation errors are properly handled and user-friendly."""
        def error_prone_node(image, factor):
            try:
                validated_image = luna_validator.validate_image_input(image)
                validated_factor = luna_validator.validate_numeric_input(
                    factor, min_value=0, max_value=1
                )
                return validated_image * validated_factor
            except ValueError as e:
                # Simulate node error handling
                raise RuntimeError(f"Node processing failed: {str(e)}")

        # Test with invalid image
        with pytest.raises(RuntimeError, match="Node processing failed"):
            error_prone_node("not an image", 0.5)

        # Test with invalid numeric value
        with pytest.raises(RuntimeError, match="Node processing failed"):
            error_prone_node(np.random.rand(64, 64, 3), 1.5)

    def test_validation_caching_integration(self):
        """Test that caching works across multiple validation calls."""
        # Clear cache first
        luna_validator.clear_cache()

        # Test with same cache key but different data
        image1 = np.random.rand(32, 32, 3)
        image2 = np.random.rand(32, 32, 3)

        # First call should validate
        result1 = luna_validator.validate_image_input(image1, cache_key='test_cache')

        # Second call with same cache key should return cached result
        result2 = luna_validator.validate_image_input(image2, cache_key='test_cache')

        # Results should be the same object (cached)
        assert result1 is result2
        assert np.array_equal(result1, result2)

    @patch('validation.luna_validator.validate_image_input')
    def test_validation_decorator_integration(self, mock_validate):
        """Test validation decorator integration."""
        mock_validate.return_value = np.random.rand(64, 64, 3)

        @validate_node_input('image', cache_key='test_image')
        def mock_node(image):
            return image

        # Call the decorated function
        result = mock_node(np.random.rand(64, 64, 3))

        # Verify validation was called
        mock_validate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])