"""
Luna Collection Input Validation System

Provides comprehensive input validation for all Luna nodes using Pydantic.
Ensures type safety, input sanitization, and user-friendly error messages.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict, model_validator
import numpy as np
from PIL import Image
import torch
import re


class ValidationCache:
    """Cache for validation results to avoid repeated expensive checks."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self._access_order = []

    def get(self, key: str) -> Optional[Any]:
        """Get cached validation result."""
        if key in self.cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached validation result."""
        if key in self.cache:
            self._access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = self._access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self._access_order.clear()


# Global validation cache instance
validation_cache = ValidationCache()


class LunaBaseInput(BaseModel):
    """Base class for all Luna node inputs with common validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator('*', mode='before')
    @classmethod
    def sanitize_input(cls, v):
        """Sanitize input values to prevent common issues."""
        if isinstance(v, str):
            # Remove null bytes and control characters
            v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
            # Trim whitespace
            v = v.strip()
        return v


class ImageInput(LunaBaseInput):
    """Validation for image inputs."""

    image: Any = Field(..., description="Input image tensor or PIL Image")

    @field_validator('image')
    @classmethod
    def validate_image(cls, v):
        """Validate image input format and dimensions."""
        if isinstance(v, np.ndarray):
            if v.ndim not in [2, 3, 4]:
                raise ValueError(f"Image array must be 2D, 3D, or 4D, got {v.ndim}D")
            if v.ndim == 3 and v.shape[-1] not in [1, 3, 4]:
                raise ValueError(f"Image channels must be 1, 3, or 4, got {v.shape[-1]}")
        elif isinstance(v, torch.Tensor):
            if v.ndim not in [2, 3, 4]:
                raise ValueError(f"Image tensor must be 2D, 3D, or 4D, got {v.ndim}D")
        elif isinstance(v, Image.Image):
            # PIL Image is valid
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(v)}")

        return v


class NumericInput(LunaBaseInput):
    """Validation for numeric inputs with range checking."""

    value: Union[int, float] = Field(..., description="Numeric value")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum allowed value")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum allowed value")

    @model_validator(mode='after')
    def validate_range(self):
        """Validate numeric value is within specified range."""
        v = self.value
        min_val = self.min_value
        max_val = self.max_value

        if min_val is not None and v < min_val:
            raise ValueError(f"Value {v} is below minimum {min_val}")
        if max_val is not None and v > max_val:
            raise ValueError(f"Value {v} is above maximum {max_val}")

        return self


class TextInput(LunaBaseInput):
    """Validation for text inputs with length and content checking."""

    text: str = Field(..., description="Text input")
    max_length: Optional[int] = Field(None, description="Maximum allowed length")
    min_length: Optional[int] = Field(1, description="Minimum allowed length")
    allow_empty: bool = Field(False, description="Whether empty strings are allowed")

    @model_validator(mode='after')
    def validate_text(self):
        """Validate text content and length."""
        v = self.text
        if not isinstance(v, str):
            raise ValueError(f"Expected string, got {type(v)}")

        min_len = self.min_length or 1
        max_len = self.max_length
        allow_empty = self.allow_empty

        if not allow_empty and len(v.strip()) == 0:
            raise ValueError("Empty text input not allowed")

        if len(v) < min_len:
            raise ValueError(f"Text too short: {len(v)} < {min_len}")

        if max_len is not None and len(v) > max_len:
            raise ValueError(f"Text too long: {len(v)} > {max_len}")

        return self


class FilePathInput(LunaBaseInput):
    """Validation for file path inputs."""

    path: str = Field(..., description="File path")
    must_exist: bool = Field(True, description="Whether file must exist")
    allowed_extensions: Optional[List[str]] = Field(None, description="Allowed file extensions")

    @model_validator(mode='after')
    def validate_path(self):
        """Validate file path exists and has correct extension."""
        import os

        v = self.path
        if not isinstance(v, str):
            raise ValueError(f"Expected string path, got {type(v)}")

        must_exist = self.must_exist
        allowed_ext = self.allowed_extensions

        if must_exist and not os.path.exists(v):
            raise ValueError(f"File does not exist: {v}")

        if allowed_ext:
            _, ext = os.path.splitext(v)
            if ext.lower() not in [e.lower() for e in allowed_ext]:
                raise ValueError(f"File extension {ext} not allowed. Allowed: {allowed_ext}")

        return self


class LunaInputValidator:
    """Main input validation class for Luna Collection nodes."""

    def __init__(self):
        self.cache = validation_cache

    def validate_image_input(self, image: Any, cache_key: Optional[str] = None) -> Any:
        """Validate image input with caching."""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            validated = ImageInput(image=image)
            if cache_key:
                self.cache.set(cache_key, validated.image)
            return validated.image
        except ValidationError as e:
            self._raise_user_friendly_error(e, "image")

    def validate_numeric_input(self, value: Union[int, float],
                             min_value: Optional[Union[int, float]] = None,
                             max_value: Optional[Union[int, float]] = None,
                             cache_key: Optional[str] = None) -> Union[int, float]:
        """Validate numeric input with range checking."""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            validated = NumericInput(value=value, min_value=min_value, max_value=max_value)
            if cache_key:
                self.cache.set(cache_key, validated.value)
            return validated.value
        except ValidationError as e:
            self._raise_user_friendly_error(e, "numeric value")
            return value  # This line will never be reached, but satisfies type checker

    def validate_text_input(self, text: str,
                          max_length: Optional[int] = None,
                          min_length: Optional[int] = 1,
                          allow_empty: bool = False,
                          cache_key: Optional[str] = None) -> str:
        """Validate text input with length and content checking."""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            validated = TextInput(text=text, max_length=max_length,
                                min_length=min_length, allow_empty=allow_empty)
            if cache_key:
                self.cache.set(cache_key, validated.text)
            return validated.text
        except ValidationError as e:
            self._raise_user_friendly_error(e, "text")
            return text  # This line will never be reached, but satisfies type checker

    def validate_file_path(self, path: str,
                          must_exist: bool = True,
                          allowed_extensions: Optional[List[str]] = None,
                          cache_key: Optional[str] = None) -> str:
        """Validate file path input."""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            validated = FilePathInput(path=path, must_exist=must_exist,
                                    allowed_extensions=allowed_extensions)
            if cache_key:
                self.cache.set(cache_key, validated.path)
            return validated.path
        except ValidationError as e:
            self._raise_user_friendly_error(e, "file path")
            return path  # This line will never be reached, but satisfies type checker

    def _raise_user_friendly_error(self, validation_error: ValidationError, input_type: str):
        """Convert Pydantic ValidationError to user-friendly error message."""
        errors = validation_error.errors()
        if errors:
            error = errors[0]
            # In Pydantic V2, the error structure is different
            loc = error.get('loc', ())
            if isinstance(loc, tuple) and len(loc) > 0:
                field = str(loc[0])
            else:
                field = 'unknown'

            msg = error.get('msg', 'Validation failed')

            user_msg = f"Invalid {input_type} for field '{field}': {msg}"
            raise ValueError(user_msg) from validation_error

    def clear_cache(self):
        """Clear validation cache."""
        self.cache.clear()


# Global validator instance
luna_validator = LunaInputValidator()


def validate_node_input(input_type: str, **kwargs):
    """
    Decorator for validating node inputs.

    Usage:
        @validate_node_input('image', cache_key='image_validation')
        def process_image(self, image):
            # image is now validated
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs_inner):
            # Extract the input value from kwargs or args
            input_value = kwargs_inner.get(input_type)
            if input_value is None and len(args) > 0:
                # Try to find the input in args by inspecting function signature
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if len(param_names) >= len(args):
                    # Find the parameter name that matches input_type
                    for i, param_name in enumerate(param_names):
                        if param_name == input_type and i < len(args):
                            input_value = args[i]
                            break

            if input_value is None:
                raise ValueError(f"Missing required input: {input_type}")

            # Validate based on input type
            cache_key = kwargs.get('cache_key')
            if input_type == 'image' or 'image' in input_type:
                validated_value = luna_validator.validate_image_input(input_value, cache_key)
            elif input_type in ['int', 'float', 'numeric'] or any(t in input_type for t in ['int', 'float', 'numeric']):
                min_val = kwargs.get('min_value')
                max_val = kwargs.get('max_value')
                validated_value = luna_validator.validate_numeric_input(
                    input_value, min_val, max_val, cache_key
                )
            elif input_type == 'text' or input_type == 'string' or 'text' in input_type or 'string' in input_type:
                max_len = kwargs.get('max_length')
                min_len = kwargs.get('min_length', 1)
                allow_empty = kwargs.get('allow_empty', False)
                validated_value = luna_validator.validate_text_input(
                    input_value, max_len, min_len, allow_empty, cache_key
                )
            elif input_type == 'file_path' or input_type == 'path' or 'path' in input_type:
                must_exist = kwargs.get('must_exist', True)
                allowed_ext = kwargs.get('allowed_extensions')
                validated_value = luna_validator.validate_file_path(
                    input_value, must_exist, allowed_ext, cache_key
                )
            else:
                # No validation for unknown types
                validated_value = input_value

            # Call the function with validated input
            # Replace the input value in the appropriate location
            if input_type in kwargs_inner:
                kwargs_inner[input_type] = validated_value
                return func(*args, **kwargs_inner)
            else:
                # For positional arguments, we need to modify args
                # Find the position of the parameter
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                try:
                    param_index = param_names.index(input_type)
                    if param_index < len(args):
                        # Replace the positional argument
                        new_args: List[Any] = list(args)
                        new_args[param_index] = validated_value
                        return func(*new_args, **kwargs_inner)
                except ValueError:
                    pass

                # Fallback: add to kwargs
                kwargs_inner[input_type] = validated_value
                return func(*args, **kwargs_inner)
        return wrapper
    return decorator