"""
Luna Collection Validation System

Comprehensive input validation with caching and error handling.
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.performance_metrics is None:
            self.performance_metrics = {}

class LunaInputValidator:
    """Advanced input validation with caching and performance monitoring."""

    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        self._validation_cache = {}

    def validate_node_input(self, input_data: Dict[str, Any], node_type: str) -> ValidationResult:
        """Validate input parameters for a specific node type."""
        import time
        start_time = time.time()

        # Create cache key
        cache_key = self._create_cache_key(input_data, node_type)

        # Check cache
        if self.enable_caching and cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            cached_result.performance_metrics['cached'] = True
            return cached_result

        errors = []
        warnings = []

        # Validate based on node type
        if node_type == "LunaSampler":
            errors.extend(self._validate_sampler_input(input_data))
        elif node_type in ["LunaSimpleUpscaler", "LunaAdvancedUpscaler"]:
            errors.extend(self._validate_upscaler_input(input_data))
        elif node_type == "LunaMediaPipeDetailer":
            errors.extend(self._validate_mediapipe_input(input_data))
        else:
            warnings.append(f"Unknown node type: {node_type}")

        # General validation
        errors.extend(self._validate_general_input(input_data))

        # Create result
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_metrics={
                'validation_time': time.time() - start_time,
                'cached': False
            }
        )

        # Cache result
        if self.enable_caching:
            self._validation_cache[cache_key] = result

        return result

    def _validate_sampler_input(self, input_data: Dict[str, Any]) -> List[str]:
        """Validate LunaSampler input parameters."""
        errors = []

        # Required parameters
        required = ['model', 'positive', 'negative', 'latent_image']
        for param in required:
            if param not in input_data:
                errors.append(f"Missing required parameter: {param}")

        # Steps validation
        if 'steps' in input_data:
            steps = input_data['steps']
            if not isinstance(steps, int) or steps < 1 or steps > 1000:
                errors.append("steps must be an integer between 1 and 1000")

        # CFG validation
        if 'cfg' in input_data:
            cfg = input_data['cfg']
            if not isinstance(cfg, (int, float)) or cfg < 0.0 or cfg > 100.0:
                errors.append("cfg must be a number between 0.0 and 100.0")

        # Denoise validation
        if 'denoise' in input_data:
            denoise = input_data['denoise']
            if not isinstance(denoise, (int, float)) or denoise < 0.0 or denoise > 1.0:
                errors.append("denoise must be a number between 0.0 and 1.0")

        return errors

    def _validate_upscaler_input(self, input_data: Dict[str, Any]) -> List[str]:
        """Validate upscaler input parameters."""
        errors = []

        # Required parameters
        required = ['image', 'upscale_model']
        for param in required:
            if param not in input_data:
                errors.append(f"Missing required parameter: {param}")

        # Scale validation
        if 'scale_by' in input_data:
            scale = input_data['scale_by']
            if not isinstance(scale, (int, float)) or scale < 1.0 or scale > 8.0:
                errors.append("scale_by must be a number between 1.0 and 8.0")

        return errors

    def _validate_mediapipe_input(self, input_data: Dict[str, Any]) -> List[str]:
        """Validate MediaPipe input parameters."""
        errors = []

        # Required parameters
        if 'image' not in input_data:
            errors.append("Missing required parameter: image")

        # Model type validation
        if 'model_type' in input_data:
            valid_types = ['face', 'eyes', 'mouth', 'hands', 'person', 'feet', 'torso']
            if input_data['model_type'] not in valid_types:
                errors.append(f"model_type must be one of: {valid_types}")

        # Confidence validation
        if 'confidence' in input_data:
            confidence = input_data['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0.1 or confidence > 1.0:
                errors.append("confidence must be a number between 0.1 and 1.0")

        return errors

    def _validate_general_input(self, input_data: Dict[str, Any]) -> List[str]:
        """General input validation."""
        errors = []

        # Check for None values in required fields
        for key, value in input_data.items():
            if value is None and key in ['model', 'positive', 'negative', 'latent_image', 'image']:
                errors.append(f"Required parameter '{key}' cannot be None")

        return errors

    def _create_cache_key(self, input_data: Dict[str, Any], node_type: str) -> str:
        """Create a cache key for validation results."""
        # Convert input data to a string representation
        input_str = str(sorted(input_data.items()))
        cache_content = f"{node_type}:{input_str}"

        # Create hash
        return hashlib.md5(cache_content.encode()).hexdigest()

    def clear_cache(self):
        """Clear the validation cache."""
        self._validation_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._validation_cache),
            'cache_enabled': self.enable_caching
        }