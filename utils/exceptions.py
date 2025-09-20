"""
Luna Collection Custom Exceptions

Custom exception classes for better error handling across Luna Collection nodes.
"""

from typing import Optional, Any


class LunaError(Exception):
    """Base exception class for all Luna Collection errors."""

    def __init__(self, message: str, node_name: Optional[str] = None, details: Optional[Any] = None):
        self.node_name = node_name
        self.details = details
        if node_name:
            super().__init__(f"[{node_name}] {message}")
        else:
            super().__init__(message)


class LunaDependencyError(LunaError):
    """Raised when a required dependency is not available."""

    def __init__(self, dependency_name: str, node_name: Optional[str] = None):
        self.dependency_name = dependency_name
        message = f"Required dependency '{dependency_name}' is not available"
        super().__init__(message, node_name)


class LunaModelError(LunaError):
    """Raised when there's an issue with model loading or inference."""

    def __init__(self, message: str, model_name: Optional[str] = None, node_name: Optional[str] = None):
        self.model_name = model_name
        if model_name:
            message = f"Model '{model_name}': {message}"
        super().__init__(message, node_name)


class LunaProcessingError(LunaError):
    """Raised when there's an error during image/text processing."""

    def __init__(self, message: str, input_data: Optional[Any] = None, node_name: Optional[str] = None):
        self.input_data = input_data
        super().__init__(message, node_name)


class LunaConfigurationError(LunaError):
    """Raised when there's a configuration issue."""

    def __init__(self, message: str, parameter: Optional[str] = None, node_name: Optional[str] = None):
        self.parameter = parameter
        if parameter:
            message = f"Configuration error for '{parameter}': {message}"
        super().__init__(message, node_name)


class LunaFileError(LunaError):
    """Raised when there's a file I/O error."""

    def __init__(self, message: str, file_path: Optional[str] = None, node_name: Optional[str] = None):
        self.file_path = file_path
        if file_path:
            message = f"File error for '{file_path}': {message}"
        super().__init__(message, node_name)


class LunaMemoryError(LunaError):
    """Raised when there's a memory-related error."""

    def __init__(self, message: str, memory_required: Optional[int] = None, node_name: Optional[str] = None):
        self.memory_required = memory_required
        if memory_required:
            message = f"Memory error (required: {memory_required}MB): {message}"
        super().__init__(message, node_name)