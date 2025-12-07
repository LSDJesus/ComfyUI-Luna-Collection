# utils/exceptions.py

## Purpose
Custom exception hierarchy for the Luna Collection providing structured error handling with node-specific context and detailed error information.

## Exports
- `LunaError`: Base exception class with node_name and details attributes
- `LunaDependencyError`: For missing required dependencies
- `LunaModelError`: For model loading/inference issues
- `LunaProcessingError`: For image/text processing errors
- `LunaConfigurationError`: For configuration parameter issues
- `LunaFileError`: For file I/O operations
- `LunaMemoryError`: For memory-related errors

## Key Imports
- `typing`: Type hints for Optional and Any

## ComfyUI Node Configuration
N/A - Exception classes, not a node.

## Input Schema
N/A - Exception classes.

## Key Methods
- `LunaError.__init__(message, node_name, details)`: Initialize with optional node context and details
- `LunaDependencyError.__init__(dependency_name, node_name)`: Initialize with dependency name
- `LunaModelError.__init__(message, model_name, node_name)`: Initialize with model context
- `LunaProcessingError.__init__(message, input_data, node_name)`: Initialize with input data context
- `LunaConfigurationError.__init__(message, parameter, node_name)`: Initialize with parameter context
- `LunaFileError.__init__(message, file_path, node_name)`: Initialize with file path context
- `LunaMemoryError.__init__(message, memory_required, node_name)`: Initialize with memory requirements

## Dependencies
- `typing`: Type annotations

## Integration Points
- Used by all Luna Collection nodes for consistent error handling
- Imported by model loaders, processors, and utility functions
- Provides structured error context for logging and user feedback
- Supports node-specific error messages with [NodeName] prefixes

## Notes
- All exceptions inherit from LunaError for consistent handling
- Optional node_name parameter adds [NodeName] prefix to messages
- Detailed attributes (model_name, file_path, memory_required) for debugging
- Designed for graceful error handling with informative user messages