# utils/__init__.py

## Purpose
Central import module for Luna Collection utilities providing unified access to all utility classes and functions with graceful import handling for optional dependencies.

## Exports
- All utility classes and functions from submodules
- Graceful handling of optional dependencies (validation, metadata_db)
- `__all__` list for controlled public API

## Key Imports
- `.segs`: Segmentation utilities (SEG, etc.)
- `.mediapipe_engine`: MediaPipe processing (optional)
- `.tiling`: Image tiling orchestration
- `.trt_engine`: TensorRT engine wrapper
- `.luna_performance_monitor`: Performance monitoring
- `.constants`: All constants and configuration
- `.luna_logger`: Logging utilities
- `.exceptions`: Custom exception classes
- `validation`: Input validation (optional)
- `.luna_metadata_db`: Metadata database (optional)

## ComfyUI Node Configuration
N/A - Import module, not a node.

## Input Schema
N/A - Import module.

## Key Methods
N/A - Import module with try/except for optional dependencies.

## Dependencies
- All utility submodules
- Optional: validation, luna_metadata_db

## Integration Points
- Imported by all Luna Collection nodes for utility access
- Provides single import point for common utilities
- Graceful degradation when optional dependencies unavailable
- Used by node __init__.py for utility exposure

## Notes
- Implements optional dependency pattern for robust imports
- Exposes all utility functions through single import
- Maintains backwards compatibility with missing modules
- Central hub for Luna Collection utility ecosystem