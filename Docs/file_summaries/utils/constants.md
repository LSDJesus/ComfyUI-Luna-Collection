# utils/constants.py

## Purpose
Centralized constants file providing consistent configuration values across the Luna Collection nodes and utilities.

## Exports
- `LUNA_BASE_CATEGORY`: Base category string "Luna"
- `CATEGORY_*`: Category constants for different node types (Preprocessing, Performance, Upscaling, etc.)
- `DEFAULT_*`: Default configuration values (confidence threshold, batch size, resolution, etc.)
- `*_EXTENSIONS`: File extension lists for embeddings, images, and models
- `PERFORMANCE_LOG_PREFIX`: Logging prefix for performance monitoring
- `LOG_FORMAT`: Standard logging format string
- `ERROR_MESSAGES`: Dictionary of standardized error messages
- `SUCCESS_MESSAGES`: Dictionary of standardized success messages

## Key Imports
None - Pure constants file.

## ComfyUI Node Configuration
N/A - Constants file, not a node.

## Input Schema
N/A - Constants file.

## Key Methods
N/A - Constants file.

## Dependencies
None.

## Integration Points
- Used by all Luna Collection nodes for consistent categorization
- Referenced by logging utilities for standardized messages
- Imported by performance monitoring and other utility modules

## Notes
- Provides centralized configuration to avoid hardcoded strings
- Includes file extension lists for model/embeddings discovery
- Contains standardized error/success message templates
- Used across preprocessing, upscaling, detailing, and loader nodes