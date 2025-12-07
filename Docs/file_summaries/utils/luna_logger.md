# utils/luna_logger.py

## Purpose
Centralized logging utility providing consistent output formatting and level control across all Luna Collection nodes.

## Exports
- `LunaLogger`: Main logger class with node-specific prefixes
- `luna_logger`: Global logger instance
- `get_logger(node_name)`: Factory function for node-specific loggers

## Key Imports
- `logging`: Python standard logging module
- `sys`: System output streams
- `typing`: Optional type hint
- `.constants`: LOG_FORMAT and LOG_DATE_FORMAT constants

## ComfyUI Node Configuration
N/A - Logging utility, not a node.

## Input Schema
N/A - Logger class.

## Key Methods
- `LunaLogger.__init__(name, level)`: Initialize logger with name and level, setup console handler
- `LunaLogger.info/warning/error/debug(message, node_name)`: Log methods with optional node prefix
- `get_logger(node_name)`: Create node-specific logger instance

## Dependencies
- `logging`: Standard Python logging
- `.constants`: For format strings

## Integration Points
- Used by all Luna Collection nodes for consistent logging
- Provides [NodeName] prefixes for better log identification
- Integrates with constants.py for standardized format strings
- Prevents log propagation to avoid duplicate messages

## Notes
- Singleton pattern with global luna_logger instance
- Node-specific loggers available via get_logger()
- Consistent timestamp format: HH:MM:SS
- Prevents duplicate handlers and propagation
- Used across loaders, processors, and utility nodes