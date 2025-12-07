# utils/luna_performance_monitor.py

## Purpose
Shared performance monitoring utility providing VRAM usage tracking, processing time measurement, and memory optimization for all Luna Collection nodes.

## Exports
- `LunaPerformanceMonitor`: Performance monitoring class
- `LUNA_PERFORMANCE_MONITOR`: Global shared instance

## Key Imports
- `torch`: GPU memory monitoring
- `psutil`: System memory and CPU monitoring
- `time`: Timing measurements
- `typing`: Type hints
- `gc`: Garbage collection control

## ComfyUI Node Configuration
N/A - Monitoring utility, not a node.

## Input Schema
N/A - Monitoring class.

## Key Methods
- `LunaPerformanceMonitor.start_monitoring(node_name)`: Begin performance tracking with garbage collection
- `LunaPerformanceMonitor.stop_monitoring(**kwargs)`: End monitoring and return comprehensive statistics
- `LunaPerformanceMonitor._get_vram_usage()`: Get current GPU memory allocation in MB
- `LunaPerformanceMonitor.get_vram_total()`: Get total GPU memory capacity
- `LunaPerformanceMonitor.optimize_memory()`: Force garbage collection and CUDA cache clearing

## Dependencies
- `torch`: GPU memory operations
- `psutil`: System resource monitoring

## Integration Points
- Used by performance-critical nodes for resource monitoring
- Provides standardized performance metrics across Luna Collection
- Integrated with constants.py PERFORMANCE_LOG_PREFIX
- Global instance allows shared monitoring across nodes

## Notes
- Tracks VRAM delta, processing time, and system resources
- Automatic garbage collection and CUDA cache clearing
- Compatible with CPU-only systems (VRAM reports as 0)
- Returns comprehensive statistics dict for logging/analysis
- Used by upscaling and processing nodes for performance optimization