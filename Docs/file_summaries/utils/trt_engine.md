# utils/trt_engine.py

## Purpose
TensorRT engine wrapper providing ONNX model conversion, engine building, and optimized inference execution with CUDA graph support and progress monitoring.

## Exports
- `Engine`: Main TensorRT engine class for model inference
- `TQDMProgressMonitor`: Progress monitoring with tqdm integration
- `numpy_to_torch_dtype_dict`: Type conversion mappings
- `torch_to_numpy_dtype_dict`: Reverse type conversion mappings

## Key Imports
- `torch`: Tensor operations and CUDA support
- `tensorrt`: NVIDIA TensorRT library
- `polygraphy`: ONNX parsing and engine building utilities
- `numpy`: Array type handling
- `tqdm`: Progress bar display
- `collections`: OrderedDict for buffer management

## ComfyUI Node Configuration
N/A - TensorRT utility, not a node.

## Input Schema
N/A - Engine class with file-based initialization.

## Key Methods
- `Engine.build(onnx_path, fp16, input_profile, ...)`: Build TensorRT engine from ONNX model
- `Engine.load()`: Load pre-built engine from file
- `Engine.activate(reuse_device_memory)`: Create execution context
- `Engine.allocate_buffers(shape_dict, device)`: Allocate GPU buffers for tensors
- `Engine.infer(feed_dict, stream, use_cuda_graph)`: Execute inference with CUDA streams
- `TQDMProgressMonitor.phase_start/phase_finish/step_complete()`: Progress tracking callbacks

## Dependencies
- `tensorrt`: NVIDIA TensorRT (optional, raises error if unavailable)
- `polygraphy`: ONNX parsing and engine utilities
- `torch`: CUDA tensor operations
- `tqdm`: Progress visualization

## Integration Points
- Used by performance-critical nodes requiring GPU acceleration
- Provides FP16 optimization and dynamic shape support
- Integrated with constants.py error messages for availability checking
- Supports CUDA graph execution for reduced latency
- Used by Luna performance nodes for hardware acceleration

## Notes
- Implements full TensorRT pipeline: ONNX → engine → inference
- Supports dynamic input shapes with optimization profiles
- Includes NVTX profiling ranges for performance analysis
- Progress monitoring with hierarchical tqdm bars
- Memory-efficient buffer allocation and reuse
- Compatible with ComfyUI's optional dependency pattern