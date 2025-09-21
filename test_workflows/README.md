# Luna Collection Test Workflows

This directory contains comprehensive test workflows designed to validate all Luna Collection nodes across different scenarios and complexity levels.

## Workflow Overview

### 01_luna_loaders_simple.json
**Purpose**: Basic functionality test of all Luna loader nodes
**Complexity**: Minimal
**Focus**: Core loading functionality
**Nodes Tested**:
- LunaCheckpointLoader
- LunaLoRAStacker
- LunaEmbeddingManager
- Basic ComfyUI integration

### 02_luna_loaders_random.json
**Purpose**: Test random/variational loader nodes
**Complexity**: Minimal
**Focus**: Seeded randomization and reproducibility
**Nodes Tested**:
- LunaLoRAStackerRandom
- LunaEmbeddingManagerRandom
- Random seed handling

### 03_luna_preprocessing.json
**Purpose**: Test text and prompt preprocessing pipeline
**Complexity**: Medium
**Focus**: Text processing and prompt enhancement
**Nodes Tested**:
- LunaPromptPreprocessor
- LunaTextProcessor
- LunaUnifiedPromptProcessor

### 04_luna_performance.json
**Purpose**: Test performance monitoring and optimization
**Complexity**: Medium
**Focus**: Performance tracking and resource management
**Nodes Tested**:
- LunaPerformanceLogger
- LunaPerformanceCondition
- LunaPerformanceDisplay
- LunaPerformanceConcat

### 05_luna_detailing.json
**Purpose**: Test face/object detection and enhancement
**Complexity**: High
**Focus**: AI-powered image enhancement
**Nodes Tested**:
- LunaDetailer
- LunaMediaPipeDetailer
- Face detection and inpainting

### 06_luna_upscaling.json
**Purpose**: Test multi-stage upscaling methods
**Complexity**: High
**Focus**: Image upscaling and quality enhancement
**Nodes Tested**:
- LunaUpscalerSimple
- LunaUpscalerAdvanced
- LunaUltimateSDUpscale

### 07_luna_full_pipeline.json
**Purpose**: Complete end-to-end workflow test
**Complexity**: Maximum
**Focus**: Full integration and compatibility
**Nodes Tested**: All Luna nodes in coordinated workflow

### 08_luna_edge_cases.json
**Purpose**: Test error handling and edge cases
**Complexity**: Variable
**Focus**: Robustness and error recovery
**Nodes Tested**: All nodes with invalid/error inputs

### 09_luna_batch_processing.json
**Purpose**: Test batch processing and multi-output
**Complexity**: High
**Focus**: Scalability and batch operations
**Nodes Tested**: Batch-capable nodes with performance monitoring

## How to Use

### Loading Workflows
1. Open ComfyUI in your browser
2. Click "Load" button in the menu
3. Select one of the JSON workflow files
4. The workflow will load with all nodes and connections

### Running Tests
1. Ensure all required models and dependencies are installed
2. Update file paths in nodes to match your actual model locations
3. Adjust parameters as needed for your hardware
4. Execute the workflow

### Expected Results
- **Simple workflows**: Should complete quickly with basic outputs
- **Complex workflows**: May take longer, test full functionality
- **Edge case workflows**: May show error handling or graceful failures
- **Batch workflows**: Generate multiple outputs for comparison

## Prerequisites

### Required Models
- SD checkpoints (Realistic Vision, etc.)
- LoRA models for testing
- Textual Inversion embeddings
- Upscaling models (4x-UltraSharp, etc.)
- Detection models (YOLO, MediaPipe)

### Dependencies
- ComfyUI with all standard nodes
- Python dependencies as listed in requirements.txt
- Sufficient VRAM for complex workflows

## Validation Checklist

For each workflow, verify:
- [ ] All nodes load without errors
- [ ] Connections are established correctly
- [ ] Parameters are within valid ranges
- [ ] Output files are generated
- [ ] No runtime errors occur
- [ ] Performance is acceptable
- [ ] Results meet quality expectations

## Troubleshooting

### Common Issues
1. **Missing Models**: Update file paths in workflow nodes
2. **Out of Memory**: Reduce batch sizes or image dimensions
3. **Import Errors**: Ensure all dependencies are installed
4. **Path Errors**: Use absolute paths or verify model locations

### Performance Optimization
- Start with simple workflows to verify basic functionality
- Gradually increase complexity
- Monitor VRAM usage with performance nodes
- Adjust batch sizes based on available resources

## Contributing

When adding new test workflows:
1. Follow the naming convention: `NN_descriptive_name.json`
2. Include comprehensive node coverage
3. Add appropriate grouping and documentation
4. Test on multiple hardware configurations
5. Update this README with new workflow details

## Support

For issues with these test workflows:
1. Check the main Luna Collection documentation
2. Verify all prerequisites are met
3. Test with simpler workflows first
4. Check ComfyUI logs for detailed error messages