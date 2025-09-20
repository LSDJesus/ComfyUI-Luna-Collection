# 🎯 Luna Collection Node Reference

This comprehensive guide provides detailed documentation for all nodes in the Luna Collection, including inputs, outputs, usage examples, and best practices.

## Table of Contents

- [Image Processing Nodes](#-image-processing-nodes)
- [Sampling & Generation Nodes](#-sampling--generation-nodes)
- [Model Management Nodes](#-model-management-nodes)
- [Text Processing Nodes](#-text-processing-nodes)
- [Computer Vision Nodes](#-computer-vision-nodes)
- [Utility Nodes](#-utility-nodes)

---

## 🖼️ Image Processing Nodes

### Luna Simple Upscaler

**Category**: `Luna Collection/Image Processing`  
**Description**: Clean, lightweight image upscaling with model-based enhancement.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `image` | `IMAGE` | Input image to upscale | Required | Image format validation |
| `upscale_model` | `UPSCALE_MODEL` | Model for upscaling | Required | Model availability check |
| `scale_by` | `FLOAT` | Scale factor (1.0-8.0) | 2.0 | Range: 1.0-8.0 |
| `resampling` | `STRING` | Resampling method | `lanczos` | Options: nearest, linear, cubic, lanczos |
| `show_preview` | `BOOLEAN` | Show preview in node | `true` | - |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `upscaled_image` | `IMAGE` | Upscaled result |

#### Usage Example
```python
# Simple 2x upscaling
upscaled = luna_simple_upscaler(
    image=input_image,
    upscale_model=upscale_model,
    scale_by=2.0
)
```

#### Performance Notes
- **Memory Usage**: ~2x input image size
- **Execution Time**: 100-500ms depending on model and scale
- **GPU Memory**: Model-dependent, typically 1-4GB

---

### Luna Advanced Upscaler

**Category**: `Luna Collection/Image Processing`  
**Description**: Professional-grade upscaling with artifact prevention and quality controls.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `image` | `IMAGE` | Input image | Required | Image validation |
| `upscale_model` | `UPSCALE_MODEL` | Upscaling model | Required | Model check |
| `scale_by` | `FLOAT` | Scale factor | 2.0 | Range: 1.0-8.0 |
| `supersample` | `BOOLEAN` | Enable supersampling | `false` | - |
| `rounding_modulus` | `INT` | Dimension rounding | 8 | Options: 1, 2, 4, 8, 16, 32 |
| `rescale_after_model` | `BOOLEAN` | Final rescale pass | `true` | - |
| `resampling` | `STRING` | Resampling method | `lanczos` | Options: nearest, linear, cubic, lanczos |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `upscaled_image` | `IMAGE` | High-quality upscaled image |

#### Advanced Features

**Supersampling**: Upscales to intermediate size then downscales for anti-aliasing
**Rounding Modulus**: Prevents artifacts by ensuring clean model input dimensions
**Rescale After Model**: Guarantees exact output dimensions

#### Performance Notes
- **Memory Usage**: 3-5x with supersampling enabled
- **Execution Time**: 200-1000ms depending on settings
- **Quality**: Superior artifact prevention vs simple upscaler

---

### Luna Ultimate SD Upscale

**Category**: `Luna Collection/Image Processing`  
**Description**: AI-powered upscaling using Stable Diffusion for intelligent enhancement.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `image` | `IMAGE` | Input image | Required | Image validation |
| `model` | `MODEL` | SD model | Required | Model compatibility |
| `positive` | `STRING` | Positive prompt | "" | Max length: 200 |
| `negative` | `STRING` | Negative prompt | "" | Max length: 200 |
| `scale_by` | `FLOAT` | Scale factor | 2.0 | Range: 1.0-4.0 |
| `steps` | `INT` | Denoising steps | 20 | Range: 1-100 |
| `denoise` | `FLOAT` | Denoise strength | 0.5 | Range: 0.0-1.0 |
| `cfg` | `FLOAT` | CFG scale | 7.0 | Range: 1.0-20.0 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `upscaled_image` | `IMAGE` | AI-enhanced upscaled image |

#### Usage Tips
- Use detailed prompts for better enhancement
- Lower denoise (0.3-0.7) for subtle improvements
- Higher CFG (8-12) for more dramatic changes

---

## 🎨 Sampling & Generation Nodes

### Luna Sampler

**Category**: `Luna Collection/Sampling`  
**Description**: Advanced KSampler with adaptive parameters and performance monitoring.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model` | `MODEL` | Diffusion model | Required | Model validation |
| `positive` | `CONDITIONING` | Positive conditioning | Required | - |
| `negative` | `CONDITIONING` | Negative conditioning | Required | - |
| `latent_image` | `LATENT` | Input latent | Required | - |
| `noise_seed` | `INT` | Random seed | Random | Range: 0-2^64 |
| `steps` | `INT` | Sampling steps | 20 | Range: 1-1000 |
| `cfg` | `FLOAT` | CFG scale | 8.0 | Range: 0.0-100.0 |
| `sampler_name` | `STRING` | Sampler algorithm | `euler` | Valid sampler names |
| `scheduler` | `STRING` | Scheduler | `normal` | Valid scheduler names |
| `denoise` | `FLOAT` | Denoise strength | 1.0 | Range: 0.0-1.0 |
| `adaptive_threshold` | `FLOAT` | Adaptive sampling | 0.0 | Range: 0.0-1.0 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `latent` | `LATENT` | Generated latent |
| `performance_metrics` | `STRING` | Performance data (JSON) |

#### Advanced Features

**Adaptive Threshold**: Dynamically adjusts sampling based on image complexity
**Performance Monitoring**: Real-time execution metrics and optimization suggestions
**Validation**: Comprehensive input validation with caching

#### Performance Metrics
```json
{
  "execution_time": 2.34,
  "memory_peak": 2048,
  "throughput": 0.427,
  "optimization_suggestions": ["Consider reducing steps for faster generation"]
}
```

---

### Luna Performance Logger

**Category**: `Luna Collection/Utilities`  
**Description**: Real-time performance monitoring and logging for optimization.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `log_filename` | `STRING` | Log file path | "performance.log" | Valid file path |
| `enable_gpu_monitoring` | `BOOLEAN` | Monitor GPU | `true` | - |
| `log_interval` | `FLOAT` | Log frequency (sec) | 1.0 | Range: 0.1-60.0 |
| `max_log_size` | `INT` | Max log size (MB) | 100 | Range: 1-1000 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `performance_report` | `STRING` | Current performance metrics |

#### Logged Metrics
- Execution time per node
- Memory usage (CPU/GPU)
- Throughput statistics
- Error rates and recovery times
- Resource utilization trends

---

## 📁 Model Management Nodes

### Luna Checkpoint Loader

**Category**: `Luna Collection/Loaders`  
**Description**: Intelligent checkpoint loading with validation and caching.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `ckpt_name` | `STRING` | Checkpoint filename | Required | File exists, valid format |
| `load_clip` | `BOOLEAN` | Load CLIP model | `true` | - |
| `load_vae` | `BOOLEAN` | Load VAE | `true` | - |
| `cache_model` | `BOOLEAN` | Enable caching | `true` | - |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `model` | `MODEL` | Loaded diffusion model |
| `clip` | `CLIP` | Loaded CLIP model |
| `vae` | `VAE` | Loaded VAE |

#### Features
- **Validation**: Checks file existence and format
- **Caching**: Intelligent model caching for performance
- **Error Handling**: Graceful fallback on load failures
- **Memory Management**: Automatic cleanup of unused models

---

### Luna LoRA Stacker

**Category**: `Luna Collection/Loaders`  
**Description**: Advanced LoRA management with weighted combinations.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `model` | `MODEL` | Base model | Required | Model validation |
| `clip` | `CLIP` | Base CLIP | Required | CLIP validation |
| `lora_1` | `STRING` | First LoRA name | "" | File exists if provided |
| `strength_1` | `FLOAT` | LoRA 1 strength | 1.0 | Range: -2.0-2.0 |
| `lora_2` | `STRING` | Second LoRA name | "" | File exists if provided |
| `strength_2` | `FLOAT` | LoRA 2 strength | 1.0 | Range: -2.0-2.0 |
| `lora_3` | `STRING` | Third LoRA name | "" | File exists if provided |
| `strength_3` | `FLOAT` | LoRA 3 strength | 1.0 | Range: -2.0-2.0 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `model` | `MODEL` | Model with LoRAs applied |
| `clip` | `CLIP` | CLIP with LoRAs applied |

#### Advanced Features
- **Weighted Combinations**: Precise control over LoRA influence
- **Validation**: Checks LoRA file existence and compatibility
- **Performance**: Efficient loading and caching
- **Flexibility**: Support for up to 3 LoRAs simultaneously

---

## 📝 Text Processing Nodes

### Luna Prompt Preprocessor

**Category**: `Luna Collection/Text Processing`  
**Description**: Advanced prompt enhancement with multiple processing modes.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `text` | `STRING` | Input prompt | Required | Max length: 200 |
| `mode` | `STRING` | Processing mode | "enhance" | Options: enhance, refine, expand, condense |
| `intensity` | `FLOAT` | Processing intensity | 0.5 | Range: 0.0-1.0 |
| `preserve_keywords` | `BOOLEAN` | Keep key terms | `true` | - |
| `max_length` | `INT` | Output length limit | 150 | Range: 10-500 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `processed_text` | `STRING` | Enhanced prompt |
| `original_length` | `INT` | Original text length |
| `processed_length` | `INT` | Processed text length |

#### Processing Modes
- **Enhance**: Improve clarity and detail
- **Refine**: Clean and optimize existing prompt
- **Expand**: Add descriptive elements
- **Condense**: Reduce length while preserving meaning

---

### Luna Unified Prompt Processor

**Category**: `Luna Collection/Text Processing`  
**Description**: All-in-one text processing pipeline with multiple enhancement stages.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `text` | `STRING` | Input text | Required | Max length: 500 |
| `enable_preprocessing` | `BOOLEAN` | Enable preprocessing | `true` | - |
| `enable_enhancement` | `BOOLEAN` | Enable enhancement | `true` | - |
| `enable_styling` | `BOOLEAN` | Enable styling | `false` | - |
| `quality_level` | `STRING` | Quality preset | "balanced" | Options: fast, balanced, quality |
| `target_length` | `INT` | Desired length | 0 | Range: 0-1000 (0 = auto) |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `processed_text` | `STRING` | Fully processed text |
| `processing_stats` | `STRING` | Processing statistics (JSON) |

#### Pipeline Stages
1. **Preprocessing**: Clean and normalize input
2. **Enhancement**: Improve clarity and detail
3. **Styling**: Apply artistic styling
4. **Optimization**: Length and quality optimization

---

## 🔍 Computer Vision Nodes

### Luna MediaPipe Detailer

**Category**: `Luna Collection/Computer Vision`  
**Description**: AI-powered image segmentation using MediaPipe for precise masking.

#### Inputs
| Input | Type | Description | Default | Validation |
|-------|------|-------------|---------|------------|
| `image` | `IMAGE` | Input image | Required | Image validation |
| `model_type` | `STRING` | Detection model | "face" | Options: face, eyes, mouth, hands, person, feet, torso |
| `confidence` | `FLOAT` | Detection confidence | 0.5 | Range: 0.1-1.0 |
| `mask_padding` | `INT` | Mask padding (px) | 10 | Range: 0-100 |
| `mask_blur` | `INT` | Mask blur radius | 5 | Range: 0-50 |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `mask` | `MASK` | Generated segmentation mask |
| `detection_count` | `INT` | Number of detections found |

#### Supported Models
- **Face**: Full face detection and masking
- **Eyes**: Precise eye region segmentation
- **Mouth**: Mouth and lip area detection
- **Hands**: Hand detection and segmentation
- **Person**: Full person segmentation (background removal)
- **Feet**: Foot and ankle detection
- **Torso**: Body torso segmentation

#### Performance Notes
- **CPU/GPU**: Runs on both CPU and GPU
- **Speed**: 50-200ms depending on model complexity
- **Accuracy**: 95%+ detection accuracy with proper confidence settings

---

## 🔧 Best Practices

### Performance Optimization
1. **Enable Caching**: Use validation caching for repeated operations
2. **Batch Processing**: Process multiple images together when possible
3. **Model Selection**: Choose appropriate model complexity for your needs
4. **Memory Management**: Monitor GPU memory usage with performance logger

### Error Handling
1. **Validation Errors**: Check input parameters before processing
2. **Model Loading**: Verify model files exist and are compatible
3. **Memory Issues**: Monitor system resources during processing
4. **Fallback Options**: Use graceful degradation when features fail

### Workflow Integration
1. **Node Ordering**: Place validation nodes early in your workflow
2. **Performance Monitoring**: Use Luna Performance Logger for optimization
3. **Error Recovery**: Implement fallback paths for critical operations
4. **Resource Management**: Clean up unused models and cache regularly

---

## 🐛 Troubleshooting

### Common Issues

**Validation Errors**
```
Solution: Check input parameters match expected types and ranges
Check: Ensure all required inputs are provided
```

**Memory Issues**
```
Solution: Reduce batch sizes or enable caching
Check: Monitor GPU memory usage with performance logger
```

**Model Loading Failures**
```
Solution: Verify model files exist and are not corrupted
Check: Use Luna Checkpoint Loader for validation
```

**Performance Degradation**
```
Solution: Enable caching and monitor with performance logger
Check: Review workflow for optimization opportunities
```

---

## 📊 Performance Benchmarks

| Node | Avg Time | Memory Usage | Throughput |
|------|----------|--------------|------------|
| Luna Simple Upscaler | 150ms | 2x input | 6.7 img/sec |
| Luna Advanced Upscaler | 300ms | 3-5x input | 3.3 img/sec |
| Luna Sampler | 2-10s | Model dependent | 0.1-0.5 img/sec |
| Luna MediaPipe Detailer | 100ms | 1.2x input | 10 img/sec |
| Validation System | <1ms | <1MB | 471K ops/sec |

*Benchmarks performed on RTX 4070 with 32GB system RAM*

---

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Development Guide](development.md)
- [API Reference](api-reference.md)
- [Troubleshooting](troubleshooting.md)

---

*For the latest updates and additional examples, visit the [Luna Collection Repository](https://github.com/LSDJesus/ComfyUI-Luna-Collection).*"