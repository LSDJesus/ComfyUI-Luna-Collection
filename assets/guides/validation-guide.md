# ✅ Validation System Guide

This comprehensive guide covers the Luna Collection's advanced validation system, including input validation, error handling, caching strategies, and best practices for robust workflow execution.

## Table of Contents

- [Overview](#-overview)
- [Core Validation Components](#-core-validation-components)
- [Input Validation](#-input-validation)
- [Error Handling](#-error-handling)
- [Caching System](#-caching-system)
- [Performance Optimization](#-performance-optimization)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 📋 Overview

The Luna Collection features a comprehensive validation system designed to ensure robust, reliable operation with intelligent error handling and performance optimization.

### Key Features

- **Multi-layer Validation**: Input, runtime, and output validation
- **Intelligent Caching**: LRU caching with automatic invalidation
- **Performance Monitoring**: Real-time validation metrics and optimization
- **Graceful Degradation**: Fallback mechanisms for validation failures
- **Type Safety**: Full type checking with Pydantic V2
- **Error Recovery**: Automatic retry mechanisms with exponential backoff

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input         │───▶│   Validation     │───▶│   Processing    │
│   Validation    │    │   Engine         │    │   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Caching       │    │   Error          │    │   Performance   │
│   System        │    │   Handling       │    │   Monitoring    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🏗️ Core Validation Components

### LunaInputValidator

The main validation engine that handles all input validation with caching and performance monitoring.

#### Key Methods

```python
class LunaInputValidator:
    def validate_node_input(self, node_input: dict, node_type: str) -> ValidationResult:
        """Validate input parameters for a specific node type."""

    def validate_image_input(self, image: np.ndarray) -> ValidationResult:
        """Validate image data format and dimensions."""

    def validate_model_compatibility(self, model_path: str, expected_type: str) -> ValidationResult:
        """Validate model file compatibility."""

    def get_validation_cache_stats(self) -> dict:
        """Get current cache performance statistics."""
```

#### Validation Types

| Validation Type | Purpose | Cache TTL | Performance Impact |
|----------------|---------|-----------|-------------------|
| `node_input` | Node parameter validation | 30 min | Low |
| `image_format` | Image data validation | 5 min | Medium |
| `model_compat` | Model compatibility check | 60 min | High |
| `file_exists` | File existence validation | 10 min | Low |

### ValidationResult

Standardized result object for all validation operations.

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    performance_metrics: dict
    cache_hit: bool
    validation_time: float
```

---

## 🔍 Input Validation

### Node Input Validation

#### Parameter Validation Rules

```python
# Example validation rules for LunaSampler
sampler_validation = {
    "steps": {
        "type": "int",
        "range": [1, 1000],
        "default": 20,
        "description": "Number of sampling steps"
    },
    "cfg": {
        "type": "float",
        "range": [0.0, 100.0],
        "default": 8.0,
        "description": "Classifier-free guidance scale"
    },
    "denoise": {
        "type": "float",
        "range": [0.0, 1.0],
        "default": 1.0,
        "description": "Denoise strength"
    }
}
```

#### Validation Categories

**Required Parameters**
- Must be present in input
- Cannot be None or empty
- Must match expected type

**Optional Parameters**
- Can be omitted (use defaults)
- Validate if present
- Type conversion if needed

**Conditional Parameters**
- Required based on other parameter values
- Example: `scale_by` required when `upscale_model` is set

### Image Validation

#### Image Format Validation
```python
image_validation_rules = {
    "supported_formats": ["RGB", "RGBA", "L", "P"],
    "max_dimensions": [8192, 8192],
    "min_dimensions": [32, 32],
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_channels": [1, 3, 4]  # Grayscale, RGB, RGBA
}
```

#### Validation Checks
- **Format Compatibility**: Check PIL/numpy array format
- **Dimension Limits**: Prevent excessive memory usage
- **Channel Validation**: Ensure correct color channels
- **Data Type**: Validate numpy dtypes
- **Memory Estimation**: Calculate expected memory usage

### Model Validation

#### Checkpoint Validation
```python
checkpoint_validation = {
    "file_extensions": [".ckpt", ".safetensors", ".pth"],
    "min_file_size": 1024 * 1024,  # 1MB
    "max_file_size": 10 * 1024 * 1024 * 1024,  # 10GB
    "required_keys": ["state_dict", "epoch", "global_step"],
    "architecture_check": True
}
```

#### LoRA Validation
```python
lora_validation = {
    "file_extensions": [".ckpt", ".safetensors", ".pt"],
    "max_rank": 1024,
    "supported_formats": ["lycoris", "standard"],
    "compatibility_check": True
}
```

---

## 🚨 Error Handling

### Error Classification

#### Validation Errors
```python
class ValidationError(Exception):
    """Base validation error with detailed context."""
    def __init__(self, message: str, error_code: str, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
```

#### Error Types

| Error Type | Code | Description | Recovery Action |
|------------|------|-------------|-----------------|
| `INVALID_TYPE` | V001 | Parameter type mismatch | Type conversion attempt |
| `OUT_OF_RANGE` | V002 | Value outside allowed range | Clamp to nearest valid value |
| `MISSING_REQUIRED` | V003 | Required parameter missing | Use default or prompt user |
| `INVALID_FORMAT` | V004 | Data format incorrect | Format conversion attempt |
| `FILE_NOT_FOUND` | V005 | Referenced file missing | Search alternatives or prompt |
| `MEMORY_LIMIT` | V006 | Memory usage too high | Reduce batch size or resolution |

### Error Recovery Strategies

#### Automatic Recovery
1. **Type Conversion**: Attempt safe type conversion
2. **Value Clamping**: Clamp out-of-range values to limits
3. **Default Fallback**: Use sensible defaults for missing parameters
4. **Format Conversion**: Convert between compatible formats

#### Graceful Degradation
```python
def graceful_validation_failure(error: ValidationError) -> ValidationResult:
    """Attempt to recover from validation failures gracefully."""
    recovery_strategies = {
        "INVALID_TYPE": try_type_conversion,
        "OUT_OF_RANGE": clamp_value,
        "MISSING_REQUIRED": use_default_value,
        "INVALID_FORMAT": convert_format
    }

    strategy = recovery_strategies.get(error.error_code)
    if strategy:
        return strategy(error)
    else:
        return ValidationResult(
            is_valid=False,
            errors=[error.message],
            warnings=["No recovery strategy available"],
            suggestions=["Check input parameters manually"]
        )
```

### Error Reporting

#### Detailed Error Messages
```python
error_templates = {
    "INVALID_TYPE": "Parameter '{param}' must be {expected_type}, got {actual_type}",
    "OUT_OF_RANGE": "Parameter '{param}' value {value} is outside range [{min}, {max}]",
    "MISSING_REQUIRED": "Required parameter '{param}' is missing",
    "INVALID_FORMAT": "Image format '{format}' not supported. Supported: {supported}"
}
```

#### Error Context
```python
error_context = {
    "node_type": "LunaSampler",
    "parameter": "steps",
    "provided_value": "not_a_number",
    "expected_type": "int",
    "validation_time": 0.023,
    "cache_hit": False
}
```

---

## 💾 Caching System

### LRU Cache Implementation

#### Cache Configuration
```python
validation_cache_config = {
    "max_size": 1000,  # Maximum cached validations
    "ttl": 1800,       # 30 minutes time-to-live
    "compression": True,  # Compress cached data
    "persistent": False,  # Don't persist across sessions
    "memory_limit": "500MB"  # Memory usage limit
}
```

#### Cache Key Generation
```python
def generate_cache_key(node_input: dict, node_type: str) -> str:
    """Generate deterministic cache key from input parameters."""
    # Sort keys for consistency
    sorted_input = json.dumps(node_input, sort_keys=True)
    key_components = [node_type, sorted_input]

    # Add version for cache invalidation
    key_components.append(str(CACHE_VERSION))

    return hashlib.sha256("|".join(key_components).encode()).hexdigest()
```

#### Cache Performance Metrics
```python
cache_stats = {
    "total_requests": 15432,
    "cache_hits": 12876,
    "cache_misses": 2556,
    "hit_rate": 83.4,
    "avg_lookup_time": 0.0012,  # seconds
    "memory_usage": "234MB",
    "eviction_count": 123
}
```

### Cache Invalidation

#### Automatic Invalidation
- **TTL-based**: Time-based expiration
- **LRU**: Least recently used eviction
- **Size-based**: Memory limit enforcement
- **Version-based**: Invalidation on code changes

#### Manual Invalidation
```python
def invalidate_cache(pattern: str = None):
    """Manually invalidate cache entries."""
    if pattern:
        # Invalidate entries matching pattern
        cache.delete_matching(pattern)
    else:
        # Clear entire cache
        cache.clear()
```

---

## ⚡ Performance Optimization

### Validation Performance Targets

| Validation Type | Target Time | Cache Hit Rate | Memory Usage |
|----------------|-------------|----------------|--------------|
| Node Input | <5ms | >90% | <1MB |
| Image Format | <10ms | >85% | <5MB |
| Model Compat | <50ms | >95% | <10MB |
| File Exists | <2ms | >80% | <1MB |

### Optimization Techniques

#### Parallel Validation
```python
async def validate_parallel(validations: List[ValidationTask]) -> List[ValidationResult]:
    """Execute multiple validations in parallel."""
    tasks = [asyncio.create_task(validate_single(task)) for task in validations]
    return await asyncio.gather(*tasks)
```

#### Batch Processing
```python
def validate_batch(inputs: List[dict], node_type: str) -> List[ValidationResult]:
    """Validate multiple inputs efficiently."""
    # Pre-compile validation rules
    rules = compile_validation_rules(node_type)

    # Batch validation with shared context
    results = []
    for input_data in inputs:
        result = validate_with_cache(input_data, rules)
        results.append(result)

    return results
```

#### Memory Optimization
- **Lazy Loading**: Load validation rules on demand
- **Object Pooling**: Reuse validation objects
- **Data Compression**: Compress cached validation results
- **Memory Mapping**: Use memory-mapped files for large rule sets

---

## ⚙️ Configuration

### Validation Configuration

#### Basic Configuration
```json
{
  "validation": {
    "enabled": true,
    "strict_mode": false,
    "cache_enabled": true,
    "parallel_processing": true,
    "max_concurrent_validations": 4
  }
}
```

#### Advanced Configuration
```json
{
  "validation": {
    "cache": {
      "max_size": 1000,
      "ttl_seconds": 1800,
      "compression": true,
      "memory_limit_mb": 500
    },
    "performance": {
      "target_latency_ms": 10,
      "batch_size": 8,
      "parallel_workers": 4
    },
    "error_handling": {
      "max_retries": 3,
      "retry_delay_ms": 100,
      "graceful_degradation": true
    }
  }
}
```

### Runtime Configuration

#### Dynamic Configuration Updates
```python
def update_validation_config(new_config: dict):
    """Update validation configuration at runtime."""
    global validation_config
    validation_config.update(new_config)

    # Reinitialize cache with new settings
    if 'cache' in new_config:
        reinitialize_cache(new_config['cache'])

    # Update performance settings
    if 'performance' in new_config:
        update_performance_settings(new_config['performance'])
```

---

## 🔧 Troubleshooting

### Common Validation Issues

#### Cache-Related Issues

**High Memory Usage**
```
Symptoms: Memory usage growing over time
Solutions:
- Reduce cache max_size
- Enable compression
- Set memory_limit_mb
- Clear cache manually
```

**Low Cache Hit Rate**
```
Symptoms: Many cache misses, slow validation
Solutions:
- Increase cache max_size
- Review cache TTL settings
- Check cache key generation
- Enable persistent cache
```

**Stale Cache Data**
```
Symptoms: Validation results don't match current state
Solutions:
- Decrease cache TTL
- Implement version-based invalidation
- Clear cache on configuration changes
- Use manual invalidation
```

#### Performance Issues

**Slow Validation**
```
Symptoms: Validation taking too long
Solutions:
- Enable parallel processing
- Increase batch size
- Optimize cache settings
- Review validation rules
```

**Memory Leaks**
```
Symptoms: Memory usage increasing over time
Solutions:
- Check for circular references
- Enable garbage collection
- Monitor object lifecycle
- Use weak references
```

#### Error Handling Issues

**Too Many False Positives**
```
Symptoms: Valid inputs being rejected
Solutions:
- Review validation rules
- Adjust error thresholds
- Enable graceful degradation
- Update validation logic
```

**Recovery Not Working**
```
Symptoms: Validation failures not recovering
Solutions:
- Check recovery strategies
- Review error context
- Enable debug logging
- Update recovery logic
```

### Debug Tools

#### Validation Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('luna.validation').setLevel(logging.DEBUG)

# Enable performance profiling
validation_config['debug'] = {
    'performance_profiling': True,
    'detailed_logging': True,
    'cache_debugging': True
}
```

#### Validation Statistics
```python
stats = validator.get_validation_stats()
print(f"""
Validation Statistics:
Total Validations: {stats['total']}
Cache Hit Rate: {stats['hit_rate']:.1f}%
Average Time: {stats['avg_time']:.3f}ms
Memory Usage: {stats['memory_usage']}MB
Error Rate: {stats['error_rate']:.2f}%
""")
```

---

## 📊 Monitoring & Metrics

### Key Performance Indicators

#### Validation Metrics
- **Throughput**: Validations per second
- **Latency**: Average validation time
- **Hit Rate**: Cache effectiveness
- **Error Rate**: Validation failure percentage
- **Memory Usage**: Current memory consumption

#### Cache Metrics
- **Size**: Current cache entries
- **Hit/Miss Ratio**: Cache effectiveness
- **Eviction Rate**: Cache turnover
- **Memory Usage**: Cache memory consumption

### Monitoring Integration

#### Prometheus Metrics
```python
# Export metrics for monitoring
validation_metrics = {
    "validation_requests_total": Counter(),
    "validation_duration_seconds": Histogram(),
    "validation_errors_total": Counter(),
    "cache_hits_total": Counter(),
    "cache_misses_total": Counter()
}
```

#### Health Checks
```python
def validation_health_check() -> dict:
    """Comprehensive health check for validation system."""
    return {
        "status": "healthy" if validator.is_healthy() else "unhealthy",
        "cache_status": cache.get_status(),
        "performance_metrics": get_performance_metrics(),
        "error_rate": calculate_error_rate(),
        "last_validation_time": get_last_activity_time()
    }
```

---

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [Performance Guide](performance-guide.md)
- [Node Reference](node-reference.md)
- [API Reference](api-reference.md)
- [Troubleshooting](troubleshooting.md)

---

## 📞 Support

For validation system support:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- Join the [Discord Community](https://discord.gg/luna-collection)

---

*Validation system performance may vary based on input complexity, cache configuration, and system resources. Monitor your specific use case for optimal configuration.*