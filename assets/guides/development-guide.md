# 🛠️ Development Guide

This comprehensive guide covers development practices, coding standards, contribution guidelines, and best practices for working with the Luna Collection codebase.

## Table of Contents

- [Getting Started](#-getting-started)
- [Development Environment](#-development-environment)
- [Code Standards](#-code-standards)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Architecture](#-architecture)
- [Debugging](#-debugging)
- [Performance](#-performance)

---

## 🚀 Getting Started

### Prerequisites

#### System Requirements
- **Python**: 3.8 or higher
- **Git**: 2.30 or higher
- **VS Code**: Latest version (recommended)
- **ComfyUI**: Latest stable version

#### Development Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Key development packages
pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    black>=23.0.0 \
    pre-commit>=3.0.0
```

### Repository Setup

#### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run initial tests
pytest
```

#### VS Code Configuration
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

---

## 🏗️ Development Environment

### Project Structure

```
ComfyUI-Luna-Collection/
├── luna_collection/           # Main package
│   ├── __init__.py
│   ├── core/                  # Core functionality
│   ├── nodes/                 # ComfyUI nodes
│   ├── validation/            # Validation system
│   ├── utils/                 # Utilities
│   └── types/                 # Type definitions
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Performance tests
├── assets/                    # Documentation and samples
│   ├── guides/               # User guides
│   ├── samples/              # Sample workflows
│   └── prompts/              # Prompt templates
├── scripts/                   # Development scripts
├── docs/                      # Documentation
└── requirements-dev.txt       # Development dependencies
```

### Development Workflow

#### Daily Development
```bash
# Start development session
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/new-node

# Make changes with tests
# ... development work ...

# Run tests and linting
pytest
flake8 .
mypy .

# Commit changes
git add .
git commit -m "feat: add new awesome node"

# Push and create PR
git push origin feature/new-node
```

#### Code Review Process
1. **Create PR**: Push feature branch and create pull request
2. **Automated Checks**: Wait for CI to pass (linting, tests, coverage)
3. **Code Review**: Address reviewer feedback
4. **Merge**: Squash merge to develop after approval

---

## 📏 Code Standards

### Python Style Guide

#### PEP 8 Compliance
```python
# Good: Proper spacing and naming
def validate_node_input(input_data: dict, node_type: str) -> ValidationResult:
    """Validate input parameters for a ComfyUI node."""
    if not isinstance(input_data, dict):
        raise ValueError("Input data must be a dictionary")

    # Validate required parameters
    required_params = ["model", "prompt"]
    for param in required_params:
        if param not in input_data:
            raise ValueError(f"Missing required parameter: {param}")

    return ValidationResult(is_valid=True)

# Bad: Poor spacing and naming
def validate(input,dict,type):
    if not isinstance(input,dict):
        raise ValueError("bad input")
    return True
```

#### Type Hints
```python
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel

# Use proper type hints
def process_image(
    image: np.ndarray,
    scale_factor: float,
    output_format: str = "RGB"
) -> np.ndarray:
    """Process image with scaling and format conversion."""
    pass

# Use generics for collections
def batch_process(
    images: List[np.ndarray],
    config: Dict[str, Any]
) -> List[np.ndarray]:
    """Process multiple images with configuration."""
    pass

# Use Optional for nullable values
def load_model(
    model_path: str,
    device: Optional[str] = None
) -> torch.nn.Module:
    """Load model with optional device specification."""
    pass
```

### Naming Conventions

#### Classes and Types
```python
# PascalCase for classes
class LunaInputValidator:
    """Validates input parameters for Luna nodes."""
    pass

class ValidationResult:
    """Result of validation operation."""
    pass

# Type aliases
ImageArray = np.ndarray
ModelConfig = Dict[str, Any]
NodeInputs = Dict[str, Any]
```

#### Functions and Variables
```python
# snake_case for functions and variables
def validate_node_input(input_data: dict) -> bool:
    """Validate input data for a node."""
    pass

def process_batch(images: List[np.ndarray]) -> List[np.ndarray]:
    """Process a batch of images."""
    pass

# Constants in UPPER_CASE
MAX_IMAGE_SIZE = 8192
DEFAULT_BATCH_SIZE = 4
VALIDATION_CACHE_TTL = 300
```

#### Private Members
```python
class ImageProcessor:
    """Process images with various transformations."""

    def __init__(self):
        self._cache = {}  # Private attribute
        self.__internal_state = None  # Name-mangled attribute

    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate image format (internal method)."""
        pass

    def __cleanup_cache(self):
        """Clean up internal cache (name-mangled method)."""
        pass
```

### Documentation Standards

#### Docstrings
```python
def upscale_image(
    image: np.ndarray,
    scale_factor: float,
    method: str = "lanczos"
) -> np.ndarray:
    """
    Upscale an image using the specified method.

    This function provides high-quality image upscaling with multiple
    interpolation methods and automatic optimization based on the
    target scale factor.

    Args:
        image: Input image as numpy array (H, W, C) format
        scale_factor: Scaling factor (1.0-8.0)
        method: Interpolation method ('nearest', 'linear', 'cubic', 'lanczos')

    Returns:
        Upscaled image as numpy array

    Raises:
        ValueError: If scale_factor is outside valid range
        TypeError: If image is not a numpy array

    Examples:
        >>> img = np.random.rand(256, 256, 3)
        >>> upscaled = upscale_image(img, 2.0, 'lanczos')
        >>> upscaled.shape
        (512, 512, 3)
    """
    pass
```

#### Module Documentation
```python
"""
Luna Collection Image Processing Module.

This module provides advanced image processing capabilities for the
Luna Collection, including upscaling, filtering, and format conversion.

Classes:
    LunaImageProcessor: Main image processing class
    UpscaleEngine: Handles image upscaling operations
    FilterEngine: Applies various image filters

Functions:
    upscale_image: Upscale single image
    batch_upscale: Upscale multiple images
    apply_filter: Apply image filter

Examples:
    Basic usage:

    >>> from luna_collection.image_processing import LunaImageProcessor
    >>> processor = LunaImageProcessor()
    >>> result = processor.upscale(image, scale_factor=2.0)
"""

__version__ = "1.0.0"
__author__ = "Luna Collection Team"
```

---

## 🧪 Testing

### Test Structure

#### Unit Tests
```python
# tests/unit/test_validation.py
import pytest
from luna_collection.validation import LunaInputValidator, ValidationResult

class TestLunaInputValidator:
    """Test cases for LunaInputValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance for tests."""
        return LunaInputValidator()

    def test_valid_input_validation(self, validator):
        """Test validation of valid input."""
        input_data = {
            "steps": 20,
            "cfg": 8.0,
            "denoise": 1.0
        }

        result = validator.validate_node_input(input_data, "LunaSampler")

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.performance_metrics["validation_time"] < 0.01

    def test_invalid_input_validation(self, validator):
        """Test validation of invalid input."""
        input_data = {
            "steps": -1,  # Invalid negative value
            "cfg": 8.0,
            "denoise": 1.0
        }

        result = validator.validate_node_input(input_data, "LunaSampler")

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "steps" in str(result.errors[0])

    @pytest.mark.parametrize("invalid_steps", [-1, 0, 1001])
    def test_steps_validation_range(self, validator, invalid_steps):
        """Test steps parameter range validation."""
        input_data = {
            "steps": invalid_steps,
            "cfg": 8.0,
            "denoise": 1.0
        }

        result = validator.validate_node_input(input_data, "LunaSampler")

        assert result.is_valid is False
        assert any("steps" in str(error) for error in result.errors)
```

#### Integration Tests
```python
# tests/integration/test_workflow_execution.py
import pytest
from luna_collection.core import WorkflowExecutor
from luna_collection.nodes import LunaSampler, LunaImageLoader

class TestWorkflowExecution:
    """Integration tests for workflow execution."""

    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing."""
        return {
            "nodes": [
                {
                    "id": "image_loader",
                    "type": "LunaImageLoader",
                    "inputs": {
                        "image_path": "test_image.png"
                    }
                },
                {
                    "id": "sampler",
                    "type": "LunaSampler",
                    "inputs": {
                        "model": "test_model",
                        "steps": 10,
                        "cfg": 7.0
                    },
                    "connections": {
                        "image": "image_loader.output"
                    }
                }
            ]
        }

    def test_workflow_execution_success(self, sample_workflow):
        """Test successful workflow execution."""
        executor = WorkflowExecutor()

        result = executor.execute_workflow(sample_workflow)

        assert result.success is True
        assert "sampler" in result.outputs
        assert result.execution_time > 0
        assert result.execution_time < 60.0  # Should complete within 1 minute

    def test_workflow_error_handling(self):
        """Test workflow error handling."""
        invalid_workflow = {
            "nodes": [
                {
                    "id": "invalid_node",
                    "type": "NonExistentNode",
                    "inputs": {}
                }
            ]
        }

        executor = WorkflowExecutor()

        result = executor.execute_workflow(invalid_workflow)

        assert result.success is False
        assert len(result.errors) > 0
        assert "NonExistentNode" in str(result.errors[0])
```

#### Performance Tests
```python
# tests/performance/test_inference_performance.py
import pytest
import time
from luna_collection.nodes import LunaSampler

class TestInferencePerformance:
    """Performance tests for inference operations."""

    @pytest.fixture
    def sampler(self):
        """Create sampler instance for performance testing."""
        return LunaSampler()

    def test_inference_speed_baseline(self, sampler, benchmark):
        """Benchmark baseline inference speed."""
        def run_inference():
            return sampler.sample(
                model="test_model",
                prompt="test prompt",
                steps=20,
                cfg=8.0
            )

        result = benchmark(run_inference)

        # Performance assertions
        assert result.stats.mean < 5.0  # Should complete in < 5 seconds
        assert result.stats.stddev < 0.5  # Should be consistent

        # Log performance metrics
        print(f"""
        Performance Results:
        Mean: {result.stats.mean:.3f}s
        StdDev: {result.stats.stddev:.3f}s
        Min: {result.stats.min:.3f}s
        Max: {result.stats.max:.3f}s
        """)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_processing_performance(self, sampler, benchmark, batch_size):
        """Test batch processing performance scaling."""
        prompts = [f"test prompt {i}" for i in range(batch_size)]

        def run_batch_inference():
            return sampler.batch_sample(
                model="test_model",
                prompts=prompts,
                steps=20,
                cfg=8.0
            )

        result = benchmark(run_batch_inference)

        # Calculate throughput
        throughput = batch_size / result.stats.mean

        print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")

        # Performance should scale reasonably with batch size
        assert throughput > 0.1  # At least 0.1 samples per second
```

### Test Coverage

#### Coverage Configuration
```ini
# .coveragerc
[run]
source = luna_collection
omit =
    */tests/*
    */migrations/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov
```

#### Coverage Goals
- **Overall Coverage**: >85%
- **Core Modules**: >90%
- **Node Classes**: >80%
- **Validation System**: >95%
- **Utility Functions**: >75%

---

## 🤝 Contributing

### Contribution Guidelines

#### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch from `develop`
3. **Make** your changes with tests
4. **Run** the full test suite locally
5. **Update** documentation if needed
6. **Commit** with conventional commit messages
7. **Push** your branch and create a PR
8. **Address** review feedback
9. **Merge** after approval

#### Commit Message Format
```bash
# Format: type(scope): description

# Examples
feat(validation): add new input validation for sampler nodes
fix(performance): optimize memory usage in image processing
docs(readme): update installation instructions
test(validation): add comprehensive test coverage for edge cases
refactor(core): simplify workflow execution logic
```

#### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Type hints are complete and correct
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Security implications are reviewed
- [ ] Breaking changes are documented

### Issue Tracking

#### Bug Reports
```markdown
## Bug Report

**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.10]
- ComfyUI: [e.g., v1.2.3]
- Luna Collection: [e.g., v1.0.0]

**Additional Context**
Any other relevant information
```

#### Feature Requests
```markdown
## Feature Request

**Problem**
Description of the problem this feature would solve

**Solution**
Description of the proposed solution

**Alternatives**
Alternative solutions considered

**Additional Context**
Mockups, examples, or additional information
```

---

## 🏛️ Architecture

### Core Architecture Principles

#### Separation of Concerns
```python
# Good: Clear separation of responsibilities
class NodeManager:
    """Manages node registration and lifecycle."""
    pass

class ValidationEngine:
    """Handles input validation and error reporting."""
    pass

class ExecutionEngine:
    """Manages workflow execution and resource allocation."""
    pass

# Bad: Mixed responsibilities
class NodeHandler:
    """Handles everything - bad design."""
    def validate_input(self): pass
    def execute_node(self): pass
    def manage_resources(self): pass
    def log_metrics(self): pass
```

#### Dependency Injection
```python
# Good: Dependency injection for testability
class LunaSampler:
    def __init__(self, validator=None, cache=None):
        self.validator = validator or LunaInputValidator()
        self.cache = cache or ValidationCache()

# Bad: Hard-coded dependencies
class LunaSampler:
    def __init__(self):
        self.validator = LunaInputValidator()  # Can't mock for testing
```

### Design Patterns

#### Factory Pattern
```python
class NodeFactory:
    """Factory for creating ComfyUI nodes."""

    @staticmethod
    def create_node(node_type: str, **kwargs) -> BaseNode:
        """Create a node instance based on type."""
        node_classes = {
            "LunaSampler": LunaSampler,
            "LunaUpscaler": LunaUpscaler,
            "LunaValidator": LunaValidator
        }

        node_class = node_classes.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")

        return node_class(**kwargs)
```

#### Strategy Pattern
```python
from abc import ABC, abstractmethod

class UpscaleStrategy(ABC):
    """Abstract base class for upscaling strategies."""

    @abstractmethod
    def upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Upscale the given image."""
        pass

class LanczosUpscaler(UpscaleStrategy):
    """Lanczos upscaling implementation."""
    def upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        # Lanczos implementation
        pass

class BicubicUpscaler(UpscaleStrategy):
    """Bicubic upscaling implementation."""
    def upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        # Bicubic implementation
        pass
```

#### Observer Pattern
```python
class PerformanceMonitor:
    """Monitor performance metrics."""

    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        """Add performance observer."""
        self._observers.append(observer)

    def notify_observers(self, metrics: dict):
        """Notify all observers of new metrics."""
        for observer in self._observers:
            observer.update(metrics)

class MetricsLogger:
    """Log performance metrics."""

    def update(self, metrics: dict):
        """Update with new metrics."""
        print(f"Performance metrics: {metrics}")
```

---

## 🔍 Debugging

### Debug Tools

#### Logging Configuration
```python
# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    """Setup comprehensive logging configuration."""

    # Create logger
    logger = logging.getLogger('luna_collection')
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Usage
logger = setup_logging(level=logging.DEBUG, log_file='debug.log')
```

#### Debug Decorators
```python
import time
import functools
from typing import Callable, Any

def debug_performance(func: Callable) -> Callable:
    """Decorator to log function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper

def debug_validation(func: Callable) -> Callable:
    """Decorator to debug validation operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.debug(f"Validation: {func.__name__}")
        logger.debug(f"Input args: {len(args)} positional, {len(kwargs)} keyword")

        result = func(*args, **kwargs)

        logger.debug(f"Validation result: {result}")
        return result

    return wrapper
```

### Debug Commands

#### Performance Profiling
```bash
# Profile specific function
python -m cProfile -s time your_script.py

# Memory profiling
python -m memory_profiler your_script.py

# Line-by-line profiling
python -c "
import line_profiler
profile = line_profiler.LineProfiler()
profile.add_function(your_function)
profile.run('your_function()')
profile.print_stats()
"
```

#### Debug Scripts
```python
# debug_workflow.py
import logging
from luna_collection.core import WorkflowExecutor

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def debug_workflow_execution(workflow_path: str):
    """Debug workflow execution with detailed logging."""
    executor = WorkflowExecutor()

    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # Execute with debugging
    logger.info("Starting workflow execution")
    result = executor.execute_workflow(workflow)

    logger.info(f"Workflow completed: {result.success}")
    if not result.success:
        logger.error(f"Errors: {result.errors}")

    return result

if __name__ == "__main__":
    debug_workflow_execution("debug_workflow.json")
```

---

## ⚡ Performance

### Performance Best Practices

#### Memory Management
```python
import gc
import psutil
from contextlib import contextmanager

@contextmanager
def memory_monitor():
    """Monitor memory usage during execution."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        logger.info(f"Memory delta: {memory_delta:.2f} MB")
        if memory_delta > 100:  # Log significant memory increases
            logger.warning(f"High memory usage: {memory_delta:.2f} MB increase")

def optimize_memory_usage():
    """Optimize memory usage."""
    # Force garbage collection
    gc.collect()

    # Clear caches if memory is low
    if psutil.virtual_memory().percent > 80:
        clear_validation_cache()
        clear_model_cache()
```

#### CPU Optimization
```python
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_processing(items: list, func: callable, max_workers: int = None):
    """Process items in parallel."""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))

    return results

def batch_processing(items: list, batch_size: int = 4):
    """Process items in batches to control memory usage."""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)

        # Memory cleanup between batches
        gc.collect()

    return results
```

#### GPU Optimization
```python
import torch

def optimize_gpu_usage():
    """Optimize GPU memory and computation."""
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set memory allocator settings
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

def gpu_memory_cleanup():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### Performance Monitoring

#### Metrics Collection
```python
import time
from collections import defaultdict
from contextlib import contextmanager

class PerformanceMonitor:
    """Monitor performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)

    @contextmanager
    def measure(self, operation: str):
        """Measure execution time of an operation."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            self.metrics[operation].append({
                'time': execution_time,
                'memory_delta': memory_delta,
                'timestamp': end_time
            })

    def get_stats(self, operation: str) -> dict:
        """Get performance statistics for an operation."""
        if operation not in self.metrics:
            return {}

        times = [m['time'] for m in self.metrics[operation]]
        memory_deltas = [m['memory_delta'] for m in self.metrics[operation]]

        return {
            'count': len(times),
            'avg_time': sum(times) / len(times),
            'max_time': max(times),
            'min_time': min(times),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'total_time': sum(times)
        }

# Usage
monitor = PerformanceMonitor()

with monitor.measure('image_processing'):
    process_image(image)

stats = monitor.get_stats('image_processing')
print(f"Average processing time: {stats['avg_time']:.3f}s")
```

---

## 🔗 Related Documentation

- [Installation Guide](installation.md)
- [Node Reference](node-reference.md)
- [Performance Guide](performance-guide.md)
- [Validation Guide](validation-guide.md)
- [CI/CD Guide](ci-cd-guide.md)

---

## 📞 Support

For development support:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- Join the [Discord Community](https://discord.gg/luna-collection)

---

*This development guide is continuously updated. Please check for the latest version and contribute improvements through pull requests.*