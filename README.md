# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**A comprehensive suite of production-ready ComfyUI nodes engineered for power, flexibility, and efficiency.**

Luna Collection represents the pinnacle of collaborative development between human creativity and AI precision. Each node is meticulously crafted to be a clean, powerful, and intuitive component in your creative workflow.

---

## ✨ Features

### 🔧 **Core Capabilities**
- **Advanced Upscaling**: Professional-grade image upscaling with artifact prevention
- **Intelligent Sampling**: KSampler with adaptive parameters and performance monitoring
- **Model Management**: Comprehensive checkpoint and LoRA loading with validation
- **Text Processing**: Unified prompt processing with multiple enhancement modes
- **Performance Monitoring**: Real-time performance tracking and optimization
- **Validation System**: Robust input validation with graceful degradation

### 🛡️ **Quality Assurance**
- **Comprehensive Testing**: 41+ automated tests covering all functionality
- **Performance Benchmarking**: Sub-millisecond validation with caching
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Type Safety**: Full Pylance compatibility with proper type annotations
- **Error Handling**: Graceful degradation when validation system is unavailable

### 📊 **Performance**
- **⚡ Sub-millisecond validation** (471K ops/sec throughput)
- **💾 Minimal memory overhead** (0.0MB for 1000+ cached validations)
- **🔄 Intelligent caching** with LRU eviction and 50% performance improvement
- **🎯 Memory efficient** with automatic cleanup and resource management

---

## 🚀 Installation

### Prerequisites
- ComfyUI (latest version recommended)
- Python 3.8+
- PyTorch with CUDA support (optional, for GPU acceleration)

### Quick Install
1. **Clone the repository:**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
   ```

2. **Install dependencies:**
   ```bash
   cd ComfyUI-Luna-Collection
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI**

The nodes will be available under the **`Luna Collection`** category.

### Optional: Performance Testing Setup
For development and performance monitoring:
```bash
pip install -r requirements-performance.txt
python run_performance_tests.py
```

---

## 🎯 Node Categories

### 🖼️ **Image Processing**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Simple Upscaler** | Clean, lightweight upscaling | Model-based scaling, resampling options |
| **Luna Advanced Upscaler** | Professional-grade upscaling | Supersampling, rounding modulus, artifact prevention |
| **Luna Ultimate SD Upscale** | AI-powered upscaling | Stable Diffusion integration, quality enhancement |

### 🎨 **Sampling & Generation**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Sampler** | Advanced KSampler | Adaptive parameters, performance monitoring, validation |
| **Luna Performance Logger** | Real-time performance tracking | Execution time, memory usage, throughput metrics |

### 📁 **Model Management**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Checkpoint Loader** | Intelligent model loading | Validation, caching, error handling |
| **Luna LoRA Stacker** | Multi-LoRA management | Weighted combinations, validation |
| **Luna Embedding Manager** | Textual inversion support | Batch processing, validation |

### 📝 **Text Processing**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Prompt Preprocessor** | Advanced prompt enhancement | Multiple processing modes, validation |
| **Luna Text Processor** | Unified text processing | Length control, content filtering |
| **Luna Unified Prompt Processor** | All-in-one text enhancement | Combined preprocessing pipeline |

### 🔍 **Computer Vision**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna MediaPipe Detailer** | AI-powered segmentation | Face, pose, hand detection |
| **Luna Face Detailer** | Facial feature enhancement | Eye, mouth, face mesh analysis |

---

## 📚 Documentation

### 📖 **Guides & Tutorials**
- **[Node Reference](assets/guides/node-reference.md)** - Detailed documentation for all nodes
- **[Performance Optimization](assets/guides/performance-guide.md)** - Optimization tips and benchmarks
- **[Validation System](assets/guides/validation-guide.md)** - Input validation and error handling
- **[CI/CD Setup](assets/guides/ci-cd-guide.md)** - Automated testing and deployment
- **[Development](assets/guides/development-guide.md)** - Contributing and development setup

### 🎯 **Quick Start Examples**
- **[Basic Workflow](assets/samples/basic-workflow.json)** - Simple upscaling workflow
- **[Advanced Pipeline](assets/samples/advanced-pipeline.json)** - Full production pipeline
- **[Performance Monitoring](assets/samples/performance-workflow.json)** - Performance tracking setup

### 📝 **Prompt Templates**
- **[Style Presets](assets/prompts/style-presets.md)** - Curated prompt collections
- **[Quality Enhancement](assets/prompts/quality-templates.md)** - Quality improvement prompts

---

## 🔧 Configuration

### Environment Variables
```bash
# Performance monitoring
LUNA_PERFORMANCE_LOG=true
LUNA_CACHE_SIZE=1000

# Validation settings
LUNA_VALIDATION_STRICT=false
LUNA_ERROR_HANDLING=graceful

# Development mode
LUNA_DEBUG=false
```

### Node Settings
Most nodes support configuration via their inputs:
- **Cache Settings**: Control validation caching behavior
- **Performance Monitoring**: Enable/disable performance tracking
- **Validation Mode**: Strict vs. graceful validation

---

## 🧪 Testing & Quality Assurance

### Automated Testing
```bash
# Run all tests
pytest tests/ -v

# Run performance benchmarks
python run_performance_tests.py

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Performance Benchmarks
- **Validation Throughput**: 471,032 operations/second
- **Memory Overhead**: < 1MB for 1000+ validations
- **Cache Hit Rate**: > 95% for repeated validations
- **Error Recovery**: < 100ms graceful degradation

### CI/CD Pipeline
- **Automated Testing**: Runs on every push/PR
- **Multi-Python Support**: Python 3.8, 3.9, 3.10, 3.11
- **Performance Regression Detection**: Automatic benchmarking
- **Code Coverage**: > 90% test coverage required

---

## 🏗️ Architecture

### Core Components
```
Luna Collection/
├── validation/          # Input validation system
├── nodes/              # Node implementations
│   ├── upscaling/      # Image upscaling nodes
│   ├── sampling/       # Generation nodes
│   ├── loaders/        # Model management
│   ├── preprocessing/  # Text processing
│   └── detailing/      # Computer vision
├── utils/              # Shared utilities
│   ├── mediapipe_engine.py  # Computer vision engine
│   └── luna_logger.py       # Logging system
├── tests/              # Comprehensive test suite
└── assets/             # Documentation and samples
    ├── guides/         # Detailed documentation
    ├── prompts/        # Prompt templates
    ├── screenshots/    # Visual examples
    └── samples/        # Workflow examples
```

### Validation System
- **Pydantic V2**: Type-safe validation with custom models
- **LRU Caching**: Intelligent caching with automatic cleanup
- **Graceful Degradation**: Continues working without validation
- **Error Recovery**: Comprehensive error handling and reporting

### Performance Monitoring
- **Real-time Metrics**: Execution time, memory usage, throughput
- **Benchmarking**: Automated performance regression detection
- **Resource Tracking**: GPU memory, CPU usage, I/O operations
- **Optimization**: Intelligent caching and resource management

---

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](assets/guides/development-guide.md) for:
- Development setup and workflow
- Coding standards and best practices
- Testing requirements
- Pull request guidelines

### Development Setup
```bash
# Clone and setup
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection

# Install development dependencies
pip install -r requirements-performance.txt

# Run tests
pytest tests/ -v

# Run performance benchmarks
python run_performance_tests.py
```

---

## 📈 Changelog

### v1.0.0 - Production Ready (2025-09-20)
- ✅ **Complete Validation System**: Comprehensive input validation with caching
- ✅ **Performance Monitoring**: Real-time performance tracking and optimization
- ✅ **CI/CD Pipeline**: Automated testing and deployment
- ✅ **Type Safety**: Full Pylance compatibility
- ✅ **Documentation**: Comprehensive guides and examples
- ✅ **41+ Automated Tests**: Complete test coverage
- ✅ **Performance Benchmarks**: 471K ops/sec validation throughput

### v0.0.2 - Advanced Features (2025-08-22)
- ➕ **Luna Advanced Upscaler**: Professional-grade upscaling controls
- 🐛 **Fixed**: Critical logic flaw in upscaling model utilization
- 📊 **Performance**: Initial performance monitoring capabilities

### v0.0.1 - Foundation (2025-08-22)
- 🎯 **Luna Simple Upscaler**: Clean, lightweight upscaling
- 🏗️ **Package Structure**: Established modular architecture
- 📦 **Initial Release**: Core functionality and documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Luna Collection** is born from the collaborative synergy between human creativity and AI precision. Special thanks to:

- The ComfyUI community for the incredible platform
- MediaPipe team for computer vision capabilities
- PyTorch ecosystem for machine learning foundations
- The open-source community for inspiration and tools

---

## 🎯 Philosophy

> "We build the tools we need, exactly as we need them. Each node is designed to be a clean, powerful, and intuitive component in a larger, more magnificent machine."

**Luna Collection** represents our commitment to:
- **Modularity**: Clean, reusable components
- **Quality**: Production-ready with comprehensive testing
- **Performance**: Optimized for speed and efficiency
- **Reliability**: Robust error handling and validation
- **Innovation**: Pushing the boundaries of what's possible

---

*Built with ❤️ by the Luna Collective*  
*Forged in the fires of creativity and precision*
