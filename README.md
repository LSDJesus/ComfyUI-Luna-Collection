# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v1.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**A comprehensive suite of ComfyUI custom nodes for advanced image processing, model management, and workflow automation.**

Luna Collection provides a modular set of tools for image upscaling, MediaPipe-based detailing, LoRA stacking, YAML wildcards, multi-instance VRAM sharing, and more. Each node is designed to be intuitive and integrate seamlessly into your ComfyUI workflows.

---

## ✨ Features

### 🔧 **Core Capabilities**
- **Advanced Upscaling**: Multiple upscaling nodes with model-based and resampling methods
- **MediaPipe Integration**: Face, hand, pose, and body segmentation and detailing
- **LoRA Management**: Advanced LoRA stacking with individual strength controls
- **YAML Wildcards**: Hierarchical prompt templates with nested path resolution
- **Luna Daemon**: Multi-instance VRAM sharing for VAE/CLIP across ComfyUI instances
- **Civitai Integration**: Automatic metadata scraping with local SQLite database
- **Prompt Processing**: Comprehensive text preprocessing and enhancement tools
- **TensorRT Support**: High-performance inference with TensorRT engines
- **Input Validation**: Pydantic-based validation for all node inputs

---

## 🚀 Installation

### Prerequisites
- ComfyUI (latest version recommended)
- Python 3.10+
- PyTorch with CUDA support (for GPU acceleration)

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

The nodes will be available under the **`Luna Collection`** or **`Luna/`** categories.

---

## 🎯 Available Nodes

### 🖼️ **Image Processing & Upscaling**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Simple Upscaler** | Clean, lightweight upscaling | Model-based scaling, multiple resampling methods |
| **Luna Advanced Upscaler** | Professional-grade upscaling | Supersampling, modulus rounding, advanced controls |
| **Luna Ultimate SD Upscale** | Multi-stage SD upscaling | Tile-based processing, seam blending |

### 🎨 **MediaPipe Detailing**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna MediaPipe Detailer** | Face/body detailing with inpainting | Flux-compatible, conditional detailing, mask generation |
| **Luna MediaPipe Segs** | Segmentation mask generation | Multi-target detection (hands, face, eyes, etc.) |
| **TensorRT Face Detailer** | High-performance TensorRT detailing | Dynamic engine support, bbox detection, SAM integration |

### 📁 **Model Management**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Checkpoint Loader** | Checkpoint loading with metadata | Model info display, efficient loading |
| **Luna LoRA Stacker** | Multi-LoRA management | Up to 4 LoRAs, individual strength/toggle controls |
| **Luna LoRA Stacker Random** | Randomized LoRA selection | Automatic variation generation |
| **Luna Embedding Manager** | Textual inversion management | Multiple embedding support |
| **Luna Embedding Manager Random** | Randomized embedding selection | Variation and experimentation |

### 📝 **Text & Prompt Processing**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Unified Prompt Processor** | All-in-one prompt enhancement | Multiple processing modes, wildcard support |
| **Luna Prompt Preprocessor** | Advanced prompt preprocessing | Style enhancement, quality boosting |
| **Luna Text Processor** | Text manipulation and filtering | Length control, content filtering |
| **Luna Wildcard Prompt Generator** | Dynamic prompt generation | Random wildcard expansion |
| **Luna Load Preprocessed** | Load saved prompts | Prompt library management |
| **Luna Save Negative Prompt** | Save negative prompts | Reusable negative prompt templates |

### 🎲 **YAML Wildcards**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna YAML Wildcard** | Hierarchical YAML wildcard expansion | Nested path resolution, templates, numeric ranges |
| **Luna YAML Wildcard Batch** | Generate multiple prompts at once | Batch processing, variation generation |
| **Luna YAML Wildcard Explorer** | Browse and preview wildcards | Interactive exploration of YAML files |
| **Luna Wildcard Builder** | Construct prompts with wildcards | Visual prompt building |
| **Luna LoRA Randomizer** | Random LoRA selection from YAML | Weighted random selection |
| **Luna Wildcard CSV Injector** | Import CSV data into YAML | Batch data import |

### 🔗 **Luna Daemon (Multi-Instance VRAM Sharing)**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Shared VAE Encode** | Encode via daemon's shared VAE | Offload VAE to separate GPU |
| **Luna Shared VAE Decode** | Decode via daemon's shared VAE | Free VRAM on main GPU |
| **Luna Shared VAE Encode (Tiled)** | Tiled encoding for large images | Memory-efficient encoding |
| **Luna Shared VAE Decode (Tiled)** | Tiled decoding for large images | Memory-efficient decoding |
| **Luna Shared CLIP Encode** | Encode via daemon's shared CLIP | Offload CLIP to separate GPU |
| **Luna Shared CLIP Encode (SDXL)** | SDXL dual CLIP encoding | SDXL-specific encoding |
| **Luna Daemon Status** | Check daemon connection status | Health monitoring |

### 🌐 **Civitai Integration**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Civitai Metadata Scraper** | Fetch metadata from Civitai | Trigger words, tags, descriptions |
| **Luna Civitai Batch Scraper** | Bulk scrape multiple models | Folder-based batch processing |

### 🔧 **Workflow & Utilities**
| Node | Description | Key Features |
|------|-------------|--------------|
| **Luna Sampler** | Advanced KSampler | Custom sampling with enhanced controls |
| **Luna Multi Saver** | Batch image saving | Multiple format support, organized output |
| **Luna Parameters Bridge** | Parameter passing between nodes | Workflow organization |
| **Luna Load Parameters** | Load saved parameters | Reusable configurations |
| **Luna Image Caption** | Automated image captioning | AI-powered descriptions |
| **Luna YOLO Annotation Exporter** | YOLO format export | Object detection workflow integration |
| **Luna Performance Monitor** | Workflow performance tracking | Execution time monitoring |
| **Luna Cache Manager** | Cache management | Memory optimization |

---

## 📚 Key Features by Node

### Luna MediaPipe Detailer
- Detects and details faces, hands, eyes, mouth, feet, torso, and full body
- Flux-compatible conditioning with pooled outputs
- Configurable mask padding, blur, and confidence thresholds
- Multiple sorting options (confidence, area, position)
- Automatic mask generation and inpainting support

### Luna LoRA Stacker
- Stack up to 4 LoRAs with individual controls
- Dropdown selection from your `models/loras` directory
- Individual enable/disable toggles per LoRA
- Separate strength controls for fine-tuning
- Compatible with ComfyUI-Impact-Pack's Apply LoRA Stack nodes

### TensorRT Face Detailer
- High-performance inference using TensorRT engines
- Dynamic engine support (min: 768, max: 1280, opt: 1024)
- ONNX bbox detector compatibility
- SAM (Segment Anything Model) integration for refinement
- Automatic region cropping and resizing

### Luna Ultimate SD Upscale
- Multi-stage upscaling with SD inpainting
- Tile-based processing for large images
- Seam blending for seamless results
- Configurable tile size and overlap
- Support for various upscaling models

### 🎲 Luna YAML Wildcard System
A powerful hierarchical wildcard system using YAML files for organized prompt generation.

**Prompt Syntax:**
- `{filename}` - Random template from `filename.yaml`'s `templates` section
- `{filename:path.to.items}` - Random item from nested path
- `{filename: text with [path.to.item] substitutions}` - Inline template
- `{1-10}` - Random integer range
- `{0.5-1.5:0.1}` - Random float with step resolution
- `__path/file__` - Legacy .txt wildcard reference

**Example YAML structure:**
```yaml
templates:
  full:
    - "a [category.item] with [another.path]"
category:
  item:
    - option_one
    - option_two
```

### 🔗 Luna Daemon (Multi-Instance VRAM Sharing)
Share VAE and CLIP models across multiple ComfyUI instances to save VRAM.

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   GPU 1 (cuda:1)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Luna VAE/CLIP Daemon                   │   │
│  │  • VAE + CLIP loaded once                       │   │
│  │  • Serves encode/decode via local socket        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ▲ Socket (127.0.0.1:19283)
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ComfyUI :8188 │ │ ComfyUI :8189 │ │ ComfyUI :8190 │
│ UNet only     │ │ UNet only     │ │ UNet only     │
└───────────────┘ └───────────────┘ └───────────────┘
```

**Usage:**
1. Start the daemon: `python luna_daemon/server.py`
2. Use `Luna Shared VAE Encode/Decode` nodes instead of standard VAE nodes
3. Multiple ComfyUI instances share the same VAE/CLIP on a separate GPU

### 🌐 Luna Metadata Database
Local SQLite database for LoRA/embedding metadata storage.

**Location:** `{ComfyUI}/user/default/ComfyUI-Luna-Collection/metadata.db`

**Features:**
- Fast hash-based lookups (Civitai tensor hash format)
- Full-text search across trigger words, tags, descriptions
- User customization: favorites, ratings, custom tags, notes
- Usage tracking: use count, last used timestamp
- Query by base model (SDXL, Pony, Illustrious, etc.)

---

## 🔧 Dependencies

### Core Requirements
- **ComfyUI** - Latest version recommended
- **PyTorch** - With CUDA support for GPU acceleration
- **MediaPipe** - For face/pose/hand detection nodes
- **OpenCV** - Image processing
- **NumPy** - Numerical operations

### Optional Dependencies
- **TensorRT** - For TensorRT Face Detailer node
- **Polygraphy** - TensorRT engine utilities
- **SAM Models** - For segmentation refinement
- **Impact Pack** - For bbox detection integration
- **Pydantic** - For input validation (v2.0+)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🏗️ Project Structure

```
ComfyUI-Luna-Collection/
├── nodes/                          # All node implementations
│   ├── loaders/                    # Model loading nodes
│   ├── upscaling/                  # Image upscaling nodes
│   ├── preprocessing/              # Text/prompt processing nodes
│   ├── detailing/                  # MediaPipe detailing nodes
│   ├── performance/                # Performance monitoring nodes
│   ├── luna_yaml_wildcard.py       # YAML wildcard system
│   ├── luna_shared_vae.py          # Shared VAE nodes (daemon)
│   ├── luna_shared_clip.py         # Shared CLIP nodes (daemon)
│   ├── luna_civitai_scraper.py     # Civitai metadata scraper
│   ├── luna_mediapipe_detailer.py  # MediaPipe face detailer
│   └── ...                         # Other node files
├── luna_daemon/                    # Multi-instance VRAM sharing daemon
│   ├── server.py                   # Daemon server
│   ├── client.py                   # Client utilities
│   └── config.py                   # Daemon configuration
├── utils/                          # Shared utilities
│   ├── luna_metadata_db.py         # SQLite metadata database
│   ├── mediapipe_engine.py         # MediaPipe processing engine
│   ├── trt_engine.py               # TensorRT engine wrapper
│   ├── luna_performance_monitor.py # Performance tracking
│   └── ...                         # Other utilities
├── validation/                     # Pydantic input validation
│   └── __init__.py                 # Validators and models
├── js/                             # Frontend JavaScript
├── tests/                          # Unit and integration tests
├── scripts/                        # Utility scripts
└── __init__.py                     # Package initialization
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add features, fix bugs, or improve documentation:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate comments.

---

## 📈 Changelog

### v1.2.0 - Current (2025-11-29)
- ✅ **YAML Wildcard System**: Hierarchical wildcards with templates, nested paths, numeric ranges
- ✅ **Luna Daemon**: Multi-instance VRAM sharing for VAE/CLIP across ComfyUI instances
- ✅ **Shared VAE/CLIP Nodes**: Encode/decode via daemon's shared models
- ✅ **Civitai Integration**: Automatic metadata scraping and embedding
- ✅ **SQLite Metadata Database**: Local storage for model metadata with full-text search
- ✅ **Input Validation**: Pydantic-based validation system for all node inputs
- ✅ **Performance Monitoring**: Execution time tracking and optimization tools
- ✅ **Project Cleanup**: Removed redundant code, fixed imports, improved structure

### v1.1.0 (2025-09-21)
- ✅ **TensorRT Integration**: High-performance TensorRT Face Detailer node
- ✅ **Enhanced LoRA Stacker**: Dropdown selection, individual toggles, proper tuple format
- ✅ **MediaPipe Improvements**: Enhanced detailer with Flux compatibility
- ✅ **Utility Functions**: Local impact_core and trt_engine utilities
- ✅ **Bug Fixes**: Fixed MediaPipe engine imports, LoRA stack format

### v1.0.0 - Initial Release (2025-08-22)
- 🎯 **Core Nodes**: Simple, Advanced, and Ultimate SD upscalers
- 🎯 **MediaPipe Integration**: Face, pose, and hand segmentation
- 🎯 **LoRA Management**: Stacking and random selection
- 🎯 **Prompt Processing**: Preprocessing and enhancement tools
- 🎯 **Workflow Tools**: Multi-saver, parameter bridge, sampler

---

## 🙏 Acknowledgments

This project builds upon the excellent work of the ComfyUI community. Special thanks to:

- **ComfyUI Team** - For the incredible platform and architecture
- **MediaPipe** - For computer vision and pose estimation capabilities
- **Impact Pack** - For bbox detection and segmentation utilities
- **ComfyUI-Impact-Pack** - For LoRA stack compatibility and detailing tools
- **TensorRT Community** - For high-performance inference optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by the Luna Collective*

