# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**A comprehensive suite of ComfyUI custom nodes for advanced image processing, model management, and workflow automation.**

Luna Collection provides a modular set of tools for image upscaling, MediaPipe-based detailing, LoRA stacking, prompt preprocessing, and more. Each node is designed to be intuitive and integrate seamlessly into your ComfyUI workflows.

---

## ✨ Features

### 🔧 **Core Capabilities**
- **Advanced Upscaling**: Multiple upscaling nodes with model-based and resampling methods
- **MediaPipe Integration**: Face, hand, pose, and body segmentation and detailing
- **LoRA Management**: Advanced LoRA stacking with individual strength controls
- **Prompt Processing**: Comprehensive text preprocessing and enhancement tools
- **Model Loading**: Intelligent checkpoint and embedding management
- **TensorRT Support**: High-performance inference with TensorRT engines
- **YOLO Integration**: Annotation export for object detection workflows
- **Image Captioning**: Automated image description generation

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
│   │   ├── luna_checkpoint_loader.py
│   │   ├── luna_lora_stacker.py
│   │   ├── luna_lora_stacker_random.py
│   │   ├── luna_embedding_manager.py
│   │   └── luna_embedding_manager_random.py
│   ├── upscaling/                  # Image upscaling nodes
│   │   ├── luna_upscaler_simple.py
│   │   ├── luna_upscaler_advanced.py
│   │   └── luna_ultimate_sd_upscale.py
│   ├── preprocessing/              # Text/prompt processing nodes
│   │   ├── luna_prompt_preprocessor.py
│   │   ├── luna_text_processor.py
│   │   └── luna_unified_prompt_processor.py
│   ├── detailing/                  # MediaPipe detailing nodes
│   ├── performance/                # Performance monitoring nodes
│   ├── luna_mediapipe_detailer.py  # MediaPipe face detailer
│   ├── luna_sampler.py             # Advanced sampler
│   ├── luna_image_caption.py       # Image captioning
│   ├── luna_multi_saver.py         # Batch saving
│   ├── luna_yolo_annotation_exporter.py
│   └── tensorrt_detailer.py        # TensorRT face detailer
├── utils/                          # Shared utilities
│   ├── mediapipe_engine.py         # MediaPipe processing engine
│   ├── trt_engine.py               # TensorRT engine wrapper
│   ├── impact_core.py              # Impact Pack integration utilities
│   └── tiling.py                   # Tiling utilities
├── js/                             # Frontend JavaScript
│   ├── luna_lora_stacker.js
│   └── luna_collection_nodes.js
├── caption-templates/              # Image captioning templates
├── test/                           # Unit tests
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

### v1.1.0 - Current (2025-09-21)
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

