# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Nodes](https://img.shields.io/badge/nodes-50+-purple.svg)

**A comprehensive suite of 50+ ComfyUI custom nodes for advanced image generation, model management, prompt engineering, and workflow automation.**

Luna Collection has evolved into a powerful toolkit spanning prompt preprocessing, YAML wildcard systems, LoRA/embedding management, MediaPipe-based detailing, multi-instance VRAM sharing, and more. Each node is designed to be modular, efficient, and seamlessly integrate into professional ComfyUI workflows.

---

## 📦 What's Included

| Category | Nodes | Description |
|----------|-------|-------------|
| **🎯 Prompt & Wildcards** | 15+ | YAML hierarchical wildcards, context-aware resolution, prompt preprocessing |
| **📁 Model Loaders** | 10+ | LoRA stackers, embedding managers, checkpoint loading with metadata |
| **🎨 MediaPipe Detailing** | 5 | Face/body/hand detection and inpainting with Flux compatibility |
| **⬆️ Upscaling** | 4 | Simple, advanced, and Ultimate SD upscale with tiling |
| **🚀 Performance** | 8 | Shared VAE/CLIP daemon, performance monitoring, caching |
| **📝 Text Processing** | 6 | Unified prompt processor, text manipulation, logic resolution |
| **🔧 Utilities** | 8+ | Multi-saver, captioning, YOLO export, parameter management |

---

## 🚀 Quick Start

### Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection
pip install -r requirements.txt
```

**Optional dependencies:**
```bash
# For MediaPipe nodes (face/body detection)
pip install mediapipe opencv-python

# For advanced performance features
pip install -r requirements-performance.txt
```

**Recommended companion extension:**
```bash
# WaveSpeed - faster than TensorRT without constraints!
# Install from: https://github.com/chengzeyi/Comfy-WaveSpeed
# Use "Apply First Block Cache" node for ~40% speedup with no model conversion
```

Restart ComfyUI and nodes will appear under `Luna/` categories.

---

## 🎯 Node Categories

### 🌿 **YAML Wildcard System** (`Luna/Wildcards`)

A powerful hierarchical wildcard system using YAML files instead of traditional .txt wildcards.

| Node | Purpose |
|------|---------|
| **Luna YAML Wildcard** | Process wildcards with `{file:path.to.items}` syntax |
| **Luna YAML Wildcard Batch** | Generate multiple variations at once |
| **Luna Wildcard Builder** | Visually construct wildcard expressions |
| **Luna LoRA Randomizer** | Randomly select LoRAs with category filtering |
| **Luna Random Int/Float Range** | Generate random numbers with step control |

**Syntax Examples:**
```
{clothing:tops.casual}           → Random item from clothing.yaml > tops > casual
{hair:styles}                    → Random hair style
{1-10}                           → Random integer 1-10
{0.5-1.5:0.1}                    → Random float with 0.1 step
__legacy/wildcard__              → Legacy .txt wildcard support
```

**SDXL Prompt Assembly Order:**

SDXL-based models (Illustrious, Pony, etc.) work best with comma-delimited atomic tags in this order:

| Priority | Category | Examples |
|----------|----------|----------|
| 1 | Quality/Score | `masterpiece, best quality, score_9` |
| 2 | Style/Medium | `anime, photorealistic, digital art` |
| 3 | Subject | `1girl, solo, <lora:character:0.8>` |
| 4 | Physical | `long blonde hair, blue eyes, slim` |
| 5 | Expression | `smile, looking at viewer, blush` |
| 6 | Clothing | `white dress, high heels, jewelry` |
| 7 | Pose/Action | `standing, walking, arms behind back` |
| 8 | Setting | `classroom, forest, simple background` |
| 9 | Props | `holding book, bag, glasses` |
| 10 | Composition | `cowboy shot, from above, close-up` |
| 11 | Lighting | `dramatic lighting, golden hour, rim light` |

> Front-load important elements - CLIP weights earlier tokens more heavily.

### 📁 **Model Loaders** (`Luna/Loaders`)

| Node | Purpose |
|------|---------|
| **Luna Checkpoint Loader** | Load checkpoints with metadata display |

### ⬆️ **Upscaling** (`Luna/Upscaling`)

| Node | Purpose |
|------|---------|
| **Luna Simple Upscaler** | Basic model-based upscaling |
| **Luna Advanced Upscaler** | Supersampling, modulus rounding, advanced controls |
| **Luna Ultimate SD Upscale** | Tile-based SD upscaling with seam blending |

### 🚀 **Performance & Daemon** (`Luna/Shared`, `Luna/Performance`)

The Luna Daemon system allows sharing VAE/CLIP models across multiple ComfyUI instances:

```
┌─────────────────────────────────────────────┐
│           GPU 1 (cuda:1)                    │
│  ┌─────────────────────────────────────┐   │
│  │     Luna VAE/CLIP Daemon            │   │
│  │  • VAE + CLIP loaded once           │   │
│  │  • Serves encode/decode requests    │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
              ▲ Socket (127.0.0.1:19283)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│:8188  │ │:8189  │ │:8190  │  Multiple ComfyUI instances
└───────┘ └───────┘ └───────┘
```

| Node | Purpose |
|------|---------|
| **Luna Daemon Config** | Configure VAE/CLIP from within workflow |
| **Luna Daemon Model Switch** | Quick preset switching (SDXL/Pony/Flux) |
| **Luna Shared VAE Encode/Decode** | Use daemon for VAE operations |
| **Luna Shared CLIP Encode** | Use daemon for text encoding |
| **Luna Daemon Status** | Monitor daemon health and VRAM |

### 📝 **Prompt Preprocessing** (`Luna/Preprocessing`)

| Node | Purpose |
|------|---------|
| **Luna Prompt Preprocessor** | Batch preprocess prompts to safetensors |
| **Luna Optimized Preprocessed Loader** | Load cached prompts with caching |
| **Luna Unified Prompt Processor** | All-in-one prompt enhancement |
| **Luna Cache Manager** | Manage embedding cache |
| **Luna Performance Monitor** | Track preprocessing performance |

### 🔗 **LoRA/Embedding Connections** (`Luna/Connections`)

Smart linking between wildcards and LoRAs/embeddings:

| Node | Purpose |
|------|---------|
| **Luna Smart LoRA Linker** | Match prompts to LoRAs via connections.json |
| **Luna Connection Matcher** | Find connections based on prompt content |
| **Luna Civitai Metadata Scraper** | Fetch and embed Civitai metadata |

### 🔧 **Utilities** (`Luna/Utils`)

| Node | Purpose |
|------|---------|
| **Luna Multi Saver** | Batch save with format options |
| **Luna Image Caption** | AI-powered captioning |
| **Luna YOLO Annotation Exporter** | Export YOLO format labels |
| **Luna Parameters Bridge** | Pass parameters between nodes |
| **Luna Load Parameters** | Load saved configurations |

---

## 📖 Detailed Guides

- **[WaveSpeed Acceleration](Docs/guides/wavespeed_acceleration.md)** - 🚀 40% faster inference (recommended!)
- **[YAML Wildcard Guide](Docs/guides/yaml_wildcards.md)** - Complete YAML wildcard syntax and examples
- **[Daemon Setup Guide](luna_daemon/README.md)** - Multi-instance VRAM sharing setup
- **[LoRA Connections Guide](Docs/guides/lora_connections.md)** - Smart LoRA/embedding linking
- **[Performance Optimization](Docs/guides/performance.md)** - Caching and performance tips
- **[Complete Node Reference](Docs/guides/node_reference.md)** - All nodes with detailed parameters

---

## 📁 Project Structure

```
ComfyUI-Luna-Collection/
├── nodes/                          # All node implementations
│   ├── loaders/                    # Model loading nodes
│   │   └── luna_checkpoint_loader.py
│   ├── preprocessing/              # Prompt processing nodes
│   │   ├── luna_prompt_preprocessor.py
│   │   └── luna_logic_resolver.py
│   ├── upscaling/                  # Image upscaling nodes
│   │   ├── luna_upscaler_simple.py
│   │   ├── luna_upscaler_advanced.py
│   │   └── luna_ultimate_sd_upscale.py
│   ├── luna_yaml_wildcard.py       # YAML wildcard system
│   ├── luna_wildcard_connections.py # LoRA/embedding linking
│   ├── luna_shared_vae.py          # Daemon VAE nodes
│   ├── luna_shared_clip.py         # Daemon CLIP nodes
│   ├── luna_daemon_config.py       # Daemon configuration
│   ├── luna_civitai_scraper.py     # Civitai metadata
│   ├── luna_sampler.py             # Custom sampler
│   └── luna_hyperlora.py           # HyperLoRA integration (experimental)
├── luna_daemon/                    # Shared model daemon
│   ├── server.py                   # Static daemon (v1)
│   ├── server_v2.py                # Dynamic scaling daemon
│   ├── client.py                   # Client library
│   └── config.py                   # Configuration
├── utils/                          # Shared utilities
│   ├── mediapipe_engine.py         # MediaPipe processing
│   ├── logic_engine.py             # Wildcard logic
│   └── luna_logger.py              # Logging utilities
├── scripts/                        # Utility scripts
│   ├── start_daemon.ps1            # Start daemon server
│   └── start_server_workflow.ps1   # Start ComfyUI with daemon
├── Docs/                           # Documentation
│   └── guides/                     # Usage guides
├── tests/                          # Unit tests
└── js/                             # Frontend JavaScript
```

---

## 🔧 Dependencies

### Core (Required)
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
Pillow>=9.0.0
pyyaml>=6.0
safetensors>=0.3.0
pydantic>=1.10.0
psutil>=5.9.0
```

### Optional
```
mediapipe        # MediaPipe nodes
opencv-python    # Image processing
aiohttp          # Civitai scraper
spandrel         # Upscaling models
```

---

## 🧪 Testing

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=luna_collection --cov-report=html

# Run specific category
pytest -m unit
pytest -m integration
```

---

## 📈 Changelog

### v2.0.0 (2025-11-28)
- ✅ **YAML Wildcard System**: Hierarchical YAML-based wildcards
- ✅ **Luna Daemon**: Multi-instance VAE/CLIP sharing with dynamic configuration
- ✅ **LoRA Connections**: Smart LoRA/embedding linking with Civitai metadata
- ✅ **Civitai Scraper**: Fetch and embed Civitai metadata into models
- ✅ **Bug Fixes**: Fixed prompt preprocessor f-string bug, missing returns
- ✅ **50+ Nodes**: Comprehensive node collection

### v1.1.0 (2025-09-21)
- ✅ Enhanced Face Detailer
- ✅ Enhanced LoRA Stacker
- ✅ MediaPipe improvements

### v1.0.0 (2025-08-22)
- 🎯 Initial release

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow the existing code style
4. Add tests for new features
5. Submit a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ by the Luna Collective*
