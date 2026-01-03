# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v2.3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**Production-grade ComfyUI infrastructure for advanced model management, multi-workflow VRAM sharing, and intelligent image generation.**

---

## ✨ What is Luna?

Luna Collection is a vertically integrated toolkit that solves real ComfyUI problems:

- **One Model Router** handles SD1.5, SDXL, Flux, SD3, and Z-IMAGE with intelligent CLIP/VAE selection
- **Precision preservation** - fp8/bf16 models stay in their original precision (no 2x VRAM bloat)
- **Multi-workflow daemon** - Share CLIP/VAE across multiple ComfyUI instances on separate GPUs
- **Workflow-aware management** - Automatically loads the right models for each workflow
- **Memory optimization** - Inference mode + intelligent caching reduces VRAM by 60-70%

---

## 🚀 Quick Start

### Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection
pip install -r requirements.txt
```

### Basic Setup
1. **Restart ComfyUI** - Luna nodes will appear under `Luna/` category
2. **Start the daemon** (optional, for multi-GPU workflows):
   ```bash
   python luna_daemon/daemon_server.py
   ```
3. **Use Luna Model Router** as your model loader instead of ComfyUI's default

### Simple Workflow
```
Luna Model Router (load your model)
    ↓
Luna Config Gateway (handle LoRAs + prompt processing)
    ↓
Luna KSampler (memory-optimized sampling)
    ↓
Luna Multi Saver (save results)
```

---

## 📦 Core Nodes

### Model Loading
- **Luna Model Router** ⚡ - Universal model loader (all architectures, precision preservation)

### Workflow Management
- **Luna Config Gateway** - Centralized parameters, LoRA weight caching, prompt processing
- **Luna KSampler** (+ Advanced/Headless variants) - Memory-optimized inference
- **Luna Reset Model Weights** - Restore pristine model state

### Vision & Prompts
- **Luna Z-IMAGE Encoder** - AI-enhanced encoding with vision modes
- **Luna Vision Node** - Extract style/description from reference images
- **Luna VLM Prompt Generator** - Generate prompts from images
- **Luna Prompt List Loader** - Batch prompts from CSV/JSON/YAML

### Image Processing
- **Luna Batch Upscale Refine** - Chess-pattern tiling with scaffolding noise
- **Luna Advanced Upscaler** - Supersampling with modulus rounding
- **Luna Ultimate SD Upscale** - Tile-based diffusion upscaling
- **Luna Multi Saver** - Batch saving with naming templates

### LoRA & Embeddings
- **Luna LoRA Stacker** (+ Random variant) - Stack up to 4 LoRAs
- **Luna Embedding Manager** (+ Random variant) - Textual inversion management
- **Luna LoRA Validator** - Validate and extract LoRA metadata

### Utilities
- **Luna YAML Wildcard** - Hierarchical templates with path resolution
- **Luna Expression Pack** - Logic and math for workflows
- **Luna GGUF Converter** - Convert models to quantized format
- **Luna Civitai Scraper** - Fetch LoRA metadata

---

## 🎯 Key Features

### Unified Model Router
Single node supporting all architectures with dynamic CLIP/VAE selection:
- **SD1.5** → CLIP-L
- **SDXL** → CLIP-L + CLIP-G  
- **Flux** → CLIP-L + CLIP-G + T5-XXL
- **SD3** → CLIP-L + CLIP-G + T5-XXL
- **Z-IMAGE** → Full Qwen3-VL with vision modes

### Smart Precision Handling
Detects and preserves your model's original precision:
- fp16, bf16, fp32
- fp8_e4m3fn, fp8_e4m3fn_scaled, fp8_e5m2
- GGUF Q8_0, Q4_K_M
- nf4, int8

No more 2.5GB models expanding to 5GB during load.

### Multi-Workflow Daemon
Run multiple ComfyUI instances on the same hardware:
- Each workflow gets its own model set
- CLIP/VAE are loaded once and shared
- Intelligent sideloading (no model unloading)
- CUDA IPC for zero-copy transfers between GPUs

### Memory Optimization
- **Inference Mode Wrapper** - 60-70% VRAM reduction via torch.inference_mode()
- **Transient LoRA** - LoRAs cached in RAM, applied with random weights, restored without disk I/O
- **First-Block Cache** - 2x speedup on final denoising steps
- **Lazy Model Unloading** - Models persist across workflows instead of reloading

---

## 📚 Documentation

- **[FEATURES.md](FEATURES.md)** - Comprehensive feature documentation (architecture, detailed guides)
- **[Docs/](Docs/)** - Technical documentation by module
- **[Docs/file_summaries/](Docs/file_summaries/)** - Auto-generated summaries of all components

---

## ⚙️ Configuration

### daemon_config.py
```python
CLIP_DEVICE = "cuda:1"        # GPU for CLIP models
VAE_DEVICE = "cuda:0"         # GPU for VAE
SHARED_DEVICE = "cuda:1"      # Shared model device
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283
```

### Model Router Settings
- **Model Source**: checkpoints / diffusion_models / unet (GGUF)
- **Model Type**: SD1.5 / SDXL / Flux / SD3 / Z-IMAGE
- **Precision**: fp16 / bf16 / fp8 / nf4 / auto
- **Daemon Mode**: auto / force_local / force_daemon

---

## 🔧 Troubleshooting

**Model won't load?**
- Check file path exists
- Try force_local mode (bypass daemon)
- Check logs for precision errors

**VRAM usage too high?**
- Use fp8/bf16 precision instead of fp32
- Enable Luna KSampler (uses inference_mode)
- Check daemon is running if multi-GPU intended

**Daemon connection failed?**
- Start daemon: `python luna_daemon/daemon_server.py`
- Check port 19283 is not blocked
- Try force_local mode to continue without daemon

---

## 🤝 Contributing

Contributions welcome! Fork, create a feature branch, and open a PR.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LSDJesus/ComfyUI-Luna-Collection/discussions)
- **Full Docs**: See [Docs/](Docs/) folder for detailed technical documentation
