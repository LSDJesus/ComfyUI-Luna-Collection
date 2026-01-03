# 🌙 ComfyUI Luna Collection - Comprehensive Feature Documentation

> **Start here:** See [README.md](README.md) for quick start and core concepts.  
> **This document:** Detailed features, complete node reference, and technical deep-dives.

![Version](https://img.shields.io/badge/version-v2.3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**Production-grade ComfyUI infrastructure for advanced model management, multi-workflow VRAM sharing, and intelligent image generation.**

---

## 📝 What's New in v2.3

🎯 **IP-Adapter TRUE BATCHING** - Each detection gets unique vision anchoring, 12× faster Semantic Detailer  
🎨 **Semantic Detailer Suite** - Surgical pyramid-based refinement with SAM3 detection  
✨ **Batch Upscale Refine** - Chess-pattern tiling with scaffolding noise + Lanczos supersampling  
🎯 **FP8 Precision Expansion** - All three variants (e4m3fn, e4m3fn_scaled, e5m2) now supported  
🦙 **Qwen3-VL GGUF Support** - Z-IMAGE works with quantized models (Q8_0, Q4_K_M)

---

## ✨ Core Features

### 🚀 **Workflow-Aware Multi-Instance Architecture**
- **Multi-Workflow Multiplexing**: Run multiple workflows simultaneously sharing CLIP/VAE models
- **Intelligent Model Routing**: Daemon tracks which models each workflow needs, sideloads new ones
- **Zero Redundancy**: Shared models used once, not duplicated across workflows
- **InferenceModeWrapper**: Automatic VRAM management for UNet models

### 🔧 **Core Infrastructure**
- **Luna Model Router**: Unified model loader for all architectures with explicit CLIP/VAE selection
- **Luna Daemon v2.0**: Workflow-aware multiplexing with per-workflow model tracking
- **Smart Precision Loading**: fp16/bf16/fp8 (all variants)/GGUF/nf4 with automatic detection
- **Transient LoRA System**: LoRAs cached in RAM, applied with randomized weights
- **Config Gateway**: Centralized workflow parameters with LoRA weight caching
- **Reset Weights Node**: Ctrl-Z for LoRA modifications between runs

### 📦 **Model Management**
- **Unified Model Router**: Single node supporting SD1.5, SDXL, Flux, SD3, Z-IMAGE with vision variants
- **Smart Precision Preservation**: Models stay in their original precision (no 2x VRAM expansion)
- **Explicit CLIP/VAE Selection**: Dynamic selectors updated based on model_type
- **Precision Conversion Cache**: Converted models saved to precision-specific directories

### 🎲 **Prompt Engineering**
- **YAML Wildcards**: Hierarchical templates with nested path resolution
- **LoRA Weight Randomization**: Randomized weights per run without reload
- **Automatic LoRA Extraction**: Extract LoRAs from prompt text (`<lora:name:weight>` syntax)
- **Batch Prompt Loading**: CSV/JSON/YAML import with seeds and LoRA stacks

### 🖼️ **Image Processing & Refinement**
- **Semantic Detailer Suite**: Surgical detection-based refinement (SAM3 + IP-Adapter)
- **Chess Refiner**: Global tile refinement with seamless blending
- **Batch Upscale Refine**: Chess-pattern tiling with scaffolding noise + supersampling
- **Advanced Upscalers**: Model-based, tile-based, and multi-stage upscaling
- **Vision-Guided Generation**: Image-to-embedding for vision-conditioned workflows

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LUNA DAEMON (Workflow Multiplexer)               │
│                                                                      │
│  workflow_A: {clip_l_A, clip_g_A, vae_shared}                       │
│  workflow_B: {clip_l_B, clip_g_B, vae_shared} ← Reused              │
│  workflow_C: {clip_l_C, clip_g_C, vae_shared} ← Reused              │
│                                                                      │
│  Benefits:                                                          │
│  • No unloading - all models stay resident                          │
│  • Shared models used once (VAE example above)                      │
│  • New workflows trigger sideloading, not replacement              │
│  • Intelligent routing ensures correct models per workflow         │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│               Model Router + Precision Preservation                  │
│                                                                      │
│  Input: model.safetensors (fp8, fp16, bf16, GGUF, nf4)             │
│  Detect: Actual precision in model file                             │
│  Load: Pass dtype to load_unet() → stays in original precision      │
│  Result: 2.5GB fp8 model stays 2.5GB (not 5GB after upcast)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Complete Node Reference

### 🔧 Model Loading (Loaders)

| Node | Purpose |
|------|---------|
| **Luna Model Router** ⚡ | Unified loader for ALL architectures (SD1.5/SDXL/Flux/SD3/Z-IMAGE) with precision preservation |
| **Luna Secondary Model Loader** | Multi-model workflows with CLIP sharing and RAM offloading |
| **Luna Model Restore** | Restore models offloaded to RAM back to VRAM |
| **Luna Dynamic Model Loader** | Legacy smart checkpoint loading with JIT precision conversion |
| **Luna Daemon VAE Loader** | Load VAE specifically from daemon |
| **Luna Daemon CLIP Loader** | Load CLIP specifically from daemon |
| **Luna Checkpoint Tunnel** | Pass MODEL through, route CLIP/VAE to daemon |
| **Luna UNet Tunnel** | UNet-only loading for advanced workflows |
| **Luna GGUF Converter** | Convert checkpoints to quantized GGUF format |
| **Luna Optimized Weights Manager** | Manage and list locally cached optimized UNets |
| **Luna INT8 Loader** | Specialized loader for INT8-quantized models |
| **Luna NF4 Loader** | Specialized loader for NF4-quantized models |

**Model Router Outputs:**
- `MODEL` - UNet (wrapped in InferenceModeWrapper or DaemonModel)
- `CLIP` - CLIP encoders (DaemonCLIP proxy or local)
- `VAE` - VAE encoder/decoder (DaemonVAE proxy or local)
- `LLM` - Full LLM for Z-IMAGE (Qwen3-VL)
- `CLIP_VISION` - Vision encoder for vision-enabled models
- `model_name` - String for metadata/Config Gateway
- `status` - Detailed loading status message

---

### 🎲 Workflow Management

| Node | Purpose |
|------|---------|
| **Luna Config Gateway** | Centralized workflow parameters + LoRA extraction + weight caching |
| **Luna KSampler** | Memory-optimized inference (uses torch.inference_mode()) |
| **Luna KSampler Advanced** | KSampler with start/end step control and noise handling |
| **Luna KSampler Headless** | KSampler with pipe-only input (zero manual connections) |
| **Luna KSampler Scaffold** | Debug variant for scaffold testing |
| **Luna Reset Model Weights** | Restore pristine model state after LoRA application |
| **Luna Multi Saver** | Batch image saving with naming templates and EXIF embedding |
| **Luna Pipe Expander** | Decompose LUNA_PIPE into individual outputs |

**Config Gateway Features:**
- Auto-extract LoRAs from prompts (`<lora:name:weight>` syntax)
- Deduplicate LoRAs (input stack + extracted)
- Cache pristine weights before LoRA application
- Apply LoRAs with random or specified weights
- Output complete workflow config tuple (LUNA_PIPE)

---

### 👁️ Vision & Prompt Processing

| Node | Purpose |
|------|---------|
| **Luna Z-IMAGE Encoder** | AI-enhanced encoding with Qwen3-VL, vision modes, noise injection |
| **Luna Z-IMAGE Processor** | Preprocessing node for Z-IMAGE workflows |
| **Luna Vision Node** | Image→embedding via vision models (describe/extract style/blend) |
| **Luna VLM Prompt Generator** | Generate prompts from reference images using vision LLM |
| **Luna YAML Wildcard** | Hierarchical wildcard expansion with templates and paths |
| **Luna YAML Wildcard Batch** | Generate multiple prompts with seeds from wildcards |
| **Luna Wildcard Builder** | Visual prompt composition with real-time preview |
| **Luna LoRA Randomizer** | Random LoRA selection from YAML files |
| **Luna YAML Path Explorer** | Debug utility for exploring YAML file structures |
| **Luna YAML Injector** | Inject wildcard results into prompts |
| **Luna Prompt List Loader** | Load prompts from CSV/JSON/YAML with seeds and LoRA stacks |
| **Luna Batch Prompt Extractor** | Extract prompts from image EXIF metadata |
| **Luna Trigger Injector** | Auto-inject LoRA trigger words into prompts |

**YAML Wildcard Syntax:**
```
{filename}                    → Random template from templates section
{filename:path.to.items}      → Random item from nested path
{filename: text [path.sub]}   → Inline template with [path] substitutions
{1-10}                        → Random integer range
{0.5-1.5:0.1}                 → Random float with step resolution
__path/file__                 → Legacy .txt wildcard (recursive resolution)
```

---

### 🎨 Image Processing & Refinement

| Node | Purpose |
|------|---------|
| **Luna Native Canvas Downscale** | Variance-corrected downscaling for draft generation |
| **Luna Prep Upscaler** | Prepare upscale models with built-in model selection |
| **Luna Scaffold Upscaler** | GPU-accelerated Lanczos with edge preservation |
| **Luna SAM3 Detector** | Semantic concept detection (SAM3) with pre-encoded conditioning |
| **Luna Semantic Detailer** | Surgical detection-based refinement (1024px crops, true batching) |
| **Luna Chess Refiner** | Global tile refinement with chess-pattern seamless blending |
| **Luna Chess Tile Test** | Debug variant for chess tiling verification |
| **Luna Simple Upscaler** | Basic model-based upscaling |
| **Luna Advanced Upscaler** | Supersampling with modulus rounding |
| **Luna Ultimate SD Upscale** | Tile-based diffusion upscaling |
| **Luna Batch Upscale Refine** | **NEW**: Chess-pattern tiling + scaffolding noise + supersampling |
| **Luna Super Upscaler** | SeedVR2-powered mega-resolution upscaling (requires SeedVR2 extension) |
| **Luna Super Upscaler Simple** | Streamlined version of Luna Super Upscaler |
| **Luna USD Clone** | Debug/test variant for USD upscaling |

**Semantic Detailer Suite Workflow:**
```
1. Pyramid Noise → 2. Draft Generation → 3. Scaffold Upscale
4. SAM3 Detection → 5. Semantic Detailer (chainable) → 6. Chess Refiner
```

---

### 📁 LoRA & Embedding Management

| Node | Purpose |
|------|---------|
| **Luna LoRA Stacker** | Stack up to 4 LoRAs with individual strength controls |
| **Luna LoRA Stacker Random** | Randomized LoRA selection from available LoRAs |
| **Luna LoRA Validator** | Validate LoRA files and extract metadata |
| **Luna LoRA Trigger Injector** | Auto-inject trigger words for LoRAs |
| **Luna Connection Matcher** | Sidebar tool for LoRA/embedding ↔ wildcard linking |
| **Luna Connection Editor** | Edit and manage LoRA/embedding connections |
| **Luna Smart LoRA Linker** | Intelligent suggestion for LoRA connections |
| **Luna Connection Stats** | Display statistics about active LoRA/embedding connections |

---

### 🔧 Utilities & Tools

| Node | Purpose |
|------|---------|
| **Luna Civitai Metadata Scraper** | Fetch LoRA metadata from Civitai |
| **Luna Civitai Batch Scraper** | Batch metadata scraping from Civitai |
| **Luna Expression Prompt Builder** | Logic and math expressions in prompts |
| **Luna Expression Slicer Saver** | Process and save expression results |
| **Luna Dimension Scaler** | Scale dimensions to model-native resolutions |
| **Luna Daemon Status** | Check daemon connection and workflow model sets |
| **Luna FB Cache Override** | Configure First-Block caching for performance |

---

### 🌐 Daemon Operations (Internal)

These are used internally by Model Router proxies:

| Proxy Class | Purpose |
|-------------|---------|
| **DaemonCLIP** | Routes text encoding to daemon worker |
| **DaemonVAE** | Routes VAE encode/decode to daemon worker |
| **DaemonModel** | Routes full model forward pass to daemon worker |
| **InferenceModeWrapper** | Local model wrapper for automatic VRAM management |

---

## 🔗 Model Type Requirements

| Type | clip_1 | clip_2 | clip_3 | clip_4 | VAE | Notes |
|------|--------|--------|--------|--------|-----|-------|
| **SD1.5** | CLIP-L | - | - | - | SD VAE | Simple single-encoder |
| **SDXL** | CLIP-L | CLIP-G | - | - | SDXL VAE | Two encoders for conditioning |
| **Flux** | CLIP-L | - | T5-XXL | - | Flux VAE | Different token requirements |
| **SD3** | CLIP-L | CLIP-G | T5-XXL | - | SD3 VAE | All three encoders combined |
| **Z-IMAGE** | Qwen3-VL* | - | - | mmproj** | Any | Full vision-language model |

*Full Qwen3-VL model (not CLIP)  
**Auto-loads if in same folder as Qwen3 model

---

## 📊 Precision Support

### Supported Precisions

| Precision | VRAM | Quality | Hardware | Notes |
|-----------|------|---------|----------|-------|
| **fp16** | 100% | Baseline | All | Standard ComfyUI |
| **bf16** | 100% | Baseline | All | Same VRAM, different numeric stability |
| **fp32** | 200% | Baseline | All | Full precision (rare) |
| **fp8_e4m3fn** | 50% | 99% | RTX 40+, Blackwell | Native hardware support |
| **fp8_e4m3fn_scaled** | 50% | 99.5% | RTX 40+ | **Recommended for 4090/5090** |
| **fp8_e5m2** | 50% | 97% | RTX 30/40+, Blackwell | Better exponent range for RTX 30 |
| **nf4** | 25% | 85% | All (via bitsandbytes) | Aggressive compression |
| **gguf_Q8_0** | 50% | 98% | All | Best quality for GGUF |
| **gguf_Q4_K_M** | 28% | 80% | All | Aggressive quantization |

---

## 💡 Key Use Cases

### High-Throughput Random Generation
```
Model Router (cache precision)
  ↓
Config Gateway (cache weights + extract LoRAs)
  ↓
Loop 2000+ times:
  • YAML Wildcard (random prompts)
  • Luna KSampler (inference)
  • Reset Weights (restore pristine)
```
✅ LoRAs cached in RAM (zero disk I/O between runs)  
✅ Model weights cached (no precision drift)  
✅ Random weights each iteration

### Multi-Workflow Production
```
Instance A (Port 8188): SDXL character generation
Instance B (Port 8189): SDXL background generation
Instance C (Port 8190): Testing/development

Daemon: Multiplexes models
  • Each instance gets correct models
  • Shared VAE reused automatically
  • No unloading between workflows
```

### Vision-Guided Refinement
```
Luna Vision Node (extract style from reference)
  ↓
Luna Config Gateway (inject into prompts)
  ↓
Luna KSampler (generate base image)
  ↓
Luna Semantic Detailer (refine details with vision anchoring)
  ↓
Luna Chess Refiner (global coherence)
```

---

## 🚀 Installation & Setup

### Quick Install
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under **`Luna/`** categories.

### Start the Daemon (Optional, for multi-GPU)
```bash
python luna_daemon/daemon_server.py
```

### Configuration
Edit `luna_daemon/config.py`:
```python
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283
CLIP_DEVICE = "cuda:1"      # GPU for CLIP
VAE_DEVICE = "cuda:0"       # GPU for VAE
```

---

## 🛠️ Troubleshooting

### Model won't load?
1. Check file path exists
2. Verify file is valid safetensors/GGUF
3. Try `force_local` mode in Model Router (bypasses daemon)
4. Check console logs for precision errors

### VRAM usage too high?
1. Use fp8/bf16 instead of fp32
2. Enable Luna KSampler (uses `torch.inference_mode()`)
3. Check daemon is running if multi-GPU intended
4. Verify no other models loading in background

### Daemon connection failed?
1. Start daemon: `python luna_daemon/daemon_server.py`
2. Check port 19283 not blocked
3. Verify network config in `luna_daemon/config.py`
4. Try `force_local` mode to continue

### LoRA weights not resetting?
1. Verify Reset Weights node is connected and executed
2. Check same model used in Config Gateway and Reset node
3. Ensure Config Gateway ran before Reset (workflow order)
4. Clear cache manually: restart ComfyUI

---

## 📚 Documentation

- **[README.md](README.md)** - Quick start and core features
- **[Docs/LUNA_PHILOSOPHY_SHIFT.md](Docs/LUNA_PHILOSOPHY_SHIFT.md)** - IP-Adapter refinement architecture
- **[Docs/file_summaries/](Docs/file_summaries/)** - Technical summaries by module

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Fork, create a feature branch, and open a PR.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LSDJesus/ComfyUI-Luna-Collection/discussions)
- **Technical Docs**: See [Docs/](Docs/) folder

