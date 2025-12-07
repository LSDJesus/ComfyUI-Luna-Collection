# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v1.5.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**A production-grade ComfyUI infrastructure for advanced model management, multi-instance VRAM sharing, and workflow automation.**

Luna Collection is a vertically integrated image generation stack designed for high-throughput workflows. It provides smart model loading with automatic precision conversion, unified model routing for all architectures (SD1.5/SDXL/Flux/SD3/Z-IMAGE), multi-GPU daemon architecture for shared VAE/CLIP, hierarchical YAML wildcards, comprehensive prompt engineering tools, and deep integration with external tools like LUNA-DiffusionToolkit (a Diffusion Toolkit Fork).

---

## ✨ Features

### 🔧 **Core Infrastructure**
- **Luna Model Router**: Unified model loader for all architectures with explicit CLIP configuration
- **Luna Daemon v1.3**: Multi-instance VRAM sharing with split CLIP/VAE architecture
- **Dynamic Model Loader**: JIT precision conversion with smart lazy evaluation
- **Secondary Model Loader**: Multi-model workflows with RAM offloading and CLIP sharing
- **CUDA IPC**: Zero-copy tensor transfer for same-GPU VAE operations
- **F-150 LoRA Support**: Transient LoRA injection for shared CLIP models
- **Connections Manager**: Sidebar UI for LoRA/embedding wildcard linking

### 📦 **Model Management**
- **Unified Model Router**: Single node supporting SD1.5, SDXL, Flux, SD3, Z-IMAGE with vision variants
- **Smart Precision Loading**: bf16, fp8, GGUF Q8_0/Q4_K_M with automatic conversion
- **Hybrid Checkpoint Loading**: CLIP/VAE from source + optimized UNet from NVMe
- **Multi-Model Workflows**: Secondary model loader with CLIP sharing and RAM offloading
- **GGUF Converter**: Convert any checkpoint to quantized GGUF format

### 🎲 **Prompt Engineering**
- **Z-IMAGE Encoder**: Unified prompt input with AI enhancement, vision, and conditioning noise injection
- **YAML Wildcards**: Hierarchical templates with nested path resolution
- **VLM Prompt Generator**: Vision-language model integration for image-guided prompting
- **PromptCraft Engine**: Constraint/modifier/expander system with LoRA linking
- **Prompt List Loader**: CSV/JSON/YAML import with pos/neg/seed/lora_stack outputs
- **Batch Prompt Extractor**: Extract prompts from image EXIF (UTF-16BE support)
- **Config Gateway**: Centralized workflow parameter management
- **Trigger Injector**: Auto-inject LoRA trigger words into prompts

### 🖼️ **Image Processing**
- **Vision Node**: Image-to-embedding for vision-enabled model workflows
- **Advanced Upscaling**: Model-based, tile-based, and multi-stage upscaling
- **Ultimate SD Upscale**: Diffusion-enhanced upscaling with seam fixing
- **Multi-Image Saver**: Batch output with naming templates and EXIF embedding

### 🔗 **External Integrations**
- **DiffusionToolkit Bridge**: API nodes for DT image library integration (planned)
- **Realtime LoRA Training**: Compatible with comfyUI-Realtime-Lora for in-workflow training
- **Civitai Metadata**: Automatic LoRA/embedding metadata scraping

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LUNA COLLECTION v1.3                                   │
│              "Production Image Generation Infrastructure"                       │
└─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                              DAEMON LAYER (Multi-Instance VRAM Sharing)
═══════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Luna Daemon v1.3                                                               │
│  ├── Split Architecture: CLIP (cuda:1 socket) + VAE (cuda:0 IPC)               │
│  ├── F-150 LoRA: TransientLoRAContext + LoRARegistry LRU (2GB cache)           │
│  ├── Length-Prefix Protocol: 4-byte header, O(n) transport                     │
│  └── CUDA IPC: Zero-copy tensor sharing for same-GPU VAE operations            │
└─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                              MODEL MANAGEMENT LAYER
═══════════════════════════════════════════════════════════════════════════════════
┌────────────────────────────┐  ┌────────────────────────────┐  ┌─────────────────┐
│  Luna Model Router ⚡      │  │  Luna Secondary Loader 🔄  │  │  GGUF Converter │
│  ├── All architectures     │  │  ├── Multi-model workflows │  │  ├── Q8_0       │
│  ├── SD1.5/SDXL/Flux/SD3   │  │  ├── CLIP sharing logic   │  │  ├── Q4_K_M     │
│  ├── Z-IMAGE + Vision      │  │  ├── RAM offload/restore  │  │  └── Q4_0       │
│  ├── Explicit CLIP config  │  │  └── Model Restore node   │  └─────────────────┘
│  └── LLM + CLIP_VISION out │  └────────────────────────────┘
└────────────────────────────┘
┌────────────────────────────┐  ┌────────────────────────────┐
│  Luna Dynamic Loader       │  │  Luna Z-IMAGE Encoder      │
│  ├── Smart lazy eval       │  │  ├── AI prompt enhancement │
│  ├── JIT UNet conversion   │  │  ├── Vision-guided prompts │
│  ├── bf16/fp8/Q8_0/Q4_K_M  │  │  ├── Noise injection       │
│  └── HDD source→NVMe opt   │  │  └── Qwen3-VL integration  │
└────────────────────────────┘  └────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                              PROMPT ENGINEERING LAYER
═══════════════════════════════════════════════════════════════════════════════════
┌────────────────────────────┐  ┌────────────────────────────┐  ┌─────────────────┐
│  Luna YAML Wildcard        │  │  Luna Prompt List Loader   │  │  Batch Prompt   │
│  ├── {file:path.to.items}  │  │  ├── CSV/JSON/YAML import  │  │  Extractor      │
│  ├── [inline.substitution] │  │  ├── pos/neg/seed outputs  │  │  ├── EXIF read  │
│  ├── {1-10} numeric ranges │  │  ├── lora_stack output     │  │  ├── Batch proc │
│  └── __legacy/txt__ compat │  │  └── index iteration       │  │  └── UTF-16BE   │
└────────────────────────────┘  └────────────────────────────┘  └─────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- ComfyUI (latest version recommended)
- Python 3.10+
- PyTorch with CUDA support
- (Optional) Multi-GPU setup for daemon architecture

### Quick Install
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
cd ComfyUI-Luna-Collection
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under **`Luna/`** categories.

---

## 🎯 Node Reference

### 🔗 **Luna Daemon (Multi-Instance VRAM Sharing)**

The daemon allows multiple ComfyUI instances to share VAE/CLIP models loaded on a separate GPU.

| Node | Description |
|------|-------------|
| **Luna Shared VAE Encode** | Encode via daemon's shared VAE |
| **Luna Shared VAE Decode** | Decode via daemon's shared VAE |
| **Luna Shared VAE Encode (Tiled)** | Memory-efficient tiled encoding |
| **Luna Shared VAE Decode (Tiled)** | Memory-efficient tiled decoding |
| **Luna Shared CLIP Encode** | Encode via daemon's shared CLIP |
| **Luna Shared CLIP Encode (SDXL)** | SDXL dual CLIP encoding with LoRA support |
| **Luna Daemon Status** | Check daemon connection and model info |

**v1.3 Split Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   GPU 1 (cuda:1)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           CLIP Daemon (:19283)                   │   │
│  │  • CLIP_L + CLIP_G loaded once                  │   │
│  │  • F-150 LoRA: transient injection per-request  │   │
│  │  • LoRARegistry LRU cache (2GB)                 │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   GPU 0 (cuda:0)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           VAE Daemon (:19284)                    │   │
│  │  • Same GPU as UNet = CUDA IPC zero-copy        │   │
│  │  • No socket serialization overhead             │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ComfyUI Instances (UNet only)                  │   │
│  │  :8188, :8189, :8190...                         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Starting the Daemon:**
```bash
# Full daemon (CLIP + VAE on one GPU)
python luna_daemon/server.py

# Split mode - CLIP on cuda:1
python luna_daemon/server.py --service-type clip --device cuda:1 --port 19283

# Split mode - VAE on cuda:0 with IPC
python luna_daemon/server.py --service-type vae --device cuda:0 --port 19284
```

### 📦 **Model Management**

| Node | Description |
|------|-------------|
| **Luna Model Router ⚡** | Unified loader for all architectures (SD1.5/SDXL/Flux/SD3/Z-IMAGE) with explicit CLIP config |
| **Luna Secondary Model Loader 🔄** | Multi-model workflows with CLIP sharing and RAM offloading |
| **Luna Model Restore 📤** | Restore models offloaded to RAM back to VRAM |
| **Luna Dynamic Model Loader** | Smart checkpoint loading with JIT precision conversion |
| **Luna Checkpoint Tunnel** | Pass MODEL through, route CLIP/VAE to daemon |
| **Luna GGUF Converter** | Convert checkpoints to quantized GGUF format |
| **Luna Optimized Weights Manager** | Manage local optimized UNet files |

**Luna Model Router** - The unified model loader:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Luna Model Router ⚡                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  MODEL SOURCE:     [checkpoints ▼] [diffusion_models ▼] [unet (gguf) ▼]    │
│  MODEL NAME:       [ponyDiffusionV6XL.safetensors ▼]                       │
│  MODEL TYPE:       [SD1.5] [SDXL] [SDXL+Vision] [Flux] [Flux+Vision] [SD3] [Z-IMAGE] │
├─────────────────────────────────────────────────────────────────────────────┤
│  DYNAMIC LOADER:   [✓ Enable] → [fp8_e4m3fn ▼] [gguf_Q8_0 ▼]               │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLIP 1:          [clip_l.safetensors ▼]     ← Required for all           │
│  CLIP 2:          [clip_g.safetensors ▼]     ← SDXL, SD3                   │
│  CLIP 3:          [t5xxl_fp16.safetensors ▼] ← Flux, SD3                   │
│  CLIP 4:          [siglip_vision.safetensors ▼] ← Vision models            │
│                                                                             │
│  Z-IMAGE: clip_1 = Full Qwen3-VL model (hidden state extraction)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS: MODEL, CLIP, VAE, LLM, CLIP_VISION, model_name, status           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**CLIP Requirements by Model Type:**
| Model Type | clip_1 | clip_2 | clip_3 | clip_4 |
|------------|--------|--------|--------|--------|
| SD1.5 | CLIP-L | - | - | - |
| SDXL | CLIP-L | CLIP-G | - | - |
| SDXL + Vision | CLIP-L | CLIP-G | - | SigLIP/CLIP-H |
| Flux | CLIP-L | - | T5-XXL | - |
| Flux + Vision | CLIP-L | - | T5-XXL | SigLIP |
| SD3 | CLIP-L | CLIP-G | T5-XXL | - |
| Z-IMAGE | Full Qwen3-VL | - | - | (auto mmproj) |

**Luna Dynamic Model Loader** - The smart precision loader:

```
┌────────────────────────────────────────────────────────┐
│             8TB HDD (Source Library)                   │
│  358 FP16 Checkpoints (6.5GB each)                     │
└────────────────────────────────────────────────────────┘
                          │
                          ▼ First use: extract UNet + convert
┌────────────────────────────────────────────────────────┐
│             NVMe (Local Optimized Weights)             │
│  models/unet/optimized/                                │
│  • illustriousXL_Q8_0.gguf (3.2GB)                     │
│  • ponyV6_fp8_e4m3fn_unet.safetensors (2.1GB)          │
└────────────────────────────────────────────────────────┘
                          │
                          ▼ Smart lazy evaluation
┌────────────────────────────────────────────────────────┐
│  • MODEL always loads optimized UNet                   │
│  • CLIP/VAE only load if outputs are connected         │
│  • No mode selection needed - just wire what you need  │
└────────────────────────────────────────────────────────┘
```

**Supported Precisions:**
| Precision | Best For | Size Reduction |
|-----------|----------|----------------|
| `bf16` | Universal, fast | ~50% |
| `fp8_e4m3fn` | Ada/Blackwell GPUs | ~75% |
| `gguf_Q8_0` | Ampere INT8 tensor cores | ~50% |
| `gguf_Q4_K_M` | Blackwell INT4 tensor cores | ~75% |

### 🎲 **YAML Wildcards**

| Node | Description |
|------|-------------|
| **Luna YAML Wildcard** | Hierarchical wildcard expansion |
| **Luna YAML Wildcard Batch** | Generate multiple prompts with seeds |
| **Luna Wildcard Builder** | Visual prompt composition |
| **Luna LoRA Randomizer** | Random LoRA selection from YAML |

**Prompt Syntax:**
```
{filename}                    → Random template from templates section
{filename:path.to.items}      → Random item from nested path
{filename: text [path.sub]}   → Inline template with substitutions
{1-10}                        → Random integer
{0.5-1.5:0.1}                 → Random float with step
__path/file__                 → Legacy .txt wildcard
```

**Example YAML (`models/wildcards/characters.yaml`):**
```yaml
templates:
  hero:
    - "a [appearance.build] [species.humanoid] with [features.eyes]"
    
appearance:
  build:
    - muscular
    - slender
    - athletic
    
species:
  humanoid:
    - elf
    - human
    - tiefling
    
features:
  eyes:
    - glowing blue eyes
    - heterochromatic eyes
```

### 📝 **Prompt Engineering**

| Node | Description |
|------|-------------|
| **Luna Z-IMAGE Encoder 🧠** | AI-enhanced encoding with Qwen3-VL, vision modes, noise injection |
| **Luna Vision Node 👁️** | Describe/extract style from reference images |
| **Luna VLM Prompt Generator 💬** | Generate prompts from images using vision LLM |
| **Luna Prompt List Loader** | Load prompts from CSV/JSON/YAML files |
| **Luna Batch Prompt Extractor** | Extract prompts from image EXIF metadata |
| **Luna Config Gateway** | Centralized workflow parameters |
| **Luna Trigger Injector** | Auto-inject LoRA trigger words |
| **Luna Expression Pack** | Logic and math expressions for workflows |

**Luna Z-IMAGE Encoder** - Unified prompt processing for Z-IMAGE models:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Luna Z-IMAGE Encoder 🧠                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  PROMPT:        "anime girl, detailed, colorful"                           │
│  AI ENHANCEMENT: [off] [subtle] [moderate] [maximum]                       │
│                                                                             │
│  VISION MODE:   [disabled] [describe] [extract_style] [blend]              │
│  IMAGE INPUT:   [optional reference image]                                 │
│                                                                             │
│  NOISE INJECTION: [✓ Enable] strength: 0.02  schedule: start_percent: 0.3  │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS: CONDITIONING (with style/noise), enhanced_prompt                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Vision Modes:**
| Mode | Description | Best For |
|------|-------------|----------|
| `disabled` | Text-only encoding | Pure text2img |
| `describe` | VLM describes image → expands prompt | Character/scene reference |
| `extract_style` | Extract artistic style → inject as suffix | Style transfer |
| `blend` | Fuse text + image embeddings (0.0-1.0) | Image variations |

**Luna Prompt List Loader** outputs:
- `positive` - Positive prompt string
- `negative` - Negative prompt string  
- `seed` - Per-prompt seed (or -1 for random)
- `lora_stack` - LoRA stack tuple for Apply LoRA Stack
- `index` - Current iteration index

### 📁 **LoRA & Embedding Management**

| Node | Description |
|------|-------------|
| **Luna LoRA Stacker** | Stack up to 4 LoRAs with strength controls |
| **Luna LoRA Stacker Random** | Randomized LoRA selection |
| **Luna Embedding Manager** | Textual inversion management |
| **Luna Embedding Manager Random** | Randomized embedding selection |
| **Luna LoRA Validator** | Validate LoRA files and extract metadata |
| **Luna Connections Manager** | Sidebar UI for LoRA/embedding ↔ wildcard linking |

### 🖼️ **Image Processing**

| Node | Description |
|------|-------------|
| **Luna Simple Upscaler** | Clean model-based upscaling |
| **Luna Advanced Upscaler** | Supersampling, modulus rounding |
| **Luna Ultimate SD Upscale** | Tile-based SD upscaling |
| **Luna Multi Saver** | Batch saving with templates |

### 🔧 **Utilities**

| Node | Description |
|------|-------------|
| **Luna Civitai Metadata Scraper** | Fetch LoRA metadata from Civitai |
| **Luna Expression Pack** | Logic and math expressions |
| **Luna Dimension Scaler** | Scale to model-native resolutions |

---

## 🔗 External Tool Integration

### Realtime LoRA Training (comfyUI-Realtime-Lora)

Luna Collection is designed to work seamlessly with [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) for in-workflow SDXL LoRA training:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 REALTIME LORA TRAINING WORKFLOW                     │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐     ┌──────────────────────────────────────┐
  │ Luna Batch Prompt    │────▶│ images_path folder with .txt captions│
  │ Extractor (export)   │     └──────────────────┬───────────────────┘
  └──────────────────────┘                        │
                                                  ▼
                               ┌──────────────────────────────────────┐
                               │ Realtime LoRA Trainer (SDXL)         │
                               │ • sd_scripts_path: D:/AI/.../sd-scripts
                               │ • ckpt_name: illustrious_v1.safetensors
                               │ • Uses kohya sd-scripts                │
                               └──────────────────┬───────────────────┘
                                                  │ lora_path
                                                  ▼
                               ┌──────────────────────────────────────┐
                               │ Apply Trained LoRA                   │
                               └──────────────────┬───────────────────┘
                                                  │
                                                  ▼
                               ┌──────────────────────────────────────┐
                               │ KSampler (generate with new LoRA)    │
                               └──────────────────────────────────────┘
```

**Setup:** Create a junction so sd-scripts can use ComfyUI's venv:
```powershell
New-Item -ItemType Junction -Path "D:\path\to\sd-scripts\.venv" -Target "D:\AI\ComfyUI\venv"
```

### DiffusionToolkit Bridge (Planned)

See [docs/LUNA_TOOLKIT_BRIDGE_NODES.md](docs/LUNA_TOOLKIT_BRIDGE_NODES.md) for planned integration nodes that enable:
- Query DT image library from ComfyUI
- Similar image search via embeddings
- Cluster-based sampling
- Caption fetching
- Metadata writeback

---

## 📚 Technical Deep Dives

### Luna Daemon Protocol

**Length-Prefix Protocol (v1.3):**
```
┌──────────────────┬─────────────────────────────────┐
│ 4-byte uint32    │ JSON payload (exact length)     │
│ payload length   │                                 │
└──────────────────┴─────────────────────────────────┘
```

Replaces the old `<<END>>` sentinel pattern which required O(n²) string scanning.

**F-150 LoRA Architecture:**
```python
# TransientLoRAContext - thread-safe LoRA injection
with TransientLoRAContext(clip_model, lora_stack, registry):
    # 1. Lock acquired
    # 2. LoRA weights loaded from registry (LRU cached)
    # 3. Weights injected via add_patches()
    # 4. Encode happens here
    # 5. Weights restored on exit
    # 6. Lock released
```

### Dynamic Loader Smart Evaluation

The loader uses ComfyUI's `check_lazy_status` to detect connected outputs:

```python
def check_lazy_status(self, ckpt_name, precision, ...):
    # Always need MODEL and unet_path
    needed = [0, 3]
    
    # Check graph for CLIP/VAE connections
    if self._is_output_connected(graph, node_id, 1):  # CLIP
        needed.append(1)
    if self._is_output_connected(graph, node_id, 2):  # VAE
        needed.append(2)
    
    return needed
```

This means:
- **MODEL only connected**: Just loads optimized UNet (~2-4GB)
- **MODEL + CLIP**: Loads UNet + extracts CLIP from source
- **MODEL + VAE**: Loads UNet + extracts VAE from source
- **All connected**: Full hybrid load

### CUDA IPC Zero-Copy

When VAE daemon runs on the same GPU as ComfyUI:

```python
# Client side
tensor.share_memory_()  # Move to shared memory
handle = tensor.storage()._share_cuda_()

# Send handle via socket (tiny metadata, not tensor data)
response = send_ipc_request(handle, shape, dtype)

# Server side - reconstructs tensor from handle
tensor = torch.zeros(shape, dtype=dtype, device=device)
tensor.storage()._set_from_cuda_ipc_handle_(handle)
```

Result: 13 VAE operations per iteration with zero serialization overhead.

---

## 🏗️ Project Structure

```
ComfyUI-Luna-Collection/
├── nodes/                          # Node implementations
│   ├── loaders/                    # Model loading nodes
│   │   ├── luna_model_router.py    # Unified multi-architecture loader
│   │   ├── luna_secondary_loader.py # Multi-model + RAM offload
│   │   ├── luna_dynamic_loader.py  # JIT precision conversion
│   │   └── luna_checkpoint_tunnel.py
│   ├── promptcraft/                # Prompt engineering nodes
│   │   ├── engine.py               # YAML parser engine
│   │   └── nodes.py                # Wildcard nodes
│   ├── upscaling/                  # Upscaler nodes
│   ├── luna_zimage_encoder.py      # Z-IMAGE AI encoder + vision
│   ├── luna_vision_node.py         # VLM-based image analysis
│   ├── luna_vlm_prompt_generator.py # Vision → prompt
│   ├── luna_yaml_wildcard.py       # YAML wildcard system
│   ├── luna_batch_prompt_extractor.py
│   ├── luna_config_gateway.py
│   ├── luna_multi_saver.py
│   └── ...
├── luna_daemon/                    # Multi-instance daemon
│   ├── server.py                   # Daemon server (dynamic scaling)
│   ├── client.py                   # Client library
│   ├── proxy.py                    # DaemonVAE/DaemonCLIP proxies
│   └── config.py                   # Configuration
├── utils/                          # Shared utilities
│   ├── luna_metadata_db.py         # SQLite metadata
│   └── ...
├── js/                             # Frontend JavaScript
├── tests/                          # Test suite
└── __init__.py
```

---

## 🔧 Configuration

### Daemon Configuration (`luna_daemon/config.py`)

```python
# Service type for split architecture
class ServiceType(Enum):
    FULL = "full"           # CLIP + VAE on same GPU
    CLIP_ONLY = "clip"      # CLIP daemon only
    VAE_ONLY = "vae"        # VAE daemon only

# Network
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283         # CLIP daemon
DAEMON_VAE_PORT = 19284     # VAE daemon (split mode)

# GPU Assignment
SHARED_DEVICE = "cuda:1"    # For CLIP
VAE_DEVICE = "cuda:0"       # For VAE (same as UNet = IPC eligible)

# Model Paths
VAE_PATH = "models/vae/sdxl_vae.safetensors"
CLIP_L_PATH = "models/clip/clip_l.safetensors"
CLIP_G_PATH = "models/clip/clip_g.safetensors"

# LoRA Cache
LORA_CACHE_MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB LRU
```

### Dynamic Loader Configuration

The loader stores optimized UNets in `models/unet/optimized/` by default.
Override with the `local_weights_dir` input.

---

## 📈 Changelog

### v1.5.0 - Current (2025-06)
- ✅ **Luna Model Router**: Unified loader for ALL architectures (SD1.5/SDXL/Flux/SD3/Z-IMAGE) with explicit 4-slot CLIP configuration
- ✅ **Luna Secondary Model Loader**: Multi-model workflows with CLIP sharing and RAM offloading via ModelMemoryManager
- ✅ **Luna Model Restore**: Companion node to restore RAM-offloaded models back to VRAM
- ✅ **Luna Z-IMAGE Encoder**: AI-enhanced prompt encoding with Qwen3-VL, vision modes (describe/extract_style/blend), built-in noise injection
- ✅ **Luna Vision Node**: Describe images or extract artistic style using VLM
- ✅ **Luna VLM Prompt Generator**: Generate prompts from reference images
- ✅ **Auto-Discovery Node Registration**: `os.walk()` based node discovery from subdirectories
- ✅ **LLM Output Support**: Model Router outputs LLM for Z-IMAGE (Qwen3-VL) workflows
- ✅ **CLIP_VISION Output**: Direct CLIP vision model output for vision-enabled architectures

### v1.4.0 (2025-12)
- ✅ **Connections Manager Sidebar**: LoRA/embedding ↔ wildcard category linking UI
- ✅ **PromptCraft Engine**: Intelligent prompt generation with constraints/modifiers/expanders
- ✅ **DynamicPrompt API Update**: Fixed compatibility with latest ComfyUI graph API
- ✅ **Realtime LoRA Training Integration**: Documentation for sd-scripts integration
- ✅ **DiffusionToolkit Bridge Spec**: Planned nodes for DT ↔ ComfyUI communication
- ✅ **Expression Pack**: Logic and math expression nodes
- ✅ **Trigger Injector**: Auto-inject LoRA trigger words into prompts

### v1.3.0 (2025-12)
- ✅ **Split Daemon Architecture**: Separate CLIP/VAE daemons for optimal GPU placement
- ✅ **CUDA IPC**: Zero-copy tensor transfer for same-GPU VAE operations
- ✅ **F-150 LoRA**: Transient LoRA injection for shared CLIP with LRU cache
- ✅ **Length-Prefix Protocol**: O(n) transport replacing O(n²) sentinel scanning
- ✅ **Luna Dynamic Model Loader**: JIT precision conversion with smart lazy evaluation
- ✅ **Smart Output Detection**: CLIP/VAE only load when outputs are connected
- ✅ **Hybrid Loading**: CLIP/VAE from FP16 source + optimized UNet from NVMe
- ✅ **GGUF Support**: Q8_0 and Q4_K_M quantization for Ampere/Blackwell

### v1.2.0 (2025-11-29)
- ✅ **YAML Wildcard System**: Hierarchical wildcards with templates
- ✅ **Luna Daemon**: Multi-instance VRAM sharing
- ✅ **Civitai Integration**: Automatic metadata scraping
- ✅ **SQLite Metadata Database**: Local storage with full-text search
- ✅ **Batch Prompt Extractor**: EXIF parsing with UTF-16BE support

### v1.1.0 (2025-09-21)
- ✅ **TensorRT Integration**: High-performance detailing
- ✅ **Enhanced LoRA Stacker**: Individual toggles, proper tuple format

### v1.0.0 (2025-08-22)
- 🎯 Initial release with upscalers, LoRA management, prompt processing

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for high-throughput image generation*

