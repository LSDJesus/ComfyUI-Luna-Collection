# 🌙 ComfyUI Luna Collection

![Version](https://img.shields.io/badge/version-v2.3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

**A production-grade ComfyUI infrastructure for advanced model management, multi-workflow VRAM sharing, and high-throughput image generation.**

Luna Collection is a vertically integrated image generation stack designed for enterprise-scale workflows. It provides workflow-aware daemon architecture for intelligent model multiplexing, unified model routing for all architectures (SD1.5/SDXL/Flux/SD3/Z-IMAGE), transient LoRA caching for zero-reload workflows, hierarchical YAML wildcards, comprehensive prompt engineering tools, and deep integration with external tools.

---

## 📝 Latest Updates (v2.3)

🎯 **IP-Adapter TRUE BATCHING Integration** - Revolutionary refinement system
- **IP-Adapter Structural Anchoring**: Proper vision→attention injection via learned projections, not naive fusion
- **TRUE BATCHING Architecture**: Batch dimension preserved - Latent[i] sees Embed[i], no averaging
- **12× Speed Improvement**: Semantic Detailer batches all detections in one sample call
- **6× Speed Improvement**: Chess Refiner batches 13 even + 12 odd tiles in two passes
- **Per-Detection Uniqueness**: Each face/object gets its own unique CLIP-ViT anchor
- **Integrated Upscale Loader**: Prep Upscaler includes built-in upscale model selection (4x-UltraSharp recommended)
- **100% Pixel-Space Refinement**: Semantic Detailer and Chess Refiner work entirely on pixels (crops → encode fresh → refine → decode → paste)
- See [LUNA_PHILOSOPHY_SHIFT.md](LUNA_PHILOSOPHY_SHIFT.md) for architectural deep-dive

🎨 **Luna Semantic Detailer Suite** - Surgical pyramid-based refinement system
- **Native Canvas Downscale**: Variable variance correction (0.0-1.0) for soft draft generation, optional area conditioning downscale
- **Scaffold Upscaler**: GPU-accelerated Lanczos, edge-preserving + texture coherence for artifact-free upscaling
- **SAM3 Detector**: Semantic concept detection with pre-encoded conditioning, per-concept prompts, hierarchical layers
- **Semantic Detailer**: Per-detection IP-Adapter anchoring, 1024px crops, true batching, chainable multi-layer refinement
- **Chess Refiner**: Global tile refinement with IP-Adapter vision anchoring, chess pattern for seamless blending
- **Full daemon integration**: SAM3 runs on secondary GPU, shared CLIP encoding for multi-detection batching

✨ **Luna Batch Upscale Refine** - Production-grade tiled upscaler with scaffolding noise + chess-pattern batching
- Auto-detect upscale factor (1x/2x/4x/8x/16x)
- Latent-space tiling for 64x smaller tensor operations
- Sigmoid blending mode + feathering control
- GPU Lanczos supersampling (e.g., refine 4x, output 2x)
- Tiled VAE decode prevents boundary artifacts

🎯 **FP8 Precision Expansion** - Now supports all three FP8 variants
- `fp8_e4m3fn` - RTX 40-series native (5090/4090)
- `fp8_e4m3fn_scaled` - RTX 40-series recommended
- `fp8_e5m2` - RTX 30-series better exponent range

🦙 **Qwen3-VL GGUF Support** - Z-IMAGE now works with quantized GGUF models
- Uses patched llama-cpp-python fork
- Q8_0 for quality, Q4_K_M for efficiency
- Auto-detect format, mmproj auto-loads

---

## ✨ Key Features

### 🚀 **Workflow-Aware Multi-Instance Architecture**
- **Multi-Workflow Multiplexing**: Run multiple workflows simultaneously sharing CLIP/VAE models
- **Intelligent Model Routing**: Daemon tracks which models each workflow needs, sideloads new ones without unloading
- **Zero Redundancy**: Workflows sharing same VAE use one loaded instance, not duplicate copies
- **InferenceModeWrapper**: Automatic VRAM management for UNet models
- **Workflow Isolation**: Each workflow gets correct model set despite shared infrastructure

### 🔧 **Core Infrastructure**
- **Luna Model Router**: Unified model loader for all architectures with explicit CLIP/VAE configuration
- **Luna Daemon v2.0**: Multi-workflow daemon with per-workflow model tracking and sideloading
- **Dynamic Precision Conversion**: JIT bf16/fp8/GGUF conversion with intelligent caching
- **Transient LoRA System**: LoRAs cached in RAM, applied with randomized weights, restored without disk I/O
- **Config Gateway**: Centralized workflow parameter management with LoRA weight caching
- **Reset Weights Node**: Ctrl-Z for LoRA modifications between workflow runs

### 📦 **Model Management**
- **Unified Model Router**: Single node supporting SD1.5, SDXL, Flux, SD3, Z-IMAGE with vision variants
- **Smart Precision Loading**: bf16, fp8, GGUF Q8_0/Q4_K_M with automatic conversion and caching
- **Explicit CLIP/VAE Selection**: Dynamic selectors updated based on model_type
- **Precision Conversion Cache**: Converted models saved to correct directories (e.g., `unet/fp8/`)
- **InferenceModeWrapper**: UNet models wrapped for automatic VRAM management

### 🎲 **Prompt Engineering**
- **YAML Wildcards**: Hierarchical templates with nested path resolution and inline substitution
- **LoRA Weight Randomization**: Same LoRAs, different random weights per run, no reload
- **Config Gateway Integration**: Automatic LoRA extraction from prompts, deduplication, caching
- **Prompt List Loader**: CSV/JSON/YAML import with pos/neg/seed/lora_stack outputs
- **Trigger Injector**: Auto-inject LoRA trigger words into prompts

### 🖼️ **Image Processing**
- **Vision Node**: Image-to-embedding for vision-enabled model workflows
- **Advanced Upscaling**: Model-based, tile-based, and multi-stage upscaling
- **Ultimate SD Upscale**: Diffusion-enhanced upscaling with seam fixing
- **Multi-Image Saver**: Batch output with naming templates and EXIF embedding

---

## 🏗️ Architecture v2.0

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LUNA COLLECTION v2.0                                   │
│          "Multi-Workflow Image Generation Infrastructure"                       │
└─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                    WORKFLOW-AWARE DAEMON (Multi-Instance Multiplexing)
═══════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Luna Daemon v2.0 - Workflow Multiplexer                                        │
│                                                                                 │
│  workflow_model_sets = {                                                        │
│    "workflow_A": {models: {clip_l: path_A, clip_g: path_A, vae: path_shared}}  │
│    "workflow_B": {models: {clip_l: path_B, clip_g: path_B, vae: path_shared}}  │
│  }                                                                              │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Worker Pool (VAE)                                                      │   │
│  │  • Worker 1: vae_shared.safetensors (shared by A & B)                  │   │
│  │  • Routes VAE ops to correct worker based on workflow_id               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Worker Pool (CLIP)                                                     │   │
│  │  • Worker 1: clip_l_A, clip_g_A                                        │   │
│  │  • Worker 2: clip_l_B, clip_g_B                                        │   │
│  │  • Routes CLIP ops to correct worker based on workflow_id              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Benefits:                                                                      │
│  • No model unloading - all models stay loaded                                 │
│  • Shared models reused across workflows (VAE example above)                   │
│  • New workflows trigger sideloading, not replacement                          │
│  • Intelligent routing ensures correct models per workflow                     │
└─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                              MODEL ROUTER & LOADING
═══════════════════════════════════════════════════════════════════════════════════
┌────────────────────────────┐  ┌────────────────────────────┐
│  Luna Model Router ⚡      │  │  InferenceModeWrapper      │
│  ├── All architectures     │  │  ├── Auto VRAM management  │
│  ├── SD1.5/SDXL/Flux/SD3   │  │  ├── Wraps loaded UNet     │
│  ├── Z-IMAGE + Vision      │  │  ├── Lazy loading support  │
│  ├── Dynamic CLIP selectors│  │  └── Transparent to nodes  │
│  ├── Precision conversion  │  └────────────────────────────┘
│  └── Daemon proxy creation │
└────────────────────────────┘

Flow: Router → Precision Convert → InferenceModeWrapper → Daemon Proxies

═══════════════════════════════════════════════════════════════════════════════════
                          TRANSIENT LORA SYSTEM (Zero-Reload Workflow)
═══════════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Config Gateway                           Reset Weights Node                    │
│  ├── Cache pristine weights (affected     ├── Restore cached weights           │
│  │   layers only, ~5-10% of model)        ├── Clear cache                      │
│  ├── Apply LoRAs with random weights      └── Prepare for next run             │
│  └── LoRAs stay in RAM (daemon cache)                                          │
│                                                                                 │
│  Workflow Run 1:                          Workflow Run 2:                       │
│  • Cache weights                          • Restore from cache                 │
│  • Apply lora_1@0.75, lora_2@1.2         • Apply lora_1@0.42, lora_2@0.88    │
│  • Inference                              • Inference                          │
│  • Reset → pristine state                 • Reset → pristine state             │
│                                                                                 │
│  Benefits:                                                                      │
│  • No disk I/O between runs (LoRAs cached in RAM)                             │
│  • No precision drift (exact clone restoration)                                │
│  • Supports randomized LoRA weights per run                                    │
│  • Minimal memory overhead (only affected layers cached)                       │
└─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                              PROMPT ENGINEERING
═══════════════════════════════════════════════════════════════════════════════════
┌────────────────────────────┐  ┌────────────────────────────┐
│  Luna YAML Wildcard        │  │  Luna Config Gateway       │
│  ├── {file:path.to.items}  │  │  ├── Auto LoRA extraction  │
│  ├── [inline.substitution] │  │  ├── LoRA deduplication    │
│  ├── {1-10} numeric ranges │  │  ├── Weight caching        │
│  └── __legacy/txt__ compat │  │  └── Centralized params    │
└────────────────────────────┘  └────────────────────────────┘
```

---

## 🎨 Luna Semantic Detailer Suite

**A hierarchical, multi-pass refinement system for surgical image enhancement.** Replaces blind tiled upscaling with semantic-aware pyramidal refinement.

### 🏗️ Workflow Architecture

```
1. Pyramid Noise Generator
   ├─ Model-aware (SDXL, SD1.5, Flux)
   ├─ Aspect ratio selection (1:1, 16:9, 3:2, etc.)
   ├─ Outputs: full_scaffold (4K), draft_scaffold (1K)
   └─ Variance correction: σ=1.0 preserved at all scales

2. Draft Generation
   ├─ KSampler on draft_scaffold (1K fast)
   ├─ VAE decode to pixels (1K neutral image)
   └─ Input to detector for fast analysis

3. Scaffold Upscaler
   ├─ Lanczos GPU-accelerated upscale (no upscale model)
   ├─ Edge preservation + texture coherence
   ├─ Creates neutral 4K canvas (no AI artifacts)
   └─ Outputs: upscaled_pixels (4K), full_scaffold_passthrough

4. SAM3 Detector
   ├─ Detects objects on 1K draft (fast)
   ├─ Per-concept prompts (face, eye, hand, etc.)
   ├─ Hierarchical layers (0=structural, 1+=details)
   ├─ Encodes prompts with CLIP upfront
   └─ Outputs: LUNA_DETECTION_PIPE (coordinates + conditioning)

5. Semantic Detailer (Chainable, Multi-Layer)
   ├─ Extracts crops from 4K canvas
   ├─ Refinement at 1024×1024 (optimal for SDXL/Flux)
   ├─ Batched sampling with per-concept conditioning
   ├─ Supports enlarge_crops for small inputs
   ├─ Outputs: refined_image + refined_latent + detection_pipe (passthrough)
   └─ Chaining: Layer 0 → Layer 1 → Layer 2 (cumulative refinement)

6. Chess Refiner (Final Global Pass)
   ├─ Chess-pattern tiling (even/odd for seamless blending)
   ├─ Uses full_scaffold for 1:1 noise density
   ├─ Optional supersampling (0.25-1.0x scale)
   ├─ Smoothstep blending (invisible seams)
   └─ Outputs: final_image (2K supersampled)
```

### 🔬 Key Mathematical Principles

**Variance Preservation:**
```
When downscaling noise: σ_new = σ_original / scale_factor
Solution: Multiply by scale_factor to restore σ = 1.0
Example: 4K→1K (4x) = multiply by 4.0
```

**1024px Standard:**
- SDXL native training resolution
- Optimal for anatomical features
- True GPU batch processing

**Smoothstep Blending:**
- Polynomial: t²(3-2t)
- C¹ continuity (no visible seams)
- Better than linear alpha blending

### 💡 Use Cases

**Pyramid Workflow (4K Refinement):**
```
Pyramid Noise (4K) → Draft (1K) → Scaffold Up (4K)
→ Detect → Semantic Detailer (surgical) → Chess (global) → 2K output
```
✅ Maximum quality  
✅ True 1:1 noise preservation  
✅ Multi-layer specialization possible

**Traditional Workflow (1K Base):**
```
1K image → batch_upscale_refine (4x to 4K)
→ Semantic Detailer (enlarge_crops=True) → Final output
```
✅ Compatible with existing workflows  
✅ Uses same detailer nodes  
✅ Upscales detected regions

**Layered LoRA Refinement:**
```
Base generation (1:1)
→ Detailer Layer 0 + face_lora (detailed faces)
→ Detailer Layer 1 + eye_lora (iris details)
→ Detailer Layer 2 + clothing_lora (fabric texture)
→ Chess Refiner (global coherence)
```
✅ Each layer specializes  
✅ Per-layer conditioning  
✅ No quality degradation from multi-pass

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

## 🎯 Core Workflows

### Single Workflow with Precision Conversion

```
[Luna Model Router]
  ├─ model_source: checkpoints
  ├─ model_name: illustriousXL.safetensors
  ├─ model_type: SDXL
  ├─ dynamic_precision: fp8_e4m3fn
  ├─ clip_1: clip_l.safetensors
  ├─ clip_2: clip_g.safetensors
  └─ vae_name: sdxl_vae.safetensors
       ↓
  OUTPUT: MODEL (InferenceModeWrapper), CLIP, VAE
       ↓
[Config Gateway] → [KSampler] → [Reset Weights]
```

### Multi-Workflow Daemon Setup

**Instance A** (Port 8188):
```
[Model Router] 
  ├─ SDXL + clip_l_A, clip_g_A, vae_shared
  └─ daemon_mode: auto
       ↓
  Daemon receives: workflow_id="A", models={clip_l: path_A, ...}
  Daemon creates: Worker 1 (clip_A), Worker 3 (vae_shared)
```

**Instance B** (Port 8189):
```
[Model Router]
  ├─ SDXL + clip_l_B, clip_g_B, vae_shared  
  └─ daemon_mode: auto
       ↓
  Daemon receives: workflow_id="B", models={clip_l: path_B, ...}
  Daemon sideloads: Worker 2 (clip_B), reuses Worker 3 (vae_shared)
```

Both workflows share VAE, each has own CLIP, no model unloading.

### High-Throughput Random Generation

```
[Model Router] → [Config Gateway] → [YAML Wildcard]
                      ↓                    ↓
                 Cache weights        Random prompts
                 Extract LoRAs        Random LoRA weights
                      ↓
                 [KSampler] → [Save Image] → [Reset Weights]
                                                  ↓
                                            Restore pristine
                                            Ready for next run
```

Run 2000+ times/day with same 6-7 LoRAs, different random weights each time, zero disk I/O.

---

## 📦 Node Reference

### 🔧 **Model Loading**

| Node | Description |
|------|-------------|
| **Luna Model Router ⚡** | Unified loader for all architectures with dynamic CLIP/VAE selectors and precision conversion |
| **Luna Dynamic Model Loader** | Legacy smart checkpoint loading (use Model Router instead) |
| **Luna GGUF Converter** | Convert checkpoints to quantized GGUF format |

**Model Router Outputs:**
- `MODEL` - UNet wrapped in InferenceModeWrapper (or DaemonModel if daemon enabled)
- `CLIP` - DaemonCLIP proxy (routes to daemon) or local CLIP
- `VAE` - DaemonVAE proxy (routes to daemon) or local VAE
- `LLM` - Full LLM for Z-IMAGE (Qwen3-VL)
- `CLIP_VISION` - Vision encoder for vision model types
- `model_name` - String for Config Gateway
- `status` - Detailed loading status

### 🌐 **Luna Daemon (Multi-Workflow Architecture)**

Daemon now uses workflow-aware multiplexing - each workflow gets its own model set, shared models are reused.

| Node | Description |
|------|-------------|
| **Luna Daemon Status** | Check daemon connection and loaded workflow model sets |

**Starting the Daemon:**
```bash
# Start daemon server
python luna_daemon/daemon_server.py

# Or use PowerShell script
.\scripts\start_daemon.ps1
```

**Client API (used by proxies):**
```python
# Request models for a workflow
daemon_client.get_model_proxies(
    workflow_id="my_workflow_123",  # Unique per ComfyUI instance
    model_type="SDXL",
    models={
        "clip_l": "/path/to/clip_l.safetensors",
        "clip_g": "/path/to/clip_g.safetensors",
        "vae": "/path/to/vae.safetensors"
    }
)

# All subsequent CLIP/VAE ops include workflow_id
daemon_client.clip_encode("prompt", workflow_id="my_workflow_123")
daemon_client.vae_decode(latents, workflow_id="my_workflow_123")
```

### 🎲 **Workflow Management**

| Node | Description |
|------|-------------|
| **Luna Config Gateway** | Centralized workflow parameters with LoRA weight caching |
| **Luna Reset Model Weights** | Restore model to pristine state after LoRA application |

**Config Gateway Features:**
- Auto-extracts LoRAs from prompts (`<lora:name:weight>` syntax)
- Deduplicates with lora_stack input
- Caches pristine weights before LoRA application
- Applies LoRAs with specified (or randomized) weights
- Outputs complete workflow config for image EXIF

**Reset Weights Node:**
- Place at end of workflow
- Restores cached weights (no disk I/O, no precision drift)
- Clears cache to free memory
- Prepares model for next run with different LoRA weights
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

### 🎲 **YAML Wildcards**

| Node | Description |
|------|-------------|
| **Luna YAML Wildcard** | Hierarchical wildcard expansion with templates and nested paths |
| **Luna YAML Wildcard Batch** | Generate multiple prompts with seeds for batch workflows |
| **Luna Wildcard Builder** | Visual prompt composition with real-time preview |
| **Luna LoRA Randomizer** | Random LoRA selection from YAML files with weight control |

**Prompt Syntax:**
```
{filename}                    → Random template from templates section
{filename:path.to.items}      → Random item from nested path
{filename: text [path.sub]}   → Inline template with [path] substitutions
{1-10}                        → Random integer range
{0.5-1.5:0.1}                 → Random float with step resolution
__path/file__                 → Legacy .txt wildcard (recursive)
```

**Example YAML** (`models/wildcards/characters.yaml`):
```yaml
templates:
  hero:
    - "a [appearance.build] [species.humanoid] with [features.eyes]"
    - "[species.humanoid] [appearance.build] character, [features.eyes]"
    
appearance:
  build: [muscular, slender, athletic, stocky]
    
species:
  humanoid: [elf, human, tiefling, dwarf]
    
features:
  eyes:
    - glowing blue eyes
    - heterochromatic eyes
    - emerald eyes
```

**Usage in Prompt:**
```
{characters:hero} warrior in armor
→ "a muscular elf with glowing blue eyes warrior in armor"
```

---

## 🔄 Migration from v1.x

### Key Changes in v2.0

**Daemon Architecture:**
- Old: Split CLIP/VAE daemons on separate GPUs
- New: Unified daemon with workflow-aware multiplexing
- Migration: Update daemon startup scripts, remove split config

**Model Loading:**
- Old: Dynamic Model Loader with lazy evaluation
- New: Model Router handles everything (precision, CLIP, VAE, daemon)
- Migration: Replace Dynamic Loader nodes with Model Router

**LoRA System:**
- Old: Manual LoRA loading, reload from disk each run
- New: Transient LoRA caching, weight restoration via Reset node
- Migration: Add Reset Weights node at end of workflows

**Config Gateway:**
- Old: Basic parameter passing
- New: LoRA weight caching, automatic extraction/deduplication
- Migration: No changes needed, just benefits from new features

---

## 💡 Use Cases

### High-Throughput Random Generation
**Scenario**: Generate 2000+ images/day with randomized prompts and LoRA weights

```
Workflow Setup:
├─ [Model Router] → Load finetuned Illustrious model with fp8 precision
├─ [Config Gateway] → Extract 6-7 LoRAs from prompt (same set each run)
├─ [YAML Wildcard] → Random prompts with {character}, {pose}, {background}
├─ [Random LoRA Weights] → Randomize strengths (0.5-1.5) each run
├─ [KSampler] → Generate image
└─ [Reset Weights] → Restore model to pristine state

Benefits:
• LoRAs cached in RAM after first load (zero disk I/O)
• Model weights cached before LoRA application
• Each run: restore cache → apply random weights → infer → reset
• Time saved: ~1 second per workflow × 2000 runs = 33 minutes/day
```

### Multi-Workflow Production Setup
**Scenario**: Multiple ComfyUI instances running different workflows simultaneously

```
Instance A (Port 8188) - Character Generation:
├─ Model: characterMix_SDXL
├─ CLIP: clip_l_custom, clip_g_custom
├─ VAE: sdxl_vae (shared)
└─ Daemon: workflow_id="char_gen"

Instance B (Port 8189) - Background Generation:
├─ Model: landscapeMix_SDXL  
├─ CLIP: clip_l_standard, clip_g_standard
├─ VAE: sdxl_vae (shared - reused from A!)
└─ Daemon: workflow_id="bg_gen"

Instance C (Port 8190) - Testing/Development:
├─ Model: testMix_SDXL
├─ CLIP: clip_l_custom (reused from A!)
├─ VAE: sdxl_vae (shared - reused from A!)
└─ Daemon: workflow_id="testing"

Daemon State:
• 5 total CLIPs loaded (clip_l_custom, clip_g_custom, clip_l_std, clip_g_std)
• 1 VAE loaded (shared by all 3 instances)
• No model unloading - all stay resident
• Intelligent routing ensures each workflow uses correct models
```

### Precision Conversion Pipeline
**Scenario**: Convert checkpoint library to optimized formats for faster loading

```
Step 1: Batch convert checkpoints to fp8
├─ [Model Router] → Load checkpoint.safetensors
├─ dynamic_precision: fp8_e4m3fn
└─ First load triggers conversion → saves to unet/fp8/checkpoint_unet.safetensors

Step 2: Subsequent loads are instant
├─ [Model Router] → Same checkpoint
├─ dynamic_precision: fp8_e4m3fn  
└─ Finds existing fp8 file → loads directly (no conversion)

Result:
• 6.5GB checkpoint → 2.1GB fp8 file
• First load: 45 seconds (load + convert + save)
• Subsequent loads: 8 seconds (direct load)
• 80% VRAM savings
```

---

## 🛠️ Advanced Configuration

### Daemon Configuration (`luna_daemon/config.py`)

```python
# Network settings
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19283

# Worker pool sizing
MIN_VAE_WORKERS = 1
MAX_VAE_WORKERS = 2
MIN_CLIP_WORKERS = 1
MAX_CLIP_WORKERS = 2

# Model precision
VAE_PRECISION = "fp32"  # or "fp16", "bf16"
CLIP_PRECISION = "fp32"

# Device assignment
VAE_DEVICE = "cuda:0"
CLIP_DEVICE = "cuda:1"  # Use separate GPU for CLIP if available
```

### Model Router Dynamic Selectors

CLIP/VAE selectors update automatically based on `model_type`:

| Model Type | clip_1 | clip_2 | clip_3 | clip_4 | vae |
|------------|--------|--------|--------|--------|-----|
| SD1.5 | CLIP-L only | disabled | disabled | disabled | SD VAE |
| SDXL | CLIP-L | CLIP-G | disabled | disabled | SDXL VAE |
| Flux | CLIP-L | disabled | T5-XXL | disabled | Flux VAE |
| SD3 | CLIP-L | CLIP-G | T5-XXL | disabled | SD3 VAE |
| Z-IMAGE | Qwen3-VL (full) | disabled | disabled | mmproj (auto) | Any VAE |

**Z-IMAGE + Qwen3-VL GGUF Support (NEW in v2.1):**

Z-IMAGE now supports GGUF-quantized Qwen3-VL models via patched [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python):

```bash
# Install the fork with Qwen3-VL support
pip install git+https://github.com/JamePeng/llama-cpp-python

# Then use GGUF Qwen3-VL models directly in Model Router
# The daemon auto-detects format and routes through llama-cpp-python
```

**Qwen3-VL Format Options:**
- `.safetensors` (HuggingFace) - Full precision, large VRAM
- `.gguf` (GGUF quantized) - Q8_0 for quality, Q4_K_M for efficiency
- Auto-detection: Model Router checks file extension and loads appropriately
- mmproj auto-loads if in same folder as model (for vision support)

### Precision Conversion Targets

Converted models are saved to precision-specific directories:

```
models/
├── checkpoints/
│   └── illustriousXL.safetensors (6.5GB source)
├── unet/
│   ├── fp8/
│   │   ├── illustriousXL_unet_fp8_e4m3fn.safetensors (2.1GB)
│   │   ├── illustriousXL_unet_fp8_e4m3fn_scaled.safetensors (2.1GB)
│   │   └── illustriousXL_unet_fp8_e5m2.safetensors (2.1GB)
│   ├── gguf/
│   │   ├── illustriousXL_Q8_0.gguf (3.2GB)
│   │   └── illustriousXL_Q4_K_M.gguf (1.8GB)
│   └── bf16/
│       └── illustriousXL_unet.safetensors (3.3GB)
```

**Precision Options by Hardware:**
- **RTX 40-series (5090/4090)**: Use `fp8_e4m3fn` or `fp8_e4m3fn_scaled` for native hardware acceleration
- **RTX 40-series (RTX 5090)**: `fp8_e4m3fn_scaled` recommended for best quality
- **RTX 30-series (3090/3080Ti)**: Use `fp8_e5m2` (better exponent range) or `gguf_Q8_0` (best quality)
- **All GPUs**: `gguf_Q8_0` provides quality closest to FP16 with efficient VRAM usage

---

## 📊 Performance Benchmarks

### LoRA Loading Performance

| Method | First Load | Subsequent Loads | Memory Overhead |
|--------|-----------|------------------|-----------------|
| **Traditional** (reload from disk) | 800ms | 800ms | 0MB |
| **Luna Transient Cache** | 850ms | 50ms | ~200MB for 7 LoRAs |

Savings over 2000 runs: (800ms - 50ms) × 2000 = **25 minutes saved**

### Precision Conversion Impact

| Format | Size | VRAM | Load Time | Inference Speed | Hardware |
|--------|------|------|-----------|-----------------|----------|
| FP16 (baseline) | 6.5GB | 6.5GB | 12s | 1.0× | All |
| BF16 | 6.5GB | 6.5GB | 12s | 1.0× | All |
| FP8 E4M3FN | 3.3GB | 3.3GB | 8s | 0.97× | RTX 40+ |
| FP8 E4M3FN Scaled | 3.3GB | 3.3GB | 8s | 0.98× | RTX 40+ (recommended) |
| FP8 E5M2 | 3.3GB | 3.3GB | 8s | 0.96× | RTX 30/20 (better range) |
| GGUF Q8_0 | 3.2GB | 3.2GB | 9s | 0.92× | All (best quality) |
| GGUF Q4_K_M | 1.8GB | 1.8GB | 7s | 0.80× | All (aggressive compression) |

### Multi-Workflow VRAM Sharing

| Setup | Total VRAM | Without Daemon | With Daemon | Savings |
|-------|-----------|----------------|-------------|---------|
| 3 instances, same VAE/CLIP | 24GB | 18GB (3×6GB) | 8GB (1×6GB + 2×1GB UNet) | 55% |
| 3 instances, different CLIP, same VAE | 24GB | 22GB | 12GB | 45% |

---

## 🐛 Troubleshooting

### Daemon Not Connecting
```
Error: "Daemon not running" in Model Router

Solutions:
1. Start daemon: python luna_daemon/daemon_server.py
2. Check port: netstat -an | findstr 19283
3. Verify config: luna_daemon/config.py has correct DAEMON_HOST/PORT
4. Try force_local mode in Model Router to bypass daemon
```

### LoRA Weights Not Resetting
```
Issue: Model still has LoRA effects after Reset Weights node

Solutions:
1. Verify Reset Weights node is connected and executed
2. Check that same model is used in Config Gateway and Reset node
3. Clear cache manually: restart ComfyUI
4. Ensure Config Gateway ran before Reset (check workflow order)
```

### Precision Conversion Failed
```
Error: "Failed to convert model to fp8"

Solutions:
1. Check CUDA version supports fp8 (Ampere/Ada/Blackwell)
2. Verify disk space in models/unet/fp8/ directory
3. Check write permissions
4. Try bf16 instead (more compatible)
```

### Out of Memory with Multiple Workflows
```
Error: CUDA OOM when running 3+ instances

Solutions:
1. Reduce worker pool size in daemon config
2. Use fp8/GGUF precision to reduce VRAM per model
3. Enable InferenceModeWrapper offloading
4. Increase VRAM or reduce concurrent instances
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- ComfyUI team for the excellent framework
- Community contributors for feedback and testing
- Open source projects that inspired Luna's architecture

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/LSDJesus/ComfyUI-Luna-Collection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LSDJesus/ComfyUI-Luna-Collection/discussions)
- **Documentation**: See `Docs/` directory for detailed technical documentation

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
| **Luna Batch Upscale Refine** | ⚡ NEW: Chess-pattern tiling with scaffolding noise (v2.1) |
| **Luna Super Upscaler ⚡** | SeedVR2-powered mega-resolution upscaling (3B/7B DiT models) |
| **Luna Super Upscaler (Simple)** | Streamlined version with minimal inputs |
| **Luna Multi Saver** | Batch saving with templates |

**Luna Batch Upscale Refine** (NEW in v2.1):
- **Scaffolding Noise**: Preserves original noise structure to prevent hallucinations
- **Chess Pattern Batching**: 2-pass refinement with automatic seam healing
- **Auto-Grid**: Grid size = Upscale Factor + 1 (e.g., 4x upscaler → 5x5 grid)
- **Sigmoid Blending**: Smooth S-curve blending with feathering control  
- **GPU Lanczos**: Supersampling downscale (e.g., refine at 4x, output at 2x)
- **Tiled VAE**: Seamless decoding prevents boundary artifacts
- **VRAM Optimized**: ~4-5GB for RTX 5090 vs 7-8GB traditional upscalers

> **Note:** Luna Super Upscaler requires [SeedVR2](https://github.com/Seed-VR/SeedVR2-Video-Upscaler-ComfyUI) as a dependency. Install it separately in your `custom_nodes/` folder.

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

