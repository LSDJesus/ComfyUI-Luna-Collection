# 🎨 Luna Semantic Detailer - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install segment-anything3  # For SAM3 detector
```

### Step 2: Download SAM3 Model
```bash
# Download to ComfyUI/models/sam3/
wget https://dl.fbaipublicfiles.com/segment_anything/sam3/models/sam3_h.safetensors
```

### Step 3: Enable Luna Daemon (Multi-GPU)
```powershell
# Start daemon on secondary GPU (GPU:1)
.\scripts\start_daemon.ps1
```

---

## Basic Workflow (Copy-Paste)

```
1. Pyramid Noise Generator
   ├─ model_type: SDXL
   ├─ aspect_ratio: 16:9
   ├─ scale_multiplier: 4
   └─ Outputs: full_scaffold, draft_scaffold, dimensions

2. KSampler
   ├─ Input: draft_scaffold
   ├─ Output: draft_latent

3. VAE Decode
   ├─ Input: draft_latent
   ├─ Output: draft_image (1K pixels)

4. SAM3 Detector
   ├─ Input: draft_image
   ├─ model_type: SDXL
   ├─ Concepts:
   │  ├─ concept: "face" (prompt: "detailed skin, realistic")
   │  ├─ concept: "eye" (prompt: empty = uses "eye")
   │  └─ concept: "hand" (prompt: empty = uses "hand")
   └─ Output: detection_pipe

5. Scaffold Upscaler
   ├─ Input: draft_image, full_scaffold
   ├─ edge_enhance: 0.3
   ├─ texture_preserve: True
   └─ Output: upscaled_image (4K)

6. Semantic Detailer (Layer 0 - Faces)
   ├─ Input: upscaled_image, full_latent, detection_pipe
   ├─ target_layers: "0"
   ├─ denoise: 0.5
   ├─ enlarge_crops: False
   └─ Output: refined_image, refined_latent, detection_pipe, mask

7. Semantic Detailer (Layer 1 - Eyes with LoRA)
   ├─ Input: refined_image, refined_latent, detection_pipe, mask
   ├─ target_layers: "1"
   ├─ denoise: 0.4
   ├─ Optional: Apply eye_lora to model before this
   └─ Output: final_refined_image, final_latent, detection_pipe, final_mask

8. Chess Refiner
   ├─ Input: final_refined_image, final_latent, full_scaffold
   ├─ tile_size: 1024
   ├─ denoise: 0.35
   ├─ scale: 0.5 (4K → 2K supersampling)
   ├─ Optional: refinement_mask = final_mask
   └─ Output: final_image (2K)

9. Save Image
   └─ Input: final_image
```

---

## Node-by-Node Reference

### 🎲 Pyramid Noise Generator

**Purpose:** Generate master noise scaffold with proper variance at all scales

**Key Inputs:**
- `model_type`: SDXL / SD1.5 / Flux
- `aspect_ratio`: 1:1 / 16:9 / 9:16 / 3:2 / 2:3 / etc.
- `scale_multiplier`: 2 / 3 / 4 / 5 / 6 (multiplier from native)

**Outputs:**
- `full_scaffold`: High-res noise (e.g., 5376×3072 for SDXL 16:9 @ 4x)
- `draft_scaffold`: Model-native noise (e.g., 1344×768, variance-corrected)
- `full_width, full_height`: For downstream nodes
- `draft_width, draft_height`: For KSampler
- `scale_factor`: For coordinate mapping

**Tips:**
- ✅ Use 4x for maximum quality refinement
- ✅ 16:9 aspect for cinematic shots
- ✅ Always check console output for dimensions

---

### 🚀 Scaffold Upscaler

**Purpose:** Create artifact-free 4K canvas from 1K draft

**Key Inputs:**
- `draft_image`: From VAE decode
- `edge_enhance`: 0.3 (subtle sharpening)
- `texture_preserve`: True (maintain fine details)
- `color_smooth`: 0.1 (reduce banding)

**Outputs:**
- `upscaled_image`: 4K pixel canvas
- `full_scaffold`: Passthrough (for downstream)

**Tips:**
- ✅ GPU-accelerated Lanczos (no upscale model needed!)
- ✅ All CUDA operations
- ⚠️ Don't skip this - establishes baseline quality

---

### 🔍 SAM3 Detector

**Purpose:** Identify and localize objects with semantic concepts

**Key Inputs:**
- `image`: 1K draft (fast detection)
- `clip`: CLIP model
- `positive`: Base conditioning
- `negative`: Base negative
- `sam3_model_name`: "sam3_h.safetensors"
- `Concepts` (dynamic inputs):
  - `concept_*_name`: "face" / "eye" / "hand" (concept name)
  - `concept_*_prompt`: "detailed skin, realistic" (optional)
  - `concept_*_layer`: 0 / 1 / 2 (refinement layer)

**Outputs:**
- `detection_pipe`: Contains coordinates + pre-encoded conditioning

**Tips:**
- ✅ Runs on secondary GPU (cuda:1) via daemon
- ✅ Custom prompts override base positive
- ✅ Empty prompt uses concept name as fallback
- ⚠️ Layer 0 first, then layers 1, 2 for semantic hierarchy

---

### 🎯 Semantic Detailer

**Purpose:** Surgical refinement of detected objects

**Key Inputs:**
- `image`: 4K upscaled image
- `full_latent`: Encoded upscaled image (for compositing)
- `full_scaffold`: Master noise
- `detection_pipe`: From SAM3
- `target_layers`: "0" or "0,1" (comma-separated)
- `denoise`: 0.5 (structural), 0.3-0.4 (details)
- `enlarge_crops`: False (4K) or True (1K)

**Outputs:**
- `refined_image`: With refinements composited
- `refined_latent`: Latent canvas (for chess refiner!)
- `detection_pipe`: Passthrough (for chaining)
- `refinement_mask`: Areas actually refined (cumulative)

**Tips:**
- ✅ Crops are always 1024×1024 (optimal size)
- ✅ Chain multiple detailers for layers
- ✅ enlarge_crops=True if input < 2K
- ⚠️ Pass refined_latent to next detailer or chess refiner!

---

### ♟️ Chess Refiner

**Purpose:** Final global refinement + supersampling

**Key Inputs:**
- `image`: From semantic detailer
- `latent`: Refined latent (WITH detailer work baked in!)
- `full_scaffold`: Master noise
- `denoise`: 0.35 (conservative for final pass)
- `scale`: 0.5 (4K→2K), 0.75 (mild), 1.0 (keep full)
- `refinement_mask`: Optional (reduces denoise where already refined)

**Outputs:**
- `final_image`: Supersampled output (2K from 4K, etc.)

**Tips:**
- ✅ Keep denoise LOW (0.25-0.35) - it's a final polish
- ✅ scale < 1.0 only (use batch_upscale_refine for upscaling)
- ✅ Chess pattern prevents seams even in complex tiles
- ⚠️ Don't skip - global coherence pass is essential

---

## Troubleshooting

### "SAM3 not found"
```bash
pip install segment-anything3
# Download: https://dl.fbaipublicfiles.com/segment_anything/sam3/models/sam3_h.safetensors
# Place in: ComfyUI/models/sam3/
```

### "Detection pipe not working"
- ✅ Verify detector outputs `detection_pipe` (not `detection_data`)
- ✅ Check console for "LunaSAM3Detector: Encoding..."
- ✅ Ensure CLIP input connected

### "Crops are too small / too large"
- `enlarge_crops=False`: Crops refined at 1024, pasted back at original size
- `enlarge_crops=True`: Crops refined at 1024, pasted at 1024 (upscales!)

### "Seams visible between tiles"
- Increase `feathering` in chess refiner (1.0 = max smoothing)
- Reduce `tile_size` (smaller = more overlap)
- Verify `scale < 1.0` (supersampling helps)

### "Out of memory"
- Reduce `tile_batch_size` (8→4)
- Use daemon on secondary GPU
- Reduce `denoise` (lower = simpler computation)

---

## Performance Benchmarks

**Hardware:** RTX 4090 (24GB VRAM)

| Operation | Time | VRAM |
|-----------|------|------|
| Pyramid Noise (4K) | 0.1s | 0.1GB |
| Draft KSampler (1K, 20 steps) | 2.5s | 8GB |
| Scaffold Upscaler (4K Lanczos) | 0.5s | 2GB |
| SAM3 Detection (1K) | 3.0s | 6GB |
| Semantic Detailer (4K, 20 steps) | 8.0s | 10GB |
| Chess Refiner (4K→2K, 20 steps) | 6.0s | 12GB |
| **Total Workflow** | **20s** | **12GB peak** |

---

## Advanced: Layered LoRA Workflow

```
1. Base generation (SDXL model, default LoRAs)
   ↓
2. Detailer Layer 0 (faces)
   + Apply face_detail_lora@0.7
   + Detect: concept="face"
   + Refine at high quality
   ↓
3. Detailer Layer 1 (eyes)
   + Replace with eye_detail_lora@0.9
   + Detect: concept="eye" (layer=1)
   + Refine at high quality
   ↓
4. Detailer Layer 2 (clothing)
   + Replace with fabric_texture_lora@0.6
   + Detect: concept="dress" + "hands" (layer=2)
   ↓
5. Chess Refiner
   + No LoRA (use base positive from draft)
   + Global coherence pass
   + Supersampling 4K→2K
   ↓
6. Final output (2304×2304 with specialized details)
```

---

## Next Steps

- Read [LSD_implementation_status.md](Docs/LSD_implementation_status.md) for mathematical details
- Check [NODES_DOCUMENTATION.md](NODES_DOCUMENTATION.md) for full node list
- Explore example workflows in `example_workflows/`

🎉 **Happy refining!**
