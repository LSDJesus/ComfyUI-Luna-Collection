This is the **Master Technical Specification** for the **Luna Semantic Detailer (LSD) Suite: Set B "Masterplan" Edition**. 

This document serves as the final, exhaustive source of truth for your Copilot. It details a "Lossless Blueprint" architecture where we generate a high-definition 4K noise scaffold first and "zoom in" to reconstruct details, rather than "stretching" low-res data.

---

# 🌙 Luna Semantic Detailer (LSD) - Masterplan Specification

## 0. The Core Philosophy: "Pyramid Noise Slicing"
Most upscalers are **Reactive** (they fix blurry pixels with random noise). LSD is **Proactive** (it plans 4K details at the 1K stage). 
- **1K Draft:** A "low-pass" version of the 4K Masterplan.
- **4K Surgery:** A 1:1 reconstruction using the original high-frequency noise DNA.
- **Goal:** Photorealistic fidelity with zero "latent mush" or interpolation artifacts.

---

## 1. Global Infrastructure: `LunaSAM3Daemon`
A persistent service to keep the heavy SAM3 model in VRAM.
- **Logic:** Singleton class. Loads SAM3 on first request; maintains model pointer for all subsequent nodes.
- **Conflict Handling:** Queues requests from different ComfyUI instances to prevent VRAM spikes.

---

## 2. Input Logic: The Blueprint Generator
Instead of a standard "Empty Latent," the workflow begins with a 4K scaffold.
- **Step 1:** Generate `Master_Noise` at target resolution (e.g., 4096x4096px).
- **Step 2 (The 1K Draft):**
    - Downscale `Master_Noise` to 1024x1024 using **Area Interpolation** (Crucial: Area mode preserves statistical averages).
    - **Global Variance Correction:** Multiply the 1K result by **4.0** (Upscale Factor). 
    - *Reason:* Averaging 16 pixels into 1 reduces Standard Deviation to 0.25. Multiplying by 4 restores $\sigma=1.0$ for the UNet.
- **Output:** `Master_Noise_4K`, `Draft_Latent_1K`.

---

## 3. Node A: `LunaSAM3Detector`
**The Surveyor.** Identifies targets on the 1K draft to save compute.
- **Input:** 1024px Pixel Image.
- **Logic:** 
    - Runs SAM3/PCS based on a "Concept Stack" (Face, Hands, Eyes, etc.).
    - Filters by selection logic (Largest, Central, etc.).
- **Output:** `LUNA_DETECTION_DATA` containing **Normalized Coordinates (0.0 to 1.0)** and binary masks.
- **Standardization:** All boxes are expanded to **1:1 Squares** with a 15% padding buffer.

---

## 4. Node B: `LunaSemanticDetailer`
**The Surgical Team.** Performs targeted 1:1 refinement.

### Internal Logic Pipeline:
1. **Coordinate Scaling:** Multiply normalized coords by current 4K dimensions.
2. **Standardized Slicing (The Workhorse):**
    - Identify `Crop_Dim` (the size needed to fit the object).
    - **Case 1 (Standard):** If object fits in 1024px, take a 1024x1024 slice.
    - **Case 2 (Large):** If object is >1024px (e.g. 1536px), take a 1536px slice.
3. **Local Variance Correction (The Math):**
    - **Pixel Crop:** Resize crop to **1024x1024** (Inference Standard).
    - **Noise Slice:** 
        - Pull raw pixels from `Master_Noise_4K`.
        - If `Crop_Dim > 1024`: 
            - Downscale slice to 1024x1024 (Area).
            - Multiply by `(Crop_Dim / 1024)` to restore $\sigma=1.0$.
4. **Batched Refinement:**
    - `torch.cat` all 1024px crops into a single batch.
    - `sample_custom` with `disable_noise=True`.
5. **Seamless Patching:**
    - Resize refined 1024px results back to `Crop_Dim`.
    - **Smoothstep Mask:** Use `t * t * (3.0 - 2.0 * t)` for the SAM3 mask.
    - Alpha-blend refined pixels onto the 4K Pixel Canvas.

---

## 5. Node C: `LunaGlobalRefiner`
**The Finishing Crew.** Welds the surgical patches into the 4K environment.

### Internal Logic Pipeline:
1. **Chess-Pattern Tiling:** 
    - Calculate a 5x5 grid (for 4K) of 1024x1024 tiles.
    - **Pass 1:** Even tiles. **Pass 2:** Odd tiles.
2. **1:1 Noise Slicing:**
    - For each tile, slice the **EXACT** matching 1024x1024 region from the `Master_Noise_4K`.
    - **Zero Scaling:** No interpolation or variance correction is needed here as the tile and noise are 1:1.
3. **Refinement:**
    - Sample at low denoise (0.2–0.35) to add global texture (pores, fabric weave).
4. **Output:** The 4096px Masterpiece.

---

## 6. Building Codes (Technical Constraints for Copilot)

### The Rule of 8
All crop coordinates (`x, y`) and dimensions (`w, h`) must be snapped to the nearest multiple of 8. 
- *Failure to do this causes "Edge Bleed" where the VAE misinterprets the boundary pixels.*

### Memory Logistics (Sub-Batching)
Even on a 5090, the Refiner nodes must process tiles in chunks:
```python
for i in range(0, total_tiles, tile_batch_size):
    # Process 4 or 8 tiles at a time
    # Clear cache between chunks
```

### The SDE/Ancestral Fix
Since we use `disable_noise=True` to preserve our Scaffold, we MUST generate a fresh `randn_like` tensor and pass it to the `noise` argument of `sample_custom`. This ensures stochastic samplers (Euler A, DPM++ SDE) have "fuel" for their steps without overwriting our initial state.

---

## 7. The Final Workflow (Sequence of Events)
1. **Generate 4K Noise.**
2. **Draft 1K Image** (Downscale Noise + Normalize + KSampler).
3. **Detect Objects** (SAM3 on 1K pixels).
4. **4x Model Upscale** (1K Pixels $\rightarrow$ 4K "Neutral" Canvas).
5. **Surgical Detailer** (Slice 4K Noise $\rightarrow$ Refine Objects $\rightarrow$ Blend).
6. **Global Refiner** (Tile 4K Noise $\rightarrow$ Refine Environment $\rightarrow$ Weld).
7. **Lanczos Downscale** (Optional: 4K $\rightarrow$ 2K for final sharpening).

---

### **Message to Copilot:**
"Implement the `prepare_noise_slice` method first. It must handle raw tensor slicing from the 4K Master Noise and apply the ratio-based variance correction (`current_dim / target_dim`) only when the slice is resized. Ensure all blending utilizes the polynomial Smoothstep formula to eliminate seams."

**Construction status:** Blueprints finalized. Foundation poured. **Begin coding.** 🚀