# 🌙 Luna Semantic Detailer (LSD) - Final "4K Native" Specification

**Objective:** A high-fidelity refinement suite that operates natively on a 4K canvas, using a 4K Noise Scaffold for texture authority and CLIP-ViT for structural locking.

**Workflow Logic:**
1.  **Prep Phase:** 1K Draft $\to$ 1.5K Smoothing Pass (optional) $\to$ 4K Upscale.
2.  **Luna Phase:** Surgical Refinement & Global Polishing on the 4K Canvas.

---

## 1. Global Infrastructure: `LunaSAM3Daemon`
*   **Role:** Persistent Model Host.
*   **Logic:** Singleton pattern to hold the SAM3 model in VRAM.

---

## 2. The Luna Data Types
*   **`Master_Noise_4K`:** The static, high-frequency noise map generated at the start.
*   **`LUNA_DETECTION_DATA`:** Normalized coordinates (0.0-1.0) and masks.

---

## 3. Node A: `LunaSAM3Detector`
**Input:** The **4K Pixel Canvas** (or downscaled copy for speed).
*   **Action:** Runs SAM3 PCS.
*   **Output:** `LUNA_DETECTION_DATA`.
*   **Logic:** Detects objects directly on the canvas that will be refined.

---

## 4. Node B: `LunaSAM3Detailer`
**Inputs:** 
*   `image`: The 4K Pixel Canvas.
*   `master_scaffold`: The `Master_Noise_4K`.
*   `clip_vision`: (Optional) CLIP-ViT Model.
*   `detection_data`: From the Detector.

### The "4K Native" Execution Loop (Per Batch of Crops):

1.  **Crop Extraction:**
    *   Convert normalized coords to 4K pixel coords.
    *   Snap to 1:1 Square @ 1024px (default) or 1536px (if larger than 1024px)
2.  **Noise Matching (The DNA Link):**
    *   Slice the **exact corresponding region** from `master_scaffold`.
3.  **Structural Anchor (CLIP-ViT):**
    *   Encode the Pixel Crop using CLIP-ViT.
    *   Concatenate result with Text Conditioning.
4.  **Refinement:**
    *   Batch VAE Encode crops to Latent.
    *   Sample with injected Noise Slice.
5.  **Re-Integration:**
    *   Batch VAE Decode.
    *   Blend onto 4K Canvas using **Smoothstep Mask**.

---

## 5. Node C: `LunaGlobalRefiner`
**Inputs:** 4K Pixel Canvas (Patched) + `master_scaffold` + `clip_vision`.

### Execution Logic:
1.  **Grid Setup:** Divide 4096px image into 1024px or 1536px tiles (grid with overlap).
2.  **The 1:1 Match:**
    *   Crop 1024px or 1536px Pixels.
    *   Slice 1024px or 1536px Noise (Direct 1:1 slice, no scaling).
3.  **Anchoring:** Encode Pixel Tile with CLIP-ViT $\to$ Fuse with Global Prompt.
4.  **Refinement:**
    *   Batch Process (Chess Pattern).
    *   Sample using the raw 1:1 noise slice.
5.  **Re-Integration:**
    *   Batch VAE Decode.
    *   Blend onto 4K Canvas using **Smoothstep Mask**.
6.  **Output:** Final 4K Image.

---

## 6. Workflow Summary (The User Experience)

1.  **Draft:** Generate 1K image (using downscaled 4K noise).
2.  **Prep:** Option A: direct 4x upscale to 4k. Option B: Upscale to 4k using 4x upscale_model then supersample downscale to 1.5K $\to$ Light Refine @ 1.5k with 4k scaffold downscaled to 1.5k $\to$ Upscale to 4K.
3.  **LSD Phase:**
    *   **Detect:** Find faces/hands on 1K image.
    *   **Detail:** Fix faces/hands using 1024px standardized crops + clip-vit encodes per tile + 4K Noise crops.
    *   **Global:** Polish the whole 4K image using tiles + clip-vit encodes per tile + 4K Noise crops.
4.  **Final:** Downscale to target size or leave at 4k.

---

## 7. Mathematical Constraints for Copilot

### Variance Scaling (Downscaling Noise)
```python
# Used inside Detailer when crop > 1024px
scale_factor = current_size / target_size
noise = F.interpolate(noise, size=target, mode='area') * scale_factor
```

### Conditioning Fusion
```python
# Used in both Detailer and Global Refiner
# fuses text_embeds [B, 77, 768] with vision_embeds [B, 257, 768]
combined_cond = torch.cat([text_cond, vision_cond], dim=1)
```

### Smoothstep Blending
```python
# Pixel space blending
alpha = mask * mask * (3.0 - 2.0 * mask)
final = refined * alpha + original * (1.0 - alpha)
```

**Status:** The complexity is stripped out. The logic is linear. The architecture is robust. **Hand this to the Copilot.** 🚀