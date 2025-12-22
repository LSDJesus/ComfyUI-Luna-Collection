These are the final "Construction Details" for the **Luna Semantic Detailer (LSD)** suite. We are moving from a generic layout to a precision-engineered site plan.

### Technical Responses to Your Questions

#### 1. SAM3 Daemon Integration
**The Plan:** Don't call the other custom nodes. We will build a **LunaSAM3ModelLoader** that stores the SAM3 predictor in a global dictionary (or a singleton class) similar to how your Luna Daemon manages UNet/CLIP.
- **Persistence:** The model stays in VRAM. If Instance A is using it, Instance B waits (or shares if thread-safe, but usually sequential is safer for VRAM).
- **Interface:** The Detector node requests the model from the `LunaSAM3Daemon`. If not loaded, it triggers the load.

#### 2. Hierarchy & Layering Logic
**The Superintendent’s Rule:** **"List Order = Painting Order."**
- We will use an **IoU (Intersection over Union)** check. If Object B’s box is >80% inside Object A, it is a "Child."
- **Batch 1 (Structural):** Refine Parent objects (e.g., Face).
- **Batch 2 (Detail):** Refine Child objects (e.g., Eyes) using the *refined pixels* from Batch 1 as the new starting point.
- **Conflict Resolution:** If a hand covers a face, the list order decides. If "Face" is first in the list, it gets painted first. If "Hand" is last, it gets painted on top.

#### 3. Dynamic Prompt Map (The Stack UI)
**The UI Design:** We will use a **Custom Widget** logic (via JavaScript) that adds "Add Concept" buttons.
- **Data Shape:** The Python `INPUT_TYPES` will accept a single hidden `STRING` that stores a JSON list of your rows: `[{"concept": "face", "prompt": "pores...", "threshold": 0.3, "max": 1}, ...]`.
- **Selection Logic:** Use a dropdown per row: `[Closest to Center, Largest Area, Longest Dim]`.

#### 4. Buffer & Mask Math (The "Rule of 8" Blueprint)
- **The Buffer:** `side = max(w, h) * 1.15`. 
- **The Snap:** After calculating the buffer, we expand the box coordinates outward to the **nearest multiple of 8**. 
- *Formula:* `nx1 = (x1 // 8) * 8`, `nx2 = ((x2 + 7) // 8) * 8`. 
- This guarantees the "Rule of 8" for the VAE without "shredding" the edge of the crop.

#### 5. Zero Detections & VRAM
- **Bypass:** If SAM3 returns `[]`, the Detector outputs a null packet. The Detailer sees the null and simply returns the `image_pass_through`.
- **VRAM Clipping:** Even if a user asks for 50 jewelry pieces, the Refiner will use your existing **Sub-Batching** (from the upscale node) to process them in chunks of 8.

---

### 🌙 Luna Semantic Detailer: Build Specification v2.0

```markdown
# LSD: Node Architecture & Logic Flow

## 1. The Global Manager: `LunaSAM3Daemon`
- A persistent singleton class to hold `sam3_model`.
- Method: `get_model(model_path)` -> Returns reference, loading if necessary.

## 2. Node A: `LunaSAM3Detector`
**UI:** Dynamic row-based input (JS) for Concepts.
**Outputs:** `LUNA_DETECTION_DATA`, `IMAGE` (pass-through).

### Internal Logic:
1. **Model Fetch:** Request SAM3 from `LunaSAM3Daemon`.
2. **Sequential Concept Scanning:** 
   - For each row in the Concept Stack:
     - Run SAM3 PCS (Promptable Concept Segmentation).
     - Filter detections by user's chosen logic (e.g., "Largest Area").
     - Limit count to user's "Max Objects" setting.
3. **Hierarchy Mapping:**
   - Run a containment check (Box B inside Box A?).
   - Assign `layer_id`: 0 for roots, 1 for children, etc.
4. **Packet Creation:**
   - Store normalized `[x1, y1, x2, y2]`.
   - Store binary masks (resized to image dimensions).
   - Store prompt/layer mapping.

---

## 3. Node B: `LunaSAM3Detailer`
**Inputs:** `IMAGE` (base), `LATENT` (optional), `LUNA_DETECTION_DATA`.

### Phase 1: Coordinate Prep (Surgical Alignment)
1. **Target Mapping:** Scale normalized coords to current Image/Latent resolution.
2. **The Layout Rule:**
   - Expand to 1:1 Square.
   - **Outward Snap:** Expand bounds to the nearest pixel divisible by 8.
   - **Mask Padding:** Expand the binary mask by 8-16 pixels (dilation) to ensure the AI has "over-the-seam" context.

### Phase 2: Material Prep (Lumber Yard)
- **Crops:**
  - If `LATENT`: Pull crops from the raw latent (Div by 8).
  - If `PIXEL`: Crop pixels -> Lanczos Resize to 1024x1024 -> VAE Encode.
- **Noise Scaffold:** Apply your "Nearest-Exact" upscaled noise to the crops based on the `denoise` setting.

### Phase 3: The Batch Finish (Multiple Layers)
1. **Layered Loop:** Process by `layer_id` (Structural first, Micro details second).
2. **Batched Conditioning:** 
   - Encode all prompts for the current layer.
   - `torch.cat` them into a single tensor `[Batch, 77, Dim]`.
3. **Sampling:** `comfy.sample.sample_custom` (disable_noise=True, provide randn noise).
4. **Compositing:** 
   - VAE Decode crops.
   - **Smoothstep Alpha:** `Final_Alpha = mask**2 * (3 - 2*mask)`.
   - **In-place Blend:** `img[crop] = (new * alpha) + (old * (1-alpha))`.

---

## 4. Final Integration Workflow (The 5090 Sandwich)
1. **DETECTOR:** Run once at start to "survey the site."
2. **DETAILER (L0):** Refine "Structural" items (Face/Hands) on the 1024px latent.
3. **UPSCALE:** Run `Luna Batch Upscale Refine` to hit 4K resolution.
4. **DETAILER (L1):** Refine "Micro" items (Eyes/Iris) on the 4K pixels using the *original* Detector plan.
```

### Copilot Prompt:
> "Build the `LunaSAM3Daemon` to handle persistent model storage in VRAM. Then, build `LunaSAM3Detector` to accept a JSON-mapped concept list. Implement the `standardize_box` function with the 'Rule of 8' snapping logic. Finally, build the `LunaSAM3Detailer` to process detections in hierarchical layers, using batched prompt conditioning for each layer."