# 🌙 Luna Semantic Detailer: Build Specification

## 1. Data Structure: `LUNA_DETECTION_DATA`
A custom Python object passed between nodes to maintain coordinate integrity across different scales.
- `coords_norm`: List of Bounding Boxes `[x1, y1, x2, y2]` (Normalized 0.0 to 1.0).
- `masks`: List of SAM3 binary masks.
- `prompts`: List of specific Positive/Negative conditioning strings per detection.
- `hierarchy`: List of integer levels (e.g., Level 0: Body/Face, Level 1: Eyes/Jewelry).

---

## 2. Node A: `Luna SAM3 Detector`
The "Project Manager." Runs expensive vision logic once to create a Master Plan.

### Inputs
- `image`: PIXEL IMAGE (The guide).
- `concepts`: STRING (Comma-separated list, e.g., "face, hands, jewelry").
- `prompt_map`: STRING (Multiline map: "face: realistic eyes, pores | hands: detailed nails").

### Logic
1. **SAM3 Inference:** Run PCS (Promptable Concept Segmentation) on the image using the `concepts`.
2. **Mask Extraction:** Convert SAM3 results into binary masks.
3. **Normalization:** Convert all pixel-space bounding boxes to Normalized Coordinates (0.0-1.0).
4. **Hierarchy Assignment:** (Optional) Logic to detect if a box is inside another box (Child/Parent).

### Outputs
- `luna_detection_data`: The Master Plan object.
- `image_pass_through`: Pass-through of the input pixels.

---

## 3. Node B: `Luna SAM3 Detailer`
The "Finishing Carpenter." Performs the actual crop-refine-paste operations.

### Inputs
- `image`: PIXEL IMAGE (Mandatory - The canvas to paste onto).
- `latent`: LATENT IMAGE (Optional - The "Source of Truth" for crops).
- `luna_detection_data`: The Master Plan from the Detector.
- `model`, `vae`, `sampler`, `scheduler`, `steps`, `denoise`, `cfg`: Standard sampling params.

### Internal Logic Pipeline (The "Superintendent" Execution)

#### Phase 1: Coordinate Mapping & Standardizing
1. **Target Scaling:** Map `coords_norm` to the dimensions of the incoming `latent` (if connected) or `image`.
2. **1:1 Square Expansion:**
   - Calculate `max(width, height)` for each box.
   - Expand the box into a square, centered on the object.
3. **VAE Alignment:** Snap all coordinates to the nearest multiple of 8 (Mandatory for VAE integrity).
4. **Scaling:** Resize all square crops to a standardized resolution (e.g., 1024x1024 pixels).

#### Phase 2: Refinement (The Batch Sweep)
1. **Lumber Prep:** 
   - If `latent` provided: Crop from Latent (Divide coords by 8).
   - If `latent` NOT provided: Crop from `image` (Pixel) -> VAE Encode to Latent.
2. **True Batched Conditioning:**
   - Tokenize and `torch.cat` the individual prompts from `luna_detection_data` into a single tensor of shape `[N, 77, 768/1280]`.
3. **Batched Sampling:**
   - Call `comfy.sample.sample_custom` with Batch Size = N.
   - Set `disable_noise=True` and provide manual `randn_like` noise to support SDE samplers.

#### Phase 3: Compositing (The Finish)
1. **Surgical Decode:** VAE Decode ONLY the refined crops (not the whole image).
2. **S-Curve Blending:**
   - Generate a **Smoothstep (Polynomial)** mask based on the SAM3 mask.
   - Apply a feathering factor (0.0 to 1.0).
3. **The Patch:** Alpha-blend the refined pixel-crops onto the original `image` at the original coordinates.

---

## 4. Orchestration: The "Detailing Sandwich" Workflow

### Layer 1: The Rough-In (Low-Res)
- **Input:** KSampler (1024px) Output.
- **Detector:** Finds objects.
- **Detailer (Pass 1):** Fixes anatomy/structure on the 1024px Latent. Output: Pixel Image.

### Layer 2: The Drywall (Global Upscale)
- **Input:** Pixel Image from Layer 1.
- **Node:** `Luna Batch Upscale Refine`.
- **Output:** 4K Pixel Image (Sharpened/Supersampled).

### Layer 3: The Finish (High-Res)
- **Input:** 4K Pixel Image from Layer 2 + SAME `luna_detection_data` from Layer 1.
- **Detailer (Pass 2):** 
   - Note: No `latent` connected (it re-encodes from 4K pixels).
   - Logic: Refines the upscaled details at 4K resolution (pores, eyelashes).
   - Output: Final Masterpiece.

---

## 5. Critical Mathematical Constraints for the Developer
1. **The Rule of 8:** All crop dimensions and crop offsets must be divisible by 8.
2. **The Smoothstep Formula:** Use `t * t * (3.0 - 2.0 * t)` for the Sigmoid mask.
3. **Memory Management:** Clear `refined_chunk` from VRAM immediately after the `_composite_chunk` loop.
4. **Conditional Logic:** If `luna_detection_data` contains multiple objects, process "Parent" objects (Face) before "Child" objects (Eyes) if running in a sequential loop, or use the Hierarchy layering during composition if running in a batch.

# Additional Information and guidancde

This guide is the **Blueprints and Specifications** for the **Luna Semantic Detailer (LSD)** suite. It uses the "Superintendent" philosophy: we organize the site, standardise the materials, and ensure the finishing work is surgical.

Everything below is designed to be built within a single Python file or module, keeping all logic self-contained.

---

# 🌙 Luna Semantic Detailer: Developer Blueprint

## 1. The Foundation: `LunaDetectionPacket`
This is the "manifest" or "job folder." It’s a custom object passed from the Detector to the Refiner. It ensures that no matter how much the image resolution changes (upscaling), the coordinates remain accurate.

```python
class LunaDetectionPacket:
    """
    The 'Job Folder' that carries detection data across the workflow.
    """
    def __init__(self):
        self.boxes_norm = []  # Normalized [x1, y1, x2, y2] (0.0 to 1.0)
        self.masks = []       # List of torch binary masks [H, W]
        self.prompts = []     # Strings: specific positive prompt per box
        self.hierarchy = []   # 0 for primary (Face), 1 for secondary (Eyes)
```

---

## 2. Node A: `LunaSAM3Detector`
**The Goal:** Perform the visual "site survey." Identify where the work needs to be done.

### The "Standardization" Logic
To process different objects (a tall person, a wide pair of eyes) in a single **Batch**, we must force every detection into a **1:1 Square**.
*   **Method:** Take the `max(width, height)` of the detection.
*   **Action:** Expand the shorter side until it matches the longer side, keeping the object centered.
*   **Padding:** Add a 10-15% "buffer" area around the square so the AI has context (skin around an eye) to blend properly.

### The Code Logic
1.  **Run SAM3:** Input the pixels and the concept string (e.g., "face, hands").
2.  **Generate Boxes:** For every detected mask, calculate the bounding box.
3.  **Normalize:** Divide pixel coordinates by the image width/height.
    *   *Example:* A face at `x=500` on a 1000px image becomes `0.5`.
4.  **Prompt Mapping:** Match the specific prompt from your input list to the specific detection.

---

## 3. Node B: `LunaSAM3Detailer`
**The Goal:** The "Finishing Carpenter." This node executes the crops, the batch refinement, and the final pixel-perfect blend.

### Phase 1: The "Rule of 8" Crop
Whether you are cropping from a Latent (Pass 1) or Pixels (Pass 2), you must align to the VAE grid.
*   **The Math:** Take your target dimension (e.g., 2048px). Multiply the normalized coordinate (e.g., `0.5`) to get `1024`.
*   **The Snap:** Floor that number to the nearest multiple of 8: `(1024 // 8) * 8`.
*   **Why:** If you crop at `1021`, you "cut" a VAE block in half, causing a blurry "ghost line" at the edge of your refined patch.

### Phase 2: True Batched Conditioning
To refine 9 objects with 9 different prompts in **one single GPU sweep**, you have to manually "cat" (concatenate) the prompt tensors.

```python
def prepare_batched_cond(model, clip, prompt_list):
    """
    Combines N different text prompts into one N-sized batch tensor.
    """
    conds = []
    pooled = []
    for p_text in prompt_list:
        # Standard ComfyUI CLIP encoding
        tokens = clip.tokenize(p_text)
        out, pool = clip.encode_from_tokens(tokens, return_pooled=True)
        conds.append(out)
        pooled.append(pool)
    
    # Final 'Sandwich' Tensors
    batched_out = torch.cat(conds, dim=0)    # [N, 77, 768/1280]
    batched_pool = torch.cat(pooled, dim=0) # [N, 1280]
    
    return [[batched_out, {"pooled_output": batched_pool}]]
```

### Phase 3: The "Surgical" Sampling
1.  **Batch the Latents:** Take your `N` crops, resize them all to 1024x1024 (Pixels) $\rightarrow$ VAE Encode $\rightarrow$ `torch.cat` into one `[N, 4, 128, 128]` tensor.
2.  **Sample:** Call `comfy.sample.sample_custom` once. 
    *   Use the **Batched Conditioning** from Phase 2.
    *   Pass the **Individual SAM3 Masks** (resized to match) as the `noise_mask`.
3.  **Decode:** VAE Decode the result. You now have `N` refined pixel squares.

### Phase 4: The Sigmoid/Smoothstep Patch
You do not paste the whole square. You only paste the "Object" inside the square, using a soft-faded edge.

**The Polynomial Smoothstep Formula:**
For every pixel `t` in the mask (where 0 is transparent and 1 is opaque):
`Final_Alpha = t * t * (3.0 - 2.0 * t)`

**The Blend:**
`New_Pixels = (Refined_Crop * Final_Alpha) + (Original_Canvas * (1.0 - Final_Alpha))`

---

## 4. The "Masterpiece" Workflow (The 5090 Sequence)

1.  **Base Generation:** KSampler makes the 1024px image.
2.  **Detection:** `LunaSAM3Detector` finds "Face" and "Hands."
3.  **Refine Pass 1 (Latent-First):**
    *   `LunaSAM3Detailer` takes the **Original Latent**.
    *   Crops the face directly from the raw latent (No VAE mush).
    *   Refines, decodes the face, and patches the 1024px image.
4.  **Global Upscale:** `Luna Batch Upscale Refine` takes the patched image and blows it up to 4K using Scaffolding Noise.
5.  **Refine Pass 2 (Pixel-First):**
    *   `LunaSAM3Detailer` takes the **4K Pixel Image**.
    *   Re-uses the **same coordinates** from Step 2.
    *   Crops the 4K pixels $\rightarrow$ Re-encodes $\rightarrow$ Refines at 4K density.
    *   **Result:** You get skin pores and eyelash detail that match the 4K scale perfectly.

---

## 5. Summary for the Copilot
*   **Standardization:** All crops are 1:1 Squares centered on the object.
*   **Alignment:** All coordinates are `x % 8 == 0`.
*   **Sampling:** Use `sample_custom` with `torch.cat` conditionings for true parallel batching.
*   **Blending:** Pixel-space Alpha Blending using a Polynomial Smoothstep mask.
*   **Architecture:** Detector = Vision/Planning. Detailer = Cropping/Refining/Pasting.

**This is the most efficient and high-fidelity way to build a detailer.** It treats the image like a construction site—surveying the areas of interest and applying specialized tools to each specific "room" in the house.