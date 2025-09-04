# ComfyUI-Luna-Collection

![Version](https://img.shields.io/badge/version-v0.0.2-blue.svg)

Welcome to the Forge.

This repository contains **ComfyUI-Luna-Collection**, a bespoke collection of custom nodes for ComfyUI, engineered for power, flexibility, and a efficient workflow. These tools are born from a collaborative project between a human architect and their AI muse, Luna.

Our philosophy is simple: build the tools we need, exactly as we need them. Each node in this pack is designed to be a clean, powerful, and intuitive component in a larger, more magnificent machine. We believe in modularity, control, and the beautiful chaos of creation.

---

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    git clone https://github.com/LSDJesus/ComfyUI-Luna-Collection.git
    ```
2.  Restart ComfyUI. The nodes will be available under the **`Luna Collection`** category.

---

## Nodes

### 1. Luna Simple Upscaler

A clean, lightweight, and powerful image upscaler.

**Description:**
This node provides a straightforward, no-frills interface for upscaling images using your chosen `UPSCALE_MODEL`. It is designed to be a simple, deterministic component for general-purpose upscaling.

**Inputs:**
*   `image`: The `IMAGE` to be upscaled.
*   `upscale_model`: The `UPSCALE_MODEL` to use for the operation.
*   `scale_by` (Float): The factor by which to scale the image.
*   `resampling` (Dropdown): A fallback resampling method for the final resize pass.
*   `show_preview` (Toggle): Enables or disables the preview image within the node.

**Outputs:**
*   `upscaled_image`: The resulting upscaled `IMAGE`.

### 2. Luna Advanced Upscaler

A precision instrument for high-fidelity, artifact-free upscaling.

**Description:**
This node exposes advanced, professional-grade controls for fine-tuning the upscaling process. It is designed for workflows where absolute quality and artifact prevention are the highest priorities.

**Inputs:**
*   All inputs from the `Simple Upscaler`, plus:
*   `supersample` (Toggle): If enabled, the image is upscaled to a much larger intermediate size and then cleanly downscaled to the target. This produces incredibly smooth and anti-aliased results, especially for fine lines.
*   `rounding_modulus` (Dropdown): Ensures the image dimensions fed to the model are a clean multiple of a specific number (e.g., 8, 16). This can prevent artifacts with certain model architectures.
*   `rescale_after_model` (Toggle): Enforces a final, clean resampling pass to ensure the output image dimensions *exactly* match the `scale_by` target, correcting for any minor size deviations from the model.

**Outputs:**
*   `upscaled_image`: The resulting upscaled `IMAGE`.

---

## Changelog

**v0.0.2** - (2025-08-22)
*   **Added:** `Luna Advanced Upscaler` node with `supersample`, `rounding_modulus`, and `rescale_after_model` controls for professional-grade results.
*   **FIXED:** Corrected a critical logic flaw in the upscaling functions of both nodes where the `upscale_model` was not being utilized. Both nodes now correctly use the provided model for the primary upscale operation.

**v0.0.1** - (2025-08-22)
*   Initial release.
*   Added `Luna Simple Upscaler` node.
*   Established package structure with `__init__.py`.

---

This collection is a living project. New tools will be forged, and existing ones will be refined. We build for the future.

Now, get back to work.
