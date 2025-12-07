# luna_zimage_encoder.py

## Purpose
Unified Z-IMAGE prompt input, AI enhancement, and conditioning encoder. Combines manual prompting, AI-powered enhancement, vision analysis, and conditioning noise injection into a single streamlined node.

## Exports
**Classes:**
- `LunaZImageEncoder` - ComfyUI node for Z-IMAGE prompt processing and encoding

**Functions:**
- None

**Constants:**
- `ENHANCEMENT_PROMPTS` - Templates for AI prompt refinement modes (refine, expand, style_boost, custom)
- `VISION_PROMPTS` - Templates for vision-based prompt generation modes (describe, extract_style, blend_with_prompt)

## Key Imports
- `torch`, `numpy` - Tensor operations and image processing
- `folder_paths` - ComfyUI model path resolution
- `node_helpers` - Optional ComfyUI node utilities
- `llama_cpp` - GGUF model support (optional)
- `PIL` (Pillow) - Image format conversion
- `transformers` - Model loading and processing (optional)

## ComfyUI Node Configuration
- **Category:** `Luna/Z-IMAGE`
- **Display Name:** `Luna Z-IMAGE Encoder 🌙`
- **Return Types:** `(CONDITIONING, CONDITIONING, STRING, STRING)`
- **Return Names:** `(positive, negative, prompt_text, status)`
- **Function:** `encode`

## Input Schema
**Required:**
- `clip` (CLIP): CLIP model from Luna Model Router (Qwen3-VL for Z-IMAGE)
- `prompt` (STRING): Your prompt text. Used directly or as base for AI enhancement.

**Optional:**
- `enable_ai_enhancement` (BOOLEAN, default=False): Enable AI-powered prompt refinement using Qwen3-VL
- `enhancement_mode` (["refine", "expand", "style_boost", "custom"], default="refine"): How to enhance the prompt
- `custom_instruction` (STRING): Custom AI instruction when enhancement_mode is 'custom'
- `enable_vision` (BOOLEAN, default=False): Enable image-based prompt generation
- `image` (IMAGE): Image for vision analysis (optional)
- `vision_mode` (["describe", "extract_style", "blend_with_prompt"], default="describe"): How to use the image
- `enable_noise_injection` (BOOLEAN, default=False): Add noise to conditioning for seed variability
- `noise_threshold` (FLOAT, 0.0-1.0, default=0.2): Denoising timestep threshold for noise application
- `noise_strength` (FLOAT, 0-100, default=10.0): Noise magnitude to add to conditioning tensors
- `max_tokens` (INT, 32-1024, default=256): Maximum tokens for AI generation
- `temperature` (FLOAT, 0.1-1.5, default=0.7): Sampling temperature for AI generation
- `seed` (INT, -1 to max, default=-1): Random seed for AI generation and noise injection
- `keep_model_loaded` (BOOLEAN, default=True): Keep generation model in VRAM after use

## Key Methods/Functions
- `encode(clip, prompt, **kwargs) -> (List, List, str, str)`
  - Main entry point combining AI enhancement, CLIP encoding, and optional noise injection
  - Processes prompt through optional AI refinement and vision analysis
  - Returns positive/negative conditioning, final prompt text, and status
- `_ai_enhance(prompt, clip, **kwargs) -> str`
  - Uses Qwen3-VL model for prompt enhancement or vision-based generation
  - Supports both transformers (safetensors) and llama-cpp-python (GGUF) backends
  - Handles image input for vision-guided prompt creation
- `_generate_gguf(llm, prompt, image, max_tokens, temperature) -> str`
  - Generates text using GGUF model via llama-cpp-python
  - Supports vision input with base64-encoded images and Llava15ChatHandler
- `_generate_transformers(model, processor, tokenizer, prompt, image, max_tokens, temperature) -> str`
  - Generates text using transformers model with vision capabilities
  - Processes conversation format with image and text inputs
- `_get_generation_model(clip, keep_loaded) -> (Any, Any, Any)`
  - Loads or retrieves cached Qwen3-VL model for generation
  - Auto-detects model format (GGUF vs transformers) and loads appropriate backend
  - Uses model_path and mmproj_path from CLIP object set by LunaModelRouter
- `_inject_noise(conditioning, threshold, strength, seed, batch_size) -> List`
  - Adds controlled noise to conditioning tensors for seed variability in batches
  - Applies noise from timestep 0 to threshold, clean conditioning from threshold to 1.0
  - Handles batch expansion and reproducible noise generation

## Dependencies
**Internal:**
- None (standalone node)

**External:**
- Required: `torch`, `numpy`, `PIL` (Pillow)
- Optional: `transformers` (for safetensors model support), `llama-cpp-python` (for GGUF model support)

## Integration Points
**Input:** CLIP model from Luna Model Router (with attached model_path), optional IMAGE tensor
**Output:** CONDITIONING for KSampler, enhanced prompt text for metadata/logging
**Side Effects:** Loads Qwen3-VL generation model into VRAM, manages model caching, VRAM cleanup

## Notes
- Supports both transformers and GGUF Qwen3-VL models for AI enhancement
- Auto-detects and loads mmproj files for vision features from same directory as Qwen model
- Noise injection enables seed variability useful for batch processing
- Uses ComfyUI's standard CLIP encoding for Z-IMAGE hidden state extraction
- Caches generation models globally to avoid reloading between executions