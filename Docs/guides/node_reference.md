# 📖 Complete Node Reference

A comprehensive reference for all Luna Collection nodes with detailed parameter documentation.

---

## Table of Contents

1. [YAML Wildcard Nodes](#yaml-wildcard-nodes)
2. [Model Loader Nodes](#model-loader-nodes)
3. [MediaPipe Detailing Nodes](#mediapipe-detailing-nodes)
4. [Upscaling Nodes](#upscaling-nodes)
5. [Daemon & Performance Nodes](#daemon--performance-nodes)
6. [Preprocessing Nodes](#preprocessing-nodes)
7. [Connection Nodes](#connection-nodes)
8. [Utility Nodes](#utility-nodes)

---

## YAML Wildcard Nodes

### Luna YAML Wildcard

**Category:** `Luna/Wildcards`

Resolves YAML-based wildcards in text with support for nested paths, templates, and numeric ranges.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | required | Input text with wildcard syntax |
| `seed` | INT | 0 | Random seed (0 = random) |
| `wildcards_dir` | STRING | "" | Custom wildcards directory |

**Outputs:** `(resolved_text: STRING)`

---

### Luna YAML Wildcard Batch

**Category:** `Luna/Wildcards`

Generate multiple wildcard variations at once.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | STRING | required | Template text with wildcards |
| `count` | INT | 10 | Number of variations (1-100) |
| `seed` | INT | 0 | Starting seed |
| `wildcards_dir` | STRING | "" | Custom directory |

**Outputs:** `(variations: STRING)` - Newline-separated

---

### Luna Random Int Range

**Category:** `Luna/Wildcards`

Generate random integers within a range.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_value` | INT | 1 | Minimum value |
| `max_value` | INT | 10 | Maximum value |
| `seed` | INT | 0 | Random seed |

**Outputs:** `(value: INT)`

---

### Luna Random Float Range

**Category:** `Luna/Wildcards`

Generate random floats with step resolution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_value` | FLOAT | 0.0 | Minimum value |
| `max_value` | FLOAT | 1.0 | Maximum value |
| `step` | FLOAT | 0.1 | Step resolution |
| `seed` | INT | 0 | Random seed |

**Outputs:** `(value: FLOAT)`

---

### Luna LoRA Randomizer

**Category:** `Luna/Wildcards`

Randomly select LoRAs with optional category filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | STRING | "" | Filter by category (from connections.json) |
| `count` | INT | 1 | Number of LoRAs to select |
| `min_weight` | FLOAT | 0.5 | Minimum strength |
| `max_weight` | FLOAT | 1.0 | Maximum strength |
| `seed` | INT | 0 | Random seed |

**Outputs:** `(lora_stack: LORA_STACK)`

---

## Model Loader Nodes

### Luna Checkpoint Loader

**Category:** `Luna/Loaders`

Load checkpoints with metadata display.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ckpt_name` | COMBO | required | Checkpoint to load |

**Outputs:** `(model: MODEL, clip: CLIP, vae: VAE, info: STRING)`

---

### Luna LoRA Stacker

**Category:** `Luna/Loaders`

Stack up to 4 LoRAs with individual controls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_1` | COMBO | None | First LoRA |
| `lora_1_strength` | FLOAT | 1.0 | Model strength |
| `lora_1_clip_strength` | FLOAT | 1.0 | CLIP strength |
| `lora_1_enabled` | BOOLEAN | True | Enable/disable |
| `lora_2...4` | ... | ... | Same for LoRAs 2-4 |

**Outputs:** `(lora_stack: LORA_STACK)`

---

### Luna LoRA Stacker Random

**Category:** `Luna/Loaders`

Randomized LoRA selection from available LoRAs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count` | INT | 2 | Number of LoRAs |
| `min_strength` | FLOAT | 0.5 | Minimum strength |
| `max_strength` | FLOAT | 1.0 | Maximum strength |
| `seed` | INT | 0 | Random seed |
| `exclude_pattern` | STRING | "" | Regex pattern to exclude |

**Outputs:** `(lora_stack: LORA_STACK)`

---

### Luna Embedding Manager

**Category:** `Luna/Loaders`

Manage and apply textual inversions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_1...5` | COMBO | None | Embeddings to manage |
| `weight_1...5` | FLOAT | 1.0 | Embedding weights |
| `enabled_1...5` | BOOLEAN | True | Enable/disable each |

**Outputs:** `(embedding_string: STRING, embedding_list: LIST)`

---

## MediaPipe Detailing Nodes

### Luna MediaPipe Detailer

**Category:** `Luna/Detailing`

Face and body detailing with inpainting support.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image |
| `detect_face` | BOOLEAN | True | Detect faces |
| `detect_hands` | BOOLEAN | False | Detect hands |
| `detect_eyes` | BOOLEAN | False | Detect eyes |
| `detect_mouth` | BOOLEAN | False | Detect mouth |
| `confidence` | FLOAT | 0.5 | Detection confidence |
| `mask_padding` | INT | 35 | Mask expansion |
| `mask_blur` | INT | 6 | Mask edge blur |
| `sort_by` | COMBO | "Confidence" | Sort detected regions |
| `max_objects` | INT | 10 | Maximum detections |

**Outputs:** `(image: IMAGE, mask: MASK, segs: SEGS)`

---

### Luna MediaPipe SEGS

**Category:** `Luna/Detailing`

Generate segmentation masks only.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image |
| `detect_*` | BOOLEAN | False | Detection targets |
| `confidence` | FLOAT | 0.30 | Detection threshold |
| `mask_padding` | INT | 35 | Mask expansion |
| `mask_blur` | INT | 6 | Edge blur |

**Outputs:** `(segs: SEGS, image: IMAGE)`

---

## Upscaling Nodes

### Luna Simple Upscaler

**Category:** `Luna/Upscaling`

Basic model-based upscaling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image |
| `upscale_model` | COMBO | required | Upscale model |
| `scale` | FLOAT | 2.0 | Scale factor |

**Outputs:** `(image: IMAGE)`

---

### Luna Advanced Upscaler

**Category:** `Luna/Upscaling`

Advanced upscaling with supersampling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image |
| `upscale_model` | COMBO | required | Upscale model |
| `scale` | FLOAT | 2.0 | Scale factor |
| `supersample` | BOOLEAN | False | Enable supersampling |
| `modulus` | INT | 8 | Round to modulus |
| `resampling` | COMBO | "lanczos" | Resampling method |

**Outputs:** `(image: IMAGE)`

---

### Luna Ultimate SD Upscale

**Category:** `Luna/Upscaling`

Tile-based SD upscaling with inpainting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Input image |
| `model` | MODEL | required | Diffusion model |
| `vae` | VAE | required | VAE for encode/decode |
| `positive` | CONDITIONING | required | Positive prompt |
| `negative` | CONDITIONING | required | Negative prompt |
| `upscale_model` | COMBO | required | Upscale model |
| `tile_size` | INT | 512 | Tile dimensions |
| `tile_overlap` | INT | 64 | Tile overlap |
| `denoise` | FLOAT | 0.35 | Denoise strength |
| `seam_fix_mode` | COMBO | "Half Tile" | Seam blending |

**Outputs:** `(image: IMAGE)`

---

## Daemon & Performance Nodes

### Luna Daemon Config

**Category:** `Luna/Shared`

Configure VAE/CLIP for the daemon from workflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae` | COMBO | "(daemon default)" | VAE model |
| `clip_l` | COMBO | "(daemon default)" | CLIP-L model |
| `clip_g` | COMBO | "(daemon default)" | CLIP-G model |
| `t5xxl` | COMBO | "(daemon default)" | T5-XXL (Flux) |
| `device` | COMBO | "cuda:1" | GPU device |
| `apply_immediately` | BOOLEAN | True | Apply now |

**Outputs:** `(config: DAEMON_CONFIG, status: STRING)`

---

### Luna Daemon Model Switch

**Category:** `Luna/Shared`

Quick preset switching for common model configs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | COMBO | "SDXL" | Model preset |
| `device` | COMBO | "cuda:1" | GPU device |

**Presets:** SDXL, SDXL (FP16 VAE), Pony, Illustrious, SD 1.5, Flux

**Outputs:** `(config: DAEMON_CONFIG, status: STRING)`

---

### Luna Shared VAE Encode

**Category:** `Luna/Shared`

VAE encode via daemon.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pixels` | IMAGE | required | Input image |

**Outputs:** `(latent: LATENT)`

---

### Luna Shared VAE Decode

**Category:** `Luna/Shared`

VAE decode via daemon.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples` | LATENT | required | Input latents |

**Outputs:** `(image: IMAGE)`

---

### Luna Shared CLIP Encode

**Category:** `Luna/Shared`

Text encoding via daemon.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive` | STRING | required | Positive prompt |
| `negative` | STRING | "" | Negative prompt |

**Outputs:** `(positive: CONDITIONING, negative: CONDITIONING)`

---

### Luna Daemon Status

**Category:** `Luna/Shared`

Monitor daemon health.

**Outputs:** `(status: STRING, is_running: BOOLEAN)`

---

## Preprocessing Nodes

### Luna Prompt Preprocessor

**Category:** `Luna/Preprocessing`

Batch preprocess prompts to safetensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | required | CLIP model |
| `prompt_list_path` | STRING | required | Path to prompt file |
| `filename_prefix` | STRING | "prompt" | Output prefix |
| `batch_size` | INT | 10 | Save frequency |
| `start_index` | INT | 0 | Resume index |
| `prepend_text` | STRING | "" | Prepend to prompts |
| `append_text` | STRING | "" | Append to prompts |
| `quantize_embeddings` | BOOLEAN | False | Use FP16 |

**Outputs:** `(json_path: STRING, count: INT)`

---

### Luna Cache Manager

**Category:** `Luna/Preprocessing`

Manage embedding cache.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | COMBO | "get_stats" | Action to perform |
| `max_cache_size` | INT | 100 | Cache limit |

**Actions:** `clear_cache`, `get_stats`, `optimize_cache`, `set_max_size`

**Outputs:** `(info: STRING)`

---

### Luna Performance Monitor

**Category:** `Luna/Preprocessing`

Track preprocessing performance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | COMBO | "get_report" | Action |
| `sample_window` | INT | 50 | Samples to analyze |

**Outputs:** `(report: STRING, avg_time: FLOAT, hit_rate: FLOAT, memory_mb: INT)`

---

## Connection Nodes

### Luna Smart LoRA Linker

**Category:** `Luna/Connections`

Auto-match LoRAs based on prompt content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | STRING | required | Prompt to match |
| `model` | MODEL | required | Model to apply to |
| `clip` | CLIP | required | CLIP to apply to |
| `match_mode` | COMBO | "both" | Match triggers/tags |
| `civitai_type_filter` | STRING | "" | Filter by type |
| `base_model_filter` | STRING | "" | Filter by base |
| `max_loras` | INT | 5 | Max LoRAs |

**Outputs:** `(model: MODEL, clip: CLIP, applied_loras: STRING)`

---

### Luna Civitai Metadata Scraper

**Category:** `Luna/Loaders`

Fetch Civitai metadata for models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | COMBO | required | Model to scrape |
| `write_to_model` | BOOLEAN | True | Embed in file |
| `update_connections` | BOOLEAN | True | Update JSON |

**Outputs:** `(metadata: STRING, status: STRING)`

---

## Utility Nodes

### Luna Multi Saver

**Category:** `Luna/Utils`

Batch save images with format options.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | required | Images to save |
| `output_dir` | STRING | "output" | Output directory |
| `filename_prefix` | STRING | "image" | Filename prefix |
| `format` | COMBO | "png" | Output format |
| `quality` | INT | 95 | JPEG quality |

**Outputs:** `(paths: STRING)`

---

### Luna Image Caption

**Category:** `Luna/Text`

Generate image captions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Image to caption |
| `model_type` | COMBO | "blip" | Caption model |

**Outputs:** `(caption: STRING)`

---

### Luna YOLO Annotation Exporter

**Category:** `Luna/Utils`

Export YOLO format annotations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | required | Source image |
| `boxes` | LIST | required | Bounding boxes |
| `labels` | LIST | required | Class labels |
| `output_path` | STRING | required | Output file |

**Outputs:** `(annotation_path: STRING)`

---

### Luna Parameters Bridge

**Category:** `Luna/Utils`

Pass parameters between nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameters` | DICT | required | Parameters dict |
| `key` | STRING | required | Key to extract |

**Outputs:** `(value: ANY)`

---

### Luna Load Parameters

**Category:** `Luna/Utils`

Load saved parameter configurations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | STRING | required | Path to config |

**Outputs:** `(parameters: DICT)`

---

## 📚 Related Guides

- [YAML Wildcards Guide](yaml_wildcards.md)
- [LoRA Connections Guide](lora_connections.md)
- [Performance Optimization](performance.md)
- [Daemon Setup](../luna_daemon/README.md)
