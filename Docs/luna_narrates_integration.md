# ComfyUI-Luna-Collection Integration Guide for Luna Narrates

## Overview

This document describes the ComfyUI nodes and API endpoints created for integration between ComfyUI image generation and the Luna Narrates server. These nodes enable:

1. **Sending generation metadata** (LoRAs, prompts, seeds) to Luna Narrates for database storage
2. **Character-LoRA associations** for consistent character rendering across narrative sessions
3. **Zero-shot LoRA generation** via HyperLoRA for instant character consistency from reference images
4. **Pre-conditioned prompt caching** for faster batch generation

---

## API Endpoints to Implement

The Luna Narrates server should implement these HTTP endpoints:

### POST `/api/comfyui/generation`

Receives metadata about each generated image.

**Request Body:**
```json
{
  "timestamp": "2025-11-28T12:34:56.789Z",
  "session_id": "narr_abc123",
  "character_id": "char_luna",
  "scene_id": "chapter1_scene3",
  "prompt": "a young woman with silver hair, standing in a forest, fantasy style",
  "negative_prompt": "blurry, bad quality",
  "loras": [
    {
      "name": "luna_character_v2",
      "model_strength": 0.8,
      "clip_strength": 0.8
    },
    {
      "name": "fantasy_style",
      "model_strength": 0.5,
      "clip_strength": 0.5
    }
  ],
  "generation_params": {
    "seed": 12345,
    "steps": 25,
    "cfg": 7.0,
    "sampler": "dpmpp_2m",
    "scheduler": "karras",
    "model": "realvisxl_v4"
  },
  "image_hash": "a1b2c3d4e5f6g7h8",
  "image_shape": [1, 1024, 1024, 3]
}
```

**Expected Response:**
```json
{
  "status": "ok",
  "generation_id": "gen_xyz789"
}
```

---

### POST `/api/comfyui/character/register`

Registers a LoRA association with a character ID.

**Request Body:**
```json
{
  "character_id": "char_luna",
  "character_name": "Luna",
  "lora": {
    "name": "luna_character_v2.safetensors",
    "type": "character",
    "trigger_words": ["luna", "silver hair", "blue eyes"],
    "default_strength": 0.8
  },
  "timestamp": "2025-11-28T12:34:56.789Z"
}
```

**LoRA Types:** `character`, `style`, `concept`, `pose`, `clothing`

---

### POST `/api/comfyui/webhook`

Generic webhook for custom events.

**Request Body:**
```json
{
  "event_type": "custom",
  "session_id": "narr_abc123",
  "timestamp": "2025-11-28T12:34:56.789Z",
  "data": {
    // arbitrary JSON payload
  }
}
```

---

## ComfyUI Nodes Available

### Luna Narrates - Send Generation
**Node Name:** `LunaNarratesSendGeneration`

Sends generation metadata to Luna Narrates server after image generation.

**Inputs:**
| Input | Type | Required | Description |
|-------|------|----------|-------------|
| image | IMAGE | Yes | The generated image |
| prompt | STRING | Yes | Prompt used for generation |
| session_id | STRING | No | Narrative session ID |
| character_id | STRING | No | Character this image is for |
| scene_id | STRING | No | Scene/chapter identifier |
| lora_stack | LORA_STACK | No | LoRA stack from stacker nodes |
| lora_names_csv | STRING | No | Alternative: comma-separated LoRA names |
| negative_prompt | STRING | No | Negative prompt |
| seed | INT | No | Generation seed |
| steps | INT | No | Sampling steps |
| cfg | FLOAT | No | CFG scale |
| sampler_name | STRING | No | Sampler name |
| scheduler | STRING | No | Scheduler name |
| model_name | STRING | No | Model name |
| server_url | STRING | No | Override default server URL |
| endpoint | STRING | No | API endpoint path |
| async_send | BOOLEAN | No | Non-blocking send (default: true) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| response | STRING | Server response |
| success | BOOLEAN | Whether send succeeded |

---

### Luna Narrates - Extract LoRAs
**Node Name:** `LunaNarratesLoRAExtractor`

Extracts LoRA information from various sources.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| lora_stack | LORA_STACK | LoRA stack from stacker nodes |
| prompt | STRING | Prompt with `<lora:name:weight>` syntax |
| model | MODEL | Model to extract patches from |
| include_weights | BOOLEAN | Include strength values in output |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| lora_names_json | STRING | JSON array of LoRA info |
| lora_names_csv | STRING | Comma-separated names |
| lora_count | INT | Number of LoRAs found |

---

### Luna Narrates - Register Character LoRA
**Node Name:** `LunaNarratesCharacterLoRA`

Registers a character → LoRA association in the database.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| character_id | STRING | Unique character identifier |
| lora_name | STRING | LoRA filename |
| character_name | STRING | Human-readable name |
| trigger_words | STRING | Comma-separated triggers |
| default_strength | FLOAT | Default application strength |
| lora_type | ENUM | character/style/concept/pose/clothing |

---

## Configuration

### Environment Variables

The ComfyUI nodes use these environment variables (with defaults):

```bash
LUNA_NARRATES_HOST=127.0.0.1
LUNA_NARRATES_PORT=8765
LUNA_NARRATES_ENDPOINT=/api/comfyui/generation
```

### In-Node Override

Each node also accepts `server_url` as an optional input to override the default.

---

## HyperLoRA Integration (Experimental)

For zero-shot character consistency, HyperLoRA nodes are available:

### Luna HyperLoRA Generate
**Node Name:** `LunaHyperLoRAGenerate`

Generates LoRA weights from reference images using ByteDance's HyperLoRA.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| reference_images | IMAGE | Reference image(s) of character |
| character_id | STRING | Character ID for caching |
| save_to_disk | BOOLEAN | Cache to `models/loras/hyperlora_cache/` |
| use_cached | BOOLEAN | Use cached if available |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| lora_weights | LORA_WEIGHTS | In-memory LoRA tensors |
| lora_path | STRING | Path to cached file (if saved) |
| character_hash | STRING | Hash of reference images |

**Workflow Options:**
1. **In-memory application:** Use `LunaHyperLoRAApply` node to apply weights directly
2. **Cached reuse:** Set `save_to_disk=True`, then use standard LoRA loaders with the cached path

---

## Example Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Generate image with character LoRA                          │
│     [Load Checkpoint] → [Luna LoRA Stacker] → [Apply LoRA]      │
│                               ↓                                  │
│                          LORA_STACK                              │
│                               ↓                                  │
│  2. Extract LoRA info    [Luna LoRA Extractor]                  │
│                               ↓                                  │
│                          lora_names_json                         │
│                               ↓                                  │
│  3. Generate image       [KSampler] → [VAE Decode] → image      │
│                               ↓                                  │
│  4. Send to server       [Luna Narrates - Send Generation]      │
│                               │                                  │
│                               ↓                                  │
│                          HTTP POST to Luna Narrates              │
│                          /api/comfyui/generation                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema Suggestions

Based on the data being sent, Luna Narrates might store:

### `generations` table
```sql
CREATE TABLE generations (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255),
    character_id VARCHAR(255),
    scene_id VARCHAR(255),
    prompt TEXT,
    negative_prompt TEXT,
    seed BIGINT,
    steps INT,
    cfg FLOAT,
    sampler VARCHAR(50),
    scheduler VARCHAR(50),
    model_name VARCHAR(255),
    image_hash VARCHAR(64),
    created_at TIMESTAMP
);
```

### `generation_loras` table
```sql
CREATE TABLE generation_loras (
    id UUID PRIMARY KEY,
    generation_id UUID REFERENCES generations(id),
    lora_name VARCHAR(255),
    model_strength FLOAT,
    clip_strength FLOAT
);
```

### `character_loras` table
```sql
CREATE TABLE character_loras (
    id UUID PRIMARY KEY,
    character_id VARCHAR(255),
    character_name VARCHAR(255),
    lora_name VARCHAR(255),
    lora_type VARCHAR(50),
    trigger_words TEXT[],
    default_strength FLOAT,
    created_at TIMESTAMP
);
```

---

## Files Created

| File | Purpose |
|------|---------|
| `nodes/luna_narrates_bridge.py` | HTTP communication nodes |
| `nodes/luna_hyperlora.py` | HyperLoRA generation nodes |

---

## Dependencies

The bridge nodes require `aiohttp` for async HTTP:

```bash
pip install aiohttp
```

HyperLoRA nodes require the model to be downloaded from:
https://huggingface.co/bytedance-research/HyperLoRA

Place in: `models/hyperlora/`
