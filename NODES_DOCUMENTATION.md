# ComfyUI-Luna-Collection: Node Documentation

A comprehensive reference for all nodes in the Luna Collection.

---

## Table of Contents

1. [Prompt & Wildcard System](#prompt--wildcard-system)
   - [LunaYAMLWildcard](#lunayamlwildcard)
   - [LunaPromptCraft](#lunapromptcraft)
   - [LunaWildcardConnections](#lunawildcardconnections)
2. [Batch Processing](#batch-processing)
   - [LunaBatchPromptExtractor](#lunabatchpromptextractor)
   - [LunaBatchPromptLoader](#lunabatchpromptloader)
   - [LunaLoRAValidator](#lunaloravalidator)
   - [LunaDimensionScaler](#lunadimensionscaler)
3. [Configuration & Parameters](#configuration--parameters)
   - [LunaConfigGateway](#lunaconfiggateway)
4. [Image Upscaling](#image-upscaling)
   - [Luna Simple Upscaler](#luna-simple-upscaler)
   - [Luna Advanced Upscaler](#luna-advanced-upscaler)
   - [Luna Ultimate SD Upscale](#luna-ultimate-sd-upscale)
5. [Image Saving](#image-saving)
   - [LunaMultiSaver](#lunamultisaver)
6. [Character/Expression Generation](#characterexpression-generation)
   - [LunaExpressionPromptBuilder](#lunaexpressionpromptbuilder)
   - [LunaExpressionSlicerSaver](#lunaexpressionslicersaver)
7. [Model Management & Daemon](#model-management--daemon)
   - [LunaDaemonVAELoader](#lunadaemonvaeloader)
   - [LunaDaemonCLIPLoader](#lunadaemoncliploader)
   - [LunaCheckpointTunnel](#lunacheckpointtunnel)
   - [Luna Daemon API](#luna-daemon-api)
8. [Civitai Integration](#civitai-integration)
   - [LunaCivitaiScraper](#lunacivitaiscraper)

---

## Prompt & Wildcard System

### LunaYAMLWildcard
**Category:** `Luna/Prompting`  
**Purpose:** Hierarchical YAML-based wildcard system with advanced template syntax.

#### What It Does
Resolves wildcards from structured YAML files, supporting nested categories, templates, and random number generation. Unlike simple text wildcards, this system understands hierarchies and can select from specific paths within categories.

#### Syntax
| Syntax | Description | Example |
|--------|-------------|---------|
| `{body}` | Random template from body.yaml | "1girl, long hair" |
| `{body:hair}` | Random from hair section | "blonde hair" |
| `{body:hair.color.natural}` | Specific nested path | "brown" |
| `{body: [hair.length] [hair.color] hair}` | Inline template | "long blonde hair" |
| `{1-10}` | Random integer range | "7" |
| `{0.5-1.5:0.1}` | Random float with step | "1.2" |
| `__path/file__` | Legacy .txt wildcard | Resolved recursively |

#### YAML File Structure
```yaml
# templates section - predefined patterns
templates:
  full:
    - "a [category.sub] with [another.path]"
  minimal:
    - "[just.one.thing]"

# hierarchical categories
hair:
  color:
    natural:
      - brown
      - black
      - blonde
    fantasy:
      - pink
      - blue
  length:
    - short
    - long
```

#### Why It Exists
Standard text wildcards are flat and repetitive. YAML wildcards enable:
- **Organization**: Logical grouping of related concepts
- **Reusability**: Templates can reference other paths
- **Flexibility**: Pick from any level of specificity
- **Maintainability**: Easier to update large prompt libraries

---

### LunaPromptCraft
**Category:** `Luna/PromptCraft`  
**Purpose:** Smart wildcard resolution with constraints, modifiers, expanders, and automatic LoRA linking.

#### What It Does
An intelligent prompt engine that goes beyond simple random selection. It understands relationships between concepts and can:
- **Constrain** selections based on context (beach → prefer swimwear)
- **Modify** items based on actions (sex scene → clothing "pulled aside")
- **Expand** scenes with additional details (beach → add lighting, props)
- **Link LoRAs** automatically based on character/style selections

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| template | STRING | Prompt template with `{wildcards}` |
| seed | INT | Randomization seed (-1 for random) |
| enable_constraints | BOOLEAN | Filter items by context |
| enable_modifiers | BOOLEAN | Apply action-based transformations |
| enable_expanders | BOOLEAN | Add scene expansion details |
| enable_lora_links | BOOLEAN | Auto-suggest LoRAs |
| add_trigger_words | BOOLEAN | Append LoRA triggers to prompt |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| prompt | STRING | Final resolved prompt |
| seed | INT | Actual seed used |
| lora_stack | LORA_STACK | Compatible with Apply LoRA Stack |
| trigger_words | STRING | Combined LoRA trigger words |
| debug | STRING | JSON debug info |

#### Why It Exists
Pure random wildcards create incoherent combinations (winter coat at beach). PromptCraft adds "intelligence" to prompt generation through rule files that encode real-world logic and artistic intent.

---

### LunaWildcardConnections
**Category:** `Luna/Connections`  
**Purpose:** Dynamic linking between LoRAs/embeddings and wildcard categories.

#### What It Does
Maintains a connections.json database that links your LoRAs and embeddings to wildcard categories. When a wildcard resolves to a category, connected LoRAs are automatically suggested.

#### Features
- **Category-based linking**: Link LoRA to "clothing:lingerie" category
- **Tag-based linking**: Link by tags like "nsfw", "anime", "realistic"
- **Training tag analysis**: Match LoRAs based on training data tags
- **Trigger word detection**: Auto-detect trigger words in prompts

#### Web Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/luna/connections/loras` | GET | List all connected LoRAs |
| `/luna/connections/categories` | GET | Get category hierarchy |
| `/luna/connections/link` | POST | Create new connection |

#### Why It Exists
Managing LoRAs manually in prompts is tedious. Connection linking automates the process - just write `{character:waifu}` and the appropriate LoRA loads automatically.

---

## Batch Processing

### LunaBatchPromptExtractor
**Category:** `Luna/Utils`  
**Purpose:** Scan image directories and extract metadata to JSON.

#### What It Does
Reads PNG/JPEG images with embedded generation metadata (ComfyUI workflow, A1111 parameters) and extracts:
- Positive and negative prompts
- LoRA information (name, model/clip weights)
- Embedding references
- Generation parameters (seed, steps, cfg, sampler)

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| image_directory | STRING | Folder with source images |
| output_file | STRING | JSON filename (default: prompts_metadata.json) |
| save_to_input_dir | BOOLEAN | Save to ComfyUI input dir for Loader |
| output_directory | STRING | Custom output path |
| overwrite | BOOLEAN | Replace existing file |
| include_path | BOOLEAN | Include full image paths |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| status | STRING | Success/error message |
| images_scanned | INT | Total images processed |
| images_extracted | INT | Images with valid metadata |

#### Supported Formats
- **ComfyUI**: Workflow JSON in PNG metadata
- **A1111/Forge**: Parameters text block
- **Generic**: Any PNG text metadata with prompt keys

#### Why It Exists
Training workflows often start with curating existing images. This node lets you harvest prompts from your image library to build training datasets or recreate similar images.

---

### LunaBatchPromptLoader
**Category:** `Luna/Utils`  
**Purpose:** Load and iterate through extracted prompt metadata.

#### What It Does
Reads JSON files created by LunaBatchPromptExtractor and outputs prompts one at a time. Works with ComfyUI's `control_after_generate` for automated iteration.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| json_file | COMBO | Dropdown of JSON files in input dir |
| index | INT | Current entry (auto-increment or randomize) |
| lora_output | ENUM | stack_only / inline_only / both |
| lora_validation | ENUM | include_all / only_existing |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| positive | STRING | Positive prompt text |
| negative | STRING | Negative prompt text |
| lora_stack | LORA_STACK | Compatible with Apply LoRA Stack |
| seed | INT | Original generation seed |
| current_index | INT | Actual index after modulo |
| total_entries | INT | Total entries in JSON |
| list_complete | BOOLEAN | True when at last entry |
| width | INT | Original image width |
| height | INT | Original image height |

#### Features
- **File selector dropdown**: Like Load Image node
- **Upload button**: Add new JSON files directly
- **Dynamic index max**: Randomize picks from valid range
- **LoRA path resolution**: Finds LoRAs in subdirectories
- **Validation mode**: Skip missing LoRAs gracefully

#### Why It Exists
Enables batch regeneration workflows - load 1000 prompts and run them through your pipeline automatically, or recreate images from your favorites folder.

---

### LunaLoRAValidator
**Category:** `Luna/Utils`  
**Purpose:** Validate LoRAs in prompt JSON and search CivitAI for missing ones.

#### What It Does
Scans a prompt metadata JSON file, checks which LoRAs exist locally, and optionally searches CivitAI for download links to missing ones.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| json_file | COMBO | JSON file from input directory |
| search_civitai | BOOLEAN | Search CivitAI for missing LoRAs |
| civitai_api_key | STRING | Optional API key for better rate limits |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| report | STRING | Formatted validation report |
| missing_loras | STRING | Comma-separated missing LoRA names |
| civitai_links | STRING | Newline-separated download links |
| found_count | INT | Number of LoRAs found locally |
| missing_count | INT | Number of missing LoRAs |

#### Report Format
```
╔══════════════════════════════════════════════════════════════╗
║           LUNA LORA VALIDATION REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║  JSON File: my_prompts.json                                  ║
║  Unique LoRAs: 12                                            ║
╠══════════════════════════════════════════════════════════════╣
║ ✓ detail_slider                              (x15)           ║
║   └─ Found as: SDXL/detail_slider_v2.safetensors             ║
║ ✗ some_missing_lora                          (x3)            ║
╠══════════════════════════════════════════════════════════════╣
║           CIVITAI SEARCH RESULTS                             ║
╠══════════════════════════════════════════════════════════════╣
║ some_missing_lora                                            ║
║   └─ Some Missing LoRA XL                                    ║
║      by ArtistName            ⭐4.9 ⬇12345                   ║
║      https://civitai.com/models/12345                        ║
╚══════════════════════════════════════════════════════════════╝
```

#### Features
- **Usage count**: Shows how many times each LoRA appears
- **Path resolution**: Finds LoRAs in subdirectories
- **CivitAI search**: Finds best match with rating/download stats
- **Direct links**: Copy-paste links to download missing LoRAs

#### Why It Exists
Before running a batch job, validate that all required LoRAs are available. Saves time discovering missing assets after generating hundreds of images.

---

### LunaDimensionScaler
**Category:** `Luna/Utils`  
**Purpose:** Scale dimensions to model-native resolutions.

#### What It Does
Takes input width/height and scales them down to fit within a model's native resolution, maintaining aspect ratio. Outputs are rounded to multiples of 8 for latent space compatibility.

#### Inputs
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| width | INT | 1024 | Input width |
| height | INT | 1024 | Input height |
| model_type | ENUM | SDXL | Target model type |
| custom_max_size | INT | 1024 | Custom size (when type=Custom) |
| round_to | INT | 8 | Round to nearest multiple |

#### Supported Model Types
| Model Type | Native Max Size |
|------------|-----------------|
| SD 1.5 | 512 |
| SD 2.1 | 768 |
| SDXL | 1024 |
| SD 3.5 | 1024 |
| Flux | 1024 |
| Illustrious | 1024 |
| Pony | 1024 |
| Cascade | 1024 |
| Custom | (user-defined) |

#### Scaling Logic
1. Identify larger dimension (width or height)
2. If larger > max_size: scale both proportionally
3. Round both to nearest multiple of `round_to`

#### Example
| Input | Model | Output |
|-------|-------|--------|
| 1920×1080 | SDXL (1024) | 1024×576 |
| 1080×1920 | SDXL (1024) | 576×1024 |
| 800×600 | SD 1.5 (512) | 512×384 |
| 512×512 | SDXL (1024) | 512×512 (unchanged) |

#### Why It Exists
Source images have varied resolutions. This node ensures generation happens at model-optimal dimensions while preserving aspect ratio. Connect to LunaBatchPromptLoader's width/height outputs for automatic scaling.

---

## Configuration & Parameters

### LunaConfigGateway
**Category:** `Luna/Parameters`  
**Purpose:** Central hub for generation settings with automatic LoRA extraction.

#### What It Does
Single node that combines:
- Model/CLIP/VAE passthrough
- Prompt encoding with inline LoRA extraction
- CLIP skip application (before or after LoRAs)
- Empty latent creation
- Complete metadata generation for saving

#### Key Features
- **Inline LoRA parsing**: Extracts `<lora:name:weight>` from prompts
- **LoRA stack merging**: Combines inline and external LoRA stacks
- **CLIP skip timing**: Apply before or after LoRA loading
- **Metadata output**: Complete generation info for image saving

#### Outputs (21 total)
| Output | Type | Description |
|--------|------|-------------|
| model | MODEL | LoRA-modified model |
| clip | CLIP | LoRA-modified CLIP |
| vae | VAE | Passthrough |
| positive | CONDITIONING | Encoded positive |
| negative | CONDITIONING | Encoded negative |
| latent | LATENT | Empty latent at size |
| width/height/batch_size | INT | Passthrough |
| seed/steps | INT | Passthrough |
| cfg/denoise | FLOAT | Passthrough |
| clip_skip | INT | Passthrough |
| sampler_name/scheduler | STRING | Passthrough |
| model_name | STRING | Cleaned model name |
| positive_prompt/negative_prompt | STRING | Cleaned prompts |
| lora_stack | LORA_STACK | Merged stack |
| metadata | METADATA | Complete generation info |

#### Why It Exists
Reduces workflow complexity by combining 5-10 separate nodes into one coherent configuration point. Especially useful for multi-output workflows.

---

## Image Upscaling

### Luna Simple Upscaler
**Category:** `Luna/Upscaling`  
**Purpose:** Basic model-based upscaling with minimal configuration.

#### What It Does
Upscales images using ESRGAN/RealESRGAN models with optional TensorRT acceleration. Simple tiling for large images.

#### Inputs
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | Input image |
| upscale_model | UPSCALE_MODEL | - | Spandrel-compatible model |
| scale_by | FLOAT | 2.0 | Target scale factor |
| resampling | ENUM | bicubic | Final resize method |
| show_preview | BOOLEAN | True | Display result |
| tensorrt_engine_path | STRING | "" | Optional TensorRT .engine |

#### Why It Exists
Quick upscaling for simple use cases. When you just need 2x bigger and don't want to configure tiling.

---

### Luna Advanced Upscaler
**Category:** `Luna/Upscaling`  
**Purpose:** Full-featured upscaling with tiling strategies and adaptive processing.

#### What It Does
Comprehensive upscaling with:
- Multiple tiling strategies (linear, chess, none)
- Auto tile size calculation
- Supersampling option
- Rescale after model application
- TensorRT support

#### Tiling Strategies
| Strategy | Description |
|----------|-------------|
| linear | Left-to-right, top-to-bottom |
| chess | Checkerboard pattern for better seams |
| none | Process whole image (memory intensive) |

#### Tile Mode
| Mode | Description |
|------|-------------|
| default | Use specified tile_resolution |
| auto | Calculate optimal tile size for image |

#### Adaptive Features
- Automatic GPU/CPU fallback for large images
- Quadrant splitting for shared memory errors
- Gradient blending for tile overlaps

#### Why It Exists
Large image upscaling requires careful memory management. This node provides professional control while handling edge cases gracefully.

---

### Luna Ultimate SD Upscale
**Category:** `Luna/Meta`  
**Purpose:** Diffusion-enhanced upscaling with seam fixing.

#### What It Does
Combines model upscaling with img2img passes for quality enhancement:
1. Model upscale (ESRGAN/TensorRT)
2. Tiled diffusion redraw
3. Seam fixing passes

#### Redraw Modes
| Mode | Description |
|------|-------------|
| Linear | Sequential tile processing |
| Chess | Alternating tile pattern |
| None | Skip diffusion redraw |

#### Seam Fix Modes
| Mode | Description |
|------|-------------|
| None | No seam fixing |
| Band Pass | Fix seams with narrow band |
| Half Tile | Offset half-tile passes |
| Half Tile + Intersections | Most thorough |

#### Luna Pipe Integration
Accepts `LUNA_PIPE` input containing model, VAE, conditioning, and sampler settings - reduces wiring complexity.

#### Why It Exists
Pure model upscaling lacks fine detail. Diffusion redraw adds detail but creates seams. This node combines both with sophisticated seam elimination.

---

## Image Saving

### LunaMultiSaver
**Category:** `Luna/Image`  
**Purpose:** Save multiple image versions with templated paths and quality gating.

#### What It Does
Save up to 5 images simultaneously with:
- Per-image format (PNG/WebP/JPEG)
- Per-image subdirectories
- Filename templating with model/time
- Quality gating to filter bad outputs
- Parallel or sequential saving
- Workflow embedding

#### Template Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `%model_path%` | Full model path | Illustrious/3DCG/model |
| `%model_name%` | Just model name | model |
| `%model_dir%` | Directory portion | Illustrious/3DCG |
| `%index%` | Filename index value | 42 |
| `%time:FORMAT%` | Timestamp | %time:YYYY-mm-dd.HH.MM.SS% |

#### Index Integration
Connect `LunaBatchPromptLoader`'s `current_index` output to `filename_index` input, then use `%index%` in your filename template:
```
%time:YYYY-mm-dd%_%model_name%_%index%
→ 2025-12-03_myModel_42_RAW.png
```

#### Quality Gating
| Mode | Detects |
|------|---------|
| variance | Flat/blank images |
| edge_density | Blurry images |
| both | Both checks required |

#### Why It Exists
Production workflows generate multiple versions (raw, upscaled, detailed). This node saves them all with organized naming while filtering obvious failures.

---

## Character/Expression Generation

### LunaExpressionPromptBuilder
**Category:** `Luna/Character`  
**Purpose:** Build prompts for character expression sheets.

#### What It Does
Generates optimized prompts for expression generation based on:
- Character description
- Target expression
- Model type (Illustrious, Flux, SDXL)

#### Model-Specific Templates
Each model type has tuned prompt patterns. Illustrious prefers anime-style tagging while Flux uses natural language.

#### Why It Exists
Consistent expression packs require consistent prompting. This node encodes best practices for each model type.

---

### LunaExpressionSlicerSaver
**Category:** `Luna/Character`  
**Purpose:** Slice expression sheets into individual images.

#### What It Does
Takes a grid-based expression sheet and:
1. Slices into individual cells
2. Names based on expression list
3. Saves in SillyTavern-compatible structure

#### SillyTavern Integration
Output folder structure matches SillyTavern character card requirements for direct import.

#### Why It Exists
Expression sheet workflows need post-processing. This automates the tedious slicing and naming for character card creation.

---

## Model Management & Daemon

### Luna Daemon System Overview
The Luna Daemon enables **multi-instance VRAM sharing**. A separate process holds VAE/CLIP models on one GPU while multiple ComfyUI instances share them via socket connections.

```
┌─────────────────────────────────────────────────────────┐
│                   GPU 1 (cuda:1)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Luna VAE/CLIP Daemon                   │   │
│  │  • VAE + CLIP loaded once                       │   │
│  │  • Serves encode/decode via local socket        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ Socket (127.0.0.1:19283)
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ComfyUI :8188 │ │ ComfyUI :8189 │ │ ComfyUI :8190 │
│ UNet only     │ │ UNet only     │ │ UNet only     │
└───────────────┘ └───────────────┘ └───────────────┘
```

### LunaDaemonVAELoader
**Category:** `Luna/Daemon`  
**Purpose:** Load VAE through daemon for shared VRAM usage.

#### What It Does
Creates a proxy VAE that routes all encode/decode operations through the daemon instead of loading the model locally.

#### Why It Exists
VAE models use 2-4GB VRAM each. Sharing one across instances saves significant memory.

---

### LunaDaemonCLIPLoader
**Category:** `Luna/Daemon`  
**Purpose:** Load CLIP through daemon for shared VRAM usage.

#### SDXL Support
For SDXL, load both clip_l and clip_g. For SD1.5, just the single clip model.

---

### LunaCheckpointTunnel
**Category:** `Luna/Daemon`  
**Purpose:** Transparent proxy that auto-routes to daemon when available.

#### Intelligent Routing
| Daemon State | Behavior |
|--------------|----------|
| Not running | Pass everything through unchanged |
| Running, no models | Load into daemon, output proxies |
| Running, matching type | Return existing proxies (share!) |
| Running, different type | Load as additional component |

#### Why It Exists
Enables "set and forget" daemon usage. Just insert after checkpoint loader - no workflow changes needed.

---

### Luna Daemon API
**Purpose:** Web endpoints for daemon control.

#### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/luna/daemon/status` | GET | Get daemon status and loaded models |
| `/luna/daemon/start` | POST | Start daemon process |
| `/luna/daemon/stop` | POST | Stop daemon process |
| `/luna/daemon/unload` | POST | Unload models from daemon |

---

## Civitai Integration

### LunaCivitaiScraper
**Category:** `Luna/Utils`  
**Purpose:** Fetch and embed Civitai metadata for LoRAs/embeddings.

#### What It Does
1. Computes SHA-256 tensor hash for safetensors file
2. Queries Civitai API using hash
3. Retrieves trigger words, tags, descriptions
4. Embeds metadata into file header (modelspec.* format)
5. Optionally writes .swarm.json sidecar

#### Metadata Retrieved
- Trigger words
- Training tags with frequencies
- Model description
- Recommended weights
- Base model compatibility
- NSFW classification

#### SwarmUI Compatibility
Uses the same metadata format as SwarmUI for cross-tool compatibility.

#### Why It Exists
Managing LoRA trigger words manually is error-prone. This node automates metadata retrieval so your LoRAs are self-documenting.

---

## Summary: Node Categories by Use Case

### Prompt Generation
- **LunaYAMLWildcard**: Structured random prompts
- **LunaPromptCraft**: Intelligent prompt generation with rules
- **LunaWildcardConnections**: LoRA/embedding linking

### Batch Workflows
- **LunaBatchPromptExtractor**: Harvest prompts from images
- **LunaBatchPromptLoader**: Iterate through prompt datasets
- **LunaLoRAValidator**: Validate LoRAs, find missing on CivitAI
- **LunaDimensionScaler**: Scale to model-native resolutions

### Configuration
- **LunaConfigGateway**: Central settings hub

### Post-Processing
- **Luna Simple/Advanced/Ultimate Upscaler**: Image upscaling
- **LunaMultiSaver**: Multi-format saving with quality gates

### Character Creation
- **LunaExpressionPromptBuilder/SlicerSaver**: Expression pack creation

### Infrastructure
- **Luna Daemon nodes**: Multi-instance VRAM sharing
- **LunaCivitaiScraper**: Model metadata management

---

## Recommended Batch Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LUNA BATCH PROCESSING PIPELINE                   │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │ Image Source Folder  │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐     ┌──────────────────────┐
  │ Luna Batch Prompt    │────▶│ prompts_metadata.json│
  │ Extractor            │     └──────────┬───────────┘
  └──────────────────────┘                │
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │ Luna LoRA Validator  │──▶ CivitAI Links
                               └──────────┬───────────┘
                                          │ (verify all LoRAs exist)
                                          ▼
  ┌──────────────────────┐     ┌──────────────────────┐
  │ Luna Checkpoint      │────▶│ Luna Batch Prompt    │
  │ Loader               │     │ Loader               │
  └──────────────────────┘     └──────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────┐         ┌───────────┐         ┌───────────┐
            │ positive  │         │ lora_stack│         │ width     │
            │ negative  │         │           │         │ height    │
            └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
                  │                     │                     │
                  │                     │                     ▼
                  │                     │           ┌──────────────────┐
                  │                     │           │ Luna Dimension   │
                  │                     │           │ Scaler           │
                  │                     │           └────────┬─────────┘
                  │                     │                    │
                  └─────────────────────┼────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │ Luna Config Gateway  │
                              │ (apply LoRAs, encode)│
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ KSampler / Generate  │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ Luna Multi Saver     │
                              │ %time%_%model%_%index%│
                              └──────────────────────┘
```

---

*Document generated from source code analysis. Last updated: December 2025*
