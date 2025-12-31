# ComfyUI-Luna-Collection: Node Documentation

A comprehensive reference for all nodes in the Luna Collection.

---

## Table of Contents

1. [Prompt & Wildcard System](#prompt--wildcard-system)
   - [LunaYAMLWildcard](#lunayamlwildcard)
   - [LunaPromptCraft](#lunapromptcraft)
   - [LunaWildcardConnections](#lunawildcardconnections)
   - [LunaTriggerInjector](#lunatriggerinjector)
   - [LunaZImageEncoder](#lunazimageencoder)
   - [LunaVisionNode](#lunavisionnode)
   - [LunaVLMPromptGenerator](#lunavlmpromptgenerator)
2. [Batch Processing](#batch-processing)
   - [LunaBatchPromptExtractor](#lunabatchpromptextractor)
   - [LunaBatchPromptLoader](#lunabatchpromptloader)
   - [LunaLoRAValidator](#lunaloravalidator)
   - [LunaDimensionScaler](#lunadimensionscaler)
3. [Configuration & Parameters](#configuration--parameters)
   - [LunaConfigGateway](#lunaconfiggateway)
   - [LunaExpressionPack](#lunaexpressionpack)
4. [Image Upscaling](#image-upscaling)
   - [Luna Simple Upscaler](#luna-simple-upscaler)
   - [Luna Advanced Upscaler](#luna-advanced-upscaler)
   - [Luna Ultimate SD Upscale](#luna-ultimate-sd-upscale)
5. [Image Saving](#image-saving)
   - [LunaMultiSaver](#lunamultisaver)
6. [Model Management & Daemon](#model-management--daemon)
   - [LunaModelRouter](#lunamodelrouter)
   - [LunaSecondaryModelLoader](#lunasecondarymodelloader)
   - [LunaModelRestore](#lunamodelrestore)
   - [LunaDynamicModelLoader](#lunadynamicmodelloader)
   - [LunaDaemonVAELoader](#lunadaemonvaeloader)
   - [LunaDaemonCLIPLoader](#lunadaemoncliploader)
   - [LunaCheckpointTunnel](#lunacheckpointtunnel)
   - [LunaGGUFConverter](#lunaggufconverter)
   - [LunaOptimizedWeightsManager](#lunaoptimizedweightsmanager)
   - [Luna Daemon API](#luna-daemon-api)
7. [Civitai Integration](#civitai-integration)
   - [LunaCivitaiScraper](#lunacivitaiscraper)
8. [External Integrations](#external-integrations)
   - [Realtime LoRA Training](#realtime-lora-training)
   - [DiffusionToolkit Bridge](#diffusiontoolkit-bridge)

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

**UI Access:** The Connections Manager is available in the ComfyUI sidebar (look for the Luna icon) or via the floating toolbar button.

---

### LunaTriggerInjector
**Category:** `Luna/Prompting`  
**Purpose:** Automatically inject LoRA trigger words into prompts.

#### What It Does
Parses incoming LoRA stack, looks up trigger words from metadata (Civitai or embedded), and injects them into the prompt at a configurable position.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| prompt | STRING | Base prompt text |
| lora_stack | LORA_STACK | LoRAs to extract triggers from |
| position | ENUM | start / end / after_quality |
| separator | STRING | Between triggers (default: ", ") |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| prompt | STRING | Prompt with triggers injected |
| triggers | STRING | Just the trigger words |

#### Why It Exists
Many LoRAs require specific trigger words. This node eliminates forgetting triggers and keeps prompts clean while ensuring LoRAs activate properly.

---

### LunaZImageEncoder
**Category:** `Luna/Encoding`  
**Purpose:** AI-enhanced prompt encoding for Z-IMAGE models with vision modes and noise injection.

#### What It Does
The unified prompt processing node for Z-IMAGE workflows. Combines multiple capabilities:
- **AI Enhancement**: Uses Qwen3-VL to enhance and expand prompts
- **Vision Modes**: Describe images, extract styles, or blend text+image
- **Noise Injection**: Built-in conditioning noise for better diversity

#### Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Luna Z-IMAGE Encoder 🧠                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:          LLM (Qwen3-VL from Model Router)                          │
│  PROMPT:         "anime girl, detailed, colorful"                          │
│  AI ENHANCEMENT: [off] [subtle] [moderate] [maximum]                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  VISION MODE:    [disabled] [describe] [extract_style] [blend]             │
│  BLEND WEIGHT:   0.5 (for blend mode only)                                 │
│  IMAGE INPUT:    [optional reference image]                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  NOISE INJECTION: [✓ Enable]                                               │
│    strength: 0.02    start_percent: 0.0    end_percent: 0.3               │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| llm | LLM | Qwen3-VL from Luna Model Router |
| positive | STRING | Base positive prompt |
| negative | STRING | Negative prompt |
| ai_enhancement | COMBO | off/subtle/moderate/maximum |
| vision_mode | COMBO | disabled/describe/extract_style/blend |
| image | IMAGE | Optional reference image (for vision modes) |
| blend_weight | FLOAT | Image/text blend ratio (0.0-1.0) |
| inject_noise | BOOLEAN | Enable conditioning noise |
| noise_strength | FLOAT | Noise intensity (0.0-0.1) |
| noise_start | FLOAT | Start injection percent |
| noise_end | FLOAT | End injection percent |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| CONDITIONING+ | CONDITIONING | Enhanced positive conditioning with noise |
| CONDITIONING- | CONDITIONING | Negative conditioning |
| enhanced_prompt | STRING | AI-enhanced prompt text |

#### Vision Modes
| Mode | Description | Use Case |
|------|-------------|----------|
| `disabled` | Text-only encoding | Standard text2img |
| `describe` | VLM describes image → expands prompt | Character/scene reference |
| `extract_style` | Extract artistic style as suffix | Style transfer |
| `blend` | Fuse text + image embeddings | Image variations |

#### AI Enhancement Levels
| Level | Description |
|-------|-------------|
| `off` | Use prompt as-is |
| `subtle` | Minor detail expansion |
| `moderate` | Add composition, lighting, style hints |
| `maximum` | Full artistic expansion with rich details |

#### Why It Exists
Z-IMAGE models use Qwen3-VL as their text encoder. This node provides the full encoding pipeline: AI-enhanced prompts, vision-aware conditioning, and the noise injection technique that improves generation diversity.

---

### LunaVisionNode
**Category:** `Luna/Vision`  
**Purpose:** Describe images or extract artistic style using vision LLM.

#### What It Does
Analyzes images using Qwen3-VL vision capabilities to generate descriptions or extract style information.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| llm | LLM | Qwen3-VL from Luna Model Router |
| image | IMAGE | Reference image to analyze |
| mode | COMBO | describe/extract_style |
| max_tokens | INT | Maximum response length |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| description | STRING | Image description or style extraction |

#### Why It Exists
Provides standalone image analysis for workflows that need VLM capabilities without full encoding pipeline.

---

### LunaVLMPromptGenerator
**Category:** `Luna/Vision`  
**Purpose:** Generate generation prompts from reference images.

#### What It Does
Takes a reference image and generates a complete prompt suitable for recreating similar images. Uses structured output to ensure proper prompt formatting.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| llm | LLM | Qwen3-VL from Luna Model Router |
| image | IMAGE | Reference image |
| style_hint | STRING | Optional style guidance |
| detail_level | COMBO | minimal/balanced/detailed |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| prompt | STRING | Generated prompt for image recreation |
| style_tags | STRING | Extracted style tags |

#### Why It Exists
Useful for creating training prompts, recreating styles from reference images, or generating variations of existing artwork.

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

### LunaExpressionPack
**Category:** `Luna/Logic`  
**Purpose:** Logic and math expressions for workflow automation.

#### What It Does
Collection of utility nodes for workflow logic:

| Node | Description |
|------|-------------|
| **Luna Compare** | Compare two values (==, !=, <, >, <=, >=) |
| **Luna Switch** | Conditional output routing |
| **Luna Math** | Basic math operations |
| **Luna String Format** | Template-based string formatting |
| **Luna Random Choice** | Random selection from list |

#### Why It Exists
Complex workflows need conditional logic. These nodes provide the building blocks for dynamic, data-driven workflows.

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

### LunaModelRouter
**Category:** `Luna/Loaders`  
**Purpose:** Unified model loader for ALL Stable Diffusion architectures with explicit CLIP configuration.

#### What It Does
The universal model loader that handles SD1.5, SDXL, Flux, SD3, and Z-IMAGE models with a single node. Features:
- **Explicit CLIP slots**: 4 dedicated CLIP inputs for full control
- **Model type detection**: Automatic or manual architecture specification
- **Dynamic Loader integration**: Optional JIT precision conversion
- **Vision model support**: Direct CLIP_VISION output for vision-enabled architectures
- **LLM output**: Qwen3-VL output for Z-IMAGE workflows
- **Daemon integration**: Optional VAE/CLIP daemon connection

#### Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Luna Model Router ⚡                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  MODEL SOURCE:     [checkpoints ▼] [diffusion_models ▼] [unet (gguf) ▼]    │
│  MODEL NAME:       [ponyDiffusionV6XL.safetensors ▼]                       │
│  MODEL TYPE:       [auto] [SD1.5] [SDXL] [SDXL+Vision] [Flux] [Flux+Vision] [SD3] [Z-IMAGE] │
├─────────────────────────────────────────────────────────────────────────────┤
│  ENABLE DYNAMIC LOADER: [✓]                                                │
│    PRECISION:      [bf16 ▼] [fp8_e4m3fn ▼] [gguf_Q8_0 ▼]                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLIP 1:          [clip_l.safetensors ▼]        ← Required for all        │
│  CLIP 2:          [clip_g.safetensors ▼]        ← SDXL, SD3               │
│  CLIP 3:          [t5xxl_fp16.safetensors ▼]    ← Flux, SD3               │
│  CLIP 4:          [siglip_vision.safetensors ▼] ← Vision architectures    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Z-IMAGE: clip_1 = Full Qwen3-VL model (hidden state extraction)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### CLIP Requirements by Model Type
| Model Type | clip_1 | clip_2 | clip_3 | clip_4 |
|------------|--------|--------|--------|--------|
| SD1.5 | CLIP-L | - | - | - |
| SDXL | CLIP-L | CLIP-G | - | - |
| SDXL + Vision | CLIP-L | CLIP-G | - | SigLIP/CLIP-H |
| Flux | CLIP-L | - | T5-XXL | - |
| Flux + Vision | CLIP-L | - | T5-XXL | SigLIP |
| SD3 | CLIP-L | CLIP-G | T5-XXL | - |
| Z-IMAGE | Full Qwen3-VL | - | - | (auto mmproj) |

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| model_source | COMBO | checkpoints/diffusion_models/unet |
| ckpt_name | COMBO | Model dropdown (filtered by source) |
| model_type | COMBO | Architecture selection or auto-detect |
| enable_dynamic | BOOLEAN | Enable JIT precision conversion |
| precision | COMBO | bf16/fp8_e4m3fn/gguf_Q8_0/gguf_Q4_K_M |
| clip_1 | COMBO | Primary CLIP model |
| clip_2 | COMBO | Secondary CLIP (SDXL/SD3) |
| clip_3 | COMBO | T5 model (Flux/SD3) |
| clip_4 | COMBO | Vision CLIP model |
| use_daemon | BOOLEAN | Connect to Luna Daemon |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| MODEL | MODEL | Loaded/optimized UNet |
| CLIP | CLIP | Combined CLIP encoder |
| VAE | VAE | VAE model (or daemon proxy) |
| LLM | LLM | Qwen3-VL for Z-IMAGE |
| CLIP_VISION | CLIP_VISION | Vision model (if applicable) |
| model_name | STRING | Loaded model name |
| status | STRING | Load status message |

#### Web Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/luna/model_router/models` | GET | List available models by source |
| `/luna/model_router/clips` | GET | List available CLIP models |
| `/luna/model_router/status` | GET | Current load status |

#### Why It Exists
Traditional loaders require different nodes for each architecture. Model Router provides a single unified interface with explicit CLIP control, eliminating the complexity of managing multiple loader types.

---

### LunaSecondaryModelLoader
**Category:** `Luna/Loaders`  
**Purpose:** Load additional models for multi-model workflows with CLIP sharing and RAM offloading.

#### What It Does
Enables complex workflows requiring multiple models:
- **CLIP Sharing**: Reuse CLIP from primary model when compatible
- **RAM Offloading**: Offload primary model to RAM when secondary loads
- **Memory Management**: ModelMemoryManager singleton tracks all loaded models
- **Smart Restoration**: Models can be restored from RAM to VRAM on demand

#### Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Luna Secondary Model Loader 🔄                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  PRIMARY MODEL:    [← Connect from Model Router]                           │
│  PRIMARY CLIP:     [← Connect for CLIP sharing check]                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  SECONDARY CKPT:   [animaginexl.safetensors ▼]                             │
│  SECONDARY TYPE:   [SDXL ▼]                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLIP SHARING:     [✓ Auto-detect] → Reuses CLIP if types match           │
│  OFFLOAD PRIMARY:  [✓ To RAM] → Primary → RAM, Secondary → VRAM            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Memory Flow
```
Initial State:
  VRAM: [Primary Model ~12GB]
  RAM:  [empty]

After Secondary Load (with offload):
  VRAM: [Secondary Model ~12GB]  
  RAM:  [Primary Model (offloaded)]

After Model Restore:
  VRAM: [Primary Model ~12GB]
  RAM:  [Secondary (if still needed)]
```

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| primary_model | MODEL | Model from primary loader |
| primary_clip | CLIP | CLIP from primary loader |
| ckpt_name | COMBO | Secondary checkpoint |
| model_type | COMBO | Secondary architecture |
| clip_sharing | COMBO | auto/force_share/force_separate |
| offload_primary | BOOLEAN | Offload primary to RAM |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| MODEL | MODEL | Secondary model |
| CLIP | CLIP | Secondary or shared CLIP |
| VAE | VAE | Secondary VAE |
| primary_ref | MODEL_REF | Reference for restoration |

#### CLIP Sharing Logic
| Primary | Secondary | Action |
|---------|-----------|--------|
| SDXL | SDXL | Share CLIP (same architecture) |
| SD1.5 | SDXL | Separate CLIP (incompatible) |
| SDXL | SDXL+Vision | Share base, add vision |

#### Why It Exists
Multi-model workflows (e.g., refiner + base, different styles) typically require loading everything into VRAM simultaneously. This node enables efficient model switching with CLIP reuse and RAM offloading.

---

### LunaModelRestore
**Category:** `Luna/Loaders`  
**Purpose:** Restore models offloaded to RAM back to VRAM.

#### What It Does
Companion node to Secondary Model Loader. Restores models that were offloaded to RAM:
- Retrieves model from ModelMemoryManager
- Moves model back to VRAM
- Optionally offloads currently-loaded model to RAM

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| model_ref | MODEL_REF | Reference from Secondary Loader |
| target_device | COMBO | cuda:0/cuda:1/auto |
| offload_current | BOOLEAN | Offload current VRAM model |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| MODEL | MODEL | Restored model |
| CLIP | CLIP | Associated CLIP |
| VAE | VAE | Associated VAE |

#### Why It Exists
After using secondary model, you often need to restore the primary. This node handles the memory choreography automatically.

---

### LunaDynamicModelLoader
**Category:** `Luna/Loaders`  
**Purpose:** Smart checkpoint loading with JIT precision conversion and lazy evaluation.

#### What It Does
The centerpiece of model management. Loads checkpoints with:
- **Smart lazy evaluation**: Only loads CLIP/VAE when those outputs are connected
- **JIT UNet conversion**: Converts UNet to optimized precision on first use
- **Hybrid loading**: CLIP/VAE from source file, UNet from cached optimized version
- **Automatic caching**: Converted UNets stored on NVMe for instant reloads

#### Architecture
```
┌────────────────────────────────────────────────────────┐
│             8TB HDD (Source Library)                   │
│  358 FP16 Checkpoints (6.5GB each)                     │
└────────────────────────────────────────────────────────┘
                          │
                          ▼ First use: extract UNet + convert
┌────────────────────────────────────────────────────────┐
│             NVMe (Local Optimized Weights)             │
│  models/unet/optimized/                                │
│  • illustriousXL_Q8_0.gguf (3.2GB)                     │
│  • ponyV6_fp8_e4m3fn_unet.safetensors (2.1GB)          │
└────────────────────────────────────────────────────────┘
                          │
                          ▼ Subsequent loads: instant from cache
┌────────────────────────────────────────────────────────┐
│  • MODEL always loads optimized UNet                   │
│  • CLIP/VAE only load if outputs are connected         │
│  • No mode selection needed - just wire what you need  │
└────────────────────────────────────────────────────────┘
```

#### Supported Precisions
| Precision | Best For | Size Reduction |
|-----------|----------|----------------|
| `bf16` | Universal, fast | ~50% |
| `fp8_e4m3fn` | Ada/Blackwell GPUs | ~75% |
| `gguf_Q8_0` | Ampere INT8 tensor cores | ~50% |
| `gguf_Q4_K_M` | Blackwell INT4 tensor cores | ~75% |

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| ckpt_name | COMBO | Checkpoint dropdown |
| precision | COMBO | Target UNet precision |
| local_weights_dir | STRING | Override cache directory |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| MODEL | MODEL | Optimized UNet |
| CLIP | CLIP | Original CLIP (lazy loaded) |
| VAE | VAE | Original VAE (lazy loaded) |
| unet_path | STRING | Path to cached UNet file |

#### Why It Exists
Loading full FP16 checkpoints is slow and memory-intensive. This node enables:
- Fast iteration with optimized models
- Large checkpoint libraries without VRAM constraints
- Seamless daemon integration (CLIP/VAE not loaded when using daemon)

---

### LunaGGUFConverter
**Category:** `Luna/Utils`  
**Purpose:** Convert checkpoints to quantized GGUF format.

#### What It Does
Extracts UNet from any checkpoint and converts to GGUF quantization:
- Q8_0 (8-bit) - Best for Ampere INT8 tensor cores
- Q4_K_M (4-bit) - Best for Blackwell INT4 tensor cores
- Q4_0 (4-bit) - Smaller, slightly lower quality

#### Why It Exists
Pre-convert your checkpoint library to optimized formats for faster loading and reduced VRAM usage.

---

### LunaOptimizedWeightsManager
**Category:** `Luna/Utils`  
**Purpose:** Manage cached optimized UNet files.

#### What It Does
Provides UI for managing the local optimized weights cache created by Luna Dynamic Model Loader and Model Router:
- **List cached files**: View all optimized UNets in cache directory
- **Delete old versions**: Remove outdated precision conversions
- **Cache statistics**: Total size, file count, age information
- **Batch operations**: Clear all caches or select specific files

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| action | COMBO | list/delete/clear_all/stats |
| file_pattern | STRING | Filter pattern (e.g., "*_Q8_0*") |
| cache_dir | STRING | Override default cache location |

#### Outputs
| Output | Type | Description |
|--------|------|-------------|
| result | STRING | Action result/file list |
| size_gb | FLOAT | Total cache size in GB |
| file_count | INT | Number of cached files |

#### Why It Exists
As you test different precision conversions, the cache grows. This node helps manage disk usage without manual file deletion.

---

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

## External Integrations

### Realtime LoRA Training

Luna Collection integrates with [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) for in-workflow SDXL LoRA training.

#### Setup
1. Clone comfyUI-Realtime-Lora to your custom_nodes folder
2. Install [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) somewhere
3. Create a junction so sd-scripts can use ComfyUI's venv:
```powershell
New-Item -ItemType Junction -Path "D:\path\to\sd-scripts\.venv" -Target "D:\AI\ComfyUI\venv"
```

#### Workflow
```
┌─────────────────────────────────────────────────────────────────────┐
│                 REALTIME LORA TRAINING WORKFLOW                     │
└─────────────────────────────────────────────────────────────────────┘

  [Image Folder with .txt captions]
             │
             ▼
  ┌──────────────────────────────────────┐
  │ Realtime LoRA Trainer (SDXL)         │
  │ • sd_scripts_path: path/to/sd-scripts│
  │ • ckpt_name: your_checkpoint.safetensors
  │ • training_steps: 500                │
  │ • learning_rate: 0.0005              │
  │ • lora_rank: 16                      │
  └──────────────────┬───────────────────┘
                     │ lora_path
                     ▼
  ┌──────────────────────────────────────┐
  │ Apply Trained LoRA                   │
  │ • strength: 1.0                      │
  └──────────────────┬───────────────────┘
                     │
                     ▼
  [KSampler with freshly trained LoRA]
```

#### Luna Integration Points
- **Luna Batch Prompt Extractor**: Export captions from existing images
- **Luna Multi Saver**: Save generated results with metadata
- **Luna Trigger Injector**: Add trained LoRA triggers to prompts

---

### DiffusionToolkit Bridge

See [docs/LUNA_TOOLKIT_BRIDGE_NODES.md](docs/LUNA_TOOLKIT_BRIDGE_NODES.md) for planned integration nodes.

#### Planned Nodes

| Node | Purpose |
|------|---------|
| **LunaDTImageLoader** | Load images from DT by ID or path |
| **LunaDTSimilarSearch** | Find similar images via DT embeddings |
| **LunaDTClusterSampler** | Sample from DT image clusters |
| **LunaDTCaptionFetcher** | Get captions from DT database |
| **LunaDTMetadataWriter** | Write gen params back to DT |
| **LunaDTPromptInjector** | Inject DT captions into prompts |
| **LunaDTControlNetCache** | Cache preprocessor outputs in DT |
| **LunaDTConnectionStatus** | Check DT API availability |

#### Use Cases
- Query 500k image library from within ComfyUI
- Find similar images for img2img workflows
- Sample from clusters for style consistency
- Write generation metadata back to central database

---

## Summary: Node Categories by Use Case

### Prompt Generation
- **LunaYAMLWildcard**: Structured random prompts
- **LunaPromptCraft**: Intelligent prompt generation with rules
- **LunaWildcardConnections**: LoRA/embedding linking
- **LunaTriggerInjector**: Auto-inject LoRA triggers

### Batch Workflows
- **LunaBatchPromptExtractor**: Harvest prompts from images
- **LunaBatchPromptLoader**: Iterate through prompt datasets
- **LunaLoRAValidator**: Validate LoRAs, find missing on CivitAI
- **LunaDimensionScaler**: Scale to model-native resolutions

### Configuration
- **LunaConfigGateway**: Central settings hub
- **LunaExpressionPack**: Logic and math operations

### Model Loading
- **LunaDynamicModelLoader**: Smart lazy loading with precision conversion
- **LunaGGUFConverter**: Convert to quantized formats
- **Luna Daemon nodes**: Multi-instance VRAM sharing

### Post-Processing
- **Luna Simple/Advanced/Ultimate Upscaler**: Image upscaling
- **LunaMultiSaver**: Multi-format saving with quality gates

### External Tools
- **Realtime LoRA Training**: In-workflow LoRA training via sd-scripts
- **DiffusionToolkit Bridge**: Image library integration (planned)

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

*Document generated from source code analysis. Last updated: December 2025 (v1.4.0)*
