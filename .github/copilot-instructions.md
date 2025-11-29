# ComfyUI Luna Collection - AI Agent Instructions

## Project Overview
A ComfyUI custom node collection for image generation workflows. Provides nodes for upscaling, MediaPipe detailing, LoRA/embedding management, prompt processing, and multi-instance VRAM sharing via the Luna Daemon.

## Architecture

### Node Registration Pattern
All nodes live in `nodes/` (organized by subdirectory: `loaders/`, `preprocessing/`, `upscaling/`, `detailing/`, `performance/`). The `__init__.py` uses `os.walk()` to auto-discover `.py` files. Each node file exports:
```python
NODE_CLASS_MAPPINGS = {"NodeName": NodeClass}
NODE_DISPLAY_NAME_MAPPINGS = {"NodeName": "Display Name"}
```

### Core Components
- **`utils/`** - Shared engines: `mediapipe_engine.py`, `trt_engine.py`, `luna_logger.py`
- **`nodes/`** - ComfyUI node implementations with `CATEGORY = "Luna/..."` hierarchy
- **`luna_daemon/`** - Separate VAE/CLIP server for multi-instance VRAM sharing (socket-based, configurable via `config.py`)
- **`luna_collection/`** - Core library code with validation utilities

### Model Path Resolution
**Always use `folder_paths` for ComfyUI model directories**, never hardcode paths:
```python
import folder_paths

# Standard ComfyUI model directories:
folder_paths.get_filename_list("loras")        # models/loras/
folder_paths.get_filename_list("embeddings")   # models/embeddings/
folder_paths.get_filename_list("vae")          # models/vae/
folder_paths.get_filename_list("clip")         # models/clip/
folder_paths.get_full_path("loras", lora_name) # Full path to specific model
```

### Luna Daemon Configuration (`luna_daemon/config.py`)
The daemon loads shared VAE/CLIP models once for multi-instance workflows. Configure paths using ComfyUI conventions:
- `SHARED_DEVICE` - GPU for shared models (e.g., `"cuda:1"`)
- `VAE_PATH`, `CLIP_L_PATH`, `CLIP_G_PATH` - Paths to shared models
- `DAEMON_HOST`/`DAEMON_PORT` - Network binding (default `127.0.0.1:19283`)
- `EMBEDDINGS_DIR` - Textual inversions directory

**Daemon Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                   GPU 1 (cuda:1)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Luna VAE/CLIP Daemon                   │   │
│  │  • VAE + CLIP loaded once                       │   │
│  │  • Serves encode/decode via local socket        │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
└─────────────────────────┼───────────────────────────────┘
                          │ Socket (127.0.0.1:19283)
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ComfyUI :8188 │ │ ComfyUI :8189 │ │ ComfyUI :8190 │
│ UNet only     │ │ UNet only     │ │ UNet only     │
└───────────────┘ └───────────────┘ └───────────────┘
              GPU 0 (cuda:0) - Main inference
```

**Shared nodes:** `Luna Shared VAE Encode/Decode`, `Luna Shared CLIP Encode`, `Luna Daemon Status`

## Key Patterns

### ComfyUI Node Structure
```python
class LunaNodeName:
    CATEGORY = "Luna/SubCategory"
    RETURN_TYPES = ("TYPE1", "TYPE2")
    RETURN_NAMES = ("output1", "output2")
    FUNCTION = "method_name"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}
    
    def method_name(self, **inputs) -> tuple:
        return (result1, result2)
```

### LoRA Stack Format
Nodes output `LORA_STACK` as `List[Tuple[str, float, float]]` → `(lora_name, model_strength, clip_strength)`. Compatible with ComfyUI-Impact-Pack's Apply LoRA Stack.

### Web Endpoints
LoRA metadata endpoint via `@PromptServer.instance.routes.get("/luna/get_lora_metadata")` pattern.

### YAML Wildcard System (`nodes/luna_yaml_wildcard.py`)
Hierarchical YAML-based wildcard files stored in `models/wildcards/`. The system supports:

**Prompt Syntax:**
- `{filename}` - Random template from `filename.yaml`'s `templates` section
- `{filename:path.to.items}` - Random item from nested path
- `{filename: text with [path.to.item] substitutions}` - Inline template with `[path]` replacements
- `{1-10}` - Random integer range
- `{0.5-1.5:0.1}` - Random float with step resolution
- `__path/file__` - Legacy .txt wildcard reference (resolved recursively)

**YAML File Structure:**
```yaml
# Templates section - predefined prompt patterns
templates:
  full:
    - "a [category.subcategory] with [another.path]"
    - "template variation two"
  minimal:
    - "[just.one.thing]"

# Hierarchical item categories
category:
  subcategory:
    leaf_items:
      - item_one
      - item_two
    another_group:
      - item_a
      - item_b

# Direct lists at any level
simple_list:
  - option_1
  - option_2
```

**Path Resolution:** `{file:category.subcategory.leaf_items}` flattens nested items. Templates use `[path]` syntax for inline substitution.

**Related Nodes:** `LunaYAMLWildcard`, `LunaYAMLWildcardBatch`, `LunaWildcardBuilder`, `LunaLoRARandomizer`

## Testing

```powershell
# Run all tests with coverage
pytest --cov=luna_collection --cov-report=html

# Specific test categories
pytest -m unit
pytest -m integration
```

Test markers: `unit`, `integration`, `slow`, `utils`, `validation`. Fixtures in `tests/conftest.py` provide `sample_image_tensor`, `temp_text_file`, `mock_validator`.

## Development Commands

```powershell
# Start Luna Daemon (for multi-instance VRAM sharing)
.\scripts\start_daemon.ps1

# Start ComfyUI with specific port
.\scripts\start_server_workflow.ps1 -Port 8188
```

## Important Conventions

1. **Imports**: Use `import folder_paths` for ComfyUI model paths. Handle ImportError gracefully for optional deps.
2. **Error Handling**: Return unchanged input on graceful failures
3. **Caching**: Use module-level dicts for metadata caching (see `LUNA_METADATA_CACHE`)
4. **Tooltips**: Add `"tooltip": "..."` in INPUT_TYPES for UI hints
5. **Logging**: Use `print(f"lunaCore: ...")` or `print(f"[LunaNodeName] ...")` prefix pattern

## File Naming
- Nodes: `luna_<feature>.py` or in subdirs like `loaders/luna_lora_stacker.py`
- Scripts: Descriptive verbs in `scripts/` (e.g., `extract_lora_metadata.py`)
