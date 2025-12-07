# luna_secondary_loader.py

## Purpose
Multi-model workflow support with CLIP sharing and RAM offloading. Enables efficient model switching for workflows using multiple model types (e.g., Flux generation → SDXL refinement).

## Exports
**Classes:**
- `LunaSecondaryModelLoader` - Loads secondary models with CLIP sharing and memory management
- `LunaModelRestore` - Restores models offloaded to RAM back to VRAM
- `ModelMemoryManager` - Singleton for managing model offloading between VRAM and system RAM

**Functions:**
- `get_shareable_clips(primary_type, secondary_type) -> (set, set)` - Determines which CLIP encoders can be shared between model types

**Constants:**
- `CLIP_REQUIREMENTS_MAP` - CLIP encoder requirements for each model type

## Key Imports
- `folder_paths` - ComfyUI model path resolution
- `comfy.sd`, `comfy.utils`, `comfy.model_management` - ComfyUI model loading and management
- `torch`, `gc` - PyTorch operations and garbage collection

## ComfyUI Node Configuration
- **LunaSecondaryModelLoader**
  - Category: `Luna/Core`
  - Display Name: `Luna Secondary Model Loader 🔄`
  - Return Types: `(MODEL, CLIP, VAE, STRING, STRING)`
  - Return Names: `(model, clip, vae, model_name, status)`
  - Function: `load`
- **LunaModelRestore**
  - Category: `Luna/Core`
  - Display Name: `Luna Model Restore 📤`
  - Return Types: `(*, MODEL, STRING)`
  - Return Names: `(passthrough, restored_model, status)`
  - Function: `restore`

## Input Schema
**LunaSecondaryModelLoader:**
- `model_source` (MODEL_SOURCES): Folder to load secondary model from
- `model_name` (all_models): Secondary model file
- `model_type` (MODEL_TYPES): Architecture of secondary model
- `unload_primary_to_ram` (BOOLEAN, default=True): Move primary model to system RAM to free VRAM
- `primary_model` (MODEL, optional): Primary model to offload to RAM
- `primary_clip` (CLIP, optional): CLIP from primary model - reuse compatible encoders
- `primary_vae` (VAE, optional): VAE from primary model - reuse if compatible
- `primary_type` (["auto"] + MODEL_TYPES, default="auto"): Type of primary model for CLIP sharing detection
- `additional_clip` (clip_list, default="None"): Additional CLIP encoder needed by secondary model
- `secondary_vae` (vae_list, default="None"): Different VAE for secondary model

**LunaModelRestore:**
- `trigger` (*): Any input - will be passed through while triggering model restore
- `model_id` (STRING, optional): Model ID to restore (leave empty for auto-detect)
- `original_model` (MODEL, optional): Original model object for state loading

## Key Methods/Functions
- `LunaSecondaryModelLoader.load(model_source, model_name, model_type, unload_primary_to_ram, **kwargs) -> (Any, Any, Any, str, str)`
  - Loads secondary model with CLIP sharing and optional RAM offloading
  - Detects primary model type, offloads to RAM if requested, builds combined CLIP
  - Returns secondary model components and status
- `ModelMemoryManager.offload_to_ram(model, model_id) -> bool`
  - Moves model from VRAM to system RAM for temporary storage
  - Stores model state dict or object for later retrieval
- `ModelMemoryManager.reload_from_ram(model_id, target_model=None, device="cuda") -> Optional[Any]`
  - Restores model from RAM cache back to VRAM
  - Can load into existing model or return state dict
- `get_shareable_clips(primary_type, secondary_type) -> (shareable, needs_loading)`
  - Analyzes CLIP requirements to determine which encoders can be shared
  - Returns sets of shareable and additional encoders needed
- `LunaModelRestore.restore(trigger, model_id="", original_model=None) -> (Any, Any, str)`
  - Restores cached model from RAM while passing through trigger input
  - Auto-detects model ID if not specified

## Dependencies
**Internal:**
- None (standalone node)

**External:**
- Required: `comfy`, `folder_paths`, `torch`
- Optional: ComfyUI-GGUF (for .gguf model loading)

## Integration Points
**Input:** Primary model components from Luna Model Router, secondary model filenames
**Output:** Secondary model components, restored models, status messages
**Side Effects:** RAM offloading of primary models, CLIP encoder sharing, VRAM management

## Notes
- CLIP sharing logic: Reuse compatible encoders between model types (e.g., SDXL→Flux shares CLIP-L, adds T5-XXL)
- RAM offloading: Move primary models to system RAM to free VRAM for secondary models, faster than disk reload
- Supports multi-model workflows like Flux generation → SDXL refinement
- Z-IMAGE uses different architecture, cannot share CLIP with standard models
- ModelMemoryManager is a singleton for global RAM cache management