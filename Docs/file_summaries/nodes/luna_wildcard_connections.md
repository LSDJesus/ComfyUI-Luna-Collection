# luna_wildcard_connections.py

## Purpose
Dynamic LoRA/embedding linking to wildcard categories with interactive connection management and web API for editing.

## Exports
**Classes:**
- `LunaConnectionMatcher` - Finds LoRAs based on categories, tags, triggers, training data, or Civitai types
- `LunaConnectionEditor` - Interactive node for viewing/editing LoRA/embedding connections
- `LunaSmartLoRALinker` - Main integration node that automatically injects LoRAs based on wildcard resolution
- `LunaConnectionStats` - Displays statistics about the connections database
- `ConnectionsDB` - Singleton database manager for connections.json

**Functions:**
- `register_routes()` - Registers web API endpoints for connection management
- `detect_model_type_from_name(model_name)` - Detects model type from checkpoint name
- `detect_model_type_from_model(model)` - Detects model type from loaded MODEL object

**Constants:**
- None

## Key Imports
- `json`, `os`, `random`, `re` - Data processing and text matching
- `folder_paths` - ComfyUI model path resolution
- `aiohttp` - Asynchronous web framework (optional)
- `PromptServer` - ComfyUI web server integration (optional)

## ComfyUI Node Configuration
- **LunaConnectionMatcher**
  - Category: `Luna/Connections`
  - Display Name: `Luna Connection Matcher`
  - Return Types: `(STRING, LORA_STACK, STRING)`
  - Return Names: `(lora_string, LORA_STACK, matched_info)`
  - Function: `match_connections`
- **LunaConnectionEditor**
  - Category: `Luna/Connections`
  - Display Name: `Luna Connection Editor`
  - Return Types: `(STRING, BOOLEAN)`
  - Return Names: `(status, success)`
  - Function: `edit_connection`
  - OUTPUT_NODE: `True`
- **LunaSmartLoRALinker**
  - Category: `Luna/Connections`
  - Display Name: `Luna Smart LoRA Linker`
  - Return Types: `(STRING, STRING, LORA_STACK, STRING, STRING)`
  - Return Names: `(enhanced_prompt, lora_string, LORA_STACK, detected_type, match_report)`
  - Function: `link_loras`
- **LunaConnectionStats**
  - Category: `Luna/Connections`
  - Display Name: `Luna Connection Stats`
  - Return Types: `(STRING,)`
  - Return Names: `(stats,)`
  - Function: `get_stats`

## Input Schema
**LunaConnectionMatcher:**
- `mode` (["by_category", "by_tags", "by_triggers", "by_training", "by_civitai_type", "combined"]): Matching strategy
- `max_loras` (INT, 1-10): Maximum LoRAs to return
- `seed` (INT): Random seed for selection
- `prompt` (STRING, optional): Text to scan for triggers/training tags
- `resolved_categories` (STRING, optional): Comma-separated category paths
- `tags` (STRING, optional): Comma-separated tags to match
- `civitai_types` (STRING, optional): Comma-separated Civitai types
- `model_type_filter` (["any", "sdxl", "pony", "illustrious", "sd15"]): Filter by model type
- `weight_mode` (["metadata_default", "metadata_random", "override"]): Weight determination method
- `weight_override` (FLOAT): Fixed weight when mode is 'override'
- `training_tag_min_matches` (INT, 1-10): Minimum training tag matches

**LunaConnectionEditor:**
- `edit_type` (["lora", "embedding"]): Type of item to edit
- `action` (["view", "add", "update", "remove"]): Action to perform
- `lora_select` (lora_list): LoRA to edit
- `embedding_select` (embedding_list): Embedding to edit
- `triggers` (STRING): Comma-separated trigger words
- `categories` (STRING): Comma-separated category paths
- `tags` (STRING): Comma-separated tags
- `model_type` (["sdxl", "pony", "illustrious", "sd15", "any"]): Model type compatibility
- `weight_default`/`weight_min`/`weight_max` (FLOAT): Weight configuration
- `notes` (STRING): Additional notes

**LunaSmartLoRALinker:**
- `prompt` (STRING): Resolved prompt from wildcard processing
- `enable_category_matching` (BOOLEAN): Match by wildcard category paths
- `enable_trigger_matching` (BOOLEAN): Match by trigger words in prompt
- `enable_training_tag_matching` (BOOLEAN): Use training data analysis
- `max_loras` (INT, 0-10): Maximum LoRAs to inject
- `seed` (INT): Random seed
- `model` (MODEL, optional): Connect to auto-detect model type
- `checkpoint_name` (STRING, optional): Checkpoint name for fallback detection
- `wildcard_metadata` (STRING, optional): JSON metadata from wildcard resolution
- `model_type_override` (["auto", "any", "sdxl", "pony", "illustrious", "sd15", "flux"]): Model type override
- `inject_mode` (["smart_triggers", "activation_text", "none"]): How to inject activation into prompt
- `civitai_type_filter` (["any", "character", "concept", "style", "poses", "clothing", "tool"]): Filter by Civitai type
- `weight_mode` (["metadata_default", "metadata_random", "override"]): Weight determination
- `weight_override` (FLOAT): Fixed weight when mode is 'override'
- `training_tag_min_matches` (INT): Minimum training tag matches
- `existing_lora_string` (STRING, optional): Existing LoRA string to append to

## Key Methods/Functions
- `ConnectionsDB.load(force_reload=False) -> Dict`
  - Loads connections.json database with caching
  - Manages LoRA/embedding connections to categories/tags
- `ConnectionsDB.save(data) -> bool`
  - Saves connections database to disk
- `ConnectionsDB.find_loras_by_category/category/tags/trigger/training_tags/civitai_tags(...) -> List[Dict]`
  - Query methods for finding LoRAs by different criteria
  - Supports category paths, tags, triggers, training data analysis, Civitai types
- `LunaConnectionMatcher.match_connections(mode, max_loras, seed, **kwargs) -> (str, List, str)`
  - Finds and returns matching LoRAs based on specified criteria
  - Supports combined matching across multiple sources
- `LunaSmartLoRALinker.link_loras(prompt, **kwargs) -> (str, str, List, str, str)`
  - Main integration node with intelligent metadata-driven matching
  - Auto-detects model type, injects activation text, builds LoRA strings
- `LunaConnectionEditor.edit_connection(edit_type, action, **kwargs) -> (str, bool)`
  - Views, adds, updates, or removes connection entries
  - Interactive editing of LoRA/embedding metadata
- `register_routes() -> None`
  - Registers web API endpoints for connection management
  - Provides REST API for frontend integration
- `detect_model_type_from_name/model_name) -> str`
  - Detects model type from checkpoint filename
- `detect_model_type_from_model(model) -> str`
  - Detects model type from loaded MODEL object architecture

## Dependencies
**Internal:**
- None (standalone nodes)

**External:**
- Required: `folder_paths`
- Optional: `aiohttp`, ComfyUI `PromptServer` (for web API)

## Integration Points
**Input:** Prompts, wildcard metadata, model objects for type detection, LoRA/embedding lists from ComfyUI
**Output:** LoRA strings in A1111 format, LORA_STACK for samplers, enhanced prompts with triggers, match reports
**Side Effects:** File I/O for connections.json database, web API registration, model type detection

## Notes
- Manages connections between LoRAs/embeddings and wildcard categories/tags
- Supports multiple matching strategies: category paths, trigger words, training data analysis, Civitai types
- Interactive editing via web interface and ComfyUI nodes
- Auto-detects model type from connected MODEL to filter incompatible LoRAs
- Injects activation text or smart triggers into prompts based on metadata
- Web API provides full CRUD operations for connection management