# promptcraft/engine.py

## Purpose
Core engine for smart wildcard resolution with constraints, modifiers, expanders, and LoRA/embedding linking.

## Exports
- `CategoryLoader`: Loads/manages wildcard categories from YAML
- `TagResolver`: Resolves tags for categories
- `ConstraintEngine`: Applies blacklist/whitelist constraints
- `ModifierEngine`: Applies text modifiers to resolved items
- `ExpanderEngine`: Adds expansions based on picked items
- `LoRALinker`: Links wildcard picks to LoRAs/embeddings
- `LunaPromptEngine`: Main orchestrator for wildcard processing
- `create_engine()`: Factory function for engine instances

## Key Imports
- `os`, `re`, `random`, `typing`, `pathlib`
- `yaml` (optional)
- `folder_paths` (optional)

## ComfyUI Node Configuration
N/A - Utility module

## Input Schema
N/A

## Key Methods
- `CategoryLoader.load_category(category_name: str) -> Dict`
- `CategoryLoader.get_items_at_path(path: str) -> List[Tuple[CategoryItem, str]]`
- `ConstraintEngine.filter_items(category: str, items: List, context: PromptContext) -> List`
- `ModifierEngine.apply_modifiers(text: str, path: str, context: PromptContext) -> str`
- `ExpanderEngine.get_expansions(context: PromptContext) -> List[Expansion]`
- `LoRALinker.get_links(context: PromptContext) -> Tuple[List[LoRALink], List[EmbeddingLink]]`
- `LoRALinker.get_lora_stack(context: PromptContext) -> List[Tuple[str, float, float]]`
- `LunaPromptEngine.process_template(template: str, seed: int, ...) -> Dict[str, Any]`
- `LunaPromptEngine.count_combinations(template: str) -> int`

## Dependencies
- PyYAML (optional for YAML loading)
- folder_paths (optional for ComfyUI integration)

## Integration Points
- Used by `nodes/promptcraft/nodes.py` for wildcard processing
- Integrates with ComfyUI model directories via folder_paths

## Notes
Core wildcard resolution logic supporting hierarchical categories, constraints, modifiers, expansions, and automatic LoRA/embedding linking.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\promptcraft\engine.md