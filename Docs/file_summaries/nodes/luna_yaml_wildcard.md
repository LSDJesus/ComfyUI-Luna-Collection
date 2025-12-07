# luna_yaml_wildcard.py

## Purpose
ComfyUI node for processing hierarchical YAML-based wildcards with support for templates, path-based selection, inline templates, random numbers, and legacy txt wildcards.

## Exports
**Classes:**
- `LunaYAMLWildcard` - Single prompt wildcard processor
- `LunaYAMLWildcardBatch` - Batch prompt generation from wildcards
- `LunaYAMLWildcardExplorer` - Explorer for available wildcard paths
- `LunaWildcardBuilder` - Interactive prompt builder with LoRA/embedding integration
- `LunaLoRARandomizer` - Random LoRA selection with weight ranges
- `LunaYAMLInjector` - Utility for injecting CSV items into YAML files
- `LunaYAMLPathExplorer` - Path exploration utility for YAML files
- `LunaYAMLWildcardParser` - Core parser for YAML wildcard processing

**Functions:**
- None

**Constants:**
- None

## Key Imports
- `yaml` - YAML file parsing and generation
- `random`, `re`, `decimal` - Text processing, regex, and randomization
- `folder_paths` - ComfyUI model path resolution (optional)

## ComfyUI Node Configuration
- **LunaYAMLWildcard**
  - Category: `Luna/Wildcards`
  - Display Name: `Luna YAML Wildcard`
  - Return Types: `(STRING,)`
  - Return Names: `(prompt,)`
  - Function: `process_wildcards`
- **LunaYAMLWildcardBatch**
  - Category: `Luna/Wildcards`
  - Display Name: `Luna YAML Wildcard Batch`
  - Return Types: `(STRING,)`
  - Return Names: `(prompts,)`
  - Output Is List: `(True,)`
  - Function: `generate_batch`
- **LunaYAMLWildcardExplorer**
  - Category: `Luna/Wildcards`
  - Display Name: `Luna YAML Wildcard Explorer`
  - Return Types: `(STRING,)`
  - Return Names: `(available_paths,)`
  - Function: `explore`
- **LunaWildcardBuilder**
  - Category: `Luna/Wildcards`
  - Display Name: `Luna Wildcard Builder`
  - Return Types: `(STRING, STRING, STRING)`
  - Return Names: `(prompt, loras_string, full_prompt)`
  - Function: `build_prompt`
- **LunaLoRARandomizer**
  - Category: `Luna/Wildcards`
  - Display Name: `Luna LoRA Randomizer`
  - Return Types: `(STRING,)`
  - Return Names: `(lora_string,)`
  - Function: `build_prompt`
- **LunaYAMLInjector**
  - Category: `Luna/Wildcards/Utils`
  - Display Name: `Luna YAML Injector`
  - Return Types: `(STRING, STRING, BOOLEAN)`
  - Return Names: `(yaml_preview, status, success)`
  - Function: `inject_items`
  - OUTPUT_NODE: `True`
- **LunaYAMLPathExplorer**
  - Category: `Luna/Wildcards/Utils`
  - Display Name: `Luna YAML Path Explorer`
  - Return Types: `(STRING,)`
  - Return Names: `(paths,)`
  - Function: `explore_paths`

## Input Schema
**LunaYAMLWildcard:**
- `prompt_template` (STRING): Prompt with {file:path} wildcards or {file: inline [path] template}
- `seed` (INT, default=0): Random seed (0 = random each time)
- `yaml_directory` (STRING, optional): Directory containing YAML wildcard files
- `txt_wildcard_directory` (STRING, optional): Directory containing legacy .txt wildcard files

**LunaYAMLWildcardBatch:**
- `prompt_template` (STRING): Prompt template with wildcards
- `count` (INT, default=10): Number of prompt variations to generate
- `seed` (INT, default=0): Base seed (0 = random). Each variation uses seed+index
- `yaml_directory` (STRING, optional): Directory containing YAML wildcard files
- `txt_wildcard_directory` (STRING, optional): Directory containing legacy .txt wildcard files
- `unique_only` (BOOLEAN, default=True): Remove duplicate prompts

**LunaWildcardBuilder:**
- `prompt_template` (STRING): Main prompt template with wildcards
- `seed` (INT, default=0): Random seed
- `yaml_directory` (STRING, optional): Directory containing YAML wildcard files
- `txt_wildcard_directory` (STRING, optional): Directory containing legacy .txt wildcard files
- `lora_string_input` (STRING, optional): LoRA string from other sources
- `lora_1/2/3/4` (lora_list): LoRA selections with weights
- `embedding_1/2/3` (embedding_list): Embedding selections
- `prefix`/`suffix` (STRING, optional): Text to add before/after prompt

**LunaLoRARandomizer:**
- `lora_count` (INT, default=2): Number of LoRAs to randomly select
- `seed` (INT, default=0): Random seed
- `weight_min`/`weight_max` (FLOAT): Weight range bounds
- `weight_step` (FLOAT, default=0.1): Weight increment step
- `pool_1-10` (lora_list): Pool of LoRAs to choose from

**LunaYAMLInjector:**
- `csv_input` (STRING): Comma-separated list of items to add
- `target_yaml` (yaml_files): Target YAML file to inject into
- `target_path` (STRING): Dot-separated path (e.g., 'hair.style.braids')
- `new_category` (STRING, optional): Create new sub-category at target path
- `format` (["list", "inline"]): Output format for items
- `yaml_directory` (STRING, optional): Directory containing YAML files
- `preview_only` (BOOLEAN, default=True): Preview changes without saving

## Key Methods/Functions
- `LunaYAMLWildcardParser.resolve_wildcard(wildcard, rng) -> str`
  - Resolves {file:path} patterns to random selections from YAML data
  - Supports templates, path selection, inline templates, and random numbers
- `LunaYAMLWildcardParser.process_prompt(prompt, seed) -> str`
  - Processes entire prompt template, replacing all {file:path} wildcards
  - Uses reproducible randomization with seed parameter
- `LunaYAMLWildcardParser.select_from_path(data, path, rng) -> str`
  - Selects random item from hierarchical YAML path
  - Flattens nested structures and applies weighting rules
- `LunaYAMLWildcardParser.resolve_inline_template(data, template, rng) -> str`
  - Processes inline templates with [path] substitutions
  - Recursively resolves nested wildcards
- `LunaYAMLWildcardParser.resolve_random_number(pattern, rng) -> Optional[str]`
  - Handles {1-10} integer ranges and {0.5-1.5:0.1} float ranges with resolution
- `LunaYAMLWildcardParser.resolve_txt_wildcard(item, rng) -> str`
  - Resolves __path/file__ references to legacy .txt wildcard files
  - Recursively processes wildcards within txt files
- `LunaYAMLWildcardParser.get_available_paths(filename) -> List[str]`
  - Returns all navigable paths in a YAML file for UI hints
- `LunaWildcardBuilder.build_prompt(**kwargs) -> (str, str, str)`
  - Builds complete prompts with wildcards, LoRAs, and embeddings
  - Outputs processed prompt, LoRA string, and full combined prompt
- `LunaYAMLInjector.inject_items(csv_input, target_yaml, target_path, ...) -> (str, str, bool)`
  - Parses CSV input and injects items into YAML file at specified path
  - Supports creating new categories and preview mode

## Dependencies
**Internal:**
- None (standalone nodes)

**External:**
- Required: `PyYAML`
- Optional: `folder_paths` (for LoRA/embedding lists)

## Integration Points
**Input:** YAML wildcard files, txt wildcard files, LoRA/embedding lists from ComfyUI, CSV data for injection
**Output:** Processed prompts, LoRA strings in A1111 format, YAML previews, path listings
**Side Effects:** File I/O for YAML loading/caching, random seed usage, optional YAML file modification

## Notes
- Supports hierarchical YAML with templates, path selection, and inline templates
- Compatible with legacy txt wildcards via __path/file__ syntax
- Integrates with LoRA and embedding systems for complete prompt building
- Includes utilities for exploring YAML structure and injecting new content
- Uses reproducible randomization for consistent batch generation