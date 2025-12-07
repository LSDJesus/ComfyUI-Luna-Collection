# utils/logic_engine.py

## Purpose
Core wildcard resolution system providing context-aware prompt generation with compatibility rules, weighted selection, and composition support. Pure deterministic logic with no AI dependencies.

## Exports
- `LogicItem`: Dataclass representing wildcard items with tags, compatibility rules, and composition
- `LunaLogicEngine`: Main engine for loading wildcards and resolving prompts

## Key Imports
- `yaml`: YAML file parsing for wildcard definitions
- `random`: Weighted random selection
- `re`: Regular expression pattern matching
- `os`, `pathlib`: File system operations
- `dataclasses`: Data structure definitions
- `typing`: Type hints

## ComfyUI Node Configuration
N/A - Utility engine, not a node.

## Input Schema
N/A - Engine class with file-based initialization.

## Key Methods
- `LunaLogicEngine.__init__(wildcards_dir)`: Initialize engine and load all YAML wildcard files
- `LunaLogicEngine._load_wildcard_file(filepath)`: Parse and validate single YAML wildcard file
- `LunaLogicEngine._check_circular_dependencies(name, items)`: Detect circular composition references
- `LunaLogicEngine.resolve_prompt(template, seed, initial_context)`: Resolve __wildcard__ patterns with context awareness
- `LogicItem.is_compatible(current_context)`: Check if item can be selected based on whitelist/blacklist rules
- `LogicItem.from_dict(data)`: Create LogicItem from YAML dictionary

## Dependencies
- `PyYAML`: YAML parsing (required)
- `pathlib`: File path handling (Python 3.4+)

## Integration Points
- Used by Luna YAML Wildcard nodes for prompt generation
- Reads wildcard files from models/wildcards/ directory
- Provides context-aware selection based on tags and compatibility rules
- Supports recursive composition for complex prompt building
- Weighted random selection with deterministic seeding

## Notes
- Implements sophisticated compatibility system with whitelist/blacklist/requires_tags
- Detects circular dependencies in wildcard compositions
- Pure deterministic logic - same seed always produces same results
- Supports payload accumulation for additional metadata
- Used by LunaYAMLWildcard, LunaYAMLWildcardBatch, and LunaWildcardBuilder nodes