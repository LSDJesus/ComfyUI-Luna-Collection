"""
Luna YAML Wildcard Node - ComfyUI node for hierarchical YAML wildcards

SYNTAX:
- {body} - Use a random premade template from body.yaml's templates section
- {body:hair} - Use a random premade template from the hair section, OR select from all hair items
- {body:hair.color} - Select random item from hair.color (all sub-items flattened)
- {body:hair.color.natural} - Select random item from hair.color.natural specifically
- {body: a woman with [hair.length] [hair.color] hair} - Inline template with path substitutions
- {1-10} - Random integer between 1 and 10
- {0.5-1.5:0.1} - Random float between 0.5 and 1.5 with 0.1 resolution
- {1-100:5} - Random integer between 1 and 100 in steps of 5
"""

import os
import yaml
import random
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None


class LunaYAMLWildcardParser:
    """
    Parser for hierarchical YAML wildcard files.
    
    Supports:
    - Premade templates: {body} selects from templates.full in body.yaml
    - Path-based selection: {body:hair.color.natural}
    - Inline templates: {body: [hair.length] [hair.color] hair}
    - Random numbers: {1-10} or {0.5-1.5:0.1}
    - Legacy txt wildcards: __path/file__ syntax in YAML values
    - Nested path selection from any depth
    """
    
    DEFAULT_TXT_WILDCARD_DIR = "D:/AI/SD Models/Wildcards"
    
    def __init__(self, yaml_dir: str, txt_wildcard_dir: str = ""):
        self.yaml_dir = yaml_dir
        self.txt_wildcard_dir = txt_wildcard_dir or self.DEFAULT_TXT_WILDCARD_DIR
        self.cache: Dict[str, dict] = {}
        self.txt_cache: Dict[str, List[str]] = {}  # Cache for .txt wildcard files
        self.rules: Optional[dict] = None
        self._load_rules()
    
    def _load_rules(self):
        """Load the wildcard_rules.yaml if it exists"""
        rules_path = os.path.join(self.yaml_dir, "wildcard_rules.yaml")
        if os.path.exists(rules_path):
            try:
                with open(rules_path, 'r', encoding='utf-8') as f:
                    self.rules = yaml.safe_load(f)
            except Exception as e:
                print(f"[LunaYAMLWildcard] Warning: Could not load rules: {e}")
                self.rules = None
    
    def load_txt_wildcard(self, wildcard_path: str) -> List[str]:
        """
        Load a .txt wildcard file and return its lines.
        
        Args:
            wildcard_path: Path like "hair/style" which maps to wildcards/hair/style.txt
        
        Returns:
            List of items from the file, or empty list if not found
        """
        if wildcard_path in self.txt_cache:
            return self.txt_cache[wildcard_path]
        
        # Convert path to file path
        # "hair/style" -> "wildcards/hair/style.txt"
        file_path = os.path.join(self.txt_wildcard_dir, f"{wildcard_path}.txt")
        
        if not os.path.exists(file_path):
            # Try without .txt extension in case it's already there
            if os.path.exists(wildcard_path):
                file_path = wildcard_path
            else:
                self.txt_cache[wildcard_path] = []
                return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                self.txt_cache[wildcard_path] = lines
                return lines
        except Exception as e:
            print(f"[LunaYAMLWildcard] Error loading txt wildcard {file_path}: {e}")
            self.txt_cache[wildcard_path] = []
            return []
    
    def resolve_txt_wildcard(self, item: str, rng: random.Random) -> str:
        """
        Check if an item is a __path/file__ reference and resolve it.
        
        Args:
            item: The item string, possibly in __path/file__ format
            rng: Random generator
        
        Returns:
            Resolved item or original if not a txt wildcard reference
        """
        # Check for __path/file__ pattern
        match = re.match(r'^__([^_]+(?:/[^_]+)*)__$', item)
        if not match:
            return item
        
        wildcard_path = match.group(1)
        items = self.load_txt_wildcard(wildcard_path)
        
        if items:
            selected = rng.choice(items)
            # Recursively resolve in case the txt file contains more __wildcards__
            return self.resolve_txt_wildcard(selected, rng)
        
        return item  # Return original if file not found
    
    def load_yaml(self, filename: str) -> Optional[dict]:
        """Load and cache a YAML file"""
        if not filename.endswith('.yaml'):
            filename = f"{filename}.yaml"
        
        filepath = os.path.join(self.yaml_dir, filename)
        if not os.path.exists(filepath):
            return None
        
        if filename not in self.cache:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.cache[filename] = yaml.safe_load(f)
            except Exception as e:
                print(f"[LunaYAMLWildcard] Error loading {filename}: {e}")
                return None
        
        return self.cache[filename]
    
    def get_by_path(self, data: dict, path: str) -> Any:
        """Navigate nested dict by dot-separated path"""
        if not path:
            return data
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return None
            else:
                return None
        
        return current
    
    def flatten_to_list(self, data: Any) -> List[str]:
        """Recursively flatten any nested structure to a list of strings"""
        if data is None:
            return []
        
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, (dict, list)):
                    result.extend(self.flatten_to_list(item))
            return result
        
        if isinstance(data, dict):
            result = []
            for key, value in data.items():
                # Skip special keys
                if key in ['templates', 'usage_notes', 'category', 'description', '_suffix']:
                    continue
                result.extend(self.flatten_to_list(value))
            return result
        
        if isinstance(data, str):
            return [data]
        
        return []
    
    def get_templates(self, data: dict, section: str = "") -> List[str]:
        """Get templates from a YAML file, optionally from a specific section"""
        templates = data.get('templates', {})
        
        if not section:
            # Get all templates flattened
            all_templates = []
            for key, value in templates.items():
                if isinstance(value, list):
                    all_templates.extend(value)
                elif isinstance(value, str):
                    all_templates.append(value)
            return all_templates
        
        # Get templates for specific section
        if section in templates:
            section_templates = templates[section]
            if isinstance(section_templates, list):
                return section_templates
            elif isinstance(section_templates, str):
                return [section_templates]
        
        return []
    
    def get_weight(self, item: str) -> float:
        """Get weight adjustment for an item from rules"""
        if not self.rules or 'weights' not in self.rules:
            return 1.0
        
        weights = self.rules['weights']
        
        for category, config in weights.items():
            if isinstance(config, dict) and 'items' in config and item in config['items']:
                return config.get('weight', 1.0)
        
        return 1.0
    
    def select_from_path(self, data: dict, path: str, rng: random.Random) -> str:
        """Select a random item from a path in the data"""
        target = self.get_by_path(data, path)
        if target is None:
            return f"[{path}]"  # Return original if not found
        
        items = self.flatten_to_list(target)
        if not items:
            return f"[{path}]"
        
        # Apply weights
        if self.rules and 'weights' in self.rules:
            weighted_items = []
            for item in items:
                weight = self.get_weight(item)
                count = max(1, int(weight * 10))
                weighted_items.extend([item] * count)
            items = weighted_items
        
        selected = rng.choice(items)
        
        # Resolve __txt/wildcard__ references if present
        return self.resolve_txt_wildcard(selected, rng)
    
    def resolve_random_number(self, pattern: str, rng: random.Random) -> Optional[str]:
        """
        Resolve a random number pattern like {1-10} or {0.5-1.5:0.1}
        
        Syntax:
        - {x-y} - Random integer between x and y (inclusive)
        - {x-y:z} - Random number between x and y with resolution z
        
        Returns None if pattern doesn't match number syntax
        """
        # Pattern: number-number or number-number:resolution
        # Supports integers and floats
        match = re.match(r'^(-?\d+\.?\d*)-(-?\d+\.?\d*)(?::(\d+\.?\d*))?$', pattern.strip())
        if not match:
            return None
        
        low_str, high_str, resolution_str = match.groups()
        
        try:
            # Determine if we're dealing with floats
            is_float = '.' in low_str or '.' in high_str or (resolution_str and '.' in resolution_str)
            
            low = float(low_str)
            high = float(high_str)
            
            if low > high:
                low, high = high, low  # Swap if reversed
            
            if resolution_str:
                resolution = float(resolution_str)
                if resolution <= 0:
                    resolution = 1
                
                # Generate steps and pick one
                steps = int((high - low) / resolution) + 1
                step_index = rng.randint(0, steps - 1)
                value = low + (step_index * resolution)
                
                # Clamp to high
                value = min(value, high)
                
                # Format based on resolution decimals
                if '.' in resolution_str:
                    decimals = len(resolution_str.split('.')[1])
                    return f"{value:.{decimals}f}"
                else:
                    return str(int(round(value)))
            else:
                # No resolution specified
                if is_float:
                    # Default to 2 decimal places for floats
                    value = rng.uniform(low, high)
                    return f"{value:.2f}"
                else:
                    # Integer range
                    return str(rng.randint(int(low), int(high)))
        
        except (ValueError, ZeroDivisionError):
            return None
    
    def resolve_inline_template(self, data: dict, template: str, rng: random.Random) -> str:
        """
        Resolve an inline template like "a woman with [hair.length] [hair.color] hair"
        
        Replaces all [path] references with random selections from the YAML data
        """
        # Find all [path] patterns
        pattern = r'\[([^\]]+)\]'
        
        def replace_match(match):
            path = match.group(1)
            return self.select_from_path(data, path, rng)
        
        return re.sub(pattern, replace_match, template)
    
    def resolve_wildcard(self, wildcard: str, rng: Optional[random.Random] = None) -> str:
        """
        Resolve a wildcard pattern.
        
        Syntax:
        - {body} - random premade template from body.yaml
        - {body:hair} - random from hair section (templates first, then items)
        - {body:hair.color.natural} - random item from specific path
        - {body: a woman with [hair.length] hair} - inline template with substitutions
        - {1-10} - random integer between 1 and 10
        - {0.5-1.5:0.1} - random float with resolution
        """
        if rng is None:
            rng = random.Random()
        
        # Strip braces
        wildcard = wildcard.strip('{}')
        
        # Check if it's a random number pattern first
        number_result = self.resolve_random_number(wildcard, rng)
        if number_result is not None:
            return number_result
        
        # Check if it's an inline template (contains [ ])
        if '[' in wildcard and ']' in wildcard:
            # Parse filename: template
            if ':' in wildcard:
                filename, template = wildcard.split(':', 1)
                filename = filename.strip()
                template = template.strip()
            else:
                return wildcard  # Invalid format
            
            data = self.load_yaml(filename)
            if data is None:
                return template  # Return template as-is if file not found
            
            return self.resolve_inline_template(data, template, rng)
        
        # Parse filename:path
        if ':' in wildcard:
            filename, path = wildcard.split(':', 1)
            filename = filename.strip()
            path = path.strip()
        else:
            filename = wildcard.strip()
            path = ""
        
        data = self.load_yaml(filename)
        if data is None:
            return ""
        
        # If no path, use templates from the file
        if not path:
            templates = self.get_templates(data)
            if templates:
                template = rng.choice(templates)
                # Resolve any [path] references in the template
                return self.resolve_inline_template(data, template, rng)
            # Fallback: flatten entire file
            items = self.flatten_to_list(data)
            return rng.choice(items) if items else ""
        
        # Check if path points to a section with templates
        # e.g., {body:hair} might have templates under templates.hair
        section_templates = self.get_templates(data, path)
        if section_templates:
            template = rng.choice(section_templates)
            return self.resolve_inline_template(data, template, rng)
        
        # Otherwise, select random item from the path
        return self.select_from_path(data, path, rng)
    
    def process_prompt(self, prompt: str, seed: int = 0) -> str:
        """
        Process a prompt template, replacing all {file:path} wildcards
        
        Args:
            prompt: Prompt template with wildcards
            seed: Random seed for reproducibility (0 = random)
        
        Returns:
            Processed prompt with wildcards replaced
        """
        rng = random.Random()
        if seed != 0:
            rng.seed(seed)
        
        # Find all {file:path} or {file: template} patterns
        # This regex handles nested brackets in inline templates
        pattern = r'\{([^{}]+)\}'
        
        def replace_match(match):
            wildcard = match.group(1)
            return self.resolve_wildcard(wildcard, rng)
        
        result = re.sub(pattern, replace_match, prompt)
        
        # Clean up spacing
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r',\s*,', ',', result)
        result = result.strip()
        
        return result
    
    def get_available_paths(self, filename: str) -> List[str]:
        """Get all available paths in a YAML file for UI hints"""
        data = self.load_yaml(filename)
        if data is None:
            return []
        
        paths = []
        
        def collect_paths(d: dict, prefix: str = ""):
            for key, value in d.items():
                if key in ['templates', 'usage_notes', 'category', 'description', '_suffix']:
                    continue
                
                current_path = f"{prefix}.{key}" if prefix else key
                paths.append(current_path)
                
                if isinstance(value, dict):
                    collect_paths(value, current_path)
        
        collect_paths(data)
        return paths


class LunaYAMLWildcard:
    """
    ComfyUI node for processing YAML-based wildcards.
    
    SYNTAX:
    - {body} - Use random premade template from body.yaml
    - {body:hair} - Use hair templates or random hair item
    - {body:hair.color.natural} - Random from specific path
    - {body: a woman with [hair.length] [hair.color] hair} - Inline template
    """
    
    CATEGORY = "Luna/Wildcards"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process_wildcards"
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    DEFAULT_TXT_WILDCARD_DIR = "D:/AI/SD Models/Wildcards"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": "{body: a beautiful woman with [hair.length] [hair.color.natural] hair, [eyes.color.natural] eyes, [skin.tone] skin}",
                    "tooltip": "Prompt with {file:path} wildcards or {file: inline [path] template}"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed (0 = random each time)"
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                    "tooltip": "Directory containing YAML wildcard files"
                }),
                "txt_wildcard_directory": ("STRING", {
                    "default": cls.DEFAULT_TXT_WILDCARD_DIR,
                    "tooltip": "Directory containing legacy .txt wildcard files (for __path/file__ syntax)"
                }),
            }
        }
    
    def process_wildcards(self, prompt_template: str, seed: int = 0, 
                          yaml_directory: str = "", txt_wildcard_directory: str = "") -> Tuple[str]:
        """Process the prompt template and replace wildcards"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        if not txt_wildcard_directory:
            txt_wildcard_directory = self.DEFAULT_TXT_WILDCARD_DIR
        
        if not os.path.exists(yaml_directory):
            print(f"[LunaYAMLWildcard] Warning: Directory not found: {yaml_directory}")
            return (prompt_template,)
        
        parser = LunaYAMLWildcardParser(yaml_directory, txt_wildcard_directory)
        processed = parser.process_prompt(prompt_template, seed)
        
        return (processed,)


class LunaYAMLWildcardBatch:
    """
    ComfyUI node for generating multiple prompts from YAML wildcards.
    """
    
    CATEGORY = "Luna/Wildcards"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_batch"
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    DEFAULT_TXT_WILDCARD_DIR = "D:/AI/SD Models/Wildcards"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": "{body: a [body_type.types] woman with [hair.length] [hair.color.natural] hair}",
                    "tooltip": "Prompt template with wildcards"
                }),
                "count": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of prompt variations to generate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Base seed (0 = random). Each variation uses seed+index"
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                }),
                "txt_wildcard_directory": ("STRING", {
                    "default": cls.DEFAULT_TXT_WILDCARD_DIR,
                    "tooltip": "Directory containing legacy .txt wildcard files"
                }),
                "unique_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove duplicate prompts"
                }),
            }
        }
    
    def generate_batch(self, prompt_template: str, count: int = 10, seed: int = 0, 
                       yaml_directory: str = "", txt_wildcard_directory: str = "",
                       unique_only: bool = True) -> Tuple[List[str]]:
        """Generate multiple prompt variations"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        if not txt_wildcard_directory:
            txt_wildcard_directory = self.DEFAULT_TXT_WILDCARD_DIR
        
        if not os.path.exists(yaml_directory):
            return ([prompt_template],)
        
        parser = LunaYAMLWildcardParser(yaml_directory, txt_wildcard_directory)
        
        prompts = []
        seen = set()
        
        for i in range(count):
            current_seed = (seed + i) if seed != 0 else 0
            processed = parser.process_prompt(prompt_template, current_seed)
            
            if unique_only:
                if processed not in seen:
                    prompts.append(processed)
                    seen.add(processed)
            else:
                prompts.append(processed)
        
        print(f"[LunaYAMLWildcardBatch] Generated {len(prompts)} prompts")
        
        return (prompts,)


class LunaYAMLWildcardExplorer:
    """
    ComfyUI node to explore available wildcards in YAML files.
    """
    
    CATEGORY = "Luna/Wildcards"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("available_paths",)
    FUNCTION = "explore"
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yaml_file": (["body", "clothing", "pose", "setting", "lighting", "expression", "composition", "action"], {
                    "default": "body",
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                }),
            }
        }
    
    def explore(self, yaml_file: str, yaml_directory: str = "") -> Tuple[str]:
        """List all available paths in a YAML file"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        
        if not os.path.exists(yaml_directory):
            return (f"Directory not found: {yaml_directory}",)
        
        parser = LunaYAMLWildcardParser(yaml_directory)
        paths = parser.get_available_paths(yaml_file)
        
        if not paths:
            return (f"No paths found in {yaml_file}.yaml",)
        
        # Format output
        lines = [
            f"# Available paths in {yaml_file}.yaml",
            "",
            "# Direct path selection:",
        ]
        for path in sorted(paths):
            lines.append(f"  {{{yaml_file}:{path}}}")
        
        lines.extend([
            "",
            "# Inline template example:",
            f"  {{{yaml_file}: a woman with [{paths[0]}] ...}}",
        ])
        
        return ("\n".join(lines),)


class LunaWildcardBuilder:
    """
    Interactive wildcard prompt builder with LoRA and embedding browser.
    
    Output is compatible with ImpactPack's WildcardEncode for processing
    <lora:name:weight> and embedding syntax.
    """
    
    CATEGORY = "Luna/Wildcards"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "loras_string", "full_prompt")
    FUNCTION = "build_prompt"
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    DEFAULT_TXT_WILDCARD_DIR = "D:/AI/SD Models/Wildcards"
    
    @classmethod
    def get_lora_list(cls) -> List[str]:
        """Get list of available LoRAs"""
        if folder_paths is None:
            return ["none"]
        try:
            loras = folder_paths.get_filename_list("loras")
            return ["none"] + sorted(loras) if loras else ["none"]
        except:
            return ["none"]
    
    @classmethod
    def get_embedding_list(cls) -> List[str]:
        """Get list of available embeddings"""
        if folder_paths is None:
            return ["none"]
        try:
            embeddings = folder_paths.get_filename_list("embeddings")
            return ["none"] + sorted(embeddings) if embeddings else ["none"]
        except:
            return ["none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = cls.get_lora_list()
        embeddings = cls.get_embedding_list()
        
        return {
            "required": {
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": "{composition:shot_type.distance}, {body: a beautiful [body_type.types] woman with [hair.length] [hair.color.natural] hair, [eyes.color.natural] eyes}",
                    "tooltip": "Main prompt template with wildcards"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                }),
                "txt_wildcard_directory": ("STRING", {
                    "default": cls.DEFAULT_TXT_WILDCARD_DIR,
                    "tooltip": "Directory containing legacy .txt wildcard files"
                }),
                # External LoRA string input (from LunaLoRARandomizer or other sources)
                "lora_string_input": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "LoRA string from Luna LoRA Randomizer or other source"
                }),
                # LoRA slots
                "lora_1": (loras, {"default": "none"}),
                "lora_1_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_2": (loras, {"default": "none"}),
                "lora_2_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3": (loras, {"default": "none"}),
                "lora_3_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_4": (loras, {"default": "none"}),
                "lora_4_weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Embedding slots
                "embedding_1": (embeddings, {"default": "none"}),
                "embedding_2": (embeddings, {"default": "none"}),
                "embedding_3": (embeddings, {"default": "none"}),
                # Additional prompt parts
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to add before the processed prompt"
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to add after the processed prompt"
                }),
            }
        }
    
    def build_prompt(self, prompt_template: str, seed: int = 0, yaml_directory: str = "",
                     txt_wildcard_directory: str = "",
                     lora_string_input: str = "",
                     lora_1: str = "none", lora_1_weight: float = 1.0,
                     lora_2: str = "none", lora_2_weight: float = 1.0,
                     lora_3: str = "none", lora_3_weight: float = 1.0,
                     lora_4: str = "none", lora_4_weight: float = 1.0,
                     embedding_1: str = "none", embedding_2: str = "none", embedding_3: str = "none",
                     prefix: str = "", suffix: str = "") -> Tuple[str, str, str]:
        """Build a complete prompt with wildcards, LoRAs, and embeddings"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        if not txt_wildcard_directory:
            txt_wildcard_directory = self.DEFAULT_TXT_WILDCARD_DIR
        
        # Process wildcards
        if os.path.exists(yaml_directory):
            parser = LunaYAMLWildcardParser(yaml_directory, txt_wildcard_directory)
            processed_prompt = parser.process_prompt(prompt_template, seed)
        else:
            processed_prompt = prompt_template
        
        # Build LoRA string in A1111 format from dropdown selections
        lora_parts = []
        for lora, weight in [(lora_1, lora_1_weight), (lora_2, lora_2_weight), 
                              (lora_3, lora_3_weight), (lora_4, lora_4_weight)]:
            if lora and lora != "none":
                # Remove .safetensors extension if present
                lora_name = lora.replace('.safetensors', '').replace('.pt', '')
                lora_parts.append(f"<lora:{lora_name}:{weight}>")
        
        dropdown_loras = " ".join(lora_parts)
        
        # Combine with input lora string (from LunaLoRARandomizer)
        all_lora_parts = []
        if lora_string_input and lora_string_input.strip():
            all_lora_parts.append(lora_string_input.strip())
        if dropdown_loras:
            all_lora_parts.append(dropdown_loras)
        
        loras_string = " ".join(all_lora_parts)
        
        # Build embedding list
        embedding_parts = []
        for emb in [embedding_1, embedding_2, embedding_3]:
            if emb and emb != "none":
                # Remove extension and format as embedding
                emb_name = emb.replace('.safetensors', '').replace('.pt', '').replace('.bin', '')
                embedding_parts.append(f"embedding:{emb_name}")
        
        # Combine everything
        full_parts = []
        
        if prefix.strip():
            full_parts.append(prefix.strip())
        
        if embedding_parts:
            full_parts.extend(embedding_parts)
        
        full_parts.append(processed_prompt)
        
        if loras_string:
            full_parts.append(loras_string)
        
        if suffix.strip():
            full_parts.append(suffix.strip())
        
        full_prompt = ", ".join(full_parts)
        
        # Clean up
        full_prompt = re.sub(r'\s+', ' ', full_prompt)
        full_prompt = re.sub(r',\s*,', ',', full_prompt)
        full_prompt = full_prompt.strip(' ,')
        
        return (processed_prompt, loras_string, full_prompt)


class LunaLoRARandomizer:
    """
    Randomly select LoRAs from a pool with randomized weights.
    
    Outputs A1111-format LoRA strings for use with ImpactPack's WildcardEncode.
    """
    
    CATEGORY = "Luna/Wildcards"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_string",)
    FUNCTION = "randomize_loras"
    
    @classmethod
    def get_lora_list(cls) -> List[str]:
        """Get list of available LoRAs"""
        if folder_paths is None:
            return ["none"]
        try:
            loras = folder_paths.get_filename_list("loras")
            return ["none"] + sorted(loras) if loras else ["none"]
        except:
            return ["none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = cls.get_lora_list()
        
        return {
            "required": {
                "lora_count": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Number of LoRAs to randomly select"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                }),
                "weight_min": ("FLOAT", {
                    "default": 0.5,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "weight_max": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "weight_step": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                # Pool of LoRAs to choose from
                "pool_1": (loras, {"default": "none"}),
                "pool_2": (loras, {"default": "none"}),
                "pool_3": (loras, {"default": "none"}),
                "pool_4": (loras, {"default": "none"}),
                "pool_5": (loras, {"default": "none"}),
                "pool_6": (loras, {"default": "none"}),
                "pool_7": (loras, {"default": "none"}),
                "pool_8": (loras, {"default": "none"}),
                "pool_9": (loras, {"default": "none"}),
                "pool_10": (loras, {"default": "none"}),
            }
        }
    
    def randomize_loras(self, lora_count: int, seed: int, 
                        weight_min: float, weight_max: float, weight_step: float,
                        pool_1: str = "none", pool_2: str = "none", pool_3: str = "none",
                        pool_4: str = "none", pool_5: str = "none", pool_6: str = "none",
                        pool_7: str = "none", pool_8: str = "none", pool_9: str = "none",
                        pool_10: str = "none") -> Tuple[str]:
        """Randomly select LoRAs and assign weights"""
        
        # Build pool
        pool = [l for l in [pool_1, pool_2, pool_3, pool_4, pool_5, 
                           pool_6, pool_7, pool_8, pool_9, pool_10] 
                if l and l != "none"]
        
        if not pool or lora_count == 0:
            return ("",)
        
        rng = random.Random()
        if seed != 0:
            rng.seed(seed)
        
        # Ensure we don't try to select more than available
        select_count = min(lora_count, len(pool))
        
        # Random selection without replacement
        selected = rng.sample(pool, select_count)
        
        # Generate random weights
        lora_parts = []
        weight_steps = int((weight_max - weight_min) / weight_step) + 1
        
        for lora in selected:
            step_idx = rng.randint(0, weight_steps - 1)
            weight = weight_min + (step_idx * weight_step)
            weight = min(weight, weight_max)
            
            lora_name = lora.replace('.safetensors', '').replace('.pt', '')
            lora_parts.append(f"<lora:{lora_name}:{weight:.2f}>")
        
        return (" ".join(lora_parts),)


class LunaYAMLInjector:
    """
    Utility node to inject CSV items into YAML wildcard files.
    
    Takes a comma-separated string (e.g., from an AI node), lets you edit it,
    select the target YAML file and path, then injects the items.
    """
    
    CATEGORY = "Luna/Wildcards/Utils"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("yaml_preview", "status", "success")
    FUNCTION = "inject_items"
    OUTPUT_NODE = True
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    
    @classmethod
    def get_yaml_files(cls) -> List[str]:
        """Get list of YAML files in the wildcard directory"""
        yaml_dir = cls.DEFAULT_YAML_DIR
        if not os.path.exists(yaml_dir):
            return ["none"]
        
        files = []
        for f in os.listdir(yaml_dir):
            if f.endswith('.yaml') and not f.startswith('_'):
                files.append(f.replace('.yaml', ''))
        
        return sorted(files) if files else ["none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        yaml_files = cls.get_yaml_files()
        
        return {
            "required": {
                "csv_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Comma-separated list of items to add (from AI or manual input)"
                }),
                "target_yaml": (yaml_files, {
                    "default": yaml_files[0] if yaml_files else "none",
                    "tooltip": "Target YAML file to inject into"
                }),
                "target_path": ("STRING", {
                    "default": "",
                    "tooltip": "Dot-separated path (e.g., 'hair.style.braids'). Leave empty for root level."
                }),
                "new_category": ("STRING", {
                    "default": "",
                    "tooltip": "Create a new sub-category at the target path. Leave empty to add items directly."
                }),
                "format": (["list", "inline"], {
                    "default": "list",
                    "tooltip": "Output format: 'list' for dashed items, 'inline' for bracketed array"
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                }),
                "preview_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, only preview changes without saving"
                }),
            }
        }
    
    def parse_csv(self, csv_string: str) -> List[str]:
        """Parse CSV string into list of items"""
        # Handle various separators
        items = []
        # Split by comma, newline, or semicolon
        raw_items = re.split(r'[,;\n]+', csv_string)
        
        for item in raw_items:
            cleaned = item.strip()
            # Remove quotes if present
            cleaned = cleaned.strip('"\'')
            # Replace spaces with underscores for SD compatibility
            cleaned = cleaned.replace(' ', '_')
            # Skip empty items
            if cleaned:
                items.append(cleaned)
        
        return items
    
    def get_nested_dict(self, data: dict, path: str) -> Tuple[dict, str]:
        """
        Navigate to the parent of the target path and return (parent_dict, final_key).
        Creates intermediate dicts if they don't exist.
        """
        if not path:
            return data, ""
        
        parts = path.split('.')
        current = data
        
        # Navigate to parent, creating dicts as needed
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert existing value to dict
                current[part] = {}
            current = current[part]
        
        return current, parts[-1]
    
    def inject_items(self, csv_input: str, target_yaml: str, target_path: str,
                     new_category: str, format: str, yaml_directory: str = "",
                     preview_only: bool = True) -> Tuple[str, str, bool]:
        """Inject CSV items into YAML file"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        
        if target_yaml == "none":
            return ("", "Error: No YAML file selected", False)
        
        # Parse the CSV input
        items = self.parse_csv(csv_input)
        if not items:
            return ("", "Error: No valid items found in input", False)
        
        # Load the target YAML
        yaml_path = os.path.join(yaml_directory, f"{target_yaml}.yaml")
        
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                return ("", f"Error loading YAML: {e}", False)
        else:
            data = {}
        
        # Navigate to target location
        parent_dict, final_key = self.get_nested_dict(data, target_path)
        
        # Determine where to inject
        if new_category:
            # Create new category
            if final_key:
                if final_key not in parent_dict:
                    parent_dict[final_key] = {}
                elif not isinstance(parent_dict[final_key], dict):
                    parent_dict[final_key] = {}
                target_container = parent_dict[final_key]
            else:
                target_container = parent_dict
            
            # Add new category with items
            if format == "inline":
                target_container[new_category] = items
            else:
                target_container[new_category] = items
        else:
            # Add items directly to the path
            if final_key:
                if final_key not in parent_dict:
                    parent_dict[final_key] = []
                
                existing = parent_dict[final_key]
                if isinstance(existing, list):
                    # Append to existing list, avoiding duplicates
                    for item in items:
                        if item not in existing:
                            existing.append(item)
                elif isinstance(existing, dict):
                    # Can't add list items to a dict directly
                    return ("", f"Error: Target path '{target_path}' is a category, not a list. Use 'new_category' to create a sub-category.", False)
                else:
                    # Replace with list
                    parent_dict[final_key] = items
            else:
                # Root level - need a category name
                return ("", "Error: Cannot add items to root level. Specify a target_path or new_category.", False)
        
        # Generate YAML preview
        try:
            yaml_preview = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            return ("", f"Error generating YAML: {e}", False)
        
        # Save if not preview only
        if not preview_only:
            try:
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                status = f"Success: Added {len(items)} items to {target_yaml}.yaml at '{target_path or 'root'}/{new_category or final_key}'"
            except Exception as e:
                return (yaml_preview, f"Error saving YAML: {e}", False)
        else:
            status = f"Preview: Would add {len(items)} items to {target_yaml}.yaml at '{target_path or 'root'}/{new_category or final_key}'"
        
        return (yaml_preview, status, True)


class LunaYAMLPathExplorer:
    """
    Utility node to explore and list available paths in a YAML file.
    Helps users find the right path for injection.
    """
    
    CATEGORY = "Luna/Wildcards/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "explore_paths"
    
    DEFAULT_YAML_DIR = "D:/AI/SD Models/wildcards_atomic"
    
    @classmethod
    def get_yaml_files(cls) -> List[str]:
        """Get list of YAML files"""
        yaml_dir = cls.DEFAULT_YAML_DIR
        if not os.path.exists(yaml_dir):
            return ["none"]
        
        files = []
        for f in os.listdir(yaml_dir):
            if f.endswith('.yaml') and not f.startswith('_'):
                files.append(f.replace('.yaml', ''))
        
        return sorted(files) if files else ["none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        yaml_files = cls.get_yaml_files()
        
        return {
            "required": {
                "yaml_file": (yaml_files, {
                    "default": yaml_files[0] if yaml_files else "none",
                }),
                "max_depth": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Maximum depth to explore"
                }),
                "show_item_counts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show number of items at each path"
                }),
            },
            "optional": {
                "yaml_directory": ("STRING", {
                    "default": cls.DEFAULT_YAML_DIR,
                }),
            }
        }
    
    def count_items(self, data: Any) -> int:
        """Recursively count leaf items"""
        if data is None:
            return 0
        if isinstance(data, str):
            return 1
        if isinstance(data, list):
            count = 0
            for item in data:
                if isinstance(item, str):
                    count += 1
                else:
                    count += self.count_items(item)
            return count
        if isinstance(data, dict):
            count = 0
            for key, value in data.items():
                if key not in ['templates', 'usage_notes', 'category', 'description', '_suffix']:
                    count += self.count_items(value)
            return count
        return 0
    
    def explore_paths(self, yaml_file: str, max_depth: int, show_item_counts: bool,
                      yaml_directory: str = "") -> Tuple[str]:
        """List all available paths in a YAML file"""
        
        if not yaml_directory:
            yaml_directory = self.DEFAULT_YAML_DIR
        
        if yaml_file == "none":
            return ("No YAML file selected",)
        
        yaml_path = os.path.join(yaml_directory, f"{yaml_file}.yaml")
        
        if not os.path.exists(yaml_path):
            return (f"File not found: {yaml_path}",)
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            return (f"Error loading YAML: {e}",)
        
        paths = []
        
        def collect_paths(d: dict, prefix: str = "", depth: int = 0):
            if depth >= max_depth:
                return
            
            for key, value in d.items():
                if key in ['templates', 'usage_notes', 'category', 'description']:
                    continue
                
                current_path = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # It's a category
                    if show_item_counts:
                        count = self.count_items(value)
                        paths.append(f"{current_path}/ ({count} items)")
                    else:
                        paths.append(f"{current_path}/")
                    collect_paths(value, current_path, depth + 1)
                elif isinstance(value, list):
                    # It's a list
                    if show_item_counts:
                        paths.append(f"{current_path} [{len(value)} items]")
                    else:
                        paths.append(current_path)
                else:
                    # It's a single value
                    paths.append(f"{current_path} = {value}")
        
        collect_paths(data)
        
        output = [
            f"# Paths in {yaml_file}.yaml",
            f"# Depth limit: {max_depth}",
            "",
        ]
        output.extend(sorted(paths))
        
        return ("\n".join(output),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaYAMLWildcard": LunaYAMLWildcard,
    "LunaYAMLWildcardBatch": LunaYAMLWildcardBatch,
    "LunaYAMLWildcardExplorer": LunaYAMLWildcardExplorer,
    "LunaWildcardBuilder": LunaWildcardBuilder,
    "LunaLoRARandomizer": LunaLoRARandomizer,
    "LunaYAMLInjector": LunaYAMLInjector,
    "LunaYAMLPathExplorer": LunaYAMLPathExplorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaYAMLWildcard": "Luna YAML Wildcard",
    "LunaYAMLWildcardBatch": "Luna YAML Wildcard Batch",
    "LunaYAMLWildcardExplorer": "Luna YAML Wildcard Explorer",
    "LunaWildcardBuilder": "Luna Wildcard Builder",
    "LunaLoRARandomizer": "Luna LoRA Randomizer",
    "LunaYAMLInjector": "Luna YAML Injector",
    "LunaYAMLPathExplorer": "Luna YAML Path Explorer",
}
