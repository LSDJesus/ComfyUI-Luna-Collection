"""
Luna PromptCraft Nodes
ComfyUI nodes for smart wildcard resolution.
These are thin wrappers around the core engine.
"""

import os
from typing import Dict, List, Tuple, Any, Optional

try:
    import folder_paths  # type: ignore
    HAS_FOLDER_PATHS = True
except ImportError:
    folder_paths = None  # type: ignore
    HAS_FOLDER_PATHS = False

from .engine import LunaPromptEngine, create_engine


# =============================================================================
# Shared Engine Instance
# =============================================================================

_engine_instance = None

def get_engine() -> LunaPromptEngine:
    """Get or create the shared engine instance"""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = create_engine()
    
    return _engine_instance


def get_wildcards_dir() -> str:
    """Get the wildcards directory path"""
    if HAS_FOLDER_PATHS:
        models_dir = getattr(folder_paths, 'models_dir', None)
        if models_dir:
            return os.path.join(models_dir, 'wildcards')
    
    # Fallback
    return os.path.join(os.path.dirname(__file__), '..', '..', 'wildcards')


# =============================================================================
# Luna Base Prompt
# =============================================================================

class LunaBasePrompt:
    """
    Creates a base prompt template with wildcard placeholders.
    
    Use {category} or {category:path} syntax for wildcards.
    Example: "{location}, 1girl, {clothing}, {action}"
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("LUNA_TEMPLATE",)
    RETURN_NAMES = ("template",)
    FUNCTION = "create_template"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "multiline": True,
                    "default": "{location}, 1girl, {clothing}, {action}",
                    "tooltip": "Prompt template with {wildcards}"
                }),
            },
        }
    
    def create_template(self, template: str) -> Tuple[Dict]:
        """Package template for downstream nodes"""
        engine = get_engine()
        
        return ({
            "template": template,
            "combinations": engine.count_combinations(template)
        },)


# =============================================================================
# Luna Conditionals
# =============================================================================

class LunaConditionals:
    """
    Loads and configures constraint rules for context-aware resolution.
    
    When connected to Luna Assembler, wildcards will be filtered
    based on what was previously picked (e.g., beach → swimwear).
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("LUNA_CONDITIONALS",)
    RETURN_NAMES = ("conditionals",)
    FUNCTION = "load_conditionals"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_compatibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter items based on location/action context"
                }),
                "enable_conflicts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Exclude mutually exclusive items"
                }),
            },
        }
    
    def load_conditionals(
        self, 
        enable_compatibility: bool,
        enable_conflicts: bool
    ) -> Tuple[Dict]:
        return ({
            "enable_compatibility": enable_compatibility,
            "enable_conflicts": enable_conflicts,
        },)


# =============================================================================
# Luna Expanders
# =============================================================================

class LunaExpanders:
    """
    Loads scene expansion rules to add contextual details.
    
    When connected, adds relevant details based on resolved wildcards
    (e.g., beach → "palm trees, ocean waves, sandy shore").
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("LUNA_EXPANDERS",)
    RETURN_NAMES = ("expanders",)
    FUNCTION = "load_expanders"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable scene expansions"
                }),
                "detail_level": (["minimal", "normal", "detailed", "maximum"], {
                    "default": "normal",
                    "tooltip": "How much detail to add"
                }),
            },
        }
    
    def load_expanders(self, enabled: bool, detail_level: str) -> Tuple[Dict]:
        return ({
            "enabled": enabled,
            "detail_level": detail_level,
        },)


# =============================================================================
# Luna Modifiers
# =============================================================================

class LunaModifiers:
    """
    Loads action-based modifier rules.
    
    When connected, applies transformations based on actions
    (e.g., action:sex → clothing gets "pulled aside" appended).
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("LUNA_MODIFIERS",)
    RETURN_NAMES = ("modifiers",)
    FUNCTION = "load_modifiers"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable action-based modifiers"
                }),
            },
        }
    
    def load_modifiers(self, enabled: bool) -> Tuple[Dict]:
        return ({
            "enabled": enabled,
        },)


# =============================================================================
# Luna LoRA Linker
# =============================================================================

class LunaLoRALinker:
    """
    Links wildcard categories to LoRAs/embeddings.
    
    When a category is resolved, automatically adds the linked LoRA
    to the output stack with configured weight.
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("LUNA_LORA_RULES",)
    RETURN_NAMES = ("lora_rules",)
    FUNCTION = "load_lora_rules"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable automatic LoRA linking"
                }),
                "weight_mode": (["optimal", "random", "min", "max"], {
                    "default": "optimal",
                    "tooltip": "How to select LoRA weights"
                }),
            },
            "optional": {
                "weight_variance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.05,
                    "tooltip": "Random variance to add to optimal weight"
                }),
            }
        }
    
    def load_lora_rules(
        self, 
        enabled: bool, 
        weight_mode: str,
        weight_variance: float = 0.0
    ) -> Tuple[Dict]:
        return ({
            "enabled": enabled,
            "weight_mode": weight_mode,
            "weight_variance": weight_variance,
        },)


# =============================================================================
# Luna Assembler (Main Node)
# =============================================================================

class LunaAssembler:
    """
    Main assembler node - resolves wildcards and builds the final prompt.
    
    Connect optional inputs to enable constraints, expanders, modifiers,
    and LoRA linking. Without optional inputs, works as a basic wildcard resolver.
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "STRING")
    RETURN_NAMES = ("prompt", "negative", "lora_stack", "debug_info")
    FUNCTION = "assemble"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("LUNA_TEMPLATE",),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random selection (-1 for random)"
                }),
            },
            "optional": {
                "negative_template": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional negative prompt template"
                }),
                "conditionals": ("LUNA_CONDITIONALS",),
                "expanders": ("LUNA_EXPANDERS",),
                "modifiers": ("LUNA_MODIFIERS",),
                "lora_linker": ("LUNA_LORA_RULES",),
            }
        }
    
    def assemble(
        self,
        base_prompt: Dict,
        seed: int,
        negative_template: str = "",
        conditionals: Optional[Dict] = None,
        expanders: Optional[Dict] = None,
        modifiers: Optional[Dict] = None,
        lora_linker: Optional[Dict] = None
    ) -> Tuple[str, str, List, str]:
        
        engine = get_engine()
        
        # Determine what's enabled
        enable_constraints = conditionals.get("enable_compatibility", False) if conditionals else False
        enable_modifiers = modifiers.get("enabled", False) if modifiers else False
        enable_expanders = expanders.get("enabled", False) if expanders else False
        detail_level = expanders.get("detail_level", "normal") if expanders else "normal"
        
        # Process the template
        template = base_prompt.get("template", "")
        result = engine.process_template(
            template=template,
            seed=seed,
            enable_constraints=enable_constraints,
            enable_modifiers=enable_modifiers,
            enable_expanders=enable_expanders,
            detail_level=detail_level
        )
        
        # Process negative template if provided
        negative = ""
        if negative_template:
            neg_result = engine.process_template(
                template=negative_template,
                seed=seed + 1,  # Different seed for variety
                enable_constraints=False,
                enable_modifiers=False,
                enable_expanders=False
            )
            negative = neg_result.get("prompt", "")
        
        # Build LoRA stack (placeholder - will be enhanced)
        lora_stack = []
        if lora_linker and lora_linker.get("enabled", False):
            # TODO: Implement LoRA linking based on resolved paths
            pass
        
        # Format debug info
        debug_lines = [
            f"Seed: {seed}",
            f"Combinations: {base_prompt.get('combinations', 'unknown')}",
            f"Constraints: {'ON' if enable_constraints else 'OFF'}",
            f"Modifiers: {'ON' if enable_modifiers else 'OFF'}",
            f"Expanders: {'ON' if enable_expanders else 'OFF'} ({detail_level})",
            "",
            "Picks:"
        ]
        
        for wildcard, value in result.get('picks', {}).items():
            path = result.get('paths', {}).get(wildcard, '')
            debug_lines.append(f"  {{{wildcard}}} → {value}")
            debug_lines.append(f"    path: {path}")
        
        if result.get('expansions'):
            debug_lines.append("")
            debug_lines.append("Expansions:")
            for exp in result['expansions']:
                debug_lines.append(f"  + {exp}")
        
        if result.get('tags'):
            debug_lines.append("")
            debug_lines.append(f"Context tags: {', '.join(sorted(result['tags']))}")
        
        debug_info = "\n".join(debug_lines)
        
        return (result.get('prompt', ''), negative, lora_stack, debug_info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaBasePrompt": LunaBasePrompt,
    "LunaConditionals": LunaConditionals,
    "LunaExpanders": LunaExpanders,
    "LunaModifiers": LunaModifiers,
    "LunaLoRALinker": LunaLoRALinker,
    "LunaAssembler": LunaAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaBasePrompt": "Luna Base Prompt",
    "LunaConditionals": "Luna Conditionals",
    "LunaExpanders": "Luna Expanders", 
    "LunaModifiers": "Luna Modifiers",
    "LunaLoRALinker": "Luna LoRA Linker",
    "LunaAssembler": "Luna Assembler",
}
