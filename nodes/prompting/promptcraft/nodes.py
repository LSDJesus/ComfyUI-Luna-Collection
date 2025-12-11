"""
Luna PromptCraft Nodes
ComfyUI nodes for smart wildcard resolution.
Single unified node - complexity lives in engine + JS panel.
"""

import os
import json
import random
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

_engine_instance: Optional[LunaPromptEngine] = None
_engine_path: Optional[str] = None


def get_engine(wildcards_path: Optional[str] = None) -> LunaPromptEngine:
    """Get or create the shared engine instance"""
    global _engine_instance, _engine_path
    
    # Use provided path or find default
    if wildcards_path is None:
        wildcards_path = get_default_wildcards_dir()
    
    # Recreate if path changed
    if _engine_instance is None or _engine_path != wildcards_path:
        _engine_instance = LunaPromptEngine(wildcards_path)
        _engine_path = wildcards_path
    
    return _engine_instance


def get_default_wildcards_dir() -> str:
    """Get the default wildcards directory path"""
    if HAS_FOLDER_PATHS:
        models_dir = getattr(folder_paths, 'models_dir', None)
        if models_dir:
            wildcards_path = os.path.join(models_dir, 'wildcards')
            if os.path.exists(wildcards_path):
                return wildcards_path
    
    # Fallback paths
    fallbacks = [
        "D:/AI/SD Models/wildcards",
        os.path.join(os.path.dirname(__file__), '..', '..', 'wildcards'),
    ]
    
    for path in fallbacks:
        if os.path.exists(path):
            return path
    
    return fallbacks[0]


# =============================================================================
# Luna PromptCraft (Main Node)
# =============================================================================

class LunaPromptCraft:
    """
    Smart wildcard resolution with constraints, modifiers, expanders, and LoRA linking.
    
    Template syntax:
        {category}           - Pick from category
        {category:path}      - Pick from specific path
        {category:path.sub}  - Pick from nested path
    
    Features (configured via JS Connection Manager panel):
        - Constraints: Filter items based on context (beach → swimwear)
        - Modifiers: Transform picks based on actions (sex → "pulled aside")
        - Expanders: Add scene details (beach → lighting, atmosphere)
        - LoRA Links: Auto-suggest LoRAs based on picks
    
    Outputs:
        - prompt: Final resolved prompt with all expansions
        - seed: Actual seed used (for reproducibility)
        - lora_stack: LORA_STACK compatible with Apply LoRA Stack
        - trigger_words: Combined trigger words from linked LoRAs
        - debug: JSON debug info with picks, paths, tags
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("STRING", "INT", "LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "seed", "lora_stack", "trigger_words", "debug")
    FUNCTION = "process"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "multiline": True,
                    "default": "{setting}, 1girl, {clothing}, {action}",
                    "tooltip": "Prompt template with {wildcards}. Use {category} or {category:path.to.items}"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random selection (-1 for random)"
                }),
            },
            "optional": {
                "wildcards_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to wildcards directory (leave empty for default)"
                }),
                "enable_constraints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter items based on context (beach → prefer swimwear)"
                }),
                "enable_modifiers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply action-based modifiers (sex → clothing 'pulled aside')"
                }),
                "enable_expanders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add scene details (beach → lighting, atmosphere)"
                }),
                "enable_lora_links": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-link LoRAs based on character/style picks"
                }),
                "add_trigger_words": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append LoRA trigger words to prompt"
                }),
            }
        }
    
    def process(
        self,
        template: str,
        seed: int,
        wildcards_path: str = "",
        enable_constraints: bool = True,
        enable_modifiers: bool = True,
        enable_expanders: bool = True,
        enable_lora_links: bool = True,
        add_trigger_words: bool = True,
    ) -> Tuple[str, int, List, str, str]:
        
        # Handle random seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        
        # Get engine
        path = wildcards_path if wildcards_path else None
        engine = get_engine(path)
        
        # Process template
        result = engine.process_template(
            template=template,
            seed=seed,
            enable_constraints=enable_constraints,
            enable_modifiers=enable_modifiers,
            enable_expanders=enable_expanders,
            enable_lora_links=enable_lora_links,
        )
        
        # Extract results
        prompt = result.get('prompt', template)
        lora_stack = result.get('lora_stack', [])
        trigger_words = result.get('trigger_words', [])
        
        # Format trigger words
        trigger_words_str = ", ".join(trigger_words) if trigger_words else ""
        
        # Add trigger words to prompt if enabled (and not already added by engine)
        # Engine already adds them, so this is just for the separate output
        
        # Build debug info
        debug_data = {
            "seed": seed,
            "wildcards_path": engine.wildcards_dir,
            "picks": result.get('picks', {}),
            "paths": result.get('paths', {}),
            "tags": result.get('tags', []),
            "expansions": result.get('expansions', []),
            "loras": [
                {"name": l[0], "model_weight": l[1], "clip_weight": l[2]}
                for l in lora_stack
            ],
            "embeddings": result.get('embeddings', []),
            "trigger_words": trigger_words,
            "settings": {
                "constraints": enable_constraints,
                "modifiers": enable_modifiers,
                "expanders": enable_expanders,
                "lora_links": enable_lora_links,
            }
        }
        
        debug_str = json.dumps(debug_data, indent=2)
        
        return (prompt, seed, lora_stack, trigger_words_str, debug_str)


# =============================================================================
# Luna PromptCraft Debug (Viewer Node)
# =============================================================================

class LunaPromptCraftDebug:
    """
    Debug viewer for PromptCraft output.
    
    Connect the debug output from Luna PromptCraft to visualize
    what was picked, which rules applied, and the final state.
    """
    
    CATEGORY = "Luna/PromptCraft"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted",)
    FUNCTION = "format_debug"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "debug_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Debug JSON from Luna PromptCraft"
                }),
            },
            "optional": {
                "show_paths": ("BOOLEAN", {"default": True}),
                "show_tags": ("BOOLEAN", {"default": True}),
                "show_loras": ("BOOLEAN", {"default": True}),
            }
        }
    
    def format_debug(
        self,
        debug_json: str,
        show_paths: bool = True,
        show_tags: bool = True,
        show_loras: bool = True,
    ) -> Tuple[str]:
        
        try:
            data = json.loads(debug_json)
        except json.JSONDecodeError:
            return (f"Invalid JSON:\n{debug_json}",)
        
        lines = [
            f"═══ Luna PromptCraft Debug ═══",
            f"Seed: {data.get('seed', 'unknown')}",
            f"Wildcards: {data.get('wildcards_path', 'unknown')}",
            "",
            "── Picks ──"
        ]
        
        picks = data.get('picks', {})
        paths = data.get('paths', {})
        
        for wildcard, value in picks.items():
            lines.append(f"  {{{wildcard}}} → {value}")
            if show_paths and wildcard in paths:
                lines.append(f"    └─ {paths[wildcard]}")
        
        if data.get('expansions'):
            lines.append("")
            lines.append("── Expansions ──")
            for exp in data['expansions']:
                lines.append(f"  + {exp}")
        
        if show_tags and data.get('tags'):
            lines.append("")
            lines.append(f"── Tags ──")
            lines.append(f"  {', '.join(sorted(data['tags']))}")
        
        if show_loras and data.get('loras'):
            lines.append("")
            lines.append("── LoRAs ──")
            for lora in data['loras']:
                lines.append(f"  {lora['name']} @ {lora['model_weight']}/{lora['clip_weight']}")
        
        if data.get('trigger_words'):
            lines.append("")
            lines.append("── Trigger Words ──")
            lines.append(f"  {', '.join(data['trigger_words'])}")
        
        settings = data.get('settings', {})
        lines.append("")
        lines.append("── Settings ──")
        for key, val in settings.items():
            status = "✓" if val else "✗"
            lines.append(f"  {status} {key}")
        
        return ("\n".join(lines),)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaPromptCraft": LunaPromptCraft,
    "LunaPromptCraftDebug": LunaPromptCraftDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPromptCraft": "Luna PromptCraft",
    "LunaPromptCraftDebug": "Luna PromptCraft Debug",
}
