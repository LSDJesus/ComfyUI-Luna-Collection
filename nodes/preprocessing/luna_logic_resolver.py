"""
Luna Logic Resolver - ComfyUI node for context-aware wildcard resolution
"""

import os
from pathlib import Path
import sys

# Add parent directory to path for imports
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

try:
    from utils.logic_engine import LunaLogicEngine
except ImportError:
    # Fallback for different import contexts
    try:
        from ...utils.logic_engine import LunaLogicEngine
    except ImportError:
        LunaLogicEngine = None
        print("lunaCore: LunaLogicEngine not available - Luna Logic Resolver disabled")


class LunaLogicResolver:
    """
    Context-aware wildcard resolver that prevents semantic contradictions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True,
                    "default": "A __character__ wearing __outfit__ in __location__"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "initial_context": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., scifi, female, heroic"
                }),
                "enable_payloads": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Include LoRAs",
                    "label_off": "Text Only"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "payloads", "debug_info")
    FUNCTION = "resolve"
    CATEGORY = "Luna/Text"
    
    DESCRIPTION = """
    Intelligent wildcard resolver with context tracking.
    
    Use __wildcard_name__ in your text to insert random items that
    are semantically compatible with each other. Tracks tags across
    selections to prevent contradictions (e.g., medieval + scifi).
    
    Initial context: Comma-separated tags to start with (optional)
    Enable payloads: Include bundled LoRA/embedding syntax
    """
    
    def __init__(self):
        # Find wildcards directory relative to this file
        self.wildcards_dir = Path(__file__).parent.parent.parent / "wildcards"
        self.engine = None
    
    def resolve(self, text, seed, initial_context="", enable_payloads=True):
        """
        Main resolution function
        
        Args:
            text: Template string with __wildcard__ patterns
            seed: Random seed for reproducibility
            initial_context: Comma-separated tags to start with
            enable_payloads: Whether to include LoRA/embedding payloads
        
        Returns:
            (resolved_prompt, payloads, debug_info)
        """
        
        # Initialize engine (lazy loading)
        if self.engine is None:
            try:
                self.engine = LunaLogicEngine(str(self.wildcards_dir))
            except Exception as e:
                error_msg = f"Failed to initialize wildcard engine: {e}"
                return (text, "", error_msg)
        
        # Parse initial context
        context_tags = set()
        if initial_context.strip():
            context_tags = {tag.strip().lower() for tag in initial_context.split(",")}
        
        # Resolve wildcards
        try:
            resolved, payloads = self.engine.resolve_prompt(text, seed, context_tags)
            
            # Build output
            if enable_payloads and payloads:
                final_prompt = f"{resolved} {payloads}"
            else:
                final_prompt = resolved
            
            # Debug info
            debug_lines = [
                f"Initial context: {context_tags if context_tags else 'none'}",
                f"Loaded wildcards: {', '.join(self.engine.get_wildcard_names())}",
                f"Seed: {seed}",
                f"Payloads included: {enable_payloads}",
            ]
            debug_info = "\n".join(debug_lines)
            
            return (final_prompt, payloads, debug_info)
            
        except Exception as e:
            error_msg = f"Resolution error: {e}"
            return (text, "", error_msg)


# Node registration - only if LunaLogicEngine is available
if LunaLogicEngine is not None:
    NODE_CLASS_MAPPINGS = {
        "LunaLogicResolver": LunaLogicResolver
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LunaLogicResolver": "🌙 Luna Logic Resolver"
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
