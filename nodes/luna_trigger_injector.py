"""
Luna LoRA Trigger Injector
Extracts trigger words from LoRAs (via CivitAI metadata or embedded modelspec)
and optionally injects them into prompts.
"""

import os
import re
import json
import struct
from typing import Dict, List, Optional, Tuple, Any

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


class LunaLoRATriggerInjector:
    """
    Extract trigger words from LoRAs and optionally inject into prompts.
    
    Supports:
    - LORA_STACK input (from stackers, loaders, etc.)
    - Inline <lora:name:weight> parsing from prompt
    - CivitAI cached metadata lookup
    - Embedded modelspec.trigger_phrase in safetensors headers
    - Configurable max triggers per LoRA
    - Multiple injection modes
    """
    
    CATEGORY = "Luna/LoRA"
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "STRING")
    RETURN_NAMES = ("prompt_with_triggers", "triggers", "lora_stack", "prompt_passthrough")
    FUNCTION = "process"
    
    # Regex for inline LoRA tags
    LORA_PATTERN = re.compile(r'<lora:([^:>]+):([^:>]+)(?::([^>]+))?>')
    
    # Cache for trigger lookups
    _trigger_cache: Dict[str, List[str]] = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "injection_mode": (["prepend", "append", "none"], {
                    "default": "prepend",
                    "tooltip": "Where to inject triggers: prepend (before prompt), append (after prompt), none (output separately only)"
                }),
                "max_triggers_per_lora": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Maximum number of trigger words/phrases to use per LoRA (some have dozens)"
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Separator between trigger words"
                }),
                "deduplicate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove duplicate triggers across all LoRAs"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "Input prompt (also scanned for inline <lora:> tags)"
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack from stacker/loader nodes"
                }),
            }
        }
    
    def _extract_inline_loras(self, prompt: str) -> Tuple[str, List[Tuple[str, float, float]]]:
        """
        Extract <lora:name:weight> tags from prompt.
        Returns (cleaned_prompt, list of (name, model_weight, clip_weight))
        """
        loras = []
        
        def extract_lora(match):
            name = match.group(1)
            weight1 = float(match.group(2))
            weight2 = float(match.group(3)) if match.group(3) else weight1
            loras.append((name, weight1, weight2))
            return ""  # Remove tag from prompt
        
        cleaned = self.LORA_PATTERN.sub(extract_lora, prompt)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned, loras
    
    def _resolve_lora_path(self, lora_name: str) -> Optional[str]:
        """Resolve a LoRA name to its full file path"""
        if not HAS_FOLDER_PATHS:
            return None
        
        try:
            # If already a path with extension
            if lora_name.endswith(('.safetensors', '.pt', '.ckpt')):
                full_path = folder_paths.get_full_path("loras", lora_name)
                if full_path and os.path.exists(full_path):
                    return full_path
            
            # Try adding extensions
            for ext in ['.safetensors', '.pt', '.ckpt']:
                test_name = lora_name + ext
                full_path = folder_paths.get_full_path("loras", test_name)
                if full_path and os.path.exists(full_path):
                    return full_path
            
            # Search in lora list
            lora_list = folder_paths.get_filename_list("loras")
            lora_name_lower = lora_name.lower()
            
            for lora_file in lora_list:
                # Exact match (without extension)
                basename = os.path.splitext(os.path.basename(lora_file))[0]
                if basename.lower() == lora_name_lower:
                    return folder_paths.get_full_path("loras", lora_file)
            
            # Partial match
            for lora_file in lora_list:
                if lora_name_lower in lora_file.lower():
                    return folder_paths.get_full_path("loras", lora_file)
                    
        except Exception as e:
            print(f"[LunaTriggerInjector] Error resolving LoRA path: {e}")
        
        return None
    
    def _read_safetensors_metadata(self, filepath: str) -> Dict[str, Any]:
        """Read metadata from safetensors file header"""
        try:
            with open(filepath, 'rb') as f:
                # Read header length (first 8 bytes, little-endian uint64)
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    return {}
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # Read header JSON
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Metadata is in __metadata__ key
                return header.get('__metadata__', {})
        except Exception as e:
            print(f"[LunaTriggerInjector] Error reading safetensors metadata: {e}")
            return {}
    
    def _get_triggers_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract trigger words from safetensors metadata"""
        triggers = []
        
        # Check modelspec format (SwarmUI compatible)
        if 'modelspec.trigger_phrase' in metadata:
            phrase = metadata['modelspec.trigger_phrase']
            if phrase:
                # Split by comma if multiple
                triggers.extend([t.strip() for t in phrase.split(',') if t.strip()])
        
        # Check ss_* format (Kohya trainer)
        if 'ss_tag_frequency' in metadata:
            try:
                tag_freq = json.loads(metadata['ss_tag_frequency'])
                # Get most frequent tags as potential triggers
                for dataset_tags in tag_freq.values():
                    if isinstance(dataset_tags, dict):
                        # Sort by frequency, take top ones
                        sorted_tags = sorted(dataset_tags.items(), key=lambda x: x[1], reverse=True)
                        for tag, freq in sorted_tags[:10]:
                            if tag not in triggers:
                                triggers.append(tag)
            except:
                pass
        
        # Check for trigger_word key (some formats)
        for key in ['trigger_word', 'trigger_words', 'activation_text']:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, str) and value:
                    triggers.extend([t.strip() for t in value.split(',') if t.strip()])
                elif isinstance(value, list):
                    triggers.extend([str(t).strip() for t in value if t])
        
        return triggers
    
    def _get_triggers_from_civitai_cache(self, lora_name: str) -> List[str]:
        """Look up triggers from CivitAI metadata cache"""
        # Check if we have the Luna metadata database
        try:
            from utils.luna_metadata_db import get_lora_metadata
            metadata = get_lora_metadata(lora_name)
            if metadata and 'trigger_words' in metadata:
                return metadata['trigger_words']
        except ImportError:
            pass
        
        # Check for .civitai.json sidecar file
        if HAS_FOLDER_PATHS:
            lora_path = self._resolve_lora_path(lora_name)
            if lora_path:
                sidecar_path = lora_path.rsplit('.', 1)[0] + '.civitai.json'
                if os.path.exists(sidecar_path):
                    try:
                        with open(sidecar_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'trainedWords' in data:
                                return data['trainedWords']
                    except:
                        pass
        
        return []
    
    def _get_triggers_for_lora(self, lora_name: str) -> List[str]:
        """Get all trigger words for a LoRA from all sources"""
        # Check cache first
        cache_key = lora_name.lower()
        if cache_key in self._trigger_cache:
            return self._trigger_cache[cache_key]
        
        triggers = []
        
        # Try CivitAI cache first (most reliable)
        civitai_triggers = self._get_triggers_from_civitai_cache(lora_name)
        triggers.extend(civitai_triggers)
        
        # Try embedded metadata
        lora_path = self._resolve_lora_path(lora_name)
        if lora_path and lora_path.endswith('.safetensors'):
            metadata = self._read_safetensors_metadata(lora_path)
            embedded_triggers = self._get_triggers_from_metadata(metadata)
            for t in embedded_triggers:
                if t not in triggers:
                    triggers.append(t)
        
        # Cache result
        self._trigger_cache[cache_key] = triggers
        
        return triggers
    
    def process(
        self,
        injection_mode: str,
        max_triggers_per_lora: int,
        separator: str,
        deduplicate: bool,
        prompt: str = "",
        lora_stack: Optional[List[Tuple[str, float, float]]] = None
    ) -> Tuple[str, str, List[Tuple[str, float, float]], str]:
        """
        Process LoRAs and extract/inject trigger words.
        
        Returns:
            - prompt_with_triggers: Prompt with triggers injected (based on mode)
            - triggers: All trigger words combined
            - lora_stack: Combined/passthrough LoRA stack
            - prompt_passthrough: Original prompt (cleaned of inline LoRA tags)
        """
        # Start with input stack or empty
        combined_stack = list(lora_stack) if lora_stack else []
        
        # Extract inline LoRAs from prompt
        cleaned_prompt = prompt or ""
        if prompt:
            cleaned_prompt, inline_loras = self._extract_inline_loras(prompt)
            
            # Add inline LoRAs to stack (avoid duplicates)
            existing_names = {l[0].lower() for l in combined_stack}
            for lora in inline_loras:
                if lora[0].lower() not in existing_names:
                    combined_stack.append(lora)
                    existing_names.add(lora[0].lower())
        
        # Collect triggers from all LoRAs
        all_triggers = []
        for lora_name, model_w, clip_w in combined_stack:
            lora_triggers = self._get_triggers_for_lora(lora_name)
            
            # Limit per-LoRA triggers
            limited_triggers = lora_triggers[:max_triggers_per_lora]
            
            for trigger in limited_triggers:
                if deduplicate:
                    # Case-insensitive dedup
                    if trigger.lower() not in [t.lower() for t in all_triggers]:
                        all_triggers.append(trigger)
                else:
                    all_triggers.append(trigger)
        
        # Build trigger string
        trigger_string = separator.join(all_triggers)
        
        # Build prompt with triggers based on mode
        if injection_mode == "prepend" and trigger_string:
            prompt_with_triggers = trigger_string + separator + cleaned_prompt if cleaned_prompt else trigger_string
        elif injection_mode == "append" and trigger_string:
            prompt_with_triggers = cleaned_prompt + separator + trigger_string if cleaned_prompt else trigger_string
        else:
            prompt_with_triggers = cleaned_prompt
        
        # Clean up any double separators
        double_sep = separator + separator
        while double_sep in prompt_with_triggers:
            prompt_with_triggers = prompt_with_triggers.replace(double_sep, separator)
        prompt_with_triggers = prompt_with_triggers.strip().strip(',').strip()
        
        # Log what we found
        if all_triggers:
            print(f"[LunaTriggerInjector] Found {len(all_triggers)} triggers from {len(combined_stack)} LoRAs")
            for lora_name, _, _ in combined_stack:
                lora_triggers = self._get_triggers_for_lora(lora_name)[:max_triggers_per_lora]
                if lora_triggers:
                    print(f"  {lora_name}: {lora_triggers}")
        
        return (prompt_with_triggers, trigger_string, combined_stack, cleaned_prompt)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaLoRATriggerInjector": LunaLoRATriggerInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaLoRATriggerInjector": "Luna LoRA Trigger Injector",
}
