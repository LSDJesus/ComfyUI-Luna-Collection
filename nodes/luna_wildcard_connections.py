"""
Luna Wildcard Connections - Dynamic LoRA/Embedding linking to Wildcard categories

This module provides nodes for:
1. Managing connections between LoRAs/embeddings and wildcard categories
2. Auto-loading relevant LoRAs when wildcards resolve to connected categories
3. Interactive connection editing via web interface
"""

import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

import folder_paths

# Try to import PromptServer for web endpoints
try:
    from server import PromptServer
    from aiohttp import web
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False
    print("LunaConnections: PromptServer not available, web endpoints disabled")

# =============================================================================
# CONNECTIONS DATABASE
# =============================================================================

class ConnectionsDB:
    """
    Manages the connections.json database that links LoRAs/embeddings 
    to wildcard categories and tags.
    """
    
    _instance = None
    _cache = None
    _cache_mtime = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_connections_path() -> str:
        """Get path to connections.json file"""
        # Try registered wildcards path first (if another node registered it)
        try:
            wildcards_paths = folder_paths.get_folder_paths("wildcards")
            if wildcards_paths:
                conn_path = os.path.join(wildcards_paths[0], "connections.json")
                if os.path.exists(conn_path):
                    return conn_path
        except (KeyError, IndexError):
            pass
        
        # Default: ComfyUI models_dir/wildcards
        models_dir = getattr(folder_paths, 'models_dir', None)
        if models_dir:
            default_path = os.path.join(models_dir, "wildcards", "connections.json")
            if os.path.exists(default_path):
                return default_path
            # Return this path even if it doesn't exist (will be created on first save)
            return default_path
        
        # Last resort fallback: alongside this node file
        return os.path.join(os.path.dirname(__file__), "..", "wildcards", "connections.json")
    
    def load(self, force_reload: bool = False) -> Dict:
        """Load connections database with caching"""
        path = self.get_connections_path()
        
        if not os.path.exists(path):
            return self._get_empty_db()
        
        try:
            mtime = os.path.getmtime(path)
            if not force_reload and self._cache and mtime <= self._cache_mtime:
                return self._cache
            
            with open(path, 'r', encoding='utf-8') as f:
                self._cache = json.load(f)
                self._cache_mtime = mtime
                return self._cache
        except Exception as e:
            print(f"LunaConnections: Error loading {path}: {e}")
            return self._get_empty_db()
    
    def save(self, data: Dict) -> bool:
        """Save connections database"""
        path = self.get_connections_path()
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self._cache = data
            self._cache_mtime = os.path.getmtime(path)
            return True
        except Exception as e:
            print(f"LunaConnections: Error saving {path}: {e}")
            return False
    
    def _get_empty_db(self) -> Dict:
        """Return empty database structure"""
        return {
            "_meta": {"version": "1.0.0"},
            "loras": {},
            "embeddings": {},
            "tag_groups": {},
            "category_aliases": {}
        }
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def find_loras_by_category(self, category: str) -> List[Dict]:
        """Find all LoRAs linked to a category path (e.g., 'clothing:lingerie.types')"""
        db = self.load()
        results = []
        
        for lora_name, config in db.get("loras", {}).items():
            categories = config.get("categories", [])
            for cat in categories:
                # Match exact or partial category paths
                if category in cat or cat.startswith(category):
                    results.append({
                        "name": lora_name,
                        "config": config,
                        "matched_category": cat
                    })
                    break
        
        return results
    
    def find_loras_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict]:
        """Find LoRAs by tags (AND or OR matching)"""
        db = self.load()
        results = []
        tag_set = set(t.lower() for t in tags)
        
        for lora_name, config in db.get("loras", {}).items():
            lora_tags = set(t.lower() for t in config.get("tags", []))
            
            if match_all:
                if tag_set.issubset(lora_tags):
                    results.append({"name": lora_name, "config": config})
            else:
                if tag_set & lora_tags:  # Any intersection
                    results.append({"name": lora_name, "config": config})
        
        return results
    
    def find_loras_by_trigger(self, text: str) -> List[Dict]:
        """Find LoRAs whose triggers appear in text"""
        db = self.load()
        results = []
        text_lower = text.lower()
        
        for lora_name, config in db.get("loras", {}).items():
            triggers = config.get("triggers", [])
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    results.append({
                        "name": lora_name,
                        "config": config,
                        "matched_trigger": trigger
                    })
                    break
        
        return results
    
    def find_loras_by_training_tags(self, prompt_words: Set[str], min_matches: int = 2) -> List[Dict]:
        """
        Find LoRAs based on training_tags frequency data.
        Uses the training data to find LoRAs that were trained on similar concepts.
        
        Args:
            prompt_words: Set of words from the prompt (lowercased)
            min_matches: Minimum number of training tag matches required
            
        Returns:
            List of matches with scores based on training tag frequency
        """
        db = self.load()
        results = []
        
        for lora_name, config in db.get("loras", {}).items():
            training_tags = config.get("training_tags", {})
            if not training_tags:
                continue
            
            # Calculate match score based on frequency-weighted matches
            match_score = 0
            matched_tags = []
            
            for tag, frequency in training_tags.items():
                # Normalize tag for matching (replace underscores, lowercase)
                normalized_tag = tag.lower().replace("_", " ")
                tag_words = set(normalized_tag.split())
                
                # Check for word overlap
                if tag_words & prompt_words:
                    # Weight by training frequency (log scale to dampen extremes)
                    import math
                    match_score += math.log1p(frequency)
                    matched_tags.append((tag, frequency))
            
            if len(matched_tags) >= min_matches:
                results.append({
                    "name": lora_name,
                    "config": config,
                    "match_score": match_score,
                    "matched_training_tags": matched_tags
                })
        
        # Sort by match score (highest first)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results
    
    def find_loras_by_civitai_tags(self, tags: List[str]) -> List[Dict]:
        """Find LoRAs by their Civitai classification tags"""
        db = self.load()
        results = []
        tag_set = set(t.lower() for t in tags)
        
        for lora_name, config in db.get("loras", {}).items():
            civitai_tags = set(t.lower() for t in config.get("civitai_tags", []))
            matched = tag_set & civitai_tags
            
            if matched:
                results.append({
                    "name": lora_name,
                    "config": config,
                    "matched_civitai_tags": list(matched)
                })
        
        return results
    
    def get_activation_text(self, lora_name: str) -> str:
        """Get the full activation text for a LoRA"""
        config = self.get_lora_config(lora_name)
        if config:
            return config.get("activation_text", "")
        return ""
    
    def get_smart_triggers(self, lora_name: str, max_triggers: int = 3) -> List[str]:
        """
        Get the most important trigger words for a LoRA.
        Prioritizes explicit triggers, falls back to top training tags.
        """
        config = self.get_lora_config(lora_name)
        if not config:
            return []
        
        # First try explicit triggers
        triggers = config.get("triggers", [])
        if triggers:
            return triggers[:max_triggers]
        
        # Fall back to top training tags
        training_tags = config.get("training_tags", {})
        if training_tags:
            sorted_tags = sorted(training_tags.items(), key=lambda x: x[1], reverse=True)
            # Filter out common/generic tags
            generic = {"1girl", "solo", "breasts", "looking at viewer", "simple background", 
                      "white background", "highres", "absurdres", "hi res"}
            filtered = [tag for tag, _ in sorted_tags if tag.lower() not in generic]
            return filtered[:max_triggers]
        
        return []
    
    def find_embeddings_by_category(self, category: str) -> List[Dict]:
        """Find embeddings linked to a category"""
        db = self.load()
        results = []
        
        for emb_name, config in db.get("embeddings", {}).items():
            categories = config.get("categories", [])
            for cat in categories:
                if category in cat or cat.startswith(category):
                    results.append({
                        "name": emb_name,
                        "config": config,
                        "matched_category": cat
                    })
                    break
        
        return results
    
    def get_lora_config(self, lora_name: str) -> Optional[Dict]:
        """Get config for a specific LoRA"""
        db = self.load()
        return db.get("loras", {}).get(lora_name)
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags from the database"""
        db = self.load()
        tags = set()
        
        for config in db.get("loras", {}).values():
            tags.update(config.get("tags", []))
        for config in db.get("embeddings", {}).values():
            tags.update(config.get("tags", []))
        
        return sorted(tags)
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories from the database"""
        db = self.load()
        categories = set()
        
        for config in db.get("loras", {}).values():
            categories.update(config.get("categories", []))
        for config in db.get("embeddings", {}).values():
            categories.update(config.get("categories", []))
        
        return sorted(categories)


# Global instance
connections_db = ConnectionsDB()


# =============================================================================
# DYNAMIC LORA MATCHER NODE
# =============================================================================

class LunaConnectionMatcher:
    """
    Analyzes a prompt or wildcard resolution and returns matching LoRAs.
    
    Can match by:
    - Category paths that were used in wildcard resolution
    - Tags associated with content
    - Trigger words found in the prompt
    - Training data (semantic matching based on what the LoRA was trained on)
    - Civitai classification tags
    """
    
    CATEGORY = "Luna/Connections"
    RETURN_TYPES = ("STRING", "LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_string", "LORA_STACK", "matched_info")
    FUNCTION = "match_connections"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["by_category", "by_tags", "by_triggers", "by_training", "by_civitai_type", "combined"], {
                    "default": "combined",
                    "tooltip": "How to find matching LoRAs"
                }),
                "max_loras": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Maximum number of LoRAs to return"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for random selection (0 = random)"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt to scan for triggers and training tag matching"
                }),
                "resolved_categories": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated category paths (e.g., 'clothing:lingerie, pose:standing')"
                }),
                "tags": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated tags to match (e.g., 'anime, colorful')"
                }),
                "civitai_types": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated Civitai types (e.g., 'character, concept, style')"
                }),
                "model_type_filter": (["any", "sdxl", "pony", "illustrious", "sd15"], {
                    "default": "any",
                    "tooltip": "Filter LoRAs by model type"
                }),
                "weight_mode": (["metadata_default", "metadata_random", "override"], {
                    "default": "metadata_random",
                    "tooltip": "How to determine weights"
                }),
                "weight_override": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Override weight when mode is 'override'"
                }),
                "training_tag_min_matches": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum training tag matches for by_training mode"
                }),
            }
        }
    
    def match_connections(self, mode: str, max_loras: int, seed: int,
                          prompt: str = "", resolved_categories: str = "",
                          tags: str = "", civitai_types: str = "",
                          model_type_filter: str = "any",
                          weight_mode: str = "metadata_random",
                          weight_override: float = 1.0,
                          training_tag_min_matches: int = 2) -> Tuple[str, List, str]:
        """Find and return matching LoRAs based on criteria"""
        
        all_matches = []
        match_info = []
        rng = random.Random(seed if seed != 0 else None)
        
        # Parse inputs
        category_list = [c.strip() for c in resolved_categories.split(",") if c.strip()]
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        civitai_list = [c.strip() for c in civitai_types.split(",") if c.strip()]
        
        # Category matching
        if mode in ["by_category", "combined"] and category_list:
            for cat in category_list:
                matches = connections_db.find_loras_by_category(cat)
                for m in matches:
                    all_matches.append(m)
                    match_info.append(f"Category '{cat}' -> {m['name']}")
        
        # Tag matching
        if mode in ["by_tags", "combined"] and tag_list:
            matches = connections_db.find_loras_by_tags(tag_list, match_all=False)
            for m in matches:
                if m not in all_matches:
                    all_matches.append(m)
                    match_info.append(f"Tags {tag_list} -> {m['name']}")
        
        # Trigger matching
        if mode in ["by_triggers", "combined"] and prompt:
            matches = connections_db.find_loras_by_trigger(prompt)
            for m in matches:
                if m not in all_matches:
                    all_matches.append(m)
                    match_info.append(f"Trigger '{m['matched_trigger']}' -> {m['name']}")
        
        # Training tag matching (semantic)
        if mode in ["by_training", "combined"] and prompt:
            prompt_words = set(re.findall(r'\b[a-z_]+\b', prompt.lower()))
            matches = connections_db.find_loras_by_training_tags(prompt_words, training_tag_min_matches)
            for m in matches:
                if m['name'] not in [x['name'] for x in all_matches]:
                    all_matches.append(m)
                    tags_preview = [t for t, _ in m.get('matched_training_tags', [])[:3]]
                    match_info.append(f"Training ({m['match_score']:.1f}) [{', '.join(tags_preview)}] -> {m['name']}")
        
        # Civitai type matching
        if mode in ["by_civitai_type", "combined"] and civitai_list:
            matches = connections_db.find_loras_by_civitai_tags(civitai_list)
            for m in matches:
                if m['name'] not in [x['name'] for x in all_matches]:
                    all_matches.append(m)
                    match_info.append(f"Civitai type {m['matched_civitai_tags']} -> {m['name']}")
        
        # Filter by model type
        if model_type_filter != "any":
            all_matches = [m for m in all_matches 
                          if m['config'].get('model_type', 'any') in (model_type_filter, 'any')]
        
        # Deduplicate by name
        seen = set()
        unique_matches = []
        for m in all_matches:
            if m['name'] not in seen:
                seen.add(m['name'])
                unique_matches.append(m)
        
        # Random selection if more than max
        if len(unique_matches) > max_loras:
            unique_matches = rng.sample(unique_matches, max_loras)
        
        # Build outputs
        lora_parts = []
        lora_stack = []
        
        for match in unique_matches:
            lora_name = match['name']
            config = match['config']
            
            # Determine weight based on mode
            if weight_mode == "override":
                weight = weight_override
            elif weight_mode == "metadata_random":
                weight_range = config.get('weight_range', {})
                if isinstance(weight_range, dict):
                    min_w = weight_range.get('min', 0.7)
                    max_w = weight_range.get('max', 1.0)
                elif isinstance(weight_range, list) and len(weight_range) >= 2:
                    min_w, max_w = weight_range[0], weight_range[1]
                else:
                    min_w, max_w = 0.7, 1.0
                weight = rng.uniform(min_w, max_w)
            else:  # metadata_default
                weight_range = config.get('weight_range', {})
                if isinstance(weight_range, dict):
                    weight = (weight_range.get('min', 0.7) + weight_range.get('max', 1.0)) / 2
                else:
                    weight = 1.0
            
            # A1111 format string
            clean_name = lora_name.replace('.safetensors', '').replace('.pt', '')
            lora_parts.append(f"<lora:{clean_name}:{weight:.2f}>")
            
            # Stack format (name, model_str, clip_str)
            lora_stack.append((lora_name, weight, weight))
        
        lora_string = " ".join(lora_parts)
        info_string = "\n".join(match_info) if match_info else "No matches found"
        
        return (lora_string, lora_stack, info_string)


# =============================================================================
# CONNECTION EDITOR NODE
# =============================================================================

class LunaConnectionEditor:
    """
    Interactive node for editing LoRA/embedding connections.
    
    Select a LoRA, view/edit its linked categories and tags,
    then save back to connections.json.
    """
    
    CATEGORY = "Luna/Connections"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "success")
    FUNCTION = "edit_connection"
    OUTPUT_NODE = True
    
    @classmethod
    def get_lora_list(cls) -> List[str]:
        """Get list of available LoRAs"""
        try:
            loras = folder_paths.get_filename_list("loras")
            return ["-- Select LoRA --"] + sorted(loras) if loras else ["-- No LoRAs Found --"]
        except:
            return ["-- No LoRAs Found --"]
    
    @classmethod
    def get_embedding_list(cls) -> List[str]:
        """Get list of available embeddings"""
        try:
            embeddings = folder_paths.get_filename_list("embeddings")
            return ["-- Select Embedding --"] + sorted(embeddings) if embeddings else ["-- No Embeddings Found --"]
        except:
            return ["-- No Embeddings Found --"]
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = cls.get_lora_list()
        embeddings = cls.get_embedding_list()
        db = connections_db.load()
        all_tags = connections_db.get_all_tags()
        
        return {
            "required": {
                "edit_type": (["lora", "embedding"], {
                    "default": "lora"
                }),
                "action": (["view", "add", "update", "remove"], {
                    "default": "view",
                    "tooltip": "Action to perform"
                }),
            },
            "optional": {
                "lora_select": (loras, {"default": loras[0]}),
                "embedding_select": (embeddings, {"default": embeddings[0]}),
                "triggers": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated trigger words"
                }),
                "categories": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Comma-separated category paths (e.g., 'clothing:lingerie, pose:standing')"
                }),
                "tags": ("STRING", {
                    "default": "",
                    "tooltip": f"Comma-separated tags. Existing: {', '.join(all_tags[:20])}"
                }),
                "model_type": (["sdxl", "pony", "illustrious", "sd15", "any"], {
                    "default": "any"
                }),
                "weight_default": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "weight_min": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "weight_max": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "notes": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    def edit_connection(self, edit_type: str, action: str,
                        lora_select: str = "", embedding_select: str = "",
                        triggers: str = "", categories: str = "", tags: str = "",
                        model_type: str = "any", weight_default: float = 1.0,
                        weight_min: float = 0.5, weight_max: float = 1.5,
                        notes: str = "") -> Tuple[str, bool]:
        """Edit a connection entry"""
        
        db = connections_db.load()
        
        # Determine target
        if edit_type == "lora":
            target = lora_select
            collection = "loras"
        else:
            target = embedding_select
            collection = "embeddings"
        
        if target.startswith("--"):
            return ("Please select a LoRA or embedding", False)
        
        # Parse inputs
        trigger_list = [t.strip() for t in triggers.split(",") if t.strip()]
        category_list = [c.strip() for c in categories.split(",") if c.strip()]
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        if action == "view":
            # Return current config
            config = db.get(collection, {}).get(target, {})
            if config:
                return (json.dumps(config, indent=2), True)
            else:
                return (f"No connection found for {target}", False)
        
        elif action == "add" or action == "update":
            # Create/update entry
            if collection not in db:
                db[collection] = {}
            
            db[collection][target] = {
                "triggers": trigger_list,
                "categories": category_list,
                "tags": tag_list,
                "model_type": model_type,
                "weight_default": weight_default,
                "weight_range": [weight_min, weight_max],
                "notes": notes
            }
            
            if connections_db.save(db):
                return (f"Successfully saved connection for {target}", True)
            else:
                return (f"Error saving connection for {target}", False)
        
        elif action == "remove":
            if target in db.get(collection, {}):
                del db[collection][target]
                if connections_db.save(db):
                    return (f"Removed connection for {target}", True)
            return (f"No connection found for {target}", False)
        
        return ("Unknown action", False)


# =============================================================================
# SMART WILDCARD LORA LINKER
# =============================================================================

class LunaSmartLoRALinker:
    """
    The main integration node - automatically injects LoRAs based on 
    wildcard resolution with intelligent metadata-driven matching.
    
    Features:
    - Category-based matching from wildcard resolution
    - Trigger word detection in prompts
    - Training tag analysis for semantic matching
    - Civitai tag filtering (concept, character, style, etc.)
    - Automatic activation text injection
    - Smart weight randomization from metadata ranges
    
    Can auto-detect model type from connected MODEL to filter incompatible LoRAs!
    """
    
    CATEGORY = "Luna/Connections"
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "lora_string", "LORA_STACK", "detected_type", "match_report")
    FUNCTION = "link_loras"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The resolved prompt from wildcard processing"
                }),
                "enable_category_matching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match LoRAs by wildcard category paths"
                }),
                "enable_trigger_matching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match LoRAs by trigger words in prompt"
                }),
                "enable_training_tag_matching": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use training data analysis to find semantically similar LoRAs (experimental)"
                }),
                "max_loras": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Connect MODEL to auto-detect type and filter LoRAs"
                }),
                "checkpoint_name": ("STRING", {
                    "default": "",
                    "tooltip": "Checkpoint name for model type detection fallback"
                }),
                "wildcard_metadata": ("STRING", {
                    "default": "",
                    "tooltip": "JSON metadata from wildcard resolution (category paths used)"
                }),
                "model_type_override": (["auto", "any", "sdxl", "pony", "illustrious", "sd15", "flux"], {
                    "default": "auto",
                    "tooltip": "Override auto-detection. 'auto' uses MODEL input, 'any' disables filtering"
                }),
                "inject_mode": (["smart_triggers", "activation_text", "none"], {
                    "default": "smart_triggers",
                    "tooltip": "How to inject LoRA activation into prompt. 'smart_triggers' picks key words, 'activation_text' uses full Civitai text"
                }),
                "civitai_type_filter": (["any", "character", "concept", "style", "poses", "clothing", "tool"], {
                    "default": "any",
                    "tooltip": "Filter by Civitai LoRA type classification"
                }),
                "weight_mode": (["metadata_default", "metadata_random", "override"], {
                    "default": "metadata_random",
                    "tooltip": "How to determine LoRA weights"
                }),
                "weight_override": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Fixed weight when weight_mode is 'override'"
                }),
                "training_tag_min_matches": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum training tag matches for semantic matching"
                }),
                "existing_lora_string": ("STRING", {
                    "default": "",
                    "tooltip": "Existing LoRA string to append to"
                }),
            }
        }
    
    def link_loras(self, prompt: str, enable_category_matching: bool,
                   enable_trigger_matching: bool, enable_training_tag_matching: bool,
                   max_loras: int, seed: int,
                   model=None, checkpoint_name: str = "",
                   wildcard_metadata: str = "", model_type_override: str = "auto",
                   inject_mode: str = "smart_triggers",
                   civitai_type_filter: str = "any",
                   weight_mode: str = "metadata_random",
                   weight_override: float = 1.0,
                   training_tag_min_matches: int = 2,
                   existing_lora_string: str = "") -> Tuple[str, str, List, str, str]:
        """Analyze prompt and link relevant LoRAs with intelligent metadata matching"""
        
        all_matches = []
        match_report_lines = []
        rng = random.Random(seed if seed != 0 else None)
        
        # Determine model type for filtering
        detected_type = "unknown"
        if model_type_override == "auto":
            if model is not None:
                detected_type = detect_model_type_from_model(model)
            if detected_type == "unknown" and checkpoint_name:
                detected_type = detect_model_type_from_name(checkpoint_name)
            model_type = detected_type if detected_type != "unknown" else "any"
        elif model_type_override == "any":
            model_type = "any"
            detected_type = "any (no filtering)"
        else:
            model_type = model_type_override
            detected_type = f"override: {model_type_override}"
        
        match_report_lines.append(f"Model Type: {detected_type}")
        
        # Parse wildcard metadata for category paths
        if enable_category_matching and wildcard_metadata:
            try:
                meta = json.loads(wildcard_metadata)
                categories = meta.get("resolved_categories", [])
                for cat in categories:
                    matches = connections_db.find_loras_by_category(cat)
                    for m in matches:
                        m["match_source"] = f"category:{cat}"
                    all_matches.extend(matches)
                    if matches:
                        match_report_lines.append(f"Category '{cat}' -> {len(matches)} matches")
            except:
                pass
        
        # Scan prompt for triggers
        if enable_trigger_matching and prompt:
            matches = connections_db.find_loras_by_trigger(prompt)
            for m in matches:
                m["match_source"] = f"trigger:{m.get('matched_trigger', '?')}"
            all_matches.extend(matches)
            if matches:
                match_report_lines.append(f"Trigger matching -> {len(matches)} matches")
        
        # Training tag semantic matching
        if enable_training_tag_matching and prompt:
            # Extract words from prompt for matching
            prompt_words = set(re.findall(r'\b[a-z_]+\b', prompt.lower()))
            matches = connections_db.find_loras_by_training_tags(prompt_words, training_tag_min_matches)
            for m in matches:
                tags_info = [f"{t}({f})" for t, f in m.get("matched_training_tags", [])[:3]]
                m["match_source"] = f"training:{','.join(tags_info)}"
            all_matches.extend(matches)
            if matches:
                match_report_lines.append(f"Training tag matching -> {len(matches)} matches")
        
        # Filter by model type
        if model_type != "any":
            before = len(all_matches)
            all_matches = [m for m in all_matches 
                          if m['config'].get('model_type', 'any') in (model_type, 'any')]
            if before != len(all_matches):
                match_report_lines.append(f"Model type filter: {before} -> {len(all_matches)}")
        
        # Filter by Civitai type
        if civitai_type_filter != "any":
            before = len(all_matches)
            all_matches = [m for m in all_matches
                          if civitai_type_filter.lower() in [t.lower() for t in m['config'].get('civitai_tags', [])]]
            if before != len(all_matches):
                match_report_lines.append(f"Civitai type filter ({civitai_type_filter}): {before} -> {len(all_matches)}")
        
        # Deduplicate by name, keeping first (highest priority) match
        seen = set()
        unique_matches = []
        for m in all_matches:
            if m['name'] not in seen:
                seen.add(m['name'])
                unique_matches.append(m)
        
        # Limit selection (random if over max)
        if len(unique_matches) > max_loras:
            unique_matches = rng.sample(unique_matches, max_loras)
        
        match_report_lines.append(f"\nSelected {len(unique_matches)} LoRAs:")
        
        # Build outputs
        lora_parts = []
        lora_stack = []
        trigger_additions = []
        
        for match in unique_matches:
            lora_name = match['name']
            config = match['config']
            
            # Determine weight based on mode
            if weight_mode == "override":
                weight = weight_override
            elif weight_mode == "metadata_random":
                weight_range = config.get('weight_range', {})
                if isinstance(weight_range, dict):
                    min_w = weight_range.get('min', 0.7)
                    max_w = weight_range.get('max', 1.0)
                elif isinstance(weight_range, list) and len(weight_range) >= 2:
                    min_w, max_w = weight_range[0], weight_range[1]
                else:
                    min_w, max_w = 0.7, 1.0
                weight = rng.uniform(min_w, max_w)
            else:  # metadata_default
                weight_range = config.get('weight_range', {})
                if isinstance(weight_range, dict):
                    weight = (weight_range.get('min', 0.7) + weight_range.get('max', 1.0)) / 2
                else:
                    weight = 1.0
            
            # A1111 format
            clean_name = lora_name.replace('.safetensors', '').replace('.pt', '')
            lora_parts.append(f"<lora:{clean_name}:{weight:.2f}>")
            
            # Stack format
            lora_stack.append((lora_name, weight, weight))
            
            # Collect triggers based on mode
            if inject_mode == "activation_text":
                activation = config.get('activation_text', '')
                if activation:
                    # Take first part of activation text (often a mess of examples)
                    first_part = activation.split(',')[0].strip()
                    if first_part and first_part not in trigger_additions:
                        trigger_additions.append(first_part)
            elif inject_mode == "smart_triggers":
                smart = connections_db.get_smart_triggers(lora_name, max_triggers=2)
                for t in smart:
                    if t and t not in trigger_additions:
                        trigger_additions.append(t)
            
            # Report line
            source = match.get('match_source', 'unknown')
            match_report_lines.append(f"  • {clean_name} @ {weight:.2f} ({source})")
        
        # Combine with existing
        if existing_lora_string:
            lora_parts.insert(0, existing_lora_string)
        
        lora_string = " ".join(lora_parts)
        
        # Enhance prompt with triggers
        enhanced_prompt = prompt
        if inject_mode != "none" and trigger_additions:
            unique_triggers = list(dict.fromkeys(trigger_additions))  # Preserve order, remove dupes
            trigger_str = ", ".join(unique_triggers[:5])  # Limit to 5 triggers
            enhanced_prompt = f"{prompt}, {trigger_str}"
            match_report_lines.append(f"\nInjected triggers: {trigger_str}")
        
        match_report = "\n".join(match_report_lines)
        
        return (enhanced_prompt, lora_string, lora_stack, detected_type, match_report)


# =============================================================================
# CONNECTION STATS NODE
# =============================================================================

class LunaConnectionStats:
    """
    Display statistics about the connections database.
    Useful for debugging and exploring what's available.
    """
    
    CATEGORY = "Luna/Connections"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "get_stats"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload of connections database"
                }),
            }
        }
    
    def get_stats(self, refresh: bool) -> Tuple[str]:
        db = connections_db.load(force_reload=refresh)
        
        lora_count = len(db.get("loras", {}))
        emb_count = len(db.get("embeddings", {}))
        all_tags = connections_db.get_all_tags()
        all_cats = connections_db.get_all_categories()
        
        # Count by model type
        model_types = {}
        for config in db.get("loras", {}).values():
            mt = config.get("model_type", "unknown")
            model_types[mt] = model_types.get(mt, 0) + 1
        
        stats = f"""Luna Connections Database
========================
Path: {connections_db.get_connections_path()}

LoRAs: {lora_count}
Embeddings: {emb_count}
Unique Tags: {len(all_tags)}
Unique Categories: {len(all_cats)}

LoRAs by Model Type:
{chr(10).join(f'  {k}: {v}' for k, v in sorted(model_types.items()))}

Top Tags: {', '.join(all_tags[:15])}

Categories: {', '.join(all_cats[:10])}..."""
        
        return (stats,)


# =============================================================================
# WEB ENDPOINTS FOR INTERACTIVE EDITING
# =============================================================================

if HAS_PROMPT_SERVER:
    
    @PromptServer.instance.routes.get("/luna/connections/list")
    async def list_connections(request):
        """List all connections"""
        db = connections_db.load(force_reload=True)
        return web.json_response({
            "loras": list(db.get("loras", {}).keys()),
            "embeddings": list(db.get("embeddings", {}).keys()),
            "tags": connections_db.get_all_tags(),
            "categories": connections_db.get_all_categories()
        })
    
    @PromptServer.instance.routes.get("/luna/connections/full")
    async def get_full_connections(request):
        """Get the full connections database"""
        db = connections_db.load(force_reload=True)
        return web.json_response({
            "loras": db.get("loras", {}),
            "embeddings": db.get("embeddings", {})
        })
    
    @PromptServer.instance.routes.get("/luna/connections/get")
    async def get_connection(request):
        """Get a specific connection"""
        item_type = request.query.get("type", "lora")
        name = request.query.get("name", "")
        
        if not name:
            return web.Response(status=400, text="name parameter required")
        
        db = connections_db.load()
        collection = "loras" if item_type == "lora" else "embeddings"
        
        config = db.get(collection, {}).get(name)
        if config:
            return web.json_response(config)
        else:
            return web.Response(status=404, text=f"No connection found for {name}")
    
    @PromptServer.instance.routes.post("/luna/connections/save")
    async def save_connection(request):
        """Save a connection"""
        try:
            data = await request.json()
            item_type = data.get("type", "lora")
            name = data.get("name", "")
            config = data.get("config", {})
            
            if not name or not config:
                return web.Response(status=400, text="name and config required")
            
            db = connections_db.load()
            collection = "loras" if item_type == "lora" else "embeddings"
            
            if collection not in db:
                db[collection] = {}
            
            db[collection][name] = config
            
            if connections_db.save(db):
                return web.json_response({"success": True})
            else:
                return web.Response(status=500, text="Failed to save")
                
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    @PromptServer.instance.routes.post("/luna/connections/delete")
    async def delete_connection(request):
        """Delete a connection"""
        try:
            data = await request.json()
            item_type = data.get("type", "lora")
            name = data.get("name", "")
            
            if not name:
                return web.Response(status=400, text="name parameter required")
            
            db = connections_db.load()
            collection = "loras" if item_type == "lora" else "embeddings"
            
            if name in db.get(collection, {}):
                del db[collection][name]
                if connections_db.save(db):
                    return web.json_response({"success": True})
            
            return web.Response(status=404, text=f"No connection found for {name}")
                
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    @PromptServer.instance.routes.post("/luna/connections/match")
    async def match_connections_api(request):
        """API endpoint to find matching LoRAs"""
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            categories = data.get("categories", [])
            tags = data.get("tags", [])
            
            results = {
                "by_trigger": [],
                "by_category": [],
                "by_tags": []
            }
            
            if prompt:
                matches = connections_db.find_loras_by_trigger(prompt)
                results["by_trigger"] = [{"name": m["name"], "trigger": m.get("matched_trigger")} 
                                         for m in matches]
            
            for cat in categories:
                matches = connections_db.find_loras_by_category(cat)
                results["by_category"].extend([{"name": m["name"], "category": cat} 
                                               for m in matches])
            
            if tags:
                matches = connections_db.find_loras_by_tags(tags)
                results["by_tags"] = [{"name": m["name"]} for m in matches]
            
            return web.json_response(results)
            
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    @PromptServer.instance.routes.post("/luna/connections/bulk_import")
    async def bulk_import_connections(request):
        """Bulk import connections from JSON"""
        try:
            data = await request.json()
            db = connections_db.load()
            
            # Merge with existing
            if "loras" in data:
                if "loras" not in db:
                    db["loras"] = {}
                db["loras"].update(data["loras"])
            
            if "embeddings" in data:
                if "embeddings" not in db:
                    db["embeddings"] = {}
                db["embeddings"].update(data["embeddings"])
            
            if connections_db.save(db):
                return web.json_response({
                    "success": True,
                    "loras_imported": len(data.get("loras", {})),
                    "embeddings_imported": len(data.get("embeddings", {}))
                })
            else:
                return web.Response(status=500, text="Failed to save")
                
        except Exception as e:
            return web.Response(status=500, text=str(e))


# =============================================================================
# MODEL TYPE DETECTION UTILITIES
# =============================================================================

def detect_model_type_from_name(model_name: str) -> str:
    """
    Attempt to detect model type from checkpoint/model name.
    Returns: 'sdxl', 'sd15', 'pony', 'illustrious', 'flux', or 'unknown'
    """
    name_lower = model_name.lower()
    
    # Check for explicit markers
    if 'flux' in name_lower:
        return 'flux'
    if 'pony' in name_lower or 'pdxl' in name_lower:
        return 'pony'
    if 'illustrious' in name_lower or 'ilxl' in name_lower or 'noob' in name_lower:
        return 'illustrious'
    if 'sdxl' in name_lower or 'xl' in name_lower:
        return 'sdxl'
    if 'sd15' in name_lower or 'sd1.5' in name_lower or 'v1-5' in name_lower:
        return 'sd15'
    if 'realistic' in name_lower and 'xl' not in name_lower:
        return 'sd15'  # Realistic Vision etc are usually SD1.5
    
    return 'unknown'


def detect_model_type_from_model(model) -> str:
    """
    Detect model type from a loaded MODEL object.
    Inspects model architecture to determine base type.
    """
    try:
        # Try to get model config
        if hasattr(model, 'model_config'):
            config = model.model_config
            
            # Check for SDXL architecture (larger UNet)
            if hasattr(config, 'unet_config'):
                unet_config = config.unet_config
                
                # SDXL has 2816 in_channels for context
                context_dim = unet_config.get('context_dim', 0)
                if context_dim == 2048:
                    return 'sdxl'  # Could be SDXL, Pony, or Illustrious
                elif context_dim == 768:
                    return 'sd15'
                elif context_dim == 4096:
                    return 'flux'
        
        # Fallback: check model size/parameters
        if hasattr(model, 'model'):
            param_count = sum(p.numel() for p in model.model.parameters())
            if param_count > 5e9:  # >5B params
                return 'flux'
            elif param_count > 2e9:  # >2B params  
                return 'sdxl'  # SDXL family
            else:
                return 'sd15'
                
    except Exception as e:
        print(f"LunaConnections: Error detecting model type: {e}")
    
    return 'unknown'


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaConnectionMatcher": LunaConnectionMatcher,
    "LunaConnectionEditor": LunaConnectionEditor,
    "LunaSmartLoRALinker": LunaSmartLoRALinker,
    "LunaConnectionStats": LunaConnectionStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaConnectionMatcher": "Luna Connection Matcher",
    "LunaConnectionEditor": "Luna Connection Editor",
    "LunaSmartLoRALinker": "Luna Smart LoRA Linker",
    "LunaConnectionStats": "Luna Connection Stats",
}
