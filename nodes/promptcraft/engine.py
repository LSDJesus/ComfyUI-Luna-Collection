"""
Luna PromptCraft Engine
Core logic for smart wildcard resolution with constraints, modifiers, and expanders.
No ComfyUI dependencies - can be used standalone.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CategoryItem:
    """A single item from a category file (matches existing YAML format)"""
    id: str
    text: str
    tags: Set[str] = field(default_factory=set)
    blacklist: Set[str] = field(default_factory=set)
    whitelist: Set[str] = field(default_factory=set)
    weight: float = 1.0
    payload: str = ""


@dataclass
class ResolvedItem:
    """Result of resolving a single wildcard"""
    value: str                      # The resolved text
    path: str                       # Full path that was picked (e.g., "location.outdoor.beach")
    tags: Set[str] = field(default_factory=set)  # Tags associated with this pick
    item_id: str = ""               # Item ID if from structured YAML
    blacklist: Set[str] = field(default_factory=set)  # Tags this item conflicts with
    whitelist: Set[str] = field(default_factory=set)  # Tags this item requires
    
    
@dataclass 
class PromptContext:
    """Tracks state across wildcard resolutions in a single prompt"""
    seed: int
    picks: Dict[str, ResolvedItem] = field(default_factory=dict)  # category -> pick
    tags: Set[str] = field(default_factory=set)                   # accumulated tags
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self._rng is None:
            self._rng = random.Random(self.seed if self.seed >= 0 else None)
    
    @property
    def rng(self) -> random.Random:
        """Get the RNG - always available after __post_init__"""
        if self._rng is None:
            self._rng = random.Random(self.seed if self.seed >= 0 else None)
        return self._rng
    
    def add_pick(self, category: str, item: ResolvedItem):
        """Record a pick and update context tags"""
        self.picks[category] = item
        self.tags.update(item.tags)


@dataclass
class ConstraintRule:
    """A single constraint rule"""
    when_tags: Set[str]              # Context must have ANY of these tags
    prefer: List[str] = field(default_factory=list)   # Boost these paths/items
    avoid: List[str] = field(default_factory=list)    # Exclude these paths/items
    require: List[str] = field(default_factory=list)  # Must pick from these


@dataclass
class ModifierRule:
    """How an action modifies other categories"""
    trigger_tags: Set[str]           # When these tags are in context
    target_category: str             # Modify items from this category
    append_any: List[str] = field(default_factory=list)  # Append one of these
    require_tags: Set[str] = field(default_factory=set)  # Target must have these tags


@dataclass
class ExpanderRule:
    """Scene expansion details"""
    path: str                        # Path this expander applies to
    base: str = ""                   # Always include this
    details: List[str] = field(default_factory=list)  # Random detail options
    detail_count: Tuple[int, int] = (1, 2)  # (min, max) details to add


# =============================================================================
# Category Loader
# =============================================================================

# Special file types identified by prefix or 'type' header field
SPECIAL_FILE_TYPES = {
    'rules_': 'rules',           # rules_clothing.yaml -> constraint rules
    'expanders_': 'expanders',   # expanders_location.yaml -> scene expanders  
    'modifiers_': 'modifiers',   # modifiers_action.yaml -> action modifiers
    'lora_': 'lora',             # lora_characters.yaml -> LoRA linking rules
    'tags_': 'tags',             # tags_clothing.yaml -> tag definitions
}


class CategoryLoader:
    """
    Loads and caches YAML/JSON category files from wildcards directory.
    
    Recursively scans all subfolders and treats files as if in root.
    So wildcards/luna/setting.yaml is accessed as {setting}.
    
    Special files are identified by:
    1. Filename prefix (rules_, expanders_, modifiers_, lora_, tags_)
    2. Or 'type' field in YAML header
    
    Supports both:
    1. Luna structured format: name, description, common_tags, items[{id, text, tags, blacklist, whitelist, weight}]
    2. Simple hierarchical format: nested dicts with string/list leaves
    """
    
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = Path(wildcards_dir)
        self._cache: Dict[str, Dict] = {}
        self._cache_mtime: Dict[str, float] = {}
        self._items_cache: Dict[str, List[CategoryItem]] = {}
        self._file_map: Dict[str, Path] = {}  # category_name -> filepath
        self._special_files: Dict[str, List[Path]] = {  # type -> [filepaths]
            'rules': [],
            'expanders': [],
            'modifiers': [],
            'lora': [],
            'tags': [],
        }
        self._scanned = False
    
    def _scan_directory(self, force: bool = False):
        """Recursively scan wildcards directory for all YAML/JSON files"""
        if self._scanned and not force:
            return
        
        self._file_map.clear()
        for key in self._special_files:
            self._special_files[key].clear()
        
        if not self.wildcards_dir.exists():
            self._scanned = True
            return
        
        # Walk through all subdirectories
        for filepath in self.wildcards_dir.rglob('*'):
            if filepath.suffix not in ('.yaml', '.yml', '.json'):
                continue
            if filepath.name.startswith('.'):
                continue
            
            # Skip folders that start with '_!' (exclusion pattern)
            if any(part.startswith('_!') for part in filepath.parts):
                continue
            
            # Check if this is a special file by prefix
            stem = filepath.stem
            is_special = False
            
            for prefix, file_type in SPECIAL_FILE_TYPES.items():
                if stem.startswith(prefix):
                    self._special_files[file_type].append(filepath)
                    is_special = True
                    break
            
            if not is_special:
                # Quick check for 'type' header in YAML
                file_type = self._check_file_type(filepath)
                if file_type and file_type in self._special_files:
                    self._special_files[file_type].append(filepath)
                else:
                    # Regular category file - use stem as category name
                    # Later files override earlier ones (allows customization)
                    self._file_map[stem] = filepath
        
        self._scanned = True
    
    def _check_file_type(self, filepath: Path) -> Optional[str]:
        """Quick check for 'type' field in YAML header without full parse"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read first few lines to check for type field
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith('type:'):
                        type_val = line.split(':', 1)[1].strip().strip('"\'')
                        return type_val
        except Exception:
            pass
        return None
    
    def get_categories_path(self) -> Path:
        """Get path to categories directory (legacy, now scans all)"""
        return self.wildcards_dir
    
    def load_yaml(self, filepath: Path) -> Dict:
        """Load a YAML or JSON file"""
        if not filepath.exists():
            return {}
        
        mtime = filepath.stat().st_mtime
        cache_key = str(filepath)
        
        if cache_key in self._cache and self._cache_mtime.get(cache_key, 0) >= mtime:
            return self._cache[cache_key]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix in ('.yaml', '.yml'):
                if not HAS_YAML:
                    print("LunaPromptCraft: PyYAML not installed, cannot load YAML files")
                    return {}
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        
        self._cache[cache_key] = data
        self._cache_mtime[cache_key] = mtime
        
        # If this has the Luna structured format, parse items
        if 'items' in data and isinstance(data['items'], list):
            items = self._parse_items(data, filepath.stem)
            self._items_cache[filepath.stem] = items
        
        return data
    
    def _parse_items(self, data: Dict, category_name: str) -> List[CategoryItem]:
        """Parse items from Luna structured YAML format"""
        items = []
        common_tags = set(data.get('common_tags', []))
        
        for item_data in data.get('items', []):
            if not isinstance(item_data, dict):
                continue
            
            # Parse item tags, blacklist, whitelist
            item_tags = set(item_data.get('tags', []))
            item_tags.update(common_tags)  # Include common tags
            
            blacklist = set()
            bl = item_data.get('blacklist', [])
            if isinstance(bl, list):
                blacklist = set(bl)
            
            whitelist = set()
            wl = item_data.get('whitelist', [])
            if isinstance(wl, list):
                whitelist = set(wl)
            
            item = CategoryItem(
                id=item_data.get('id', ''),
                text=item_data.get('text', ''),
                tags=item_tags,
                blacklist=blacklist,
                whitelist=whitelist,
                weight=float(item_data.get('weight', 1.0)),
                payload=item_data.get('payload', '')
            )
            items.append(item)
        
        return items
    
    def load_category(self, name: str) -> Dict:
        """Load a category file by name (e.g., 'setting' loads setting.yaml from any subfolder)"""
        self._scan_directory()
        
        if name in self._file_map:
            return self.load_yaml(self._file_map[name])
        
        return {}
    
    def get_category_items(self, name: str) -> List[CategoryItem]:
        """Get parsed items for a category"""
        # Ensure category is loaded
        self.load_category(name)
        return self._items_cache.get(name, [])
    
    def list_categories(self) -> List[str]:
        """List all available category files (excludes special files)"""
        self._scan_directory()
        return sorted(self._file_map.keys())
    
    def get_special_files(self, file_type: str) -> List[Path]:
        """Get all special files of a given type (rules, expanders, modifiers, lora, tags)"""
        self._scan_directory()
        return self._special_files.get(file_type, [])
    
    def get_items_at_path(self, path: str) -> List[Tuple[CategoryItem, str]]:
        """
        Get all items under a path.
        Returns list of (CategoryItem, category_name) tuples.
        
        For Luna structured format, the 'path' is just the category name.
        Example: "location_outdoor" returns all items from location_outdoor.yaml
        """
        parts = path.split('.')
        if not parts:
            return []
        
        category_name = parts[0]
        
        # Try to get structured items first
        items = self.get_category_items(category_name)
        if items:
            return [(item, category_name) for item in items]
        
        # Fallback to hierarchical format
        category_data = self.load_category(category_name)
        
        if not category_data:
            return []
        
        # Navigate to the target node
        node = category_data
        current_path = category_name
        
        for part in parts[1:]:
            if isinstance(node, dict) and part in node:
                node = node[part]
                current_path = f"{current_path}.{part}"
            else:
                return []
        
        # Collect all leaf items under this node
        return self._collect_leaves_as_items(node, current_path)
    
    def _collect_leaves_as_items(self, node: Any, path: str) -> List[Tuple[CategoryItem, str]]:
        """Recursively collect all leaf string values as CategoryItems"""
        results = []
        
        if isinstance(node, str):
            item = CategoryItem(id=path, text=node)
            results.append((item, path))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                if isinstance(item, str):
                    cat_item = CategoryItem(id=f"{path}_{i}", text=item)
                    results.append((cat_item, path))
                elif isinstance(item, dict):
                    # Handle complex item with value/weight/tags
                    if 'value' in item:
                        cat_item = CategoryItem(
                            id=item.get('id', f"{path}_{i}"),
                            text=item['value'],
                            tags=set(item.get('tags', [])),
                            weight=float(item.get('weight', 1.0))
                        )
                        results.append((cat_item, path))
        elif isinstance(node, dict):
            for key, value in node.items():
                # Skip metadata fields
                if key in ('name', 'description', 'common_tags', 'items'):
                    continue
                child_path = f"{path}.{key}"
                results.extend(self._collect_leaves_as_items(value, child_path))
        
        return results


# =============================================================================
# Tag Resolver
# =============================================================================

class TagResolver:
    """Manages tag mappings for paths"""
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._tags: Dict[str, Set[str]] = {}
        self._loaded = False
    
    def load_tags(self, force: bool = False):
        """Load all tag mapping files"""
        if self._loaded and not force:
            return
        
        self._tags.clear()
        
        # Use category loader's special file discovery if available
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('tags'):
                self._load_tag_file(filepath)
        else:
            # Fallback to legacy tags/ subdirectory
            tags_path = self.wildcards_dir / "tags"
            if tags_path.exists():
                for filepath in tags_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml', '.json'):
                        self._load_tag_file(filepath)
        
        self._loaded = True
    
    def _load_tag_file(self, filepath: Path):
        """Load a single tag file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix in ('.yaml', '.yml'):
                if not HAS_YAML:
                    return
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        
        # Flatten nested structure to path -> tags
        self._flatten_tags(data, "")
    
    def _flatten_tags(self, node: Any, prefix: str):
        """Flatten nested tag definitions"""
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, list):
                    # This is a tag list for this path
                    self._tags[path] = set(value)
                elif isinstance(value, dict):
                    # Recurse into nested structure
                    self._flatten_tags(value, path)
    
    def get_tags(self, path: str) -> Set[str]:
        """Get tags for a path, inheriting from parent paths"""
        self.load_tags()
        
        tags = set()
        parts = path.split('.')
        
        # Accumulate tags from root to leaf
        for i in range(len(parts)):
            check_path = '.'.join(parts[:i+1])
            if check_path in self._tags:
                tags.update(self._tags[check_path])
        
        return tags


# =============================================================================
# Constraint Engine
# =============================================================================

class ConstraintEngine:
    """Filters items based on context-aware rules and item blacklist/whitelist"""
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._rules: Dict[str, List[ConstraintRule]] = {}
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load constraint rules"""
        if self._loaded and not force:
            return
        
        self._rules.clear()
        
        # Use category loader's special file discovery if available
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('rules'):
                self._load_rules_file(filepath)
        else:
            # Fallback to legacy rules/ subdirectory
            rules_path = self.wildcards_dir / "rules"
            if rules_path.exists():
                for filepath in rules_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml', '.json'):
                        self._load_rules_file(filepath)
        
        self._loaded = True
    
    def _load_rules_file(self, filepath: Path):
        """Load rules from a YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        for category, cat_rules in data.items():
            if category not in self._rules:
                self._rules[category] = []
            
            prefs = cat_rules.get('context_preferences', [])
            for pref in prefs:
                rule = ConstraintRule(
                    when_tags=set(pref.get('when', [])),
                    prefer=pref.get('prefer', []),
                    avoid=pref.get('avoid', []),
                    require=pref.get('require', [])
                )
                self._rules[category].append(rule)
    
    def filter_items(
        self, 
        category: str, 
        items: List[Tuple[CategoryItem, str]], 
        context: PromptContext
    ) -> List[Tuple[CategoryItem, str]]:
        """
        Filter items based on current context.
        Uses both explicit rules AND item blacklist/whitelist from YAML.
        """
        self.load_rules()
        
        filtered: List[Tuple[CategoryItem, str]] = []
        
        for item, cat_name in items:
            # Check item's blacklist - if context has ANY blacklisted tag, exclude
            if item.blacklist and item.blacklist.intersection(context.tags):
                continue
            
            # Check item's whitelist - if specified, context must have at least one
            if item.whitelist and not item.whitelist.intersection(context.tags):
                continue
            
            filtered.append((item, cat_name))
        
        # Apply additional constraint rules if any
        if category in self._rules:
            for rule in self._rules[category]:
                # Check if rule applies (context has any of the trigger tags)
                if not rule.when_tags.intersection(context.tags):
                    continue
                
                # Apply avoid filter
                if rule.avoid:
                    filtered = [
                        (item, cat) for item, cat in filtered
                        if not any(avoid in item.id or avoid in item.text for avoid in rule.avoid)
                    ]
                
                # Apply require filter
                if rule.require:
                    required = [
                        (item, cat) for item, cat in filtered
                        if any(req in item.id or req in item.text or req in item.tags for req in rule.require)
                    ]
                    if required:  # Only apply if we'd have results
                        filtered = required
        
        return filtered if filtered else items  # Fallback to original if too restrictive


# =============================================================================
# Modifier Engine
# =============================================================================

class ModifierEngine:
    """Applies action-based modifications to resolved items"""
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._rules: List[ModifierRule] = []
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load modifier rules"""
        if self._loaded and not force:
            return
        
        self._rules.clear()
        
        # Use category loader's special file discovery if available
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('modifiers'):
                self._load_modifier_file(filepath)
        else:
            # Fallback to legacy path
            rules_path = self.wildcards_dir / "rules" / "modifiers.yaml"
            if rules_path.exists():
                self._load_modifier_file(rules_path)
        
        self._loaded = True
    
    def _load_modifier_file(self, filepath: Path):
        """Load modifiers from a YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        # Parse action modifiers
        for action, action_rules in data.get('action', {}).items():
            for target_cat, mods in action_rules.get('modify_clothing', {}).items():
                if isinstance(mods, list):
                    rule = ModifierRule(
                        trigger_tags={action},
                        target_category='clothing',
                        append_any=mods
                    )
                    self._rules.append(rule)
    
    def apply_modifiers(
        self, 
        value: str, 
        path: str, 
        context: PromptContext
    ) -> str:
        """Apply relevant modifiers to a resolved value"""
        self.load_rules()
        
        for rule in self._rules:
            # Check if rule applies
            if not rule.trigger_tags.intersection(context.tags):
                continue
            
            # Check if this item's category matches
            if not path.startswith(rule.target_category):
                continue
            
            # Apply modification
            if rule.append_any:
                modifier = context.rng.choice(rule.append_any)
                value = f"{value} {modifier}"
        
        return value


# =============================================================================
# Expander Engine
# =============================================================================

class ExpanderEngine:
    """Adds contextual scene details"""
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._rules: Dict[str, ExpanderRule] = {}
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load expander rules"""
        if self._loaded and not force:
            return
        
        self._rules.clear()
        
        # Use category loader's special file discovery if available
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('expanders'):
                self._load_expander_file(filepath)
        else:
            # Fallback to legacy expanders/ subdirectory
            expanders_path = self.wildcards_dir / "expanders"
            if expanders_path.exists():
                for filepath in expanders_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml'):
                        self._load_expander_file(filepath)
        
        self._loaded = True
    
    def _load_expander_file(self, filepath: Path):
        """Load expander definitions from YAML"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        for path, exp_data in data.items():
            count = exp_data.get('count', [1, 2])
            if isinstance(count, list) and len(count) == 2:
                detail_count = (count[0], count[1])
            else:
                detail_count = (1, 2)
            
            rule = ExpanderRule(
                path=path,
                base=exp_data.get('base', ''),
                details=exp_data.get('details', []),
                detail_count=detail_count
            )
            self._rules[path] = rule
    
    def expand(self, path: str, context: PromptContext) -> str:
        """Get expansion details for a path"""
        self.load_rules()
        
        # Find matching expander (exact match or parent match)
        rule = None
        parts = path.split('.')
        
        for i in range(len(parts), 0, -1):
            check_path = '.'.join(parts[:i])
            if check_path in self._rules:
                rule = self._rules[check_path]
                break
        
        if not rule:
            return ""
        
        # Build expansion
        expansion_parts = []
        
        if rule.base:
            expansion_parts.append(rule.base)
        
        if rule.details:
            min_count, max_count = rule.detail_count
            count = context.rng.randint(min_count, max_count)
            selected = context.rng.sample(
                rule.details, 
                min(count, len(rule.details))
            )
            expansion_parts.extend(selected)
        
        return ", ".join(expansion_parts)


# =============================================================================
# Main Prompt Engine
# =============================================================================

class LunaPromptEngine:
    """
    Main engine for smart wildcard resolution.
    
    Usage:
        engine = LunaPromptEngine("/path/to/wildcards")
        result = engine.process_template(
            "{location}, 1girl, {clothing}, {action}",
            seed=12345,
            enable_constraints=True,
            enable_modifiers=True,
            enable_expanders=True
        )
    """
    
    # Pattern to match {category} or {category:path}
    WILDCARD_PATTERN = re.compile(r'\{([^}]+)\}')
    
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = wildcards_dir
        self.categories = CategoryLoader(wildcards_dir)
        # Pass category loader to sub-engines for unified special file discovery
        self.tags = TagResolver(wildcards_dir, self.categories)
        self.constraints = ConstraintEngine(wildcards_dir, self.categories)
        self.modifiers = ModifierEngine(wildcards_dir, self.categories)
        self.expanders = ExpanderEngine(wildcards_dir, self.categories)
    
    def process_template(
        self,
        template: str,
        seed: int = -1,
        enable_constraints: bool = True,
        enable_modifiers: bool = True,
        enable_expanders: bool = True,
        detail_level: str = "normal"
    ) -> Dict[str, Any]:
        """
        Process a template with wildcards.
        
        Returns dict with:
            - prompt: Final resolved prompt string
            - picks: Dict of category -> ResolvedItem
            - expansions: List of expansion strings added
            - debug: Debug information
        """
        context = PromptContext(seed=seed)
        result_parts = []
        expansions = []
        debug_info = []
        
        last_end = 0
        
        for match in self.WILDCARD_PATTERN.finditer(template):
            # Add text before this match
            result_parts.append(template[last_end:match.start()])
            
            # Parse the wildcard
            wildcard = match.group(1)
            resolved = self._resolve_wildcard(
                wildcard, 
                context,
                enable_constraints,
                enable_modifiers
            )
            
            result_parts.append(resolved.value)
            context.add_pick(wildcard, resolved)
            
            debug_info.append({
                'wildcard': wildcard,
                'resolved': resolved.value,
                'path': resolved.path,
                'tags': list(resolved.tags)
            })
            
            # Get expansion if enabled
            if enable_expanders:
                expansion = self.expanders.expand(resolved.path, context)
                if expansion:
                    expansions.append(expansion)
            
            last_end = match.end()
        
        # Add remaining text
        result_parts.append(template[last_end:])
        
        # Build final prompt
        prompt = ''.join(result_parts)
        
        # Add expansions
        if expansions:
            prompt = prompt + ", " + ", ".join(expansions)
        
        return {
            'prompt': prompt,
            'picks': {k: v.value for k, v in context.picks.items()},
            'paths': {k: v.path for k, v in context.picks.items()},
            'tags': list(context.tags),
            'expansions': expansions,
            'debug': debug_info
        }
    
    def _resolve_wildcard(
        self,
        wildcard: str,
        context: PromptContext,
        enable_constraints: bool,
        enable_modifiers: bool
    ) -> ResolvedItem:
        """Resolve a single wildcard pattern"""
        
        # Parse pattern: "category" or "category:path"
        if ':' in wildcard:
            category, path = wildcard.split(':', 1)
            full_path = f"{category}.{path}" if not path.startswith(category) else path
        else:
            category = wildcard
            full_path = wildcard
        
        # Get all items under this path (returns List[Tuple[CategoryItem, str]])
        items = self.categories.get_items_at_path(full_path)
        
        if not items:
            # Fallback: return the wildcard as-is
            return ResolvedItem(
                value=f"{{{wildcard}}}",
                path=full_path,
                tags=set()
            )
        
        # Apply constraints if enabled (handles blacklist/whitelist from items)
        if enable_constraints:
            items = self.constraints.filter_items(category, items, context)
        
        # Random selection - returns (CategoryItem, category_name)
        picked_item, picked_cat = context.rng.choice(items)
        picked_path = f"{picked_cat}.{picked_item.id}" if picked_item.id else picked_cat
        
        # Use item's tags, then also check for path-based tags
        tags = set(picked_item.tags)
        tags.update(self.tags.get_tags(picked_cat))
        
        # Get the text value
        text_value = picked_item.text
        
        # Apply modifiers if enabled
        if enable_modifiers:
            text_value = self.modifiers.apply_modifiers(text_value, picked_path, context)
        
        return ResolvedItem(
            value=text_value,
            path=picked_path,
            tags=tags,
            item_id=picked_item.id,
            blacklist=picked_item.blacklist,
            whitelist=picked_item.whitelist
        )
    
    def count_combinations(self, template: str) -> int:
        """Count total possible combinations for a template"""
        total = 1
        
        for match in self.WILDCARD_PATTERN.finditer(template):
            wildcard = match.group(1)
            
            if ':' in wildcard:
                category, path = wildcard.split(':', 1)
                full_path = f"{category}.{path}" if not path.startswith(category) else path
            else:
                full_path = wildcard
            
            items = self.categories.get_items_at_path(full_path)
            if items:
                total *= len(items)
        
        return total
    
    def get_category_tree(self) -> Dict:
        """Get the full category tree structure for UI"""
        tree = {}
        
        for category_name in self.categories.list_categories():
            data = self.categories.load_category(category_name)
            tree[category_name] = data
        
        return tree


# =============================================================================
# Convenience Functions
# =============================================================================

def create_engine(wildcards_dir: Optional[str] = None) -> LunaPromptEngine:
    """Create an engine instance with default or specified wildcards directory"""
    if wildcards_dir is None:
        # Try to find wildcards directory
        try:
            import folder_paths  # type: ignore
            models_dir = getattr(folder_paths, 'models_dir', None)
            if models_dir:
                wildcards_dir = os.path.join(models_dir, 'wildcards')
        except ImportError:
            pass
    
    if wildcards_dir is None:
        wildcards_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'wildcards')
    
    return LunaPromptEngine(wildcards_dir)
