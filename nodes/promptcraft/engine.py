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
    """A single constraint rule - legacy format"""
    when_tags: Set[str]              # Context must have ANY of these tags
    prefer: List[str] = field(default_factory=list)   # Boost these paths/items
    avoid: List[str] = field(default_factory=list)    # Exclude these paths/items
    require: List[str] = field(default_factory=list)  # Must pick from these


@dataclass
class PathConstraint:
    """
    New format: source_path -> target_category -> compatible/incompatible lists
    
    Example:
        location.outdoor.beach:
          clothing:
            compatible: [10::swimwear, 10::summer, 5::nude]
            incompatible: [winter, formal, jacket]
    """
    source_path: str                 # The path that triggers this constraint
    target_category: str             # Category to filter (e.g., "clothing")
    compatible: List[Tuple[str, float]] = field(default_factory=list)   # (item, weight)
    incompatible: List[str] = field(default_factory=list)  # Items to exclude


@dataclass
class ModifierRule:
    """How an action modifies other categories"""
    trigger_tags: Set[str]           # When these tags are in context
    target_category: str             # Modify items from this category
    append_any: List[str] = field(default_factory=list)  # Append one of these
    require_tags: Set[str] = field(default_factory=set)  # Target must have these tags


@dataclass
class ExpanderRule:
    """
    Expanders: if setting:X AND expanders=true → ADD lighting:Y, item:Z
    
    Scene expansion that adds picks to OTHER categories.
    
    Example YAML:
        setting.location.outdoor.beach:
          lighting:
            paths: [lighting.natural.golden_hour, lighting.natural.bright_daylight]
            weight: 0.8  # 80% chance to add lighting
          props:
            paths: [item.beach.towel, item.beach.umbrella, item.drink.cocktail]
            count: [0, 2]  # add 0-2 props
          atmosphere:
            values: ["tropical atmosphere", "summer vibes", "relaxing mood"]
            count: [1, 2]
    """
    source_path: str                                        # Trigger path
    expansions: Dict[str, Dict] = field(default_factory=dict)  # category -> expansion config


# =============================================================================
# Category Loader
# =============================================================================

# Special file types identified by suffix (preferred) or prefix
# Suffix format: name.type.yaml (e.g., location.constraint.yaml, action.modifier.yaml)
# Prefix format: type_name.yaml (e.g., rules_location.yaml) - legacy support
SPECIAL_FILE_SUFFIXES = {
    '.constraint': 'constraints',   # location.constraint.yaml
    '.modifier': 'modifiers',       # action.modifier.yaml
    '.expander': 'expanders',       # location.expander.yaml
    '.chain': 'chains',             # outfit.chain.yaml
    '.tags': 'tags',                # master.tags.yaml
    '.lora': 'lora',                # character.lora.yaml
}

SPECIAL_FILE_PREFIXES = {
    'rules_': 'constraints',        # rules_clothing.yaml (legacy)
    'expanders_': 'expanders',      # expanders_location.yaml (legacy)
    'modifiers_': 'modifiers',      # modifiers_action.yaml (legacy)
    'lora_': 'lora',                # lora_characters.yaml (legacy)
    'tags_': 'tags',                # tags_clothing.yaml (legacy)
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
            'constraints': [],
            'expanders': [],
            'modifiers': [],
            'chains': [],
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
            
            # Check if this is a special file by SUFFIX first (preferred)
            # e.g., "location.constraint.yaml" -> stem is "location.constraint"
            stem = filepath.stem
            is_special = False
            
            for suffix, file_type in SPECIAL_FILE_SUFFIXES.items():
                if stem.endswith(suffix) or f"_{suffix[1:]}" in stem:
                    # Matches .constraint or _constraint
                    self._special_files[file_type].append(filepath)
                    is_special = True
                    break
            
            # Fallback: check by PREFIX (legacy support)
            if not is_special:
                for prefix, file_type in SPECIAL_FILE_PREFIXES.items():
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
    """
    Filters items based on context-aware rules.
    
    Supports two formats:
    1. New format (path-based): source_path -> target -> compatible/incompatible
       location.outdoor.beach:
         clothing:
           compatible: [10::swimwear, 5::summer]
           incompatible: [winter, formal]
    
    2. Legacy format (tag-based): category -> context_preferences -> when/prefer/avoid
    """
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._path_constraints: Dict[str, List[PathConstraint]] = {}  # source_path -> constraints
        self._legacy_rules: Dict[str, List[ConstraintRule]] = {}  # category -> rules
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load constraint rules"""
        if self._loaded and not force:
            return
        
        self._path_constraints.clear()
        self._legacy_rules.clear()
        
        # Use category loader's special file discovery
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('constraints'):
                self._load_constraint_file(filepath)
        else:
            # Fallback to rules/ subdirectory
            rules_path = self.wildcards_dir / "rules"
            if rules_path.exists():
                for filepath in rules_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml', '.json'):
                        if 'constraint' in filepath.stem:
                            self._load_constraint_file(filepath)
        
        self._loaded = True
    
    def _parse_weighted_item(self, item: str) -> Tuple[str, float]:
        """Parse 'weight::item' format, returns (item, weight)"""
        if '::' in item:
            parts = item.split('::', 1)
            try:
                weight = float(parts[0])
                return (parts[1], weight)
            except ValueError:
                return (item, 1.0)
        return (item, 1.0)
    
    def _load_constraint_file(self, filepath: Path):
        """Load constraints from a YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        # Parse each source path
        for source_path, targets in data.items():
            # Skip comment keys or metadata
            if source_path.startswith('#') or source_path in ('name', 'description', 'enabled'):
                continue
            
            if not isinstance(targets, dict):
                continue
            
            # Each target category (clothing, action, lighting, etc.)
            for target_cat, rules in targets.items():
                if not isinstance(rules, dict):
                    continue
                
                # Parse compatible list with weights
                compatible = []
                for item in rules.get('compatible', []):
                    compatible.append(self._parse_weighted_item(str(item)))
                
                # Parse incompatible list (no weights needed)
                incompatible = [str(i) for i in rules.get('incompatible', [])]
                
                constraint = PathConstraint(
                    source_path=source_path,
                    target_category=target_cat,
                    compatible=compatible,
                    incompatible=incompatible
                )
                
                if source_path not in self._path_constraints:
                    self._path_constraints[source_path] = []
                self._path_constraints[source_path].append(constraint)
    
    def get_constraints_for_context(self, context: PromptContext, target_category: str) -> List[PathConstraint]:
        """Get all constraints that apply based on current picks"""
        self.load_rules()
        
        applicable = []
        
        # Check each pick in context against our constraint rules
        for pick_category, resolved in context.picks.items():
            picked_path = resolved.path
            
            # Check for exact match and parent path matches
            parts = picked_path.split('.')
            for i in range(len(parts), 0, -1):
                check_path = '.'.join(parts[:i])
                if check_path in self._path_constraints:
                    for constraint in self._path_constraints[check_path]:
                        if constraint.target_category == target_category:
                            applicable.append(constraint)
        
        return applicable
    
    def filter_items(
        self, 
        category: str, 
        items: List[Tuple[CategoryItem, str]], 
        context: PromptContext
    ) -> List[Tuple[CategoryItem, str]]:
        """
        Filter items based on current context.
        Uses path constraints and item blacklist/whitelist from YAML.
        """
        self.load_rules()
        
        filtered: List[Tuple[CategoryItem, str]] = []
        
        # First pass: basic blacklist/whitelist from items themselves
        for item, cat_name in items:
            # Check item's blacklist - if context has ANY blacklisted tag, exclude
            if item.blacklist and item.blacklist.intersection(context.tags):
                continue
            
            # Check item's whitelist - if specified, context must have at least one
            if item.whitelist and not item.whitelist.intersection(context.tags):
                continue
            
            filtered.append((item, cat_name))
        
        # Second pass: apply path-based constraints
        constraints = self.get_constraints_for_context(context, category)
        
        if constraints:
            # Collect all incompatible items
            all_incompatible: Set[str] = set()
            all_compatible: Dict[str, float] = {}  # item -> weight
            
            for constraint in constraints:
                all_incompatible.update(constraint.incompatible)
                for item_name, weight in constraint.compatible:
                    # Take highest weight if item appears multiple times
                    if item_name not in all_compatible or weight > all_compatible[item_name]:
                        all_compatible[item_name] = weight
            
            # Filter out incompatible items
            if all_incompatible:
                filtered = [
                    (item, cat) for item, cat in filtered
                    if not any(
                        inc.lower() in item.id.lower() or 
                        inc.lower() in item.text.lower() or
                        inc.lower() in cat.lower()
                        for inc in all_incompatible
                    )
                ]
            
            # If we have compatible items specified, prefer those
            if all_compatible and filtered:
                compatible_items = [
                    (item, cat) for item, cat in filtered
                    if any(
                        comp.lower() in item.id.lower() or
                        comp.lower() in item.text.lower() or
                        comp.lower() in cat.lower()
                        for comp in all_compatible.keys()
                    )
                ]
                # Only use compatible filter if it leaves us with results
                if compatible_items:
                    # Update weights on matched items
                    for item, cat in compatible_items:
                        for comp, weight in all_compatible.items():
                            if comp.lower() in item.id.lower() or comp.lower() in item.text.lower():
                                item.weight = item.weight * weight
                    filtered = compatible_items
        
        return filtered if filtered else items  # Fallback to original if too restrictive


# =============================================================================
# Modifier Engine
# =============================================================================

@dataclass
class PathModifier:
    """
    Modifiers: if X AND Y then ALSO Z
    
    When a trigger path is picked, modify target category items.
    
    Example YAML:
        action.intimate.sex:
          clothing:
            modifiers: ["pulled aside", "pulled down", "lifted up"]
            compatible: [bikini, underwear, dress]  # only modify these
            incompatible: [formal, pristine]        # never modify these
            allow_states: [nude, partially_clothed] # context hint
    """
    source_path: str                                    # Trigger path (e.g., action.intimate.sex)
    target_category: str                                # What to modify (e.g., clothing)
    modifiers: List[str] = field(default_factory=list)  # Text to append
    compatible: List[str] = field(default_factory=list) # Only modify items matching these
    incompatible: List[str] = field(default_factory=list)  # Never modify items matching these
    allow_states: List[str] = field(default_factory=list)  # Hint for state (not used in modifier)


class ModifierEngine:
    """Applies action-based modifications to resolved items"""
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._path_modifiers: Dict[str, List[PathModifier]] = {}  # source_path -> modifiers
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load modifier rules"""
        if self._loaded and not force:
            return
        
        self._path_modifiers.clear()
        
        # Use category loader's special file discovery
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('modifiers'):
                self._load_modifier_file(filepath)
        else:
            # Fallback to rules/ subdirectory
            rules_path = self.wildcards_dir / "rules"
            if rules_path.exists():
                for filepath in rules_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml') and 'modifier' in filepath.stem:
                        self._load_modifier_file(filepath)
        
        self._loaded = True
    
    def _load_modifier_file(self, filepath: Path):
        """Load modifiers from a YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        # Parse each source path
        for source_path, targets in data.items():
            # Skip comment keys or metadata
            if source_path.startswith('#') or source_path in ('name', 'description', 'enabled'):
                continue
            
            if not isinstance(targets, dict):
                continue
            
            # Each target category
            for target_cat, rules in targets.items():
                if not isinstance(rules, dict):
                    continue
                
                modifier = PathModifier(
                    source_path=source_path,
                    target_category=target_cat,
                    modifiers=rules.get('modifiers', []),
                    compatible=rules.get('compatible', []),
                    incompatible=rules.get('incompatible', []),
                    allow_states=rules.get('allow_states', [])
                )
                
                if source_path not in self._path_modifiers:
                    self._path_modifiers[source_path] = []
                self._path_modifiers[source_path].append(modifier)
    
    def get_modifiers_for_context(self, context: PromptContext, target_category: str) -> List[PathModifier]:
        """Get all modifiers that apply based on current picks"""
        self.load_rules()
        
        applicable = []
        
        for pick_category, resolved in context.picks.items():
            picked_path = resolved.path
            
            # Check for exact match and parent path matches
            parts = picked_path.split('.')
            for i in range(len(parts), 0, -1):
                check_path = '.'.join(parts[:i])
                if check_path in self._path_modifiers:
                    for modifier in self._path_modifiers[check_path]:
                        if modifier.target_category == target_category:
                            applicable.append(modifier)
        
        return applicable
    
    def apply_modifiers(
        self, 
        value: str, 
        path: str, 
        context: PromptContext
    ) -> str:
        """
        Apply relevant modifiers to a resolved value.
        
        Logic: if trigger_path is in context AND target matches, append modifier.
        
        Args:
            value: The picked text value (e.g., "red bikini")
            path: The path of the picked item (e.g., "clothing.swimwear.bikini")
            context: Current prompt context with all picks
        
        Returns:
            Modified value with appended modifier text
        """
        self.load_rules()
        
        # Get target category from path (first segment)
        target_cat = path.split('.')[0] if '.' in path else path
        
        # Find all modifiers that target this category
        modifiers = self.get_modifiers_for_context(context, target_cat)
        
        if not modifiers:
            return value
        
        # Check each modifier to see if it should apply
        for mod in modifiers:
            if not mod.modifiers:
                continue
            
            # Check incompatible - if value matches any, skip this modifier
            if mod.incompatible:
                is_incompatible = any(
                    inc.lower() in value.lower() or inc.lower() in path.lower()
                    for inc in mod.incompatible
                )
                if is_incompatible:
                    continue
            
            # Check compatible - if specified, value must match at least one
            if mod.compatible:
                is_compatible = any(
                    comp.lower() in value.lower() or comp.lower() in path.lower()
                    for comp in mod.compatible
                )
                if not is_compatible:
                    continue
            
            # Apply a random modifier from the list
            modifier_text = context.rng.choice(mod.modifiers)
            value = f"{value}, {modifier_text}"
            break  # Only apply one modifier per item (first match wins)
        
        return value


# =============================================================================
# Expander Engine
# =============================================================================

@dataclass
class Expansion:
    """Result of expanding a context"""
    category: str           # Target category (e.g., "lighting", "props")
    value: str              # The expanded value
    path: str = ""          # Path if from wildcard resolution
    is_text: bool = False   # True if raw text, False if resolved from path


class ExpanderEngine:
    """
    Adds contextual scene details to OTHER categories.
    
    Unlike modifiers (which mutate picked values), expanders ADD new picks.
    
    Example: picking "beach" can auto-add:
      - lighting: golden_hour (from lighting category)
      - props: beach_towel, umbrella (from item category)
      - atmosphere: "tropical vibes" (raw text)
    """
    
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
        """
        Load expander definitions from YAML.
        
        Supports two formats:
        
        1. New format (category-based):
            setting.location.outdoor.beach:
              lighting:
                paths: [lighting.natural.golden_hour]
                weight: 0.8
              props:
                paths: [item.beach.towel, item.beach.umbrella]
                count: [0, 2]
              atmosphere:
                values: ["tropical vibes", "summer mood"]
                count: [1, 1]
        
        2. Legacy format (details/atmosphere lists):
            location.outdoor.beach:
              details: ["soft sand", "ocean waves"]
              atmosphere: ["tropical atmosphere"]
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        for source_path, exp_data in data.items():
            # Skip non-dict entries (comments, metadata)
            if not isinstance(exp_data, dict):
                continue
            if source_path.startswith('#'):
                continue
            
            expansions: Dict[str, Dict] = {}
            
            # Check for new format (has category keys with paths/values)
            has_new_format = any(
                isinstance(v, dict) and ('paths' in v or 'values' in v)
                for v in exp_data.values()
            )
            
            if has_new_format:
                # New format: each key is a target category
                for target_cat, config in exp_data.items():
                    if not isinstance(config, dict):
                        continue
                    expansions[target_cat] = {
                        'paths': config.get('paths', []),
                        'values': config.get('values', []),
                        'count': config.get('count', [1, 1]),
                        'weight': config.get('weight', 1.0),
                    }
            else:
                # Legacy format: details and atmosphere lists become "scene_details"
                details = list(exp_data.get('details', []))
                atmosphere = exp_data.get('atmosphere', [])
                if details or atmosphere:
                    all_values = details + atmosphere
                    count = exp_data.get('count', [1, 3])
                    expansions['scene_details'] = {
                        'paths': [],
                        'values': all_values,
                        'count': count if isinstance(count, list) else [count, count],
                        'weight': 1.0,
                    }
            
            if expansions:
                rule = ExpanderRule(
                    source_path=source_path,
                    expansions=expansions
                )
                self._rules[source_path] = rule
    
    def get_expansions(self, context: PromptContext) -> List[Expansion]:
        """
        Get all expansions that should be added based on current context.
        
        Returns list of Expansion objects to add to the prompt.
        """
        self.load_rules()
        
        all_expansions: List[Expansion] = []
        
        # Check each pick in context against our expander rules
        for pick_category, resolved in context.picks.items():
            picked_path = resolved.path
            
            # Find matching expander (exact or parent path match)
            rule = self._find_matching_rule(picked_path)
            if not rule:
                continue
            
            # Process each expansion category
            for target_cat, config in rule.expansions.items():
                # Check weight (probability to apply)
                weight = config.get('weight', 1.0)
                if weight < 1.0 and context.rng.random() > weight:
                    continue
                
                # Determine count
                count_range = config.get('count', [1, 1])
                if isinstance(count_range, list) and len(count_range) >= 2:
                    min_count, max_count = int(count_range[0]), int(count_range[1])
                elif isinstance(count_range, (int, float)):
                    min_count = max_count = int(count_range)
                else:
                    min_count = max_count = 1
                
                count = context.rng.randint(min_count, max_count)
                if count == 0:
                    continue
                
                # Collect candidates from paths and values
                candidates: List[Tuple[str, bool]] = []  # (value, is_path)
                
                for path in config.get('paths', []):
                    candidates.append((path, True))
                for value in config.get('values', []):
                    candidates.append((value, False))
                
                if not candidates:
                    continue
                
                # Pick random candidates
                selected = context.rng.sample(
                    candidates,
                    min(count, len(candidates))
                )
                
                for item, is_path in selected:
                    if is_path:
                        # Resolve from category path
                        resolved_value = self._resolve_path(item, context)
                        if resolved_value:
                            all_expansions.append(Expansion(
                                category=target_cat,
                                value=resolved_value,
                                path=item,
                                is_text=False
                            ))
                    else:
                        # Raw text value
                        all_expansions.append(Expansion(
                            category=target_cat,
                            value=item,
                            path="",
                            is_text=True
                        ))
        
        return all_expansions
    
    def _find_matching_rule(self, path: str) -> Optional[ExpanderRule]:
        """Find expander rule matching path (exact or parent match)"""
        parts = path.split('.')
        
        for i in range(len(parts), 0, -1):
            check_path = '.'.join(parts[:i])
            if check_path in self._rules:
                return self._rules[check_path]
        
        return None
    
    def _resolve_path(self, path: str, context: PromptContext) -> str:
        """Resolve a category path to a value"""
        if not self.category_loader:
            return ""
        
        items = self.category_loader.get_items_at_path(path)
        if not items:
            return ""
        
        # Pick random item
        item, _ = context.rng.choice(items)
        return item.text
    
    def expand(self, path: str, context: PromptContext) -> str:
        """
        Legacy method - get expansion as comma-separated string.
        
        For backward compatibility. New code should use get_expansions().
        """
        expansions = self.get_expansions(context)
        if not expansions:
            return ""
        
        return ", ".join(exp.value for exp in expansions)


# =============================================================================
# LoRA Linker Engine
# =============================================================================

@dataclass
class LoRALink:
    """A LoRA suggestion linked to a pick"""
    lora_name: str                  # Filename (without path)
    model_weight: float = 1.0       # Weight for model
    clip_weight: float = 1.0        # Weight for CLIP
    trigger_words: List[str] = field(default_factory=list)  # Trigger words to add
    source_path: str = ""           # What pick triggered this
    priority: int = 0               # Higher = more important (for stacking)


@dataclass
class EmbeddingLink:
    """An embedding suggestion linked to a pick"""
    embedding_name: str             # Filename (without path)
    weight: float = 1.0             # Embedding weight
    trigger_format: str = ""        # How to format in prompt (e.g., "(embedding:name:1.0)")
    source_path: str = ""           # What pick triggered this


@dataclass
class LoRARule:
    """
    LoRA Linker: when a path is picked, suggest LoRAs/embeddings
    
    Example YAML:
        character.fantasy.elf:
          loras:
            - name: "elf_ears_v2.safetensors"
              model_weight: 0.7
              clip_weight: 0.7
              trigger: "elf ears, pointed ears"
              priority: 10
            - name: "fantasy_style.safetensors"
              model_weight: 0.5
              clip_weight: 0.5
          embeddings:
            - name: "elvish_beauty"
              weight: 0.8
        
        style.anime:
          loras:
            - name: "anime_style_v3.safetensors"
              model_weight: 0.8
              clip_weight: 0.8
    """
    source_path: str
    loras: List[Dict] = field(default_factory=list)
    embeddings: List[Dict] = field(default_factory=list)


class LoRALinker:
    """
    Links wildcard picks to LoRAs and embeddings.
    
    When you pick "elf" from characters, this can suggest:
    - LoRAs: elf_ears.safetensors at weight 0.7
    - Embeddings: elvish_beauty at weight 0.8
    - Trigger words to add to the prompt
    """
    
    def __init__(self, wildcards_dir: str, category_loader: Optional['CategoryLoader'] = None):
        self.wildcards_dir = Path(wildcards_dir)
        self.category_loader = category_loader
        self._rules: Dict[str, LoRARule] = {}
        self._loaded = False
    
    def load_rules(self, force: bool = False):
        """Load LoRA linking rules"""
        if self._loaded and not force:
            return
        
        self._rules.clear()
        
        # Use category loader's special file discovery
        if self.category_loader:
            for filepath in self.category_loader.get_special_files('lora'):
                self._load_lora_file(filepath)
        else:
            # Fallback to rules/ subdirectory
            rules_path = self.wildcards_dir / "rules"
            if rules_path.exists():
                for filepath in rules_path.iterdir():
                    if filepath.suffix in ('.yaml', '.yml') and 'lora' in filepath.stem:
                        self._load_lora_file(filepath)
        
        self._loaded = True
    
    def _load_lora_file(self, filepath: Path):
        """Load LoRA rules from YAML"""
        with open(filepath, 'r', encoding='utf-8') as f:
            if not HAS_YAML:
                return
            data = yaml.safe_load(f) or {}
        
        for source_path, config in data.items():
            # Skip comments/metadata
            if source_path.startswith('#') or source_path in ('name', 'description', 'enabled'):
                continue
            if not isinstance(config, dict):
                continue
            
            rule = LoRARule(
                source_path=source_path,
                loras=config.get('loras', []),
                embeddings=config.get('embeddings', [])
            )
            self._rules[source_path] = rule
    
    def get_links(self, context: PromptContext) -> Tuple[List[LoRALink], List[EmbeddingLink]]:
        """
        Get all LoRA and embedding links based on current context.
        
        Returns:
            Tuple of (lora_links, embedding_links)
        """
        self.load_rules()
        
        lora_links: List[LoRALink] = []
        embedding_links: List[EmbeddingLink] = []
        seen_loras: Set[str] = set()
        seen_embeddings: Set[str] = set()
        
        # Check each pick against our rules
        for pick_category, resolved in context.picks.items():
            picked_path = resolved.path
            
            # Find matching rule (exact or parent match)
            rule = self._find_matching_rule(picked_path)
            if not rule:
                continue
            
            # Process LoRAs
            for lora_config in rule.loras:
                name = lora_config.get('name', '')
                if not name or name in seen_loras:
                    continue
                
                seen_loras.add(name)
                
                # Parse trigger words
                trigger = lora_config.get('trigger', '')
                trigger_words = [t.strip() for t in trigger.split(',')] if trigger else []
                
                link = LoRALink(
                    lora_name=name,
                    model_weight=float(lora_config.get('model_weight', 1.0)),
                    clip_weight=float(lora_config.get('clip_weight', 1.0)),
                    trigger_words=trigger_words,
                    source_path=picked_path,
                    priority=int(lora_config.get('priority', 0))
                )
                lora_links.append(link)
            
            # Process embeddings
            for emb_config in rule.embeddings:
                name = emb_config.get('name', '')
                if not name or name in seen_embeddings:
                    continue
                
                seen_embeddings.add(name)
                
                link = EmbeddingLink(
                    embedding_name=name,
                    weight=float(emb_config.get('weight', 1.0)),
                    trigger_format=emb_config.get('format', ''),
                    source_path=picked_path
                )
                embedding_links.append(link)
        
        # Sort LoRAs by priority (highest first)
        lora_links.sort(key=lambda x: x.priority, reverse=True)
        
        return lora_links, embedding_links
    
    def _find_matching_rule(self, path: str) -> Optional[LoRARule]:
        """Find rule matching path (exact or parent match)"""
        parts = path.split('.')
        
        for i in range(len(parts), 0, -1):
            check_path = '.'.join(parts[:i])
            if check_path in self._rules:
                return self._rules[check_path]
        
        return None
    
    def get_lora_stack(self, context: PromptContext) -> List[Tuple[str, float, float]]:
        """
        Get LoRA stack in ComfyUI format: List[(name, model_weight, clip_weight)]
        
        Compatible with Luna LoRA Stacker and Impact Pack's Apply LoRA Stack.
        """
        lora_links, _ = self.get_links(context)
        return [(link.lora_name, link.model_weight, link.clip_weight) for link in lora_links]
    
    def get_trigger_words(self, context: PromptContext) -> List[str]:
        """Get all trigger words from linked LoRAs"""
        lora_links, _ = self.get_links(context)
        triggers = []
        for link in lora_links:
            triggers.extend(link.trigger_words)
        return triggers
    
    def format_embeddings(self, context: PromptContext, style: str = "a1111") -> List[str]:
        """
        Format embeddings for prompt injection.
        
        Styles:
            - a1111: (embedding:name:weight)
            - comfy: embedding:name
            - raw: just the name
        """
        _, embedding_links = self.get_links(context)
        formatted = []
        
        for link in embedding_links:
            if link.trigger_format:
                # Use custom format if specified
                formatted.append(link.trigger_format)
            elif style == "a1111":
                if link.weight != 1.0:
                    formatted.append(f"(embedding:{link.embedding_name}:{link.weight})")
                else:
                    formatted.append(f"embedding:{link.embedding_name}")
            elif style == "comfy":
                formatted.append(f"embedding:{link.embedding_name}")
            else:
                formatted.append(link.embedding_name)
        
        return formatted


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
        self.lora_linker = LoRALinker(wildcards_dir, self.categories)
    
    def process_template(
        self,
        template: str,
        seed: int = -1,
        enable_constraints: bool = True,
        enable_modifiers: bool = True,
        enable_expanders: bool = True,
        enable_lora_links: bool = True,
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
        
        # Get LoRA/embedding links if enabled
        lora_stack = []
        trigger_words = []
        embeddings = []
        
        if enable_lora_links:
            lora_links, embedding_links = self.lora_linker.get_links(context)
            lora_stack = [(l.lora_name, l.model_weight, l.clip_weight) for l in lora_links]
            
            # Collect trigger words
            for link in lora_links:
                trigger_words.extend(link.trigger_words)
            
            # Format embeddings
            embeddings = self.lora_linker.format_embeddings(context)
            
            # Optionally add trigger words to prompt
            if trigger_words:
                prompt = prompt + ", " + ", ".join(trigger_words)
        
        return {
            'prompt': prompt,
            'picks': {k: v.value for k, v in context.picks.items()},
            'paths': {k: v.path for k, v in context.picks.items()},
            'tags': list(context.tags),
            'expansions': expansions,
            'lora_stack': lora_stack,
            'embeddings': embeddings,
            'trigger_words': trigger_words,
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
