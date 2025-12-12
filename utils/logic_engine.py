"""
Luna Logic Engine - Core wildcard resolution system
No AI dependencies - pure deterministic logic
"""

import yaml
import random
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path


@dataclass
class LogicItem:
    """A single wildcard item with context-aware compatibility rules"""
    id: str
    text: str
    tags: Set[str] = field(default_factory=set)
    whitelist: Set[str] = field(default_factory=set)
    blacklist: Set[str] = field(default_factory=set)
    requires_tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    payload: str = ""
    composition: List[str] = field(default_factory=list)
    
    def is_compatible(self, current_context: Set[str]) -> bool:
        """Check if this item can be selected given the current context"""
        # Blacklist check: If ANY blacklisted tag is in context, reject
        if not self.blacklist.isdisjoint(current_context):
            return False
        
        # Whitelist check: If whitelist exists and NONE are in context, reject
        if self.whitelist and current_context.isdisjoint(self.whitelist):
            return False
        
        # Requires_tags check: ALL required tags must be in context
        if not all(tag in current_context for tag in self.requires_tags):
            return False
        
        return True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LogicItem':
        """Create LogicItem from parsed YAML dict"""
        return cls(
            id=data['id'],
            text=data['text'],
            tags=set(tag.lower() for tag in data.get('tags', [])),
            whitelist=set(tag.lower() for tag in data.get('whitelist', [])),
            blacklist=set(tag.lower() for tag in data.get('blacklist', [])),
            requires_tags=[tag.lower() for tag in data.get('requires_tags', [])],
            weight=float(data.get('weight', 1.0)),
            payload=data.get('payload', ''),
            composition=data.get('composition', [])
        )


class LunaLogicEngine:
    """Main resolution engine for context-aware wildcard replacement"""
    
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = Path(wildcards_dir)
        self.wildcards: Dict[str, List[LogicItem]] = {}
        self._load_all_wildcards()
    
    def _load_all_wildcards(self):
        """Scan wildcards directory and load all YAML files"""
        if not self.wildcards_dir.exists():
            print(f"Warning: Wildcards directory not found: {self.wildcards_dir}")
            return
        
        for yaml_file in self.wildcards_dir.glob("*.yaml"):
            try:
                self._load_wildcard_file(yaml_file)
            except Exception as e:
                print(f"Error loading {yaml_file.name}: {e}")
    
    def _load_wildcard_file(self, filepath: Path):
        """Load a single wildcard YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data or 'name' not in data or 'items' not in data:
            raise ValueError(f"Invalid YAML structure in {filepath.name}")
        
        name = data['name']
        items = [LogicItem.from_dict(item_data) for item_data in data['items']]
        
        # Check for duplicate IDs
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs found in {filepath.name}")
        
        # Detect circular dependencies in composition
        self._check_circular_dependencies(name, items)
        
        self.wildcards[name] = items
        print(f"Loaded wildcard: {name} ({len(items)} items)")
    
    def _check_circular_dependencies(self, name: str, items: List[LogicItem]):
        """Detect if any items have circular composition references (A→B→C→A)"""
        # Direct self-reference check
        for item in items:
            if name in item.composition:
                raise ValueError(f"Circular dependency detected: {name} references itself in {item.id}")
        
        # Build dependency graph for this wildcard file
        # Note: We can only fully validate after all files are loaded,
        # but we can detect cycles within what's already loaded
        self._detect_cycles_dfs(name, items)
    
    def _detect_cycles_dfs(self, name: str, items: List[LogicItem]):
        """Use DFS to detect circular dependencies in composition chains"""
        # Collect all composition references from this wildcard's items
        deps = set()
        for item in items:
            deps.update(item.composition)
        
        if not deps:
            return  # No compositions, no cycles possible
        
        # DFS cycle detection: start from each dependency and see if we can reach 'name'
        visited = set()
        rec_stack = set()
        
        def dfs(current: str, path: List[str]) -> Optional[List[str]]:
            """Returns the cycle path if found, None otherwise"""
            if current not in self.wildcards:
                return None  # Unknown wildcard, can't check further
            
            if current in rec_stack:
                # Found a cycle - return the path
                return path + [current]
            
            if current in visited:
                return None  # Already fully explored
            
            visited.add(current)
            rec_stack.add(current)
            
            # Get all dependencies of this wildcard
            for item in self.wildcards[current]:
                for dep in item.composition:
                    cycle = dfs(dep, path + [current])
                    if cycle:
                        return cycle
            
            rec_stack.remove(current)
            return None
        
        # Check if any dependency of 'name' leads back to 'name'
        # First, temporarily add the new wildcard to check against it
        temp_items = self.wildcards.get(name)
        self.wildcards[name] = items
        
        try:
            for item in items:
                for dep in item.composition:
                    cycle = dfs(dep, [name])
                    if cycle:
                        cycle_str = " → ".join(cycle)
                        raise ValueError(
                            f"Circular dependency detected in '{name}' item '{item.id}': {cycle_str}"
                        )
        finally:
            # Restore previous state
            if temp_items is None:
                del self.wildcards[name]
            else:
                self.wildcards[name] = temp_items
    
    def resolve_prompt(self, template: str, seed: int, initial_context: Optional[Set[str]] = None) -> Tuple[str, str]:
        """
        Resolve all __wildcard__ patterns in template
        
        Returns:
            (resolved_text, combined_payloads)
        """
        random.seed(seed)
        
        # Initialize context
        context = set(tag.lower() for tag in initial_context) if initial_context else set()
        
        # Track payloads separately
        payloads = []
        
        # Find all __wildcard__ patterns
        pattern = r'__([a-zA-Z0-9_]+)__'
        
        def replace_wildcard(match):
            wildcard_name = match.group(1)
            
            if wildcard_name not in self.wildcards:
                print(f"Warning: Wildcard '{wildcard_name}' not found")
                return match.group(0)  # Return unchanged
            
            # Get compatible items
            candidates = [
                item for item in self.wildcards[wildcard_name]
                if item.is_compatible(context)
            ]
            
            if not candidates:
                print(f"Warning: No compatible items for '{wildcard_name}' in current context: {context}")
                return match.group(0)  # Return unchanged
            
            # Weighted random selection
            weights = [item.weight for item in candidates]
            selected = random.choices(candidates, weights=weights, k=1)[0]
            
            # Update context with selected item's tags
            context.update(selected.tags)
            
            # Store payload if present
            if selected.payload:
                payloads.append(selected.payload)
            
            # Handle composition (recursive replacement)
            result_text = selected.text
            if selected.composition:
                # Recursively resolve any wildcards in the composed text
                result_text = re.sub(pattern, replace_wildcard, result_text)
            
            return result_text
        
        # Resolve all wildcards
        resolved = re.sub(pattern, replace_wildcard, template)
        
        # Combine payloads
        combined_payloads = " ".join(payloads)
        
        return resolved, combined_payloads
    
    def get_wildcard_names(self) -> List[str]:
        """Get list of all loaded wildcard names"""
        return list(self.wildcards.keys())
    
    def get_items_for_wildcard(self, name: str) -> List[LogicItem]:
        """Get all items for a specific wildcard"""
        return self.wildcards.get(name, [])


# Example usage and testing
if __name__ == "__main__":
    # Test with example wildcards directory
    engine = LunaLogicEngine("../../wildcards")
    
    # Test resolution
    template = "A __character__ wearing __outfit__ in __location__, __lighting__"
    initial_context = {"fantasy", "female"}
    
    resolved, payloads = engine.resolve_prompt(template, seed=12345, initial_context=initial_context)
    
    print(f"\nTemplate: {template}")
    print(f"Context: {initial_context}")
    print(f"\nResolved: {resolved}")
    print(f"Payloads: {payloads}")
