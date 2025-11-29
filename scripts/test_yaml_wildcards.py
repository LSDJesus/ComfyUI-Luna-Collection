#!/usr/bin/env python3
"""
Test script for new YAML wildcard format
Tests parsing and selection logic without ComfyUI
"""

import yaml
import random
import sys
import os
from pathlib import Path

# Try to get wildcards path from folder_paths
def get_default_wildcards_dir():
    try:
        base_dir = Path(__file__).parent.parent
        comfyui_root = base_dir.parent.parent
        sys.path.insert(0, str(comfyui_root))
        import folder_paths
        paths = folder_paths.get_folder_paths("wildcards")
        if paths:
            return paths[0]
    except ImportError:
        pass
    # Fallback to ComfyUI models directory
    return str(Path(__file__).parent.parent.parent.parent / 'models' / 'wildcards')

class YAMLWildcardTester:
    def __init__(self, yaml_dir=None, rules_file=None):
        self.yaml_dir = Path(yaml_dir or get_default_wildcards_dir())
        self.rules = self._load_rules(rules_file or self.yaml_dir / "wildcard_rules.yaml")
        self.cache = {}
    
    def _load_rules(self, rules_path):
        """Load the wildcard_rules.yaml file"""
        if Path(rules_path).exists():
            with open(rules_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_yaml(self, file_path):
        """Load a YAML wildcard file"""
        if file_path not in self.cache:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.cache[file_path] = yaml.safe_load(f)
        return self.cache[file_path]
    
    def get_group_items(self, data, group_path):
        """
        Get items from a group path like 'size.realistic'
        Returns list of item IDs
        """
        groups = data.get('groups', {})
        parts = group_path.split('.')
        
        current = groups
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, [])
            else:
                return []
        
        return current if isinstance(current, list) else []
    
    def get_items_by_tag(self, data, tag):
        """Get all items matching a tag"""
        items = data.get('items', [])
        return [item['id'] for item in items if tag in item.get('tags', [])]
    
    def select_random(self, data, selector):
        """
        Select random items based on selector
        
        Selector examples:
        - 'size.realistic' = pick from size.realistic group
        - 'size.realistic&&shape' = pick one from size.realistic AND one from shape
        - 'nsfw' = pick from items tagged 'nsfw'
        - '!size' = pick from items NOT in 'size' group
        """
        items = data.get('items', [])
        
        # Handle && (multiple groups)
        if '&&' in selector:
            parts = selector.split('&&')
            results = []
            for part in parts:
                group_items = self.get_group_items(data, part.strip())
                if group_items:
                    results.append(random.choice(group_items))
            return results
        
        # Handle ! (exclusion)
        if selector.startswith('!'):
            exclude_group = selector[1:]
            exclude_ids = set(self.get_group_items(data, exclude_group))
            available = [item['id'] for item in items if item['id'] not in exclude_ids]
            return [random.choice(available)] if available else []
        
        # Handle tag selection
        if '.' not in selector:
            # Try as tag first
            tagged = self.get_items_by_tag(data, selector)
            if tagged:
                return [random.choice(tagged)]
        
        # Handle group selection
        group_items = self.get_group_items(data, selector)
        if group_items:
            return [random.choice(group_items)]
        
        return []
    
    def resolve_to_text(self, data, item_ids):
        """Convert item IDs to their text representations"""
        items_map = {item['id']: item['text'] for item in data.get('items', [])}
        return [items_map.get(item_id, item_id) for item_id in item_ids]

def test_breast_yaml():
    """Test the breast YAML file"""
    tester = YAMLWildcardTester()
    breast_file = tester.yaml_dir / "tier2_body_breasts_v2.yaml"
    
    if not breast_file.exists():
        print(f"❌ File not found: {breast_file}")
        return
    
    print(f"✅ Loading: {breast_file}")
    data = tester.load_yaml(breast_file)
    
    print(f"\n📊 Category: {data.get('category')} / {data.get('subcategory')}")
    print(f"📊 Items: {len(data.get('items', []))}")
    print(f"📊 Groups: {list(data.get('groups', {}).keys())}")
    
    # Test cases
    tests = [
        ("size.realistic", "Random realistic breast size"),
        ("size.anime_fantasy", "Random anime/fantasy size"),
        ("shape.youthful_firm", "Random youthful firm shape"),
        ("shape.mature_natural", "Random mature natural shape"),
        ("spacing.revealing", "Random revealing spacing"),
        ("motion", "Random motion descriptor"),
        ("nipples.color", "Random nipple color"),
        ("size.realistic&&shape.youthful_firm", "Size + youthful shape combo"),
        ("size.realistic&&shape&&spacing", "Size + any shape + any spacing"),
        ("!size", "Any non-size descriptor"),
        ("nsfw", "Any NSFW-tagged item"),
    ]
    
    print("\n" + "="*70)
    print("TESTING SELECTIONS")
    print("="*70)
    
    for selector, description in tests:
        print(f"\n🎲 Test: {description}")
        print(f"   Selector: {selector}")
        
        # Generate 3 random selections
        for i in range(3):
            item_ids = tester.select_random(data, selector)
            texts = tester.resolve_to_text(data, item_ids)
            print(f"   Result {i+1}: {', '.join(texts)}")

def test_rules():
    """Test the wildcard_rules.yaml file"""
    tester = YAMLWildcardTester()
    rules_file = tester.yaml_dir / "wildcard_rules.yaml"
    
    if not rules_file.exists():
        print(f"❌ Rules file not found: {rules_file}")
        return
    
    print(f"\n✅ Loading: {rules_file}")
    rules = tester.rules
    
    print(f"\n📋 Context Requirements:")
    for context, items in rules.get('context_requirements', {}).items():
        print(f"   {context}: {len(items)} items")
    
    print(f"\n📋 Exclusions:")
    exclusions = rules.get('exclusions', [])
    print(f"   {len(exclusions)} exclusion rules")
    
    print(f"\n📋 LoRA Triggers:")
    for lora, config in rules.get('lora_triggers', {}).items():
        print(f"   {lora}: when={config.get('when')}, weight={config.get('weight')}")
    
    print(f"\n📋 Embedding Triggers:")
    for emb, config in rules.get('embedding_triggers', {}).items():
        print(f"   {emb}: when={config.get('when')}")

if __name__ == "__main__":
    print("="*70)
    print("YAML WILDCARD FORMAT TESTER")
    print("="*70)
    
    # Test breast YAML
    test_breast_yaml()
    
    # Test rules
    test_rules()
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)
