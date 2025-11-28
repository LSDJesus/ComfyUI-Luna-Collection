#!/usr/bin/env python3
"""
Test script for new hierarchical YAML wildcard format
Tests parsing, path resolution, and random selection
"""

import yaml
import random
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

class HierarchicalWildcardParser:
    """Parser for the new hierarchical YAML wildcard format"""
    
    def __init__(self, yaml_dir: str = "D:/AI/SD Models/wildcards_atomic"):
        self.yaml_dir = Path(yaml_dir)
        self.cache: Dict[str, dict] = {}
        self.errors: List[str] = []
    
    def load_yaml(self, filename: str) -> Optional[dict]:
        """Load a YAML file and cache it"""
        filepath = self.yaml_dir / filename
        if not filepath.exists():
            self.errors.append(f"File not found: {filepath}")
            return None
        
        if filename not in self.cache:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.cache[filename] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.errors.append(f"YAML parse error in {filename}: {e}")
                return None
        
        return self.cache[filename]
    
    def get_by_path(self, data: dict, path: str) -> Any:
        """
        Navigate nested dict by dot-separated path
        Example: 'hair.color.natural' -> data['hair']['color']['natural']
        """
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
        """
        Flatten any nested structure to a list of strings
        - List: return as-is
        - Dict: recursively flatten all values
        - String: wrap in list
        """
        if data is None:
            return []
        
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    result.extend(self.flatten_to_list(item))
                elif isinstance(item, list):
                    result.extend(self.flatten_to_list(item))
            return result
        
        if isinstance(data, dict):
            result = []
            for key, value in data.items():
                # Skip special keys
                if key in ['templates', 'usage_notes', 'category', 'description', '_suffix']:
                    continue
                # Skip suffix definitions (strings with [*])
                if isinstance(value, str) and '[*]' in value:
                    continue
                result.extend(self.flatten_to_list(value))
            return result
        
        if isinstance(data, str):
            # Skip suffix templates
            if '[*]' in data:
                return []
            return [data]
        
        return []
    
    def get_suffix(self, data: dict, category: str) -> str:
        """Get the suffix from _suffix key in a category"""
        if category in data:
            cat_data = data[category]
            if isinstance(cat_data, dict) and '_suffix' in cat_data:
                return cat_data['_suffix']
        return ""
    
    def select_random(self, filename: str, path: str = "") -> List[str]:
        """
        Select random items from a YAML file
        
        Examples:
        - select_random('body.yaml', 'hair.color.natural') -> ['black']
        - select_random('body.yaml', 'hair') -> random from all hair items
        """
        data = self.load_yaml(filename)
        if data is None:
            return []
        
        # Get the target data
        if path:
            target = self.get_by_path(data, path)
        else:
            target = data
        
        if target is None:
            self.errors.append(f"Path not found: {path} in {filename}")
            return []
        
        # Flatten to list and pick random
        items = self.flatten_to_list(target)
        if not items:
            return []
        
        return [random.choice(items)]
    
    def resolve_with_suffix(self, filename: str, path: str) -> str:
        """Select random and apply suffix if applicable"""
        data = self.load_yaml(filename)
        if data is None:
            return ""
        
        # Get the category (first part of path)
        category = path.split('.')[0] if path else ""
        suffix = self.get_suffix(data, category)
        
        # Get random selection
        items = self.select_random(filename, path)
        if not items:
            return ""
        
        result = ' '.join(items)
        if suffix:
            result = f"{result}{suffix}"
        
        return result
    
    def get_structure(self, data: dict, prefix: str = "", depth: int = 0) -> List[str]:
        """Get the structure of a YAML file for display"""
        lines = []
        indent = "    " * depth
        
        for key, value in data.items():
            # Skip special keys
            if key in ['templates', 'usage_notes', 'category', 'description', '_suffix']:
                continue
            
            path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, str) and '[*]' in value:
                lines.append(f"{indent}{key}: {value}")
            elif isinstance(value, dict):
                suffix_note = ""
                if '_suffix' in value:
                    suffix_note = f" (suffix: '{value['_suffix']}')"
                lines.append(f"{indent}{key}:{suffix_note}")
                lines.extend(self.get_structure(value, path, depth + 1))
            elif isinstance(value, list):
                count = len(value)
                lines.append(f"{indent}{key}: [{count} items]")
        
        return lines


def test_yaml_file(parser: HierarchicalWildcardParser, filename: str):
    """Test a single YAML file"""
    print(f"\n{'='*70}")
    print(f"TESTING: {filename}")
    print('='*70)
    
    data = parser.load_yaml(filename)
    if data is None:
        print(f"❌ Failed to load {filename}")
        for error in parser.errors:
            print(f"   {error}")
        parser.errors.clear()
        return False
    
    print(f"✅ Loaded successfully")
    print(f"   Category: {data.get('category', 'N/A')}")
    
    # Show structure
    print(f"\n📊 Structure:")
    structure = parser.get_structure(data)
    for line in structure[:20]:  # First 20 lines
        print(f"   {line}")
    if len(structure) > 20:
        print(f"   ... and {len(structure) - 20} more lines")
    
    # Show templates if present
    templates = data.get('templates', {})
    if templates:
        print(f"\n📝 Templates:")
        for name, tpls in templates.items():
            if isinstance(tpls, list):
                print(f"   {name}: {len(tpls)} templates")
    
    return True


def test_selections(parser: HierarchicalWildcardParser):
    """Test random selections from various paths"""
    print(f"\n{'='*70}")
    print("TESTING RANDOM SELECTIONS")
    print('='*70)
    
    test_cases = [
        # (filename, path, description)
        ("body.yaml", "hair.length", "Hair length"),
        ("body.yaml", "hair.color.natural", "Natural hair color"),
        ("body.yaml", "hair.color.fantasy", "Fantasy hair color"),
        ("body.yaml", "eyes.color.natural.warm", "Warm eye color"),
        ("body.yaml", "breasts.size.realistic", "Realistic breast size"),
        ("body.yaml", "skin.tone.pale", "Pale skin tone"),
        ("clothing.yaml", "tops.types.casual", "Casual top"),
        ("clothing.yaml", "bottoms.types.skirts", "Skirt type"),
        ("clothing.yaml", "lingerie.types.bras", "Bra type"),
        ("pose.yaml", "posture.type.standing.casual", "Casual standing pose"),
        ("pose.yaml", "seductive.poses.teasing", "Teasing pose"),
        ("setting.yaml", "location.indoor.residential", "Residential location"),
        ("setting.yaml", "genre.fantasy.high_fantasy", "Fantasy setting"),
        ("lighting.yaml", "type.natural.sunlight", "Natural sunlight"),
        ("lighting.yaml", "type.dramatic.atmospheric", "Dramatic lighting"),
        ("expression.yaml", "emotion.positive.happy", "Happy emotion"),
        ("expression.yaml", "seductive.sultry", "Sultry expression"),
        ("composition.yaml", "shot_type.distance.close", "Close shot"),
        ("composition.yaml", "angle.vertical", "Vertical angle"),
        ("action.yaml", "activity.everyday.casual", "Casual activity"),
        ("action.yaml", "intimate.teasing", "Teasing action"),
    ]
    
    for filename, path, description in test_cases:
        print(f"\n🎲 {description}")
        print(f"   Path: {filename} -> {path}")
        
        for i in range(3):
            result = parser.resolve_with_suffix(filename, path)
            if result:
                print(f"   Result {i+1}: {result}")
            else:
                print(f"   Result {i+1}: ❌ No result")
                if parser.errors:
                    print(f"   Errors: {parser.errors}")
                    parser.errors.clear()


def test_full_composition(parser: HierarchicalWildcardParser):
    """Test composing a full prompt from multiple files"""
    print(f"\n{'='*70}")
    print("TESTING FULL COMPOSITION")
    print('='*70)
    
    for i in range(5):
        print(f"\n🖼️ Composition {i+1}:")
        
        parts = []
        
        # Body attributes
        hair_color = parser.resolve_with_suffix("body.yaml", "hair.color.natural")
        hair_length = parser.resolve_with_suffix("body.yaml", "hair.length")
        eye_color = parser.resolve_with_suffix("body.yaml", "eyes.color.natural")
        skin = parser.resolve_with_suffix("body.yaml", "skin.tone")
        
        # Clothing
        top = parser.resolve_with_suffix("clothing.yaml", "tops.types.casual")
        bottom = parser.resolve_with_suffix("clothing.yaml", "bottoms.types.skirts")
        
        # Pose
        pose = parser.resolve_with_suffix("pose.yaml", "posture.type.standing")
        
        # Setting
        location = parser.resolve_with_suffix("setting.yaml", "location.indoor")
        
        # Lighting
        lighting = parser.resolve_with_suffix("lighting.yaml", "type.natural")
        
        # Expression
        expression = parser.resolve_with_suffix("expression.yaml", "emotion.positive")
        
        # Composition
        shot = parser.resolve_with_suffix("composition.yaml", "shot_type.distance.medium")
        
        prompt = f"a woman with {hair_length} {hair_color}, {eye_color}, {skin}, wearing {top} and {bottom}, {pose}, {expression}, in {location}, {lighting}, {shot}"
        
        print(f"   {prompt}")


def main():
    print("="*70)
    print("HIERARCHICAL YAML WILDCARD TESTER")
    print("="*70)
    
    parser = HierarchicalWildcardParser()
    
    # Test each YAML file
    yaml_files = [
        "body.yaml",
        "clothing.yaml", 
        "pose.yaml",
        "setting.yaml",
        "lighting.yaml",
        "expression.yaml",
        "composition.yaml",
        "action.yaml",
    ]
    
    all_passed = True
    for filename in yaml_files:
        if not test_yaml_file(parser, filename):
            all_passed = False
    
    if all_passed:
        # Test random selections
        test_selections(parser)
        
        # Test full composition
        test_full_composition(parser)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    if parser.errors:
        print(f"❌ {len(parser.errors)} errors encountered:")
        for error in parser.errors:
            print(f"   - {error}")
    else:
        print("✅ All tests passed!")
    
    print(f"\n📁 Files tested: {len(yaml_files)}")
    print(f"📊 Files in cache: {len(parser.cache)}")


if __name__ == "__main__":
    main()
