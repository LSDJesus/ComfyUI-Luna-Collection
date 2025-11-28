"""
Manual conversion assistant - I (Copilot) will analyze and convert files directly

This script helps me process wildcards by:
1. Reading the source file
2. Analyzing its structure and content
3. Converting items with proper semantic understanding
4. Writing to appropriate Luna Logic categories

You just need to add the wildcard directories to the workspace and I'll handle the rest.
"""

import yaml
import re
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict


class ManualConverter:
    def __init__(self, output_dir: Path, target_categories: List[str]):
        self.output_dir = output_dir
        self.target_categories = target_categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing category data
        self.existing_data = {}
        self.existing_ids = {}
        
        for cat in target_categories:
            yaml_path = output_dir / f"{cat}.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    self.existing_data[cat] = data
                    self.existing_ids[cat] = {item['id'] for item in data.get('items', []) if 'id' in item}
            else:
                self.existing_data[cat] = {'name': cat, 'items': []}
                self.existing_ids[cat] = set()
    
    def add_item(self, category: str, item: Dict[str, Any]) -> bool:
        """Add an item to a category"""
        if category not in self.existing_data:
            print(f"⚠ Warning: Category '{category}' not in target list, creating it...")
            self.existing_data[category] = {'name': category, 'items': []}
            self.existing_ids[category] = set()
        
        # Check for duplicate ID
        item_id = item.get('id', '')
        if item_id in self.existing_ids[category]:
            return False  # Skip duplicate
        
        # Ensure required fields
        if 'id' not in item or 'text' not in item:
            print(f"⚠ Skipping invalid item: {item}")
            return False
        
        # Set defaults
        item.setdefault('tags', [])
        item.setdefault('blacklist', [])
        item.setdefault('whitelist', [])
        item.setdefault('weight', 1.0)
        item.setdefault('payload', '')
        
        # Add to data
        self.existing_data[category]['items'].append(item)
        self.existing_ids[category].add(item_id)
        return True
    
    def save_all(self):
        """Save all categories to YAML files"""
        stats = {}
        
        for category, data in self.existing_data.items():
            if not data['items']:
                continue  # Skip empty categories
            
            yaml_path = self.output_dir / f"{category}.yaml"
            
            try:
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                
                stats[category] = len(data['items'])
            except Exception as e:
                print(f"✗ Failed to save {category}: {e}")
        
        return stats
    
    def get_stats(self) -> Dict[str, int]:
        """Get current item counts per category"""
        return {cat: len(data['items']) for cat, data in self.existing_data.items() if data['items']}


def read_dynamic_prompts_yaml(file_path: Path) -> Dict[str, Any]:
    """Read a Dynamic Prompts YAML file and return structured data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return None


def read_txt_wildcard(file_path: Path) -> List[str]:
    """Read a .txt wildcard file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return lines
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return []


def clean_dynamic_prompts_syntax(text: str) -> str:
    """Remove Dynamic Prompts syntax from text"""
    # Remove {option1|option2} - extract first option
    text = re.sub(r'\{([^}|]+)\|[^}]*\}', r'\1', text)
    # Remove weights like 2::
    text = re.sub(r'\d+::', '', text)
    # Remove __wildcards__
    text = re.sub(r'__[^_]+__', '', text)
    # Clean up extra whitespace and commas
    text = re.sub(r'\s*,\s*,\s*', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().strip(',').strip()


def make_id(text: str) -> str:
    """Generate a snake_case ID from text"""
    # Take first 60 chars
    text = text[:60]
    # Convert to lowercase
    text = text.lower()
    # Replace non-alphanumeric with underscore
    text = re.sub(r'[^a-z0-9]+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Manual conversion helper for Copilot")
    parser.add_argument("--output", "-o", required=True, help="Output directory for Luna Logic YAML")
    parser.add_argument("--list-files", "-l", help="Directory to scan for wildcard files")
    parser.add_argument("--show-stats", "-s", action="store_true", help="Show current statistics")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Target categories
    target_categories = [
        "clothing_tops", "clothing_bottoms", "clothing_full", "clothing_footwear", "clothing_materials",
        "accessories", "body_type", "body_features", "face", "expression", "eyes", "hair", "age",
        "character_types", "creatures_humanoid", "creatures_animals",
        "pose_sfw", "pose_nsfw", "action_general",
        "location_indoor", "location_outdoor", "location_scenario",
        "camera", "lighting", "composition",
        "art_style", "style_aesthetic", "colors",
        "adjectives", "details", "objects",
        "quality_tags", "negative"
    ]
    
    # Initialize converter
    converter = ManualConverter(output_dir, target_categories)
    
    if args.show_stats:
        print("\n📊 Current Statistics:")
        stats = converter.get_stats()
        if stats:
            for cat, count in sorted(stats.items()):
                print(f"  {cat}: {count} items")
            print(f"\nTotal: {sum(stats.values())} items across {len(stats)} categories")
        else:
            print("  No items yet")
        return
    
    if args.list_files:
        list_dir = Path(args.list_files)
        if not list_dir.exists():
            print(f"✗ Directory not found: {list_dir}")
            return
        
        print(f"\n📁 Scanning: {list_dir}")
        print("\nYAML files:")
        for yaml_file in sorted(list_dir.glob("**/*.yaml")):
            print(f"  {yaml_file.relative_to(list_dir)}")
        
        print("\nTXT files:")
        for txt_file in sorted(list_dir.glob("**/*.txt")):
            if txt_file.name not in ['README.txt', 'tree.txt']:
                print(f"  {txt_file.relative_to(list_dir)}")
        
        return
    
    print("\n🤖 Manual Conversion Helper Ready")
    print(f"Output: {output_dir}")
    print(f"Target categories: {len(target_categories)}")
    print("\nUse this script to help with manual conversions.")
    print("The converter object is available for programmatic use.")


if __name__ == "__main__":
    main()
