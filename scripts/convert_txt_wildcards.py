#!/usr/bin/env python3
"""
Convert Personal .txt Wildcards to Luna Logic YAML Format
Handles your personal wildcard collection with intelligent categorization
"""

import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re
from collections import defaultdict

class TxtWildcardConverter:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'items_converted': 0,
            'duplicates_skipped': 0,
            'by_category': defaultdict(int)
        }
        
        # Existing items cache for deduplication
        self.existing_items = {}
        
        # Directory to Luna Logic category mapping
        self.category_map = {
            # Clothing
            'clothing': 'clothing_full',
            'outfit': 'clothing_full',
            
            # Accessories
            'accessory': 'accessories',
            
            # Body/Appearance
            'Body': 'body_features',
            'age': 'age',
            'eyes': 'eyes',
            'Hair': 'hair',
            'skin': 'body_features',
            
            # Expressions/Emotions
            'Expression': 'expression',
            'expression': 'expression',
            
            # Actions/Poses
            'Action': 'action_general',
            'action': 'action_general',
            'Pose': 'pose_sfw',
            
            # Locations/Settings
            'Setting': 'location_scenario',
            'location': 'location_scenario',
            
            # Objects
            'Object': 'objects',
            
            # Style/Aesthetic
            'style': 'style_aesthetic',
            
            # Animals/Creatures
            'animals': 'creatures_animals',
            
            # Characters (regular)
            'character': 'character_types',
            
            # Embeddings (special handling)
            'Vicious_characters': 'character_embeddings',
            
            # Technical
            'composition': 'composition',
            'lighting': 'lighting',
            'colors': 'colors',
            'color': 'colors',
            'details': 'details',
            'weights': None,  # Skip weights files
            
            # Adjectives
            'adjective': 'adjectives',
        }
    
    def load_existing_items(self, category: str) -> Set[str]:
        """Load existing items from a Luna Logic YAML file"""
        if category in self.existing_items:
            return self.existing_items[category]
        
        yaml_file = self.output_dir / f'{category}.yaml'
        items = set()
        
        if yaml_file.exists():
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'items' in data:
                        for item in data['items']:
                            items.add(self.normalize_text(item['text']))
            except Exception as e:
                print(f"  Warning: Could not load {yaml_file}: {e}")
        
        self.existing_items[category] = items
        return items
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def determine_category(self, file_path: Path) -> Tuple[Optional[str], bool]:
        """Determine Luna Logic category from file path"""
        parts = file_path.relative_to(self.input_dir).parts
        
        # Skip certain directories
        skip_dirs = {'Yaml', 'Rubbish', 'full prompt', 'wildcard_packs'}
        if any(skip_dir in parts for skip_dir in skip_dirs):
            return None, True
        
        # Try to match directory to category
        for part in parts:
            if part in self.category_map:
                category = self.category_map[part]
                if category is None:
                    return None, True  # Explicitly skip
                return category, False
        
        # File name based matching
        filename = file_path.stem.lower()
        
        # Clothing patterns
        if any(x in filename for x in ['bot-', 'top-', 'dress', 'suit', 'uniform', 'outfit', 'cloth']):
            if 'bot-' in filename or 'bottom' in filename or 'pants' in filename or 'skirt' in filename:
                return 'clothing_bottoms', False
            elif 'top-' in filename or 'shirt' in filename or 'blouse' in filename:
                return 'clothing_tops', False
            elif 'foot' in filename or 'shoe' in filename or 'boot' in filename:
                return 'clothing_footwear', False
            else:
                return 'clothing_full', False
        
        # BDSM/NSFW
        if 'bdsm' in filename or 'fetish' in filename or 'bondage' in filename:
            return 'bdsm_accessory', False
        
        # Pose detection
        if 'pose' in filename or 'position' in filename:
            if 'nsfw' in filename or 'sex' in filename or 'explicit' in filename:
                return 'pose_nsfw', False
            return 'pose_sfw', False
        
        # Location detection
        if 'location' in filename or 'setting' in filename or 'place' in filename or 'scene' in filename:
            if 'indoor' in filename or 'room' in filename or 'building' in filename:
                return 'location_indoor', False
            elif 'outdoor' in filename or 'nature' in filename or 'landscape' in filename:
                return 'location_outdoor', False
            return 'location_scenario', False
        
        # Default to adjectives for descriptive words
        if any(x in filename for x in ['adj', 'descriptor', 'quality', 'attribute']):
            return 'adjectives', False
        
        # If no match, use parent directory or return None
        if len(parts) > 1:
            parent = parts[0]
            return self.category_map.get(parent, None), False
        
        return None, False
    
    def extract_tags(self, text: str, category: str) -> List[str]:
        """Extract semantic tags based on content and category"""
        tags = []
        text_lower = text.lower()
        
        # Category-based base tags
        if 'clothing' in category:
            tags.append('clothing')
            if 'nsfw' in text_lower or 'revealing' in text_lower or 'sexy' in text_lower:
                tags.append('nsfw')
            else:
                tags.append('sfw')
        
        elif 'pose' in category:
            tags.append('pose')
            if 'nsfw' in category or any(x in text_lower for x in ['sex', 'explicit', 'nude']):
                tags.append('nsfw')
            else:
                tags.append('sfw')
        
        elif 'bdsm' in category:
            tags.extend(['bdsm', 'fetish', 'nsfw'])
        
        elif 'body' in category or 'appearance' in category:
            tags.append('body')
        
        elif 'expression' in category:
            tags.append('expression')
        
        elif 'location' in category:
            tags.append('location')
        
        elif 'action' in category:
            tags.append('action')
        
        # Content-based tags
        if any(word in text_lower for word in ['cute', 'adorable', 'sweet']):
            tags.append('cute')
        
        if any(word in text_lower for word in ['sexy', 'erotic', 'sensual', 'seductive']):
            tags.append('sexy')
        
        if any(word in text_lower for word in ['dark', 'goth', 'emo', 'edgy']):
            tags.append('dark')
        
        if any(word in text_lower for word in ['elegant', 'classy', 'sophisticated']):
            tags.append('elegant')
        
        return sorted(list(set(tags)))
    
    def parse_txt_file(self, file_path: Path) -> List[str]:
        """Parse a .txt wildcard file and return list of items"""
        items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    # Skip nested wildcard references (we'll handle those separately)
                    if line.startswith('__') and line.endswith('__'):
                        continue
                    
                    # Remove weights if present (format: "text:weight")
                    if '::' in line:
                        line = line.split('::')[0].strip()
                    
                    # Clean up
                    line = line.strip(',').strip()
                    
                    if line:
                        items.append(line)
        
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
        
        return items
    
    def create_luna_item(self, text: str, category: str, weight: float = 1.0) -> Dict:
        """Create a Luna Logic formatted item"""
        tags = self.extract_tags(text, category)
        
        # Special handling for embeddings
        if category == 'character_embeddings':
            # Wrap embedding name in the embedding: syntax
            # Format: embedding:EmbeddingName
            item = {
                'text': f'embedding:{text.strip()}',
                'weight': weight,
                'tags': ['embedding', 'character', 'nsfw'],
                'metadata': {
                    'type': 'embedding',
                    'embedding_name': text.strip()
                }
            }
        else:
            item = {
                'text': text.strip(),
                'weight': weight,
                'tags': tags if tags else ['general']
            }
            
            # Add blacklist for NSFW content
            if 'nsfw' in tags or 'bdsm' in tags or 'fetish' in tags:
                item['blacklist'] = ['sfw', 'safe']
        
        return item
    
    def convert_file(self, file_path: Path) -> Tuple[Optional[str], List[Dict]]:
        """Convert a single .txt file to Luna Logic format"""
        category, should_skip = self.determine_category(file_path)
        
        if should_skip or category is None:
            return None, []
        
        # Parse the file
        raw_items = self.parse_txt_file(file_path)
        
        if not raw_items:
            return None, []
        
        # Load existing items for deduplication
        existing = self.load_existing_items(category)
        
        # Convert and deduplicate
        new_items = []
        for text in raw_items:
            normalized = self.normalize_text(text)
            
            if normalized not in existing:
                item = self.create_luna_item(text, category)
                new_items.append(item)
                existing.add(normalized)
            else:
                self.stats['duplicates_skipped'] += 1
        
        return category, new_items
    
    def save_category(self, category: str, new_items: List[Dict]):
        """Append new items to existing Luna Logic YAML file"""
        yaml_file = self.output_dir / f'{category}.yaml'
        
        # Load existing data
        if yaml_file.exists():
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Ensure metadata exists
            if 'metadata' not in data:
                data['metadata'] = {
                    'version': '1.0',
                    'description': f'{category.replace("_", " ").title()} wildcards',
                }
            
            # Ensure items exists
            if 'items' not in data:
                data['items'] = []
        else:
            data = {
                'metadata': {
                    'version': '1.0',
                    'description': f'{category.replace("_", " ").title()} wildcards',
                    'total_items': 0
                },
                'items': []
            }
        
        # Append new items
        data['items'].extend(new_items)
        data['metadata']['total_items'] = len(data['items'])
        
        # Save
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
    
    def convert(self, skip_wildcard_packs: bool = True):
        """Main conversion logic"""
        print("="*60)
        print("PERSONAL .TXT WILDCARD CONVERSION")
        print("="*60)
        
        # Collect all .txt files
        txt_files = list(self.input_dir.rglob('*.txt'))
        
        if skip_wildcard_packs:
            txt_files = [f for f in txt_files if 'wildcard_packs' not in f.parts]
            print(f"Skipping wildcard_packs (already extracted)")
        
        print(f"Found {len(txt_files)} .txt files to process\n")
        
        # Group files by category
        items_by_category = defaultdict(list)
        
        for file_path in txt_files:
            category, new_items = self.convert_file(file_path)
            
            if category and new_items:
                items_by_category[category].extend(new_items)
                self.stats['files_processed'] += 1
                self.stats['items_converted'] += len(new_items)
                self.stats['by_category'][category] += len(new_items)
                
                rel_path = str(file_path.relative_to(self.input_dir))
                print(f"[OK] {rel_path:60s} -> {category:30s} ({len(new_items)} new items)")
        
        # Save all categories
        print(f"\n{'='*60}")
        print("Saving to Luna Logic YAML files...")
        print(f"{'='*60}")
        
        for category, items in items_by_category.items():
            self.save_category(category, items)
            print(f"  {category:40s}: +{len(items):4d} items")
        
        # Print summary
        print(f"\n{'='*60}")
        print("CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Files processed:     {self.stats['files_processed']}")
        print(f"Items converted:     {self.stats['items_converted']}")
        print(f"Duplicates skipped:  {self.stats['duplicates_skipped']}")
        print(f"Categories updated:  {len(items_by_category)}")


def main():
    input_dir = Path(r"D:\AI\SD Models\Wildcards")
    output_dir = Path(r"D:\AI\SD Models\wildcards_yaml")
    
    converter = TxtWildcardConverter(input_dir, output_dir)
    
    # Skip wildcard_packs since we already extracted those from YAMLs
    converter.convert(skip_wildcard_packs=True)


if __name__ == '__main__':
    main()
