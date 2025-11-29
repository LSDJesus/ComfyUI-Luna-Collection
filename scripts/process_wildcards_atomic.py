"""
Atomic Wildcard Processor - Intelligent decomposition of existing wildcards
into atomic, composable elements with proper prompt ordering.

Uses LoRA/Embedding metadata to:
1. Identify concepts already covered by models (skip redundant wildcards)
2. Extract atomic elements from verbose composite wildcards
3. Apply compositional patterns from existing successful wildcards
4. Generate structured YAML with tags, blacklists, and composition rules
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re


# 10-Tier Prompt Order (from our established structure)
PROMPT_TIERS = {
    'quality': 1,           # masterpiece, 8k, best quality
    'subject': 2,           # character count, basic subject
    'face': 3,              # expression, gaze, facial features
    'hair': 4,              # color, style, length
    'clothing': 5,          # main garments, colors, details
    'pose': 6,              # body position, action
    'location': 7,          # setting, environment
    'lighting': 8,          # light quality, atmosphere
    'camera': 9,            # angle, framing, focus
    'details': 10           # enhancement, micro-details
}

# Category mapping to tiers
CATEGORY_TO_TIER = {
    # Tier 1 - Quality (skip, use embeddings)
    'quality': 'quality',
    
    # Tier 2 - Subject
    'subject': 'subject',
    'character_count': 'subject',
    'age': 'subject',
    'gender': 'subject',
    
    # Tier 3 - Face
    'expression': 'face',
    'eyes': 'face',
    'gaze': 'face',
    'mouth': 'face',
    'facial_feature': 'face',
    'emotion': 'face',
    
    # Tier 4 - Hair
    'hair_color': 'hair',
    'hair_style': 'hair',
    'hair_length': 'hair',
    'hair_accessory': 'hair',
    
    # Tier 5 - Clothing
    'clothing': 'clothing',
    'lingerie': 'clothing',
    'outfit': 'clothing',
    'color': 'clothing',
    'fabric': 'clothing',
    'pattern': 'clothing',
    'accessory': 'clothing',
    
    # Tier 6 - Pose
    'pose': 'pose',
    'action': 'pose',
    'position': 'pose',
    'gesture': 'pose',
    
    # Tier 7 - Location
    'location': 'location',
    'environment': 'location',
    'setting': 'location',
    'background': 'location',
    'furniture': 'location',
    
    # Tier 8 - Lighting
    'lighting': 'lighting',
    'atmosphere': 'lighting',
    'time_of_day': 'lighting',
    'weather': 'lighting',
    
    # Tier 9 - Camera
    'camera': 'camera',
    'angle': 'camera',
    'framing': 'camera',
    'focus': 'camera',
    'shot_type': 'camera',
    
    # Tier 10 - Details (mostly covered by LoRAs)
    'detail': 'details',
}


class LoRARegistry:
    """Track what concepts are covered by LoRAs to avoid redundant wildcards"""
    
    def __init__(self, lora_metadata_dir: Path):
        self.loras = {}
        self.triggers_by_concept = defaultdict(list)
        self.training_tags_by_concept = defaultdict(set)
        self.load_lora_metadata(lora_metadata_dir)
    
    def load_lora_metadata(self, metadata_dir: Path):
        """Load all LoRA metadata files"""
        if not metadata_dir.exists():
            print(f"WARNING: LoRA metadata directory not found: {metadata_dir}")
            return
        
        yaml_files = list(metadata_dir.glob('lora_*.yaml'))
        print(f"\nLoading LoRA metadata from {len(yaml_files)} files...")
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                for lora in data.get('loras', []):
                    filename = lora['filename']
                    self.loras[filename] = lora
                    
                    # Index by trigger words
                    for trigger in lora.get('trigger_words', []):
                        concept = trigger.lower().strip()
                        self.triggers_by_concept[concept].append(filename)
                    
                    # Index by training tags
                    for tag in lora.get('training_tags', {}).keys():
                        concept = tag.lower().strip()
                        self.training_tags_by_concept[concept].add(filename)
            
            except Exception as e:
                print(f"  ERROR loading {yaml_file.name}: {e}")
        
        print(f"  Loaded {len(self.loras)} LoRAs")
        print(f"  Indexed {len(self.triggers_by_concept)} trigger concepts")
        print(f"  Indexed {len(self.training_tags_by_concept)} training tag concepts")
    
    def is_covered_by_lora(self, text: str) -> Tuple[bool, List[str]]:
        """Check if a concept is covered by LoRAs"""
        text_lower = text.lower().strip()
        
        # Check exact trigger match
        if text_lower in self.triggers_by_concept:
            return True, self.triggers_by_concept[text_lower]
        
        # Check training tags
        if text_lower in self.training_tags_by_concept:
            return True, list(self.training_tags_by_concept[text_lower])
        
        # Check partial matches in triggers
        for trigger in self.triggers_by_concept.keys():
            if text_lower in trigger or trigger in text_lower:
                return True, self.triggers_by_concept[trigger]
        
        return False, []
    
    def get_lora_categories(self) -> Dict[str, int]:
        """Get counts of LoRAs by category"""
        categories = defaultdict(int)
        for lora in self.loras.values():
            cat = lora.get('category', 'unknown')
            categories[cat] += 1
        return dict(categories)


class EmbeddingRegistry:
    """Track what concepts are covered by embeddings"""
    
    def __init__(self, embedding_metadata_dir: Path):
        self.embeddings = {}
        self.character_embeddings = set()
        self.quality_embeddings = set()
        self.load_embedding_metadata(embedding_metadata_dir)
    
    def load_embedding_metadata(self, metadata_dir: Path):
        """Load all embedding metadata files"""
        if not metadata_dir.exists():
            print(f"WARNING: Embedding metadata directory not found: {metadata_dir}")
            return
        
        yaml_files = list(metadata_dir.glob('embedding_*.yaml'))
        print(f"\nLoading embedding metadata from {len(yaml_files)} files...")
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                category = data.get('category', '')
                
                for emb in data.get('embeddings', []):
                    trigger = emb['trigger_word']
                    self.embeddings[trigger] = emb
                    
                    # Categorize embeddings
                    if any(x in category.lower() for x in ['egirl', 'character']):
                        self.character_embeddings.add(trigger)
                    
                    tags = emb.get('tags', [])
                    if any(x in tags for x in ['quality up', 'quality enhancer', 'positive embedding']):
                        self.quality_embeddings.add(trigger)
            
            except Exception as e:
                print(f"  ERROR loading {yaml_file.name}: {e}")
        
        print(f"  Loaded {len(self.embeddings)} embeddings")
        print(f"  {len(self.character_embeddings)} character embeddings")
        print(f"  {len(self.quality_embeddings)} quality embeddings")


class WildcardAnalyzer:
    """Analyze existing wildcards to extract compositional patterns"""
    
    def __init__(self):
        self.patterns = defaultdict(list)
        self.element_combinations = defaultdict(set)
        self.blacklist_patterns = defaultdict(set)
    
    def analyze_verbose_wildcard(self, wildcard_data: Dict) -> Dict:
        """
        Analyze a verbose wildcard file to extract:
        - Compositional patterns (how elements combine)
        - Element relationships (what pairs with what)
        - Common modifiers and their positions
        """
        items = wildcard_data.get('items', [])
        
        analysis = {
            'total_items': len(items),
            'unique_elements': set(),
            'element_positions': defaultdict(list),
            'common_patterns': [],
            'suggested_atomics': []
        }
        
        for item in items:
            text = item if isinstance(item, str) else item.get('text', '')
            
            # Split into components
            parts = [p.strip() for p in re.split(r',|\band\b', text) if p.strip()]
            
            for i, part in enumerate(parts):
                analysis['unique_elements'].add(part)
                analysis['element_positions'][part].append(i)
        
        # Identify elements that always appear in same position
        positional_elements = {}
        for element, positions in analysis['element_positions'].items():
            if len(set(positions)) == 1:  # Always same position
                positional_elements[element] = positions[0]
        
        # Suggest atomic wildcards based on unique elements
        analysis['suggested_atomics'] = self._suggest_atomic_categories(
            analysis['unique_elements']
        )
        
        return analysis
    
    def _suggest_atomic_categories(self, elements: Set[str]) -> List[Dict]:
        """Suggest atomic wildcard categories from element analysis"""
        suggestions = []
        
        # Group elements by detected category
        categorized = {
            'colors': [],
            'materials': [],
            'styles': [],
            'modifiers': [],
            'objects': []
        }
        
        color_keywords = {'red', 'blue', 'green', 'black', 'white', 'pink', 'purple'}
        material_keywords = {'lace', 'silk', 'cotton', 'leather', 'satin', 'mesh'}
        style_keywords = {'elegant', 'sexy', 'cute', 'casual', 'formal'}
        
        for element in elements:
            element_lower = element.lower()
            
            if any(color in element_lower for color in color_keywords):
                categorized['colors'].append(element)
            elif any(mat in element_lower for mat in material_keywords):
                categorized['materials'].append(element)
            elif any(style in element_lower for style in style_keywords):
                categorized['styles'].append(element)
            else:
                categorized['objects'].append(element)
        
        # Generate suggestions
        for category, items in categorized.items():
            if items:
                suggestions.append({
                    'category': category,
                    'count': len(items),
                    'examples': items[:5]
                })
        
        return suggestions


class AtomicWildcardGenerator:
    """Generate atomic wildcard YAML files with proper structure"""
    
    def __init__(self, lora_registry: LoRARegistry, embedding_registry: EmbeddingRegistry):
        self.lora_registry = lora_registry
        self.embedding_registry = embedding_registry
        self.wildcard_analyzer = WildcardAnalyzer()
    
    def process_wildcard_file(self, input_file: Path) -> Optional[Dict]:
        """Process a single wildcard file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return None
            
            # Analyze the wildcard
            analysis = self.wildcard_analyzer.analyze_verbose_wildcard(data)
            
            # Generate atomic wildcards
            atomic_wildcards = self._generate_atomic_wildcards(data, analysis)
            
            return {
                'source_file': input_file.name,
                'analysis': analysis,
                'atomic_wildcards': atomic_wildcards
            }
        
        except Exception as e:
            print(f"ERROR processing {input_file.name}: {e}")
            return None
    
    def _generate_atomic_wildcards(self, original_data: Dict, analysis: Dict) -> List[Dict]:
        """Generate atomic wildcard structures from analysis"""
        atomics = []
        
        # Get original category
        original_category = original_data.get('category', 'unknown')
        tier = CATEGORY_TO_TIER.get(original_category.lower(), 'unknown')
        
        items = original_data.get('items', [])
        
        for item in items:
            text = item if isinstance(item, str) else item.get('text', '')
            
            # Check if covered by LoRA
            is_lora_covered, lora_files = self.lora_registry.is_covered_by_lora(text)
            
            # Skip if this is a quality term (use embeddings instead)
            if tier == 'quality':
                continue
            
            atomic = {
                'id': self._generate_id(text),
                'text': text,
                'tags': self._generate_tags(text, original_category),
                'tier': tier,
                'tier_order': PROMPT_TIERS.get(tier, 999)
            }
            
            # Add LoRA information if covered
            if is_lora_covered:
                atomic['lora_covered'] = True
                atomic['lora_files'] = lora_files[:3]  # Top 3
                atomic['notes'] = f"Concept covered by {len(lora_files)} LoRA(s). May be redundant."
            
            # Add blacklist if appropriate
            blacklist = self._generate_blacklist(text, original_category)
            if blacklist:
                atomic['blacklist'] = blacklist
            
            # Add whitelist/compatible elements
            compatible = self._generate_compatible(text, original_category)
            if compatible:
                atomic['compatible'] = compatible
            
            atomics.append(atomic)
        
        return atomics
    
    def _generate_id(self, text: str) -> str:
        """Generate a safe ID from text"""
        # Remove special characters, lowercase, replace spaces with underscores
        safe_id = re.sub(r'[^\w\s-]', '', text.lower())
        safe_id = re.sub(r'[-\s]+', '_', safe_id)
        return safe_id
    
    def _generate_tags(self, text: str, category: str) -> List[str]:
        """Generate appropriate tags for an element"""
        tags = [category.lower()]
        
        text_lower = text.lower()
        
        # Add semantic tags based on content
        if any(x in text_lower for x in ['sexy', 'revealing', 'lingerie', 'nude']):
            tags.append('nsfw')
        if any(x in text_lower for x in ['cute', 'innocent', 'sweet']):
            tags.append('sfw')
        if any(x in text_lower for x in ['elegant', 'formal', 'classy']):
            tags.append('elegant')
        if any(x in text_lower for x in ['dark', 'night', 'shadow']):
            tags.append('dark')
        if any(x in text_lower for x in ['bright', 'sunny', 'light']):
            tags.append('bright')
        
        return list(set(tags))
    
    def _generate_blacklist(self, text: str, category: str) -> List[str]:
        """Generate blacklist of incompatible elements"""
        blacklist = []
        
        text_lower = text.lower()
        
        # Location blacklists
        if 'indoor' in text_lower:
            blacklist.extend(['outdoor', 'nature', 'sky'])
        if 'outdoor' in text_lower:
            blacklist.extend(['indoor', 'bedroom', 'furniture'])
        
        # Lighting blacklists
        if 'bright' in text_lower or 'day' in text_lower:
            blacklist.extend(['dark', 'night', 'moonlight'])
        if 'dark' in text_lower or 'night' in text_lower:
            blacklist.extend(['bright', 'sunny', 'daylight'])
        
        # Pose blacklists
        if 'standing' in text_lower:
            blacklist.extend(['sitting', 'lying', 'kneeling'])
        if 'sitting' in text_lower:
            blacklist.extend(['standing', 'walking', 'running'])
        
        return blacklist
    
    def _generate_compatible(self, text: str, category: str) -> List[str]:
        """Generate list of compatible/complementary elements"""
        compatible = []
        
        text_lower = text.lower()
        
        # Clothing compatibility
        if 'lingerie' in text_lower:
            compatible.extend(['bedroom', 'intimate', 'soft_lighting'])
        if 'dress' in text_lower:
            compatible.extend(['elegant', 'formal', 'heels'])
        
        # Location compatibility
        if 'bedroom' in text_lower:
            compatible.extend(['bed', 'intimate', 'soft_lighting'])
        if 'outdoor' in text_lower:
            compatible.extend(['nature', 'daylight', 'trees'])
        
        return compatible


def main():
    """Main processing pipeline"""
    
    print("="*80)
    print("ATOMIC WILDCARD PROCESSOR")
    print("="*80)
    
    # Paths - try folder_paths first
    base_dir = Path(__file__).parent.parent
    lora_metadata_dir = base_dir / 'lora_metadata'
    embedding_metadata_dir = base_dir / 'embedding_metadata'
    output_dir = base_dir / 'wildcards_atomic'
    
    try:
        import sys
        comfyui_root = base_dir.parent.parent
        sys.path.insert(0, str(comfyui_root))
        import folder_paths
        wildcards_paths = folder_paths.get_folder_paths("wildcards")
        input_wildcard_dir = Path(wildcards_paths[0]) if wildcards_paths else None
    except ImportError:
        input_wildcard_dir = None
    
    if not input_wildcard_dir or not input_wildcard_dir.exists():
        input_wildcard_dir = base_dir.parent.parent / 'models' / 'wildcards'
    
    print(f"Using wildcards directory: {input_wildcard_dir}")
    
    # Initialize registries
    print("\n" + "="*80)
    print("LOADING MODEL METADATA")
    print("="*80)
    
    lora_registry = LoRARegistry(lora_metadata_dir)
    embedding_registry = EmbeddingRegistry(embedding_metadata_dir)
    
    # Print coverage summary
    print("\n" + "="*80)
    print("MODEL COVERAGE SUMMARY")
    print("="*80)
    
    lora_cats = lora_registry.get_lora_categories()
    print("\nLoRA Coverage by Category:")
    for cat, count in sorted(lora_cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} LoRAs")
    
    print(f"\nEmbedding Coverage:")
    print(f"  Characters: {len(embedding_registry.character_embeddings)} embeddings")
    print(f"  Quality: {len(embedding_registry.quality_embeddings)} embeddings")
    
    # Process wildcards
    print("\n" + "="*80)
    print("PROCESSING WILDCARDS")
    print("="*80)
    
    if not input_wildcard_dir.exists():
        print(f"ERROR: Input directory not found: {input_wildcard_dir}")
        return
    
    generator = AtomicWildcardGenerator(lora_registry, embedding_registry)
    
    yaml_files = list(input_wildcard_dir.glob('*.yaml'))
    print(f"\nFound {len(yaml_files)} wildcard files to process")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    for yaml_file in sorted(yaml_files)[:10]:  # Process first 10 as test
        print(f"\nProcessing: {yaml_file.name}")
        
        result = generator.process_wildcard_file(yaml_file)
        
        if result:
            # Save analysis
            analysis_file = output_dir / f"analysis_{yaml_file.stem}.yaml"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"  ✓ Analyzed: {result['analysis']['total_items']} items")
            print(f"  ✓ Generated: {len(result['atomic_wildcards'])} atomic elements")
            print(f"  ✓ Saved to: {analysis_file.name}")
            
            processed_count += 1
        else:
            skipped_count += 1
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
