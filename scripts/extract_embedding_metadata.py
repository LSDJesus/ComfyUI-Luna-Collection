"""
Extract embedding (textual inversion) metadata from safetensors files.
Embeddings store trigger words and training information in the __metadata__ section.
"""

import json
import yaml
import struct
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def read_safetensors_metadata(safetensors_path: Path) -> Optional[Dict]:
    """
    Read metadata from safetensors file by parsing the header directly.
    Returns the __metadata__ section if present.
    """
    try:
        with open(safetensors_path, 'rb') as f:
            # Read header size (8 bytes, little-endian unsigned long long)
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return None
            
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            
            # Read header JSON
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                return None
            
            header_json = header_bytes.decode('utf-8')
            header = json.loads(header_json)
            
            # Return the __metadata__ section
            return header.get('__metadata__', {})
    except Exception as e:
        print(f"  WARNING: Could not read {safetensors_path.name}: {e}")
        return None


def extract_embedding_metadata(embedding_path: Path) -> Dict:
    """
    Extract metadata from a single embedding file.
    Returns structured metadata dict.
    """
    metadata = {
        'filename': embedding_path.name,
        'file_path': str(embedding_path),
        'trigger_word': '',
        'description': '',
        'model_id': None,
        'base_model': '',
        'tags': [],
        'nsfw': False,
        'type': 'EMBEDDING',
        'training_info': {},
        'notes': ''
    }
    
    # Try to read safetensors metadata
    if embedding_path.suffix.lower() == '.safetensors':
        st_metadata = read_safetensors_metadata(embedding_path)
        
        if st_metadata:
            # Extract trigger word from filename or metadata
            # Embeddings typically use the filename as the trigger word
            trigger = embedding_path.stem
            
            # Check for common metadata fields
            if 'name' in st_metadata:
                trigger = st_metadata['name']
            elif 'trigger_word' in st_metadata:
                trigger = st_metadata['trigger_word']
            
            metadata['trigger_word'] = trigger
            
            # Extract base model info
            if 'sd_checkpoint' in st_metadata:
                metadata['base_model'] = st_metadata['sd_checkpoint']
            elif 'modelspec.architecture' in st_metadata:
                arch = st_metadata['modelspec.architecture']
                if 'xl' in arch.lower():
                    metadata['base_model'] = 'sdxl'
                elif 'sd-v1' in arch.lower():
                    metadata['base_model'] = 'sd1.5'
                else:
                    metadata['base_model'] = arch
            
            # Extract training info
            training_fields = [
                'ss_num_train_images', 'ss_num_epochs', 'ss_learning_rate',
                'ss_resolution', 'ss_num_vectors_per_token', 'ss_num_tokens',
                'modelspec.resolution', 'modelspec.date', 'modelspec.title',
                'modelspec.description', 'modelspec.tags'
            ]
            
            for field in training_fields:
                if field in st_metadata:
                    metadata['training_info'][field] = st_metadata[field]
            
            # Extract tags
            if 'modelspec.tags' in st_metadata:
                tags_str = st_metadata['modelspec.tags']
                if isinstance(tags_str, str):
                    metadata['tags'] = [t.strip() for t in tags_str.split(',') if t.strip()]
                elif isinstance(tags_str, list):
                    metadata['tags'] = tags_str
            
            # Extract description
            if 'modelspec.description' in st_metadata:
                metadata['description'] = st_metadata['modelspec.description']
            elif 'description' in st_metadata:
                metadata['description'] = st_metadata['description']
    
    # Also check for .metadata.json sidecar file
    json_path = embedding_path.with_suffix('.metadata.json')
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                civitai = json_data.get('civitai', {}) or {}
                if civitai:
                    if not metadata['model_id'] and 'modelId' in civitai:
                        metadata['model_id'] = civitai['modelId']
                    if not metadata['description'] and 'description' in civitai:
                        metadata['description'] = civitai['description']
                    if not metadata['tags'] and 'tags' in civitai:
                        metadata['tags'] = civitai['tags']
                    if 'nsfw' in civitai:
                        metadata['nsfw'] = civitai['nsfw']
        except:
            pass
    
    return metadata


def scan_embedding_directory(base_path: Path) -> Dict[str, List[Dict]]:
    """
    Recursively scan directory for embedding files and extract metadata.
    Returns dict mapping category names to lists of embedding metadata.
    """
    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_path}")
        return {}
    
    print(f"\nScanning embedding directory: {base_path}")
    
    # Find all embedding files
    embedding_files = []
    for ext in ['.safetensors', '.pt', '.bin']:
        embedding_files.extend(base_path.rglob(f'*{ext}'))
    
    # Organize by subdirectory (category)
    categories = defaultdict(list)
    processed = 0
    
    for embedding_file in sorted(embedding_files):
        # Skip temp files and backups
        if embedding_file.name.startswith('.') or embedding_file.name.endswith('.bkup'):
            continue
        
        # Extract metadata
        metadata = extract_embedding_metadata(embedding_file)
        
        # Determine category from subdirectory structure
        try:
            relative_path = embedding_file.relative_to(base_path)
            if len(relative_path.parts) > 1:
                # Has subdirectory - use as category
                category = '/'.join(relative_path.parts[:-1])
            else:
                # In root directory
                category = 'root'
        except ValueError:
            category = 'root'
        
        metadata['category'] = category
        categories[category].append(metadata)
        
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} embeddings...")
    
    # Calculate statistics
    total = sum(len(items) for items in categories.values())
    with_triggers = sum(1 for items in categories.values() for item in items if item['trigger_word'])
    
    print(f"\nFound {total} embeddings in {len(categories)} categories")
    if total > 0:
        print(f"  {with_triggers} with trigger words ({with_triggers/total*100:.1f}%)")
    
    return dict(categories)


def save_metadata_yaml(categories: Dict[str, List[Dict]], output_dir: Path):
    """
    Save metadata to YAML files organized by category.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving metadata to: {output_dir}")
    
    # Create master index file
    index = {
        'total_embeddings': sum(len(items) for items in categories.values()),
        'total_categories': len(categories),
        'categories': {}
    }
    
    for category, items in categories.items():
        index['categories'][category] = {
            'count': len(items),
            'file': f'embedding_{category.replace("/", "_")}.yaml'
        }
    
    index_file = output_dir / 'embedding_index.yaml'
    with open(index_file, 'w', encoding='utf-8') as f:
        yaml.dump(index, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Saved master index: {index_file}")
    
    # Save individual category files
    for category, items in sorted(categories.items()):
        category_file = output_dir / f'embedding_{category.replace("/", "_")}.yaml'
        
        category_data = {
            'category': category,
            'total_embeddings': len(items),
            'embeddings': items
        }
        
        with open(category_file, 'w', encoding='utf-8') as f:
            yaml.dump(category_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"  Saved {len(items):3d} embeddings to: {category_file.name}")
    
    print(f"\nComplete! Saved {len(categories)} category files")


def print_summary(categories: Dict[str, List[Dict]]):
    """
    Print summary statistics
    """
    total = sum(len(items) for items in categories.values())
    with_triggers = sum(1 for items in categories.values() for item in items if item['trigger_word'])
    with_base_model = sum(1 for items in categories.values() for item in items if item['base_model'])
    with_metadata = sum(1 for items in categories.values() for item in items if item['model_id'] or item['description'])
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total embeddings found: {total}")
    print(f"Embeddings with trigger words: {with_triggers} ({with_triggers/total*100:.1f}%)")
    print(f"Embeddings with base model info: {with_base_model} ({with_base_model/total*100:.1f}%)")
    print(f"Embeddings with CivitAI metadata: {with_metadata} ({with_metadata/total*100:.1f}%)")
    
    print("\nCategories:")
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {cat}: {len(items)} embeddings")
    
    # Print some examples
    print("\nExample embeddings with trigger words:")
    count = 0
    for category, items in categories.items():
        for item in items:
            if item['trigger_word'] and count < 5:
                print(f"\n  {item['filename']}")
                print(f"    Trigger: {item['trigger_word']}")
                print(f"    Base Model: {item['base_model'] or 'not specified'}")
                print(f"    Category: {category}")
                if item['tags']:
                    print(f"    Tags: {', '.join(item['tags'][:5])}")
                count += 1
            if count >= 5:
                break
        if count >= 5:
            break


def main():
    """Main extraction process"""
    
    print("="*80)
    print("Embedding Metadata Extractor - Safetensors + JSON")
    print("="*80)
    
    # Paths
    base_dir = Path(__file__).parent
    embedding_root_path = Path('D:/AI/SD Models/embeddings')
    output_dir = base_dir / 'embedding_metadata'
    
    if not embedding_root_path.exists():
        print(f"ERROR: Embedding root directory not found: {embedding_root_path}")
        return
    
    # Scan all subdirectories
    all_categories = {}
    subdirs = [d for d in embedding_root_path.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(subdirs)} embedding directories:")
    for subdir in sorted(subdirs):
        print(f"  - {subdir.name}")
    
    for subdir in sorted(subdirs):
        print(f"\n{'='*80}")
        print(f"Scanning: {subdir.name}")
        print(f"{'='*80}")
        
        categories = scan_embedding_directory(subdir)
        
        # Merge categories with base model type prefix
        for cat, items in categories.items():
            # Add base model type to category name
            if cat == 'root':
                prefixed_cat = subdir.name
            else:
                prefixed_cat = f"{subdir.name}/{cat}"
            
            if prefixed_cat not in all_categories:
                all_categories[prefixed_cat] = []
            all_categories[prefixed_cat].extend(items)
    
    if not all_categories:
        print("\nERROR: No embeddings found!")
        return
    
    # Save to YAML files
    save_metadata_yaml(all_categories, output_dir)
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY - ALL EMBEDDING TYPES")
    print("="*80)
    print_summary(all_categories)


if __name__ == '__main__':
    main()
