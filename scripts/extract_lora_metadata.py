"""
Extract LoRA metadata from SwarmUI's LiteDB database
Reads CivitAI metadata including trigger words, descriptions, and suggested weights
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

try:
    # Try importing litedb (Python LiteDB client)
    from litedb import LiteDatabase
    HAS_LITEDB = True
except ImportError:
    HAS_LITEDB = False
    print("WARNING: litedb not installed. Attempting manual parsing...")


def read_litedb_manual(db_path: Path) -> Dict:
    """
    Manual parsing of LiteDB file structure
    LiteDB stores data in BSON-like format
    """
    print(f"Reading LiteDB file: {db_path}")
    
    with open(db_path, 'rb') as f:
        data = f.read()
    
    # Look for JSON-like structures in the binary data
    # LiteDB stores strings as UTF-8
    text_data = data.decode('utf-8', errors='ignore')
    
    # Find all JSON objects (very naive approach)
    results = []
    start = 0
    while True:
        start = text_data.find('{', start)
        if start == -1:
            break
        
        # Try to find matching closing brace
        depth = 1
        pos = start + 1
        while pos < len(text_data) and depth > 0:
            if text_data[pos] == '{':
                depth += 1
            elif text_data[pos] == '}':
                depth -= 1
            pos += 1
        
        if depth == 0:
            try:
                json_str = text_data[start:pos]
                obj = json.loads(json_str)
                results.append(obj)
            except:
                pass
        
        start = pos
    
    return results


def extract_civitai_metadata(db_path: Path) -> Dict[str, Dict]:
    """
    Extract CivitAI metadata from LiteDB
    Returns dict mapping model filename -> metadata
    """
    
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return {}
    
    print(f"\nExtracting metadata from: {db_path.name}")
    print(f"File size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    metadata_by_file = {}
    
    try:
        # Try manual parsing first (more reliable)
        objects = read_litedb_manual(db_path)
        
        print(f"Found {len(objects)} potential JSON objects")
        
        for obj in objects:
            # Look for model metadata structures
            if isinstance(obj, dict):
                # Common keys in CivitAI metadata
                filename = obj.get('filename') or obj.get('fileName') or obj.get('name')
                
                if filename:
                    # Extract useful fields
                    metadata = {
                        'filename': filename,
                        'trigger_words': obj.get('trainedWords', []) or obj.get('triggerWords', []),
                        'description': obj.get('description', ''),
                        'model_id': obj.get('modelId') or obj.get('id'),
                        'version_id': obj.get('versionId'),
                        'model_name': obj.get('modelName') or obj.get('name'),
                        'base_model': obj.get('baseModel', ''),
                        'tags': obj.get('tags', []),
                        'nsfw': obj.get('nsfw', False),
                        'type': obj.get('type', ''),
                        'suggested_weight': extract_suggested_weight(obj),
                        'raw_data': obj  # Keep full data for inspection
                    }
                    
                    metadata_by_file[filename] = metadata
        
        print(f"\nExtracted metadata for {len(metadata_by_file)} models")
        
    except Exception as e:
        print(f"ERROR extracting metadata: {e}")
        import traceback
        traceback.print_exc()
    
    return metadata_by_file


def extract_suggested_weight(obj: Dict) -> Optional[float]:
    """
    Try to extract suggested weight from various fields
    """
    # Common fields that might contain weight suggestions
    for key in ['weight', 'suggestedWeight', 'strength', 'defaultWeight']:
        if key in obj and obj[key]:
            try:
                return float(obj[key])
            except:
                pass
    
    # Check description for weight mentions
    desc = obj.get('description', '')
    if 'weight' in desc.lower():
        # Simple regex to find patterns like "weight: 0.8" or "strength 0.7"
        import re
        match = re.search(r'(?:weight|strength)[\s:]+([0-9.]+)', desc, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    
    return None


def organize_by_category(metadata: Dict[str, Dict], lora_base_path: Path) -> Dict[str, List[Dict]]:
    """
    Organize LoRAs by directory structure (category)
    """
    categories = defaultdict(list)
    
    for filename, meta in metadata.items():
        # Try to find the actual file to determine category
        found = False
        
        for lora_file in lora_base_path.rglob('*.safetensors'):
            if lora_file.name == filename:
                # Get relative path from base
                rel_path = lora_file.relative_to(lora_base_path)
                category = str(rel_path.parent).replace('\\', '/')
                
                meta['file_path'] = str(lora_file)
                meta['category'] = category
                categories[category].append(meta)
                found = True
                break
        
        if not found:
            # Fallback to 'Unknown' category
            meta['category'] = 'Unknown'
            categories['Unknown'].append(meta)
    
    return categories


def save_metadata_yaml(categories: Dict[str, List[Dict]], output_dir: Path):
    """
    Save organized metadata to YAML files
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving metadata to: {output_dir}")
    
    # Save master index
    master_index = {
        'total_loras': sum(len(items) for items in categories.values()),
        'categories': {cat: len(items) for cat, items in categories.items()}
    }
    
    master_file = output_dir / 'lora_index.yaml'
    with open(master_file, 'w', encoding='utf-8') as f:
        yaml.dump(master_index, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Saved master index: {master_file}")
    
    # Save each category
    for category, items in categories.items():
        # Clean category name for filename
        safe_name = category.replace('/', '_').replace('\\', '_').replace('!', 'fav')
        if not safe_name:
            safe_name = 'root'
        
        category_file = output_dir / f'lora_{safe_name}.yaml'
        
        # Prepare data for YAML (remove raw_data to keep files clean)
        clean_items = []
        for item in items:
            clean_item = {k: v for k, v in item.items() if k != 'raw_data'}
            clean_items.append(clean_item)
        
        category_data = {
            'category': category,
            'total_loras': len(clean_items),
            'loras': clean_items
        }
        
        with open(category_file, 'w', encoding='utf-8') as f:
            yaml.dump(category_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"  Saved {len(items)} LoRAs to: {category_file.name}")
    
    print(f"\nComplete! Saved {len(categories)} category files")


def main():
    """Main extraction process"""
    
    print("="*80)
    print("LoRA Metadata Extractor - CivitAI Data from SwarmUI LiteDB")
    print("="*80)
    
    # Paths - try folder_paths first, then fallback
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / 'scripts' / 'model_metadata.ldb'
    output_dir = base_dir / 'lora_metadata'
    
    try:
        import sys
        comfyui_root = base_dir.parent.parent
        if str(comfyui_root) not in sys.path:
            sys.path.insert(0, str(comfyui_root))
        import folder_paths
        lora_paths = folder_paths.get_folder_paths("loras")
        lora_base_path = Path(lora_paths[0]) if lora_paths else None
    except ImportError:
        lora_base_path = None
    
    if not lora_base_path or not lora_base_path.exists():
        # Fallback - use ComfyUI default structure
        lora_base_path = base_dir.parent.parent / 'models' / 'loras'
        if not lora_base_path.exists():
            print(f"ERROR: LoRA directory not found. Please specify path.")
            print(f"Tried: {lora_base_path}")
            return
    
    print(f"Using LoRA directory: {lora_base_path}")
    
    # Extract metadata
    metadata = extract_civitai_metadata(db_path)
    
    if not metadata:
        print("\nERROR: No metadata extracted!")
        print("Trying alternate database...")
        db_path_alt = base_dir / 'model_metadata-log.ldb'
        metadata = extract_civitai_metadata(db_path_alt)
    
    if not metadata:
        print("\nERROR: Could not extract any metadata from database files!")
        return
    
    # Organize by category
    print("\nOrganizing LoRAs by category...")
    categories = organize_by_category(metadata, lora_base_path)
    
    # Print summary
    print("\nCategory Summary:")
    for cat, items in sorted(categories.items()):
        print(f"  {cat}: {len(items)} LoRAs")
    
    # Save to YAML files
    save_metadata_yaml(categories, output_dir)
    
    # Print examples
    print("\nExample metadata:")
    for category, items in list(categories.items())[:2]:
        if items:
            example = items[0]
            print(f"\n{category}/{example['filename']}:")
            print(f"  Trigger words: {example['trigger_words']}")
            print(f"  Suggested weight: {example['suggested_weight']}")
            print(f"  Type: {example['type']}")
            print(f"  NSFW: {example['nsfw']}")


if __name__ == '__main__':
    main()
