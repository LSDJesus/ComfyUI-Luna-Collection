"""
Extract LoRA metadata from multiple sources:
1. LiteDB database (SwarmUI cached metadata - fastest, includes CivitAI data)
2. Safetensors __metadata__ section (training info, trigger phrases)
3. .metadata.json files (SwarmUI sidecar files)
"""

import json
import yaml
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


try:
    import safetensors
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not installed, will only read .metadata.json files")


# LiteDB parsing functions
def parse_litedb_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Parse a LiteDB string: [int32_length][utf8_data]\\0"""
    if offset + 4 > len(data):
        return "", offset
    length = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    if offset + length > len(data):
        return "", offset
    string = data[offset:offset+length].decode('utf-8', errors='ignore').rstrip('\x00')
    return string, offset + length


def parse_litedb_field(data: bytes, offset: int) -> Tuple[Optional[Tuple[str, Any]], int]:
    """Parse a single BSON-like field from LiteDB"""
    if offset >= len(data):
        return None, offset
    
    field_type = data[offset]
    offset += 1
    
    # Read field name (null-terminated)
    field_name_end = data.find(b'\x00', offset)
    if field_name_end == -1:
        return None, offset
    field_name = data[offset:field_name_end].decode('utf-8', errors='ignore')
    offset = field_name_end + 1
    
    # Parse value based on type
    value = None
    try:
        if field_type == 0x02:  # String
            value, offset = parse_litedb_string(data, offset)
        elif field_type == 0x08:  # Boolean
            value = bool(data[offset])
            offset += 1
        elif field_type == 0x10:  # Int32
            value = struct.unpack_from('<i', data, offset)[0]
            offset += 4
        elif field_type == 0x12:  # Int64
            value = struct.unpack_from('<q', data, offset)[0]
            offset += 8
        else:
            # Unknown type, skip
            return None, offset
    except (struct.error, IndexError):
        return None, offset
    
    return (field_name, value), offset


def parse_litedb_document(data: bytes, offset: int) -> Tuple[Optional[Dict], int]:
    """Parse a single BSON-like document from LiteDB"""
    if offset + 4 > len(data):
        return None, offset
    
    doc_length = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    doc_end = offset + doc_length - 4
    
    document = {}
    while offset < doc_end and offset < len(data):
        field, offset = parse_litedb_field(data, offset)
        if field is None:
            break
        field_name, field_value = field
        document[field_name] = field_value
    
    return document, doc_end


def extract_litedb_metadata(litedb_path: Path, lora_directory: Path) -> Dict[str, Dict]:
    """
    Extract metadata from LiteDB database file.
    Returns dict mapping file paths to metadata.
    """
    if not litedb_path.exists():
        return {}
    
    print(f"Parsing LiteDB database: {litedb_path}")
    metadata_by_path = {}
    all_paths = []  # Debug: collect all paths found
    
    try:
        with open(litedb_path, 'rb') as f:
            data = f.read()
        
        # LiteDB uses 8KB pages
        PAGE_SIZE = 8192
        
        # Search through all pages for documents
        for page_offset in range(0, len(data), PAGE_SIZE):
            page_data = data[page_offset:page_offset + PAGE_SIZE]
            
            # Look for document markers (try multiple offsets within page)
            for doc_offset in range(0, len(page_data) - 4, 4):
                try:
                    # Try to parse a document
                    doc, _ = parse_litedb_document(page_data, doc_offset)
                    if doc and '_id' in doc:
                        file_path = doc.get('_id', '')
                        all_paths.append(file_path)  # Debug: collect all paths
                        # Check if this is a LoRA in our directory
                        if file_path and lora_directory.as_posix() in file_path.replace('\\', '/'):
                            metadata_by_path[file_path] = doc
                except:
                    continue
        
        print(f"Found {len(metadata_by_path)} LoRA entries in LiteDB")
        
        # Debug: Print sample paths to see the format
        if all_paths and len(metadata_by_path) == 0:
            print(f"\nDEBUG: Found {len(all_paths)} total entries but none matched directory")
            print(f"Looking for: {lora_directory.as_posix()}")
            print("\nSample paths from database (first 5):")
            for path in all_paths[:5]:
                if path and '.safetensors' in path.lower():
                    print(f"  {path}")

        return metadata_by_path
        
    except Exception as e:
        print(f"Error parsing LiteDB: {e}")
        return {}


def read_safetensors_metadata(safetensors_path: Path) -> Optional[Dict]:
    """
    Read metadata from safetensors file header (__metadata__ section).
    This contains training info, trigger phrases, and tag frequencies.
    """
    if not HAS_SAFETENSORS:
        return None
    
    try:
        # Read the header manually to get __metadata__
        with open(safetensors_path, 'rb') as f:
            # First 8 bytes = header size
            header_size = struct.unpack('<Q', f.read(8))[0]
            # Read header JSON
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            
            # Return the __metadata__ section if it exists
            return header.get('__metadata__', {})
    except Exception as e:
        # print(f"Could not read {safetensors_path.name}: {e}")
        return None


def read_metadata_json(json_path: Path) -> Optional[Dict]:
    """
    Read metadata from .metadata.json file (created by SwarmUI)
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {json_path.name}: {e}")
        return None


def extract_lora_metadata(lora_path: Path, litedb_metadata: Optional[Dict[str, Dict]] = None) -> Dict:
    """
    Extract all available metadata for a LoRA file.
    Priority: LiteDB (SwarmUI cache) > Safetensors metadata > .metadata.json sidecar
    
    Args:
        lora_path: Path to the .safetensors file
        litedb_metadata: Optional dict of metadata from LiteDB database
    """
    metadata = {
        'filename': lora_path.name,
        'file_path': str(lora_path),
        'trigger_words': [],
        'description': '',
        'model_id': None,
        'model_name': '',
        'base_model': '',
        'tags': [],
        'nsfw': False,
        'type': 'LORA',
        'suggested_weight': None,
        'activation_text': '',
        'training_tags': {},
        'resolution': '',
        'notes': ''
    }
    
    # 1. Check LiteDB database first (most complete - has CivitAI data)
    if litedb_metadata:
        file_key = str(lora_path).replace('\\', '/')
        litedb_doc = None
        
        # Try to find this file in the LiteDB data
        for db_path, db_doc in litedb_metadata.items():
            if file_key in db_path or lora_path.name in db_path:
                litedb_doc = db_doc
                break
        
        if litedb_doc:
            metadata['model_name'] = litedb_doc.get('Title', '')
            metadata['description'] = litedb_doc.get('Description', '')
            trigger_phrase = litedb_doc.get('TriggerPhrase', '')
            if trigger_phrase:
                metadata['trigger_words'] = [trigger_phrase]
            metadata['model_id'] = litedb_doc.get('ModelId')
            metadata['base_model'] = litedb_doc.get('ModelClassType', '')
            metadata['resolution'] = f"{litedb_doc.get('StandardWidth', '')}x{litedb_doc.get('StandardHeight', '')}"
    
    # 2. Check safetensors __metadata__ section (training info, trigger phrases)
    st_metadata = read_safetensors_metadata(lora_path)
    if st_metadata:
        # Extract trigger phrase from modelspec
        trigger_phrase = st_metadata.get('modelspec.trigger_phrase', '')
        if trigger_phrase and not metadata['trigger_words']:
            # Split on common separators
            triggers = [t.strip() for t in trigger_phrase.replace(',', ' ').split() if t.strip()]
            metadata['trigger_words'] = triggers[:5]  # Limit to first 5
            metadata['activation_text'] = trigger_phrase
        
        # Extract training tags frequency
        tag_freq_str = st_metadata.get('ss_tag_frequency', '')
        if tag_freq_str:
            try:
                # Parse the JSON-like tag frequency data
                import ast
                tag_freq = ast.literal_eval(tag_freq_str) if isinstance(tag_freq_str, str) else tag_freq_str
                if isinstance(tag_freq, dict):
                    # Get top 20 most frequent tags
                    all_tags = {}
                    for dataset, tags in tag_freq.items():
                        if isinstance(tags, dict):
                            all_tags.update(tags)
                    
                    sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
                    metadata['training_tags'] = dict(sorted_tags[:20])
            except:
                pass
        
        # Extract other useful metadata
        if not metadata['base_model']:
            metadata['base_model'] = st_metadata.get('ss_base_model_version', '')
            if not metadata['base_model']:
                arch = st_metadata.get('modelspec.architecture', '')
                if 'xl' in arch.lower():
                    metadata['base_model'] = 'SDXL'
                elif 'sd-v1' in arch.lower() or 'sd1' in arch.lower():
                    metadata['base_model'] = 'SD1.5'
        
        if not metadata['resolution']:
            res = st_metadata.get('modelspec.resolution', '') or st_metadata.get('ss_resolution', '')
            metadata['resolution'] = res
        
        # Extract tags from modelspec
        tags_str = st_metadata.get('modelspec.tags', '')
        if tags_str and not metadata['tags']:
            metadata['tags'] = [t.strip() for t in tags_str.split(',') if t.strip()]
    
    # 3. Check .metadata.json sidecar file (SwarmUI creates these, but often empty)
    json_path = lora_path.with_suffix('.metadata.json')
    json_data = read_metadata_json(json_path)
    
    if json_data and json_data.get('civitai'):
        civitai = json_data['civitai']
        # Only use JSON data if LiteDB didn't have it
        if not metadata['model_name']:
            metadata['model_name'] = civitai.get('modelName', '')
        if not metadata['description']:
            metadata['description'] = civitai.get('description', '')
        if not metadata['trigger_words']:
            metadata['trigger_words'] = civitai.get('trainedWords', [])
        if not metadata['base_model']:
            metadata['base_model'] = civitai.get('baseModel', '')
        if not metadata['tags']:
            metadata['tags'] = civitai.get('tags', [])
        metadata['nsfw'] = civitai.get('nsfw', False)
        
        # Try to extract suggested weight from description
        if metadata['description']:
            desc = metadata['description'].lower()
            if 'weight' in desc or 'strength' in desc:
                import re
                match = re.search(r'(?:weight|strength)[\s:]+([0-9.]+)', desc, re.IGNORECASE)
                if match:
                    try:
                        metadata['suggested_weight'] = float(match.group(1))
                    except:
                        pass
    
    return metadata


def scan_lora_directory(base_path: Path, litedb_path: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """
    Scan directory for all LoRA files and extract metadata.
    Organize by subdirectory structure.
    
    Args:
        base_path: Root directory containing LoRA files
        litedb_path: Optional path to LiteDB database file
    """
    print(f"\nScanning LoRA directory: {base_path}")
    
    # First, try to extract all metadata from LiteDB if available
    litedb_metadata = {}
    if litedb_path and litedb_path.exists():
        litedb_metadata = extract_litedb_metadata(litedb_path, base_path)
    
    categories = defaultdict(list)
    total_count = 0
    has_metadata_count = 0
    has_triggers_count = 0
    
    for lora_file in base_path.rglob('*.safetensors'):
        # Skip if it's a checkpoint/model (too large)
        if lora_file.stat().st_size > 500 * 1024 * 1024:  # Skip files > 500MB
            continue
        
        # Get category from directory structure
        rel_path = lora_file.relative_to(base_path)
        category = str(rel_path.parent).replace('\\', '/')
        if category == '.':
            category = 'root'
        
        # Extract metadata (pass LiteDB data)
        metadata = extract_lora_metadata(lora_file, litedb_metadata)
        metadata['category'] = category
        
        # Track stats
        if metadata.get('model_name') or metadata.get('description'):
            has_metadata_count += 1
        if metadata.get('trigger_words'):
            has_triggers_count += 1
        
        categories[category].append(metadata)
        total_count += 1
        
        if total_count % 50 == 0:
            print(f"  Processed {total_count} LoRAs...")
    
    print(f"\nFound {total_count} LoRAs in {len(categories)} categories")
    print(f"  {has_metadata_count} with metadata ({has_metadata_count/total_count*100:.1f}%)")
    print(f"  {has_triggers_count} with trigger words ({has_triggers_count/total_count*100:.1f}%)")
    
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
        'categories': {cat: len(items) for cat, items in sorted(categories.items())}
    }
    
    master_file = output_dir / 'lora_index.yaml'
    with open(master_file, 'w', encoding='utf-8') as f:
        yaml.dump(master_index, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Saved master index: {master_file}")
    
    # Save each category
    for category, items in sorted(categories.items()):
        # Clean category name for filename
        safe_name = category.replace('/', '_').replace('\\', '_').replace('!', 'fav')
        if not safe_name or safe_name == 'root':
            safe_name = 'root'
        
        category_file = output_dir / f'lora_{safe_name}.yaml'
        
        category_data = {
            'category': category,
            'total_loras': len(items),
            'loras': items
        }
        
        with open(category_file, 'w', encoding='utf-8') as f:
            yaml.dump(category_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"  Saved {len(items):3d} LoRAs to: {category_file.name}")
    
    print(f"\nComplete! Saved {len(categories)} category files")


def print_summary(categories: Dict[str, List[Dict]]):
    """
    Print summary statistics
    """
    total = sum(len(items) for items in categories.values())
    with_triggers = sum(1 for items in categories.values() for item in items if item['trigger_words'])
    with_weight = sum(1 for items in categories.values() for item in items if item['suggested_weight'])
    with_metadata = sum(1 for items in categories.values() for item in items if item['model_id'])
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total LoRAs found: {total}")
    print(f"LoRAs with CivitAI metadata: {with_metadata} ({with_metadata/total*100:.1f}%)")
    print(f"LoRAs with trigger words: {with_triggers} ({with_triggers/total*100:.1f}%)")
    print(f"LoRAs with suggested weight: {with_weight} ({with_weight/total*100:.1f}%)")
    
    print("\nCategories:")
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {cat}: {len(items)} LoRAs")
    
    # Print some examples
    print("\nExample LoRAs with trigger words:")
    count = 0
    for category, items in categories.items():
        for item in items:
            if item['trigger_words'] and count < 5:
                print(f"\n  {item['filename']}")
                print(f"    Triggers: {', '.join(item['trigger_words'][:5])}")
                print(f"    Weight: {item['suggested_weight'] or 'not specified'}")
                print(f"    Category: {category}")
                count += 1
            if count >= 5:
                break
        if count >= 5:
            break


def main():
    """Main extraction process"""
    
    print("="*80)
    print("LoRA Metadata Extractor - 3-Tier Extraction (LiteDB + Safetensors + JSON)")
    print("="*80)
    
    # Paths - try folder_paths first, then fallback
    base_dir = Path(__file__).parent.parent
    litedb_path = base_dir / 'scripts' / 'model_metadata.ldb'
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
    
    # Check for LiteDB database
    if not litedb_path.exists():
        print(f"WARNING: LiteDB database not found: {litedb_path}")
        print("Continuing with safetensors + JSON extraction only...\n")
        litedb_path = None
    
    # Scan directory with optional LiteDB
    categories = scan_lora_directory(lora_base_path, litedb_path)
    
    if not categories:
        print("\nERROR: No LoRAs found!")
        return
    
    # Save to YAML files
    save_metadata_yaml(categories, output_dir)
    
    # Print summary
    print_summary(categories)


if __name__ == '__main__':
    main()
