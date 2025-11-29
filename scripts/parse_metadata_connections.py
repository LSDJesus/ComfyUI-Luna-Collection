#!/usr/bin/env python3
"""
Parse LoRA/embedding metadata YAML files and auto-populate connections.json.

This enhanced version reads the extracted Civitai metadata from YAML files
to get accurate trigger words, suggested weights, and training tags.

Usage:
    python scripts/parse_metadata_connections.py
    
The script will:
1. Read lora_metadata/*.yaml files for LoRA info
2. Read embedding_metadata/*.yaml files for embedding info
3. Map folder structure to wildcard categories
4. Extract trigger words and suggested weights from metadata
5. Generate/update wildcards_atomic/connections.json
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import yaml

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LORA_METADATA_DIR = PROJECT_ROOT / "lora_metadata"
EMBEDDING_METADATA_DIR = PROJECT_ROOT / "embedding_metadata"

# Output path for connections.json
CONNECTIONS_OUTPUT = Path(r"D:\AI\SD Models\wildcards_atomic\connections.json")


@dataclass
class Connection:
    """Represents a LoRA or embedding connection"""
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    model_type: str = "any"
    weight_range: Dict[str, float] = field(default_factory=lambda: {"min": 0.3, "max": 1.0})
    # New fields from metadata
    activation_text: str = ""
    training_tags: Dict[str, int] = field(default_factory=dict)
    civitai_tags: List[str] = field(default_factory=list)


# Category mapping from folder/category names to wildcard categories
CATEGORY_MAP = {
    # Body-related 
    "body": ("body", ["anatomy", "physique"]),
    "age": ("body.age", ["age", "slider"]),
    "breasts": ("body.breasts", ["anatomy", "chest"]),
    "tits": ("body.breasts", ["anatomy", "chest"]),
    "eyes": ("body.eyes", ["anatomy", "face"]),
    "face": ("body.face", ["anatomy", "appearance"]),
    "full_body": ("body.full", ["anatomy", "physique"]),
    "hair": ("body.hair", ["appearance", "style"]),
    "penis": ("body.penis", ["anatomy", "nsfw"]),
    "pussy": ("body.pussy", ["anatomy", "nsfw"]),
    "birth": ("body.pregnancy", ["pregnancy", "nsfw"]),
    
    # Character folders
    "characters": ("character", ["character", "persona"]),
    "characters/bug noir": ("character.miraculous", ["miraculous", "ladybug"]),
    "characters/bug_noir": ("character.miraculous", ["miraculous", "ladybug"]),
    "characters/dark_arts": ("character.dark_arts", ["oc", "original"]),
    "characters/eternum": ("character.eternum", ["game", "visual_novel"]),
    
    # Clothing
    "clothes": ("clothing", ["outfit", "fashion"]),
    "clothing": ("clothing", ["outfit", "fashion"]),
    
    # Position/Pose
    "position_pose": ("pose", ["position", "composition"]),
    "position": ("pose.position", ["position", "sex"]),
    "pose": ("pose", ["posture", "stance"]),
    
    # Effects
    "cum": ("effect.cum", ["nsfw", "fluid"]),
    "squirt": ("effect.squirt", ["nsfw", "fluid"]),
    "glow": ("effect.glow", ["lighting", "special"]),
    
    # Detail/Style
    "detail": ("detail", ["enhancement", "quality"]),
    "realism": ("style.realism", ["style", "quality"]),
    "style": ("style", ["art_style", "aesthetic"]),
    "lighting": ("lighting", ["atmosphere", "mood"]),
    "reijil": ("style.reijil", ["artistic", "quality"]),
    
    # Special
    "futanari": ("nsfw.futanari", ["nsfw", "futa"]),
    "toy": ("nsfw.toy", ["nsfw", "accessory"]),
    
    # HDA 
    "hda": ("style.hda", ["quality", "detailed"]),
    "hda/illus": ("style.hda.illustrious", ["quality", "detailed"]),
    
    # Favorites/Sliders
    "!favorites": ("favorite", ["slider", "utility"]),
    "!incs": ("style.incs", ["style", "artistic"]),
    "!try": ("experimental", ["testing", "trial"]),
    
    # Root level
    "root": ("misc", []),
    
    # Embeddings
    "all": ("embedding.quality", ["quality", "general"]),
    "egirls": ("embedding.character.egirl", ["character", "vicious", "realistic"]),
    "illustrious": ("embedding.quality.illustrious", ["quality", "illustrious"]),
    "pony": ("embedding.quality.pony", ["quality", "pony"]),
    "sdxl": ("embedding.quality.sdxl", ["quality", "sdxl"]),
}


def detect_model_type(base_model: str, file_path: str, category: str = "") -> str:
    """Detect model type from base_model field and file path."""
    path_lower = file_path.lower() if file_path else ""
    base_lower = base_model.lower() if base_model else ""
    cat_lower = category.lower() if category else ""
    
    # Check base_model field first
    if 'pony' in base_lower or 'pdxl' in base_lower:
        return "pony"
    if 'illustrious' in base_lower or 'noob' in base_lower:
        return "illustrious"
    
    # Check file path
    if '\\pony\\' in path_lower or '/pony/' in path_lower:
        return "pony"
    if '\\illustrious\\' in path_lower or '/illustrious/' in path_lower:
        return "illustrious"
    if '\\sdxl\\' in path_lower or '/sdxl/' in path_lower:
        return "sdxl"
    if '\\sd15\\' in path_lower or '/sd15/' in path_lower or '\\sd1.5\\' in path_lower:
        return "sd15"
    if '\\flux\\' in path_lower or '/flux/' in path_lower:
        return "flux"
    
    # Check category for embeddings
    if 'pony' in cat_lower:
        return "pony"
    if 'illustrious' in cat_lower or 'illus' in cat_lower:
        return "illustrious"
    if 'sdxl' in cat_lower:
        return "sdxl"
    
    # Default based on sdxl_base
    if 'sdxl' in base_lower:
        return "sdxl"  # Could be pony or illustrious, but we know it's XL
    
    return "any"


def clean_trigger_words(trigger_words: List[str]) -> List[str]:
    """Clean and deduplicate trigger words."""
    if not trigger_words:
        return []
    
    cleaned = []
    seen = set()
    
    for word in trigger_words:
        if not word or not isinstance(word, str):
            continue
        
        # Clean the word
        word = word.strip().lower()
        
        # Skip very short or generic words
        if len(word) < 2:
            continue
        if word in {'a', 'an', 'the', 'and', 'or', 'with', 'at', 'on', 'in', 'to', 'for'}:
            continue
        
        # Skip if already seen
        if word in seen:
            continue
        
        seen.add(word)
        cleaned.append(word)
    
    return cleaned[:10]  # Limit to 10 most important triggers


def extract_weight_range(suggested_weight: Optional[float], tags: List[str]) -> Dict[str, float]:
    """Extract weight range from metadata or infer from tags."""
    if suggested_weight is not None:
        # Create a range around the suggested weight
        min_w = max(0.1, suggested_weight - 0.3)
        max_w = min(2.0, suggested_weight + 0.3)
        return {"min": round(min_w, 2), "max": round(max_w, 2), "suggested": suggested_weight}
    
    # Infer from tags
    tags_lower = [t.lower() for t in tags] if tags else []
    
    if 'slider' in tags_lower:
        return {"min": -1.5, "max": 1.5}  # Sliders typically use negative to positive
    if 'style' in tags_lower or 'concept' in tags_lower:
        return {"min": 0.5, "max": 1.2}
    if 'character' in tags_lower:
        return {"min": 0.7, "max": 1.0}
    
    return {"min": 0.3, "max": 1.0}


def get_top_training_tags(training_tags: Dict[str, int], limit: int = 10) -> Dict[str, int]:
    """Get the top N training tags by frequency."""
    if not training_tags:
        return {}
    
    # Sort by count and take top N
    sorted_tags = sorted(training_tags.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_tags[:limit])


def map_category_to_wildcards(category: str) -> Tuple[List[str], List[str]]:
    """Map a metadata category to wildcard categories and tags."""
    cat_lower = category.lower().replace(' ', '_')
    
    categories = set()
    tags = set()
    
    # Direct match
    if cat_lower in CATEGORY_MAP:
        cat, cat_tags = CATEGORY_MAP[cat_lower]
        categories.add(cat)
        tags.update(cat_tags)
    
    # Try parent categories
    parts = cat_lower.split('/')
    for i, part in enumerate(parts):
        part_clean = part.replace('!', '').strip()
        if part_clean in CATEGORY_MAP:
            cat, cat_tags = CATEGORY_MAP[part_clean]
            categories.add(cat)
            tags.update(cat_tags)
        elif part_clean:
            # Use as-is
            categories.add(part_clean)
    
    return sorted(list(categories)), sorted(list(tags))


def process_lora_metadata() -> Dict[str, Connection]:
    """Process all LoRA metadata YAML files."""
    print("\n📂 Processing LoRA metadata...")
    
    connections = {}
    total_count = 0
    triggers_count = 0
    
    # Find all YAML files except index
    yaml_files = list(LORA_METADATA_DIR.glob("lora_*.yaml"))
    yaml_files = [f for f in yaml_files if 'index' not in f.name]
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"   ⚠ Error reading {yaml_file.name}: {e}")
            continue
        
        if not data or 'loras' not in data:
            continue
        
        category = data.get('category', 'root')
        
        for lora in data['loras']:
            filename = lora.get('filename', '')
            if not filename:
                continue
            
            # Extract name without extension
            name = filename.replace('.safetensors', '')
            
            # Get categories and tags from folder structure
            categories, folder_tags = map_category_to_wildcards(category)
            
            # Get trigger words - clean them up
            raw_triggers = lora.get('trigger_words', [])
            triggers = clean_trigger_words(raw_triggers)
            if triggers:
                triggers_count += 1
            
            # Get Civitai tags
            civitai_tags = lora.get('tags', [])
            all_tags = set(folder_tags)
            all_tags.update([t.lower().replace(' ', '_') for t in civitai_tags if t])
            
            # Detect model type
            model_type = detect_model_type(
                lora.get('base_model', ''),
                lora.get('file_path', ''),
                category
            )
            
            # Get weight range
            weight_range = extract_weight_range(
                lora.get('suggested_weight'),
                civitai_tags
            )
            
            # Get top training tags
            training_tags = get_top_training_tags(lora.get('training_tags', {}), 15)
            
            # Add tags from training_tags keys
            for ttag in list(training_tags.keys())[:5]:
                clean_tag = ttag.lower().replace(' ', '_')
                if len(clean_tag) > 2 and clean_tag not in {'1girl', '1boy', 'solo'}:
                    all_tags.add(clean_tag)
            
            conn = Connection(
                categories=categories,
                tags=sorted(list(all_tags)),
                triggers=triggers,
                model_type=model_type,
                weight_range=weight_range,
                activation_text=lora.get('activation_text', '')[:500],  # Truncate long text
                training_tags=training_tags,
                civitai_tags=civitai_tags
            )
            
            connections[name] = conn
            total_count += 1
    
    print(f"   ✓ Processed {total_count} LoRAs from {len(yaml_files)} files")
    print(f"   ✓ Found trigger words for {triggers_count} LoRAs")
    return connections


def process_embedding_metadata() -> Dict[str, Connection]:
    """Process all embedding metadata YAML files."""
    print("\n📂 Processing Embedding metadata...")
    
    connections = {}
    total_count = 0
    triggers_count = 0
    
    # Find all YAML files except index
    yaml_files = list(EMBEDDING_METADATA_DIR.glob("embedding_*.yaml"))
    yaml_files = [f for f in yaml_files if 'index' not in f.name]
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"   ⚠ Error reading {yaml_file.name}: {e}")
            continue
        
        if not data or 'embeddings' not in data:
            continue
        
        category = data.get('category', 'root')
        
        for emb in data['embeddings']:
            filename = emb.get('filename', '')
            if not filename:
                continue
            
            # Extract name without extension
            name = filename.replace('.safetensors', '')
            
            # Get categories and tags from folder structure
            categories, folder_tags = map_category_to_wildcards(category)
            
            # Add embedding-specific categories
            categories = list(set(categories))
            if not any(c.startswith('embedding.') for c in categories):
                categories.append('embedding')
            
            # Get trigger word (singular for embeddings)
            trigger_word = emb.get('trigger_word', '')
            triggers = [trigger_word.lower()] if trigger_word else []
            if triggers:
                triggers_count += 1
            
            # Get tags from metadata
            civitai_tags = emb.get('tags', [])
            all_tags = set(folder_tags)
            all_tags.update([t.lower().replace(' ', '_') for t in civitai_tags if t])
            
            # Detect if positive or negative embedding
            if '-neg' in filename.lower() or 'negative' in filename.lower():
                all_tags.add('negative')
                categories.append('embedding.negative')
            else:
                all_tags.add('positive')
                if 'embedding.positive' not in categories:
                    categories.append('embedding.positive')
            
            # Detect model type from base_model or path
            base_model = emb.get('base_model', '')
            model_type = detect_model_type(
                base_model,
                emb.get('file_path', ''),
                category
            )
            
            # Embeddings typically use different weight range
            weight_range = {"min": 0.5, "max": 1.5}
            
            # Special handling for character embeddings
            if 'vicious' in name.lower():
                all_tags.add('vicious_character')
                all_tags.add('realistic')
            if 'tower13' in name.lower():
                all_tags.add('tower13')
            if 'hda' in category.lower():
                all_tags.add('hda_style')
            
            conn = Connection(
                categories=sorted(list(set(categories))),
                tags=sorted(list(all_tags)),
                triggers=triggers,
                model_type=model_type,
                weight_range=weight_range,
                activation_text='',
                training_tags={},
                civitai_tags=civitai_tags
            )
            
            connections[name] = conn
            total_count += 1
    
    print(f"   ✓ Processed {total_count} embeddings from {len(yaml_files)} files")
    print(f"   ✓ Found trigger words for {triggers_count} embeddings")
    return connections


def main():
    """Main entry point."""
    print("🔍 Luna Connection Auto-Populator (Metadata Version)")
    print("=" * 60)
    
    # Check for metadata directories
    if not LORA_METADATA_DIR.exists():
        print(f"❌ LoRA metadata directory not found: {LORA_METADATA_DIR}")
        return
    
    if not EMBEDDING_METADATA_DIR.exists():
        print(f"❌ Embedding metadata directory not found: {EMBEDDING_METADATA_DIR}")
        return
    
    # Process metadata
    all_loras = process_lora_metadata()
    all_embeddings = process_embedding_metadata()
    
    # Convert to JSON-serializable format
    def conn_to_dict(conn: Connection) -> dict:
        d = asdict(conn)
        # Only include non-empty fields
        return {k: v for k, v in d.items() if v}
    
    loras_json = {name: conn_to_dict(conn) for name, conn in all_loras.items()}
    embeddings_json = {name: conn_to_dict(conn) for name, conn in all_embeddings.items()}
    
    # Create output structure
    connections_data = {
        "version": "2.0",
        "description": "Auto-generated from Civitai metadata with trigger words and weights",
        "generated_from": "lora_metadata/ and embedding_metadata/ YAML files",
        "loras": loras_json,
        "embeddings": embeddings_json
    }
    
    # Write to file
    CONNECTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONNECTIONS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(connections_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated connections.json")
    print(f"   📁 Location: {CONNECTIONS_OUTPUT}")
    print(f"   📊 Total LoRAs: {len(all_loras)}")
    print(f"   📊 Total Embeddings: {len(all_embeddings)}")
    
    # Print sample with trigger words
    print(f"\n📋 Sample LoRAs with trigger words:")
    sample_count = 0
    for name, conn in all_loras.items():
        if conn.triggers and sample_count < 5:
            print(f"   • {name}")
            print(f"     Triggers: {conn.triggers}")
            print(f"     Categories: {conn.categories}")
            print(f"     Model: {conn.model_type}")
            if 'suggested' in conn.weight_range:
                print(f"     Suggested Weight: {conn.weight_range['suggested']}")
            sample_count += 1
    
    # Print embedding samples
    print(f"\n📋 Sample embeddings with triggers:")
    sample_count = 0
    for name, conn in all_embeddings.items():
        if conn.triggers and conn.triggers[0] and sample_count < 5:
            print(f"   • {name}")
            print(f"     Trigger: {conn.triggers[0]}")
            print(f"     Categories: {conn.categories}")
            sample_count += 1
    
    # Print category summary
    print(f"\n📊 Category Summary:")
    all_categories = set()
    for conn in all_loras.values():
        all_categories.update(conn.categories)
    for conn in all_embeddings.values():
        all_categories.update(conn.categories)
    
    for cat in sorted(all_categories)[:25]:
        print(f"   • {cat}")
    if len(all_categories) > 25:
        print(f"   ... and {len(all_categories) - 25} more")


if __name__ == "__main__":
    main()
