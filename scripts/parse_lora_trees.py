#!/usr/bin/env python3
"""
Parse LoRA tree files and auto-populate connections.json with intelligent category/tag mappings.

This script reads the tree_lora_*.txt files and creates structured connections 
based on folder hierarchy. Each folder name becomes a category/tag, and the
model type is inferred from the tree file name (pony, illustrious, etc.).

Usage:
    python scripts/parse_lora_trees.py
    
The script will:
1. Parse tree_lora_pony.txt → model_type: "pony"
2. Parse tree_lora_illustrious.txt → model_type: "illustrious" 
3. Parse tree_embeddings.txt → embeddings with model detection
4. Generate/update wildcards_atomic/connections.json
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "Docs"

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


# Category mapping from folder names to wildcard categories
# Maps folder names (lowercase) to (category, additional_tags)
FOLDER_CATEGORY_MAP = {
    # Body-related folders
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
    "bug noir": ("character.miraculous", ["miraculous", "ladybug"]),
    "bug_noir": ("character.miraculous", ["miraculous", "ladybug"]),
    "bug girl": ("character.miraculous", ["miraculous", "ladybug"]),
    "dark_arts": ("character.dark_arts", ["oc", "original"]),
    "eternum": ("character.eternum", ["game", "visual_novel"]),
    "stableyogi instagirls": ("character.instagirls", ["realistic", "influencer"]),
    
    # Clothing folders
    "clothes": ("clothing", ["outfit", "fashion"]),
    "clothing": ("clothing", ["outfit", "fashion"]),
    
    # Position/Pose folders
    "position_pose": ("pose", ["position", "composition"]),
    "position": ("pose.position", ["position", "sex"]),
    "pose": ("pose", ["posture", "stance"]),
    
    # Effect folders
    "cum": ("effect.cum", ["nsfw", "fluid"]),
    "squirt": ("effect.squirt", ["nsfw", "fluid"]),
    "glow": ("effect.glow", ["lighting", "special"]),
    
    # Detail/Style folders
    "detail": ("detail", ["enhancement", "quality"]),
    "realism": ("style.realism", ["style", "quality"]),
    "style": ("style", ["art_style", "aesthetic"]),
    "lighting": ("lighting", ["atmosphere", "mood"]),
    
    # Special folders
    "futanari": ("nsfw.futanari", ["nsfw", "futa"]),
    "toy": ("nsfw.toy", ["nsfw", "accessory"]),
    
    # HDA (High Detail Art) folders  
    "hda": ("style.hda", ["quality", "detailed"]),
    "illus": ("style.hda.illustrious", ["quality", "detailed"]),
    
    # Favorites/Sliders
    "!favorites": ("favorite", ["slider", "utility"]),
    "!incs": ("style.incs", ["style", "artistic"]),
    "!try": ("experimental", ["testing", "trial"]),
    
    # Reijil style
    "reijil": ("style.reijil", ["artistic", "quality"]),
}


# Keywords in filenames that map to tags
FILENAME_TAG_PATTERNS = {
    # Slider patterns
    r"slider": ["slider", "adjustable"],
    r"_v\d": ["versioned"],
    
    # Body/Anatomy
    r"breast|boob|tit": ["breasts", "chest"],
    r"nipple": ["nipples", "anatomy"],
    r"ass|butt": ["ass", "anatomy"],
    r"pussy|vagina|vulva": ["pussy", "anatomy"],
    r"penis|cock|dick": ["penis", "anatomy"],
    r"clitor": ["clitoris", "anatomy"],
    r"pregnant|belly": ["pregnant", "pregnancy"],
    r"muscle|toned|athletic": ["muscle", "fit"],
    r"skin": ["skin", "texture"],
    r"eye": ["eyes", "face"],
    r"hair": ["hair", "appearance"],
    r"face|beauty": ["face", "beauty"],
    r"age": ["age", "appearance"],
    r"weight|body": ["body_type", "physique"],
    
    # Clothing
    r"lingerie": ["lingerie", "underwear"],
    r"bikini": ["bikini", "swimwear"],
    r"dress": ["dress", "clothing"],
    r"bodysuit": ["bodysuit", "clothing"],
    r"harness": ["harness", "accessory"],
    r"latex|leather": ["latex", "material"],
    r"stocking": ["stockings", "legwear"],
    
    # Positions/Actions
    r"cowgirl": ["cowgirl", "position"],
    r"dogg[iy]": ["doggystyle", "position"],
    r"missionary": ["missionary", "position"],
    r"fellatio|blowjob|oral": ["oral", "action"],
    r"finger": ["fingering", "action"],
    r"masturbat": ["masturbation", "solo"],
    r"cum|creampie": ["cum", "fluid"],
    r"squirt|ejaculat": ["squirt", "fluid"],
    
    # Effects/Style
    r"glow": ["glowing", "effect"],
    r"neon": ["neon", "lighting"],
    r"detail": ["detailed", "quality"],
    r"realism|realistic": ["realistic", "style"],
    r"anime": ["anime", "style"],
    r"disney": ["disney", "cartoon"],
    r"smooth": ["smooth", "quality"],
    
    # Special
    r"futa": ["futanari", "nsfw"],
    r"dildo|toy|vibrat": ["toy", "nsfw"],
    r"bondage|bdsm|shibari": ["bondage", "bdsm"],
}


def parse_tree_file(file_path: Path) -> Dict[str, List[str]]:
    """
    Parse a tree file and extract folder -> files mapping.
    
    Returns:
        Dict mapping folder paths to list of .safetensors files
    """
    if not file_path.exists():
        print(f"  ⚠ File not found: {file_path}")
        return {}
    
    folders: Dict[str, List[str]] = {}
    current_path: List[str] = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
            
        # Check for folder marker (+---)
        if '+---' in line:
            folder_name = line.split('+---')[-1].strip()
            # Determine depth by counting tree characters
            depth = line.count('│') + line.count('�')
            
            # Adjust current path to match depth
            while len(current_path) > depth:
                current_path.pop()
            current_path.append(folder_name)
            
            folder_key = '/'.join(current_path)
            if folder_key not in folders:
                folders[folder_key] = []
        
        # Check for .safetensors file
        elif '.safetensors' in line and '.sha256' not in line:
            filename = line.split('│')[-1].split('�')[-1].strip()
            if filename.endswith('.safetensors'):
                folder_key = '/'.join(current_path) if current_path else "root"
                if folder_key not in folders:
                    folders[folder_key] = []
                folders[folder_key].append(filename)
    
    return folders


def extract_lora_name(filename: str) -> str:
    """Extract clean LoRA name from filename."""
    # Remove extension
    name = filename.replace('.safetensors', '')
    return name


def detect_model_type_from_filename(filename: str) -> Optional[str]:
    """Try to detect model type from filename patterns."""
    filename_lower = filename.lower()
    
    if any(x in filename_lower for x in ['pony', 'pdxl', 'pny']):
        return "pony"
    elif any(x in filename_lower for x in ['illust', 'ilxl', 'ixl', 'noob']):
        return "illustrious"
    elif any(x in filename_lower for x in ['sdxl', 'xl']):
        return "sdxl"
    elif any(x in filename_lower for x in ['sd15', 'sd1.5']):
        return "sd15"
    elif 'flux' in filename_lower:
        return "flux"
    
    return None


def get_categories_and_tags(folder_path: str, filename: str) -> Tuple[List[str], List[str]]:
    """
    Determine categories and tags from folder path and filename.
    
    Returns:
        Tuple of (categories, tags)
    """
    categories = set()
    tags = set()
    
    # Process folder path
    folder_parts = folder_path.lower().split('/')
    for part in folder_parts:
        if part in FOLDER_CATEGORY_MAP:
            cat, folder_tags = FOLDER_CATEGORY_MAP[part]
            categories.add(cat)
            tags.update(folder_tags)
        elif part and part != "root":
            # Use folder name as-is for unknown folders
            categories.add(part.replace(' ', '_'))
            tags.add(part.replace(' ', '_').replace('!', ''))
    
    # Process filename patterns
    filename_lower = filename.lower()
    for pattern, pattern_tags in FILENAME_TAG_PATTERNS.items():
        if re.search(pattern, filename_lower):
            tags.update(pattern_tags)
    
    # Infer some categories from tags
    if 'slider' in tags:
        categories.add('utility.slider')
    
    return sorted(list(categories)), sorted(list(tags))


def extract_triggers_from_filename(filename: str) -> List[str]:
    """Try to extract trigger words from LoRA filename."""
    triggers = []
    
    # Common patterns that might be triggers
    name = filename.replace('.safetensors', '')
    
    # Check for character names (usually PascalCase or underscored)
    if re.match(r'^[A-Z][a-z]+(_[A-Z][a-z]+)*', name):
        # Might be a character name
        clean_name = name.split('_')[0]
        if len(clean_name) > 2:
            triggers.append(clean_name.lower())
    
    return triggers


def process_tree_for_model_type(tree_file: Path, model_type: str) -> Dict[str, Connection]:
    """
    Process a tree file and create connections for each LoRA.
    
    Returns:
        Dict mapping LoRA name to Connection
    """
    print(f"\n📂 Processing {tree_file.name} (model_type: {model_type})")
    
    folders = parse_tree_file(tree_file)
    connections = {}
    
    for folder_path, files in folders.items():
        for filename in files:
            lora_name = extract_lora_name(filename)
            
            # Skip metadata or hash files
            if lora_name.endswith('.metadata') or lora_name.endswith('.sha256'):
                continue
            
            # Get categories and tags
            categories, tags = get_categories_and_tags(folder_path, filename)
            
            # Override model_type if detected from filename
            detected_type = detect_model_type_from_filename(filename)
            final_model_type = detected_type or model_type
            
            # Extract potential triggers
            triggers = extract_triggers_from_filename(filename)
            
            # Create connection
            conn = Connection(
                categories=categories,
                tags=tags,
                triggers=triggers,
                model_type=final_model_type,
                weight_range={"min": 0.3, "max": 1.0}
            )
            
            connections[lora_name] = conn
    
    print(f"   ✓ Found {len(connections)} LoRAs")
    return connections


def process_embeddings_tree(tree_file: Path) -> Dict[str, Connection]:
    """
    Process embeddings tree file and create connections.
    
    Returns:
        Dict mapping embedding name to Connection
    """
    print(f"\n📂 Processing {tree_file.name} (embeddings)")
    
    folders = parse_tree_file(tree_file)
    connections = {}
    
    # Embedding folder -> category/tags mapping
    EMBEDDING_FOLDER_MAP = {
        "all": ("embedding.quality", ["quality", "general"]),
        "characters": ("embedding.character", ["character", "persona"]),
        "egirls": ("embedding.character.egirl", ["character", "vicious", "realistic"]),
        "hda": ("embedding.style.hda", ["style", "quality", "detailed"]),
        "illustrious": ("embedding.quality.illustrious", ["quality", "illustrious"]),
        "pony": ("embedding.quality.pony", ["quality", "pony"]),
        "sdxl": ("embedding.quality.sdxl", ["quality", "sdxl"]),
        "illus": ("embedding.style.hda.illustrious", ["style", "detailed"]),
    }
    
    for folder_path, files in folders.items():
        for filename in files:
            if not filename.endswith('.safetensors'):
                continue
                
            emb_name = extract_lora_name(filename)
            
            # Get folder-based categories
            categories = set()
            tags = set()
            
            folder_parts = folder_path.lower().split('/')
            for part in folder_parts:
                if part in EMBEDDING_FOLDER_MAP:
                    cat, folder_tags = EMBEDDING_FOLDER_MAP[part]
                    categories.add(cat)
                    tags.update(folder_tags)
                elif part and part != "root":
                    categories.add(f"embedding.{part}")
                    tags.add(part)
            
            # Detect model type from filename/folder
            model_type = "any"
            for part in folder_parts:
                if 'pony' in part.lower():
                    model_type = "pony"
                elif 'illust' in part.lower():
                    model_type = "illustrious"
                elif 'sdxl' in part.lower():
                    model_type = "sdxl"
            
            # Check filename for model hints
            detected = detect_model_type_from_filename(filename)
            if detected:
                model_type = detected
            
            # Detect if negative embedding
            if '-neg' in filename or 'negative' in filename.lower():
                categories.add("embedding.negative")
                tags.add("negative")
            else:
                categories.add("embedding.positive")
                tags.add("positive")
            
            # Character embedding patterns
            if any(x in folder_path.lower() for x in ['character', 'egirl', 'vicious']):
                tags.add("character")
                # Try to extract character name
                if 'DV_' in emb_name or 'DV2_' in emb_name or 'DV3_' in emb_name or 'DV4_' in emb_name:
                    tags.add("vicious_character")
                elif 'Tower13_' in emb_name:
                    tags.add("tower13")
                elif 'TrualityCampus_' in emb_name:
                    tags.add("truality_campus")
            
            conn = Connection(
                categories=sorted(list(categories)),
                tags=sorted(list(tags)),
                triggers=[],  # Embeddings don't typically have triggers
                model_type=model_type,
                weight_range={"min": 0.5, "max": 1.5}
            )
            
            connections[emb_name] = conn
    
    print(f"   ✓ Found {len(connections)} embeddings")
    return connections


def main():
    """Main entry point."""
    print("🔍 Luna Connection Auto-Populator")
    print("=" * 50)
    
    all_loras = {}
    all_embeddings = {}
    
    # Process LoRA trees
    lora_trees = [
        (DOCS_DIR / "tree_lora_pony.txt", "pony"),
        (DOCS_DIR / "tree_lora_illustrious.txt", "illustrious"),
    ]
    
    for tree_file, model_type in lora_trees:
        if tree_file.exists():
            connections = process_tree_for_model_type(tree_file, model_type)
            all_loras.update(connections)
    
    # Process embeddings tree
    embeddings_tree = DOCS_DIR / "tree_embeddings.txt"
    if embeddings_tree.exists():
        all_embeddings = process_embeddings_tree(embeddings_tree)
    
    # Convert to JSON-serializable format
    loras_json = {name: asdict(conn) for name, conn in all_loras.items()}
    embeddings_json = {name: asdict(conn) for name, conn in all_embeddings.items()}
    
    # Create output structure
    connections_data = {
        "version": "1.0",
        "description": "Auto-generated LoRA/embedding connections based on folder structure",
        "loras": loras_json,
        "embeddings": embeddings_json
    }
    
    # Write to file
    CONNECTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONNECTIONS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(connections_data, f, indent=2)
    
    print(f"\n✅ Generated connections.json")
    print(f"   📁 Location: {CONNECTIONS_OUTPUT}")
    print(f"   📊 Total LoRAs: {len(all_loras)}")
    print(f"   📊 Total Embeddings: {len(all_embeddings)}")
    
    # Print category summary
    print(f"\n📊 Category Summary:")
    all_categories = set()
    for conn in all_loras.values():
        all_categories.update(conn.categories)
    for conn in all_embeddings.values():
        all_categories.update(conn.categories)
    
    for cat in sorted(all_categories)[:20]:
        print(f"   • {cat}")
    if len(all_categories) > 20:
        print(f"   ... and {len(all_categories) - 20} more")


if __name__ == "__main__":
    main()
