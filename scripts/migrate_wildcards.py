"""
Wildcard Migration Tool - Convert .txt wildcards to YAML format using LM Studio

This script processes your existing wildcard .txt files and uses a local LLM
to classify, enhance, and convert them to the new YAML format.

Requirements:
- LM Studio running with OpenAI-compatible API enabled (default: http://localhost:1234)
- A suitable model loaded (recommended: Llama 3.2 3B, Mistral 7B, or similar)

Usage:
    python migrate_wildcards.py --input /path/to/txt/wildcards --output ./wildcards

Process:
1. Scans input directory for .txt files
2. Reads each file and sends contents to LLM
3. LLM analyzes and suggests:
   - YAML category (existing or new)
   - Tag classification
   - Whitelist/blacklist rules
   - Weights and payloads (if applicable)
4. Shows preview and asks for confirmation
5. Merges into existing YAML or creates new file
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import json

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)


class WildcardMigrator:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", model: str = None):
        """
        Initialize migrator with LM Studio connection
        
        Args:
            lm_studio_url: Base URL for LM Studio API
            model: Model name (if None, uses whatever is loaded)
        """
        self.client = OpenAI(
            base_url=lm_studio_url,
            api_key="lm-studio"  # LM Studio doesn't validate this
        )
        self.model = model or "local-model"  # LM Studio ignores this usually
        
        # Test connection
        try:
            self.client.models.list()
            print(f"✓ Connected to LM Studio at {lm_studio_url}")
        except Exception as e:
            print(f"✗ Failed to connect to LM Studio: {e}")
            print("Make sure LM Studio is running with API server enabled")
            sys.exit(1)
    
    def load_existing_wildcards(self, wildcards_dir: Path) -> Dict[str, Dict]:
        """Load all existing YAML wildcards to provide context to LLM"""
        existing = {}
        
        if not wildcards_dir.exists():
            wildcards_dir.mkdir(parents=True, exist_ok=True)
            return existing
        
        for yaml_file in wildcards_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'name' in data:
                        existing[data['name']] = {
                            'file': yaml_file,
                            'description': data.get('description', ''),
                            'items': data.get('items', []),
                            'tags': self._extract_all_tags(data.get('items', []))
                        }
            except Exception as e:
                print(f"Warning: Could not load {yaml_file.name}: {e}")
        
        return existing
    
    def _extract_all_tags(self, items: List[Dict]) -> set:
        """Extract unique tags from all items"""
        tags = set()
        for item in items:
            tags.update(item.get('tags', []))
        return tags
    
    def analyze_txt_wildcard(self, txt_content: str, txt_filename: str, 
                            existing_wildcards: Dict[str, Dict]) -> Optional[Dict]:
        """
        Send .txt wildcard to LLM for analysis and YAML conversion
        
        Returns dict with:
        - line_items: List of dicts, each with category and item data
        - Can assign different lines to different categories
        """
        
        # Build context about existing wildcards with descriptions
        category_list = []
        for name, data in existing_wildcards.items():
            desc = data.get('description', '')
            category_list.append(f"  - {name}: {desc}" if desc else f"  - {name}")
        
        existing_tags = set()
        for wc in existing_wildcards.values():
            existing_tags.update(wc['tags'])
        
        # Construct prompt
        system_prompt = """You are a wildcard classification assistant. Your job is to analyze text-based wildcards and convert them to structured YAML format with semantic tags.

IMPORTANT: Each line can be assigned to a DIFFERENT category based on what it describes. Don't force all lines into one category.

Guidelines:
- Match each line to the most appropriate existing category
- Only suggest new categories if no existing category fits
- Assign relevant tags (lowercase, single words like: scifi, medieval, fantasy, modern, nature, urban, combat, elegant)
- Set blacklist tags for incompatible contexts (e.g., medieval ↔ scifi)
- Set whitelist tags for required contexts (optional, only if strictly needed)
- Assign weights (0.5-2.0, default 1.0) based on how "default" or "specialized" each item is
- Extract any LoRA/embedding syntax and put in payload field

Output ONLY valid JSON in this exact format:
{
  "line_items": [
    {
      "category": "category_name",
      "item": {
        "id": "unique_snake_case_id",
        "text": "the prompt text",
        "tags": ["tag1", "tag2"],
        "blacklist": ["incompatible_tag"],
        "whitelist": [],
        "weight": 1.0,
        "payload": "<lora:...> if present or empty string"
      }
    }
  ],
  "reasoning": "brief explanation of categorization choices"
}"""
        
        user_prompt = f"""Analyze this wildcard file and convert to YAML format:

**Filename:** {txt_filename}
(Use the filename as context for understanding the theme and purpose of these wildcards)

**Available categories:**
{chr(10).join(category_list) if category_list else 'none - you may suggest new categories'}

**Existing tags in library:** {', '.join(sorted(existing_tags)[:50]) if existing_tags else 'none'}

**Wildcard content:**
```
{txt_content[:2000]}  # Limit to first 2000 chars
```

IMPORTANT: Analyze each line individually and assign it to the MOST APPROPRIATE category. Different lines can go to different categories. The filename "{txt_filename}" provides context but don't force all lines into one category if they describe different things (e.g., a file might contain both clothing and accessories)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower = more deterministic
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON (LLM might wrap it in markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Convert new format to old format for compatibility
            # Group items by category
            if 'line_items' in result:
                categories = {}
                for line_item in result['line_items']:
                    cat = line_item['category']
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(line_item['item'])
                
                # Return list of results, one per category
                return [{
                    'category': cat,
                    'items': items,
                    'reasoning': result.get('reasoning', '')
                } for cat, items in categories.items()]
            else:
                # Old format compatibility
                return [result]
            
        except json.JSONDecodeError as e:
            print(f"✗ LLM returned invalid JSON: {e}")
            print(f"Raw response: {content[:500]}")
            return None
        except Exception as e:
            print(f"✗ LLM request failed: {e}")
            return None
    
    def merge_into_yaml(self, category: str, new_items: List[Dict], 
                       wildcards_dir: Path, existing_wildcards: Dict[str, Dict]) -> bool:
        """
        Merge new items into existing YAML or create new file
        
        Returns True if successful
        """
        yaml_path = wildcards_dir / f"{category}.yaml"
        
        if category in existing_wildcards:
            # Load existing
            with open(existing_wildcards[category]['file'], 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Get existing IDs to avoid duplicates
            existing_ids = {item['id'] for item in data['items']}
            
            # Add new items (skip duplicates)
            added_count = 0
            for item in new_items:
                if item['id'] not in existing_ids:
                    data['items'].append(item)
                    added_count += 1
                else:
                    print(f"  ⊘ Skipping duplicate ID: {item['id']}")
            
            print(f"  + Adding {added_count} new items to existing {category}.yaml")
        else:
            # Create new file
            data = {
                'name': category,
                'items': new_items
            }
            print(f"  ✓ Creating new wildcard: {category}.yaml")
        
        # Write YAML
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"  ✗ Failed to write YAML: {e}")
            return False
    
    def migrate_file(self, txt_path: Path, wildcards_dir: Path, 
                    existing_wildcards: Dict[str, Dict], auto_approve: bool = False) -> bool:
        """
        Migrate a single .txt wildcard file
        
        Returns True if successful
        """
        print(f"\n{'='*60}")
        print(f"Processing: {txt_path.name}")
        print(f"{'='*60}")
        
        # Read file
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"✗ Could not read file: {e}")
            return False
        
        # Skip empty files
        if not content.strip():
            print("⊘ Empty file, skipping")
            return False
        
        # Analyze with LLM
        print("🤖 Analyzing with LLM...")
        results = self.analyze_txt_wildcard(content, txt_path.name, existing_wildcards)
        
        if not results:
            print("✗ Analysis failed")
            return False
        
        # Show preview for each category
        print(f"\n📋 Analysis Results:")
        print(f"  Found {len(results)} categories")
        
        for result in results:
            print(f"\n  Category: {result['category']}")
            print(f"  Items to add: {len(result['items'])}")
            print(f"  Reasoning: {result['reasoning']}")
            
            # Show first few items as preview
            print(f"\n  Preview (first 3 items):")
            for i, item in enumerate(result['items'][:3]):
                print(f"    {i+1}. {item['id']}")
                print(f"       Text: {item['text'][:60]}...")
                print(f"       Tags: {', '.join(item['tags'])}")
                if item.get('blacklist'):
                    print(f"       Blacklist: {', '.join(item['blacklist'])}")
        
        # Ask for confirmation
        if not auto_approve:
            response = input("\n❓ Add these items to YAML? [y/N/s(kip)]: ").lower()
            
            if response == 's':
                print("⊘ Skipped")
                return False
            elif response != 'y':
                print("⊘ Cancelled")
                return False
        
        # Merge all categories into YAML
        success_count = 0
        for result in results:
            success = self.merge_into_yaml(
                result['category'],
                result['items'],
                wildcards_dir,
                existing_wildcards
            )
            
            if success:
                success_count += 1
                # Reload existing wildcards for next file
                yaml_path = wildcards_dir / f"{result['category']}.yaml"
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    existing_wildcards[result['category']] = {
                        'file': yaml_path,
                        'description': data.get('description', ''),
                        'items': data.get('items', []),
                        'tags': self._extract_all_tags(data.get('items', []))
                    }
        
        if success_count == len(results):
            print(f"✓ Successfully migrated {txt_path.name} to {len(results)} categories")
            return True
        else:
            print(f"⚠ Partially migrated {txt_path.name}: {success_count}/{len(results)} categories")
            return False


def main():
    parser = argparse.ArgumentParser(description="Migrate .txt wildcards to YAML format using LM Studio")
    parser.add_argument("--input", "-i", required=True, help="Directory containing .txt wildcard files")
    parser.add_argument("--output", "-o", default="./wildcards", help="Output directory for YAML files")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1", help="LM Studio API URL")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--auto-approve", "-y", action="store_true", help="Auto-approve all migrations")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N files (for testing)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Initialize migrator
    print(f"\n🚀 Luna Wildcard Migration Tool")
    print(f"{'='*60}")
    migrator = WildcardMigrator(args.lm_studio_url, args.model)
    
    # Load existing wildcards
    print(f"\n📂 Loading existing wildcards from: {output_dir}")
    existing = migrator.load_existing_wildcards(output_dir)
    if existing:
        print(f"  Found {len(existing)} existing wildcard files")
    else:
        print(f"  No existing wildcards (will create new)")
    
    # Find .txt files
    txt_files = list(input_dir.glob("*.txt"))
    if args.limit:
        txt_files = txt_files[:args.limit]
    
    if not txt_files:
        print(f"\n✗ No .txt files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n📝 Found {len(txt_files)} .txt files to process")
    
    # Process each file
    success_count = 0
    for txt_file in txt_files:
        if migrator.migrate_file(txt_file, output_dir, existing, args.auto_approve):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Migration complete!")
    print(f"  Successfully migrated: {success_count}/{len(txt_files)} files")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
