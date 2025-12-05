"""
Luna LoRA Validator
Scans a prompt JSON file and validates which LoRAs exist locally.
Optionally searches CivitAI for missing LoRAs and provides download links.
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
import urllib.parse
import ssl
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

if TYPE_CHECKING:
    import folder_paths

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    folder_paths = None  # type: ignore
    HAS_FOLDER_PATHS = False

# SSL context for HTTPS requests
SSL_CONTEXT = ssl.create_default_context()

# CivitAI API
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_SEARCH_URL = f"{CIVITAI_API_BASE}/models"


class LunaLoRAValidator:
    """
    Validate LoRAs in a prompt JSON file against locally installed LoRAs.
    
    Features:
    - Scans JSON and extracts all unique LoRA names
    - Checks which exist locally (with path resolution)
    - Optionally searches CivitAI for missing LoRAs
    - Outputs formatted report with download links
    """
    
    CATEGORY = "Luna/Utils"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("report", "missing_loras", "civitai_links", "found_count", "missing_count")
    FUNCTION = "validate_loras"
    OUTPUT_NODE = True
    
    # Cache for discovered LoRAs (name -> list of relative paths)
    _lora_cache: Dict[str, List[str]] = {}
    _cache_initialized = False
    
    @classmethod
    def _get_json_files(cls) -> List[str]:
        """Get list of JSON files from ComfyUI input directory"""
        if not HAS_FOLDER_PATHS:
            return []
        
        try:
            input_dir = folder_paths.get_input_directory()
            files = []
            for f in os.listdir(input_dir):
                if f.endswith('.json') and os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)
            return sorted(files) if files else ["No JSON files found"]
        except Exception as e:
            print(f"[LunaLoRAValidator] Error scanning input directory: {e}")
            return ["Error scanning directory"]
    
    @classmethod
    def INPUT_TYPES(cls):
        json_files = cls._get_json_files()
        
        return {
            "required": {
                "json_file": (json_files, {
                    "tooltip": "Select a JSON metadata file to validate"
                }),
                "search_civitai": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Search CivitAI for missing LoRAs"
                }),
                "civitai_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional CivitAI API key for faster/more reliable searches"
                }),
            }
        }
    
    @classmethod
    def _build_lora_cache(cls):
        """Scan loras directory to build filename -> path cache"""
        if cls._cache_initialized or not HAS_FOLDER_PATHS:
            return
        
        cls._lora_cache = {}
        
        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            for base_path in lora_paths:
                if not os.path.isdir(base_path):
                    continue
                for root, dirs, files in os.walk(base_path):
                    for filename in files:
                        if filename.endswith(('.safetensors', '.pt', '.ckpt', '.bin')):
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, base_path)
                            # Store by filename (without extension) for matching
                            name_no_ext = os.path.splitext(filename)[0]
                            name_lower = name_no_ext.lower()
                            if name_lower not in cls._lora_cache:
                                cls._lora_cache[name_lower] = []
                            cls._lora_cache[name_lower].append(rel_path)
        except Exception as e:
            print(f"[LunaLoRAValidator] Error scanning loras: {e}")
        
        cls._cache_initialized = True
        print(f"[LunaLoRAValidator] Cached {len(cls._lora_cache)} unique LoRA names")
    
    def lora_exists(self, lora_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a LoRA exists locally.
        Returns (exists, resolved_path) tuple.
        """
        self._build_lora_cache()
        
        # Handle if it's already a path (contains / or \)
        if '/' in lora_name or '\\' in lora_name:
            if HAS_FOLDER_PATHS:
                try:
                    # Try with extension
                    full_path = folder_paths.get_full_path("loras", lora_name)
                    if full_path and os.path.exists(full_path):
                        return True, lora_name
                    # Try adding extension
                    for ext in ['.safetensors', '.pt', '.ckpt']:
                        test_name = lora_name + ext
                        full_path = folder_paths.get_full_path("loras", test_name)
                        if full_path and os.path.exists(full_path):
                            return True, test_name
                except:
                    pass
            return False, None
        
        # Strip extension if present
        name_no_ext = os.path.splitext(lora_name)[0]
        name_lower = name_no_ext.lower()
        
        # Look up in cache
        if name_lower in self._lora_cache:
            return True, self._lora_cache[name_lower][0]
        
        # Try partial match (name appears in filename)
        for cached_name, paths in self._lora_cache.items():
            if name_lower in cached_name or cached_name in name_lower:
                return True, paths[0]
        
        return False, None
    
    def search_civitai(self, lora_name: str, api_key: str = "") -> Optional[Dict[str, Any]]:
        """
        Search CivitAI for a LoRA by name.
        Returns dict with model info if found.
        """
        # Clean the search query
        search_query = os.path.splitext(lora_name)[0]
        # Remove common suffixes that might interfere
        for suffix in ['_v1', '_v2', '_v3', '-v1', '-v2', '-v3', '_lora', '-lora']:
            if search_query.lower().endswith(suffix):
                search_query = search_query[:-len(suffix)]
        
        # Build search URL
        params = {
            "query": search_query,
            "types": "LORA",
            "limit": 5,
            "sort": "Highest Rated"
        }
        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{CIVITAI_SEARCH_URL}?{query_string}"
        
        headers = {"User-Agent": "ComfyUI-Luna-Collection/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=10, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                items = data.get("items", [])
                if items:
                    # Return the first/best match
                    model = items[0]
                    return {
                        "name": model.get("name", "Unknown"),
                        "id": model.get("id"),
                        "type": model.get("type", "LORA"),
                        "nsfw": model.get("nsfw", False),
                        "url": f"https://civitai.com/models/{model.get('id')}",
                        "creator": model.get("creator", {}).get("username", "Unknown"),
                        "download_count": model.get("stats", {}).get("downloadCount", 0),
                        "rating": model.get("stats", {}).get("rating", 0),
                    }
        except urllib.error.HTTPError as e:
            print(f"[LunaLoRAValidator] CivitAI HTTP error for '{lora_name}': {e.code}")
        except urllib.error.URLError as e:
            print(f"[LunaLoRAValidator] CivitAI URL error for '{lora_name}': {e.reason}")
        except Exception as e:
            print(f"[LunaLoRAValidator] CivitAI search error for '{lora_name}': {e}")
        
        return None
    
    def validate_loras(
        self,
        json_file: str,
        search_civitai: bool,
        civitai_api_key: str
    ) -> Tuple[str, str, str, int, int]:
        """
        Validate all LoRAs in a JSON file.
        
        Returns:
            - report: Full validation report
            - missing_loras: Comma-separated list of missing LoRA names
            - civitai_links: Newline-separated CivitAI links for missing LoRAs
            - found_count: Number of LoRAs found locally
            - missing_count: Number of missing LoRAs
        """
        # Resolve file path
        if HAS_FOLDER_PATHS:
            file_path = folder_paths.get_annotated_filepath(json_file)
        else:
            file_path = json_file
        
        if not os.path.isfile(file_path):
            return (f"Error: {json_file} not found", "", "", 0, 0)
        
        # Load JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
        except Exception as e:
            return (f"Error loading JSON: {e}", "", "", 0, 0)
        
        if not isinstance(metadata_list, list):
            return ("Error: JSON is not a list", "", "", 0, 0)
        
        # Collect all unique LoRAs
        all_loras: Dict[str, Dict] = {}  # name -> {model_strength, clip_strength, count}
        
        for entry in metadata_list:
            loras = entry.get("loras", [])
            for lora in loras:
                name = lora.get("name", "")
                if not name:
                    continue
                
                name_lower = name.lower()
                if name_lower not in all_loras:
                    all_loras[name_lower] = {
                        "original_name": name,
                        "model_strength": lora.get("model_strength", 1.0),
                        "clip_strength": lora.get("clip_strength", 1.0),
                        "count": 1
                    }
                else:
                    all_loras[name_lower]["count"] += 1
        
        if not all_loras:
            return ("No LoRAs found in JSON file", "", "", 0, 0)
        
        # Validate each LoRA
        found_loras = []
        missing_loras = []
        civitai_results = []
        
        report_lines = [
            f"╔══════════════════════════════════════════════════════════════╗",
            f"║           LUNA LORA VALIDATION REPORT                        ║",
            f"╠══════════════════════════════════════════════════════════════╣",
            f"║  JSON File: {json_file[:45]:<45} ║",
            f"║  Total Entries: {len(metadata_list):<44} ║",
            f"║  Unique LoRAs: {len(all_loras):<45} ║",
            f"╠══════════════════════════════════════════════════════════════╣",
        ]
        
        # Check each LoRA
        for name_lower, info in sorted(all_loras.items()):
            original_name = info["original_name"]
            exists, resolved_path = self.lora_exists(original_name)
            
            if exists:
                found_loras.append(original_name)
                report_lines.append(f"║ ✓ {original_name[:40]:<40} (x{info['count']}) ║")
                if resolved_path and resolved_path != original_name:
                    report_lines.append(f"║   └─ Found as: {resolved_path[:43]:<43} ║")
            else:
                missing_loras.append(original_name)
                report_lines.append(f"║ ✗ {original_name[:40]:<40} (x{info['count']}) ║")
        
        report_lines.append(f"╠══════════════════════════════════════════════════════════════╣")
        report_lines.append(f"║  Found: {len(found_loras):<5}  Missing: {len(missing_loras):<5}                          ║")
        
        # Search CivitAI for missing LoRAs
        civitai_links = []
        if search_civitai and missing_loras:
            report_lines.append(f"╠══════════════════════════════════════════════════════════════╣")
            report_lines.append(f"║           CIVITAI SEARCH RESULTS                             ║")
            report_lines.append(f"╠══════════════════════════════════════════════════════════════╣")
            
            for lora_name in missing_loras:
                # Rate limit - be nice to CivitAI
                time.sleep(0.5)
                
                result = self.search_civitai(lora_name, civitai_api_key)
                if result:
                    civitai_links.append(result["url"])
                    report_lines.append(f"║ {lora_name[:30]:<30}                              ║")
                    report_lines.append(f"║   └─ {result['name'][:50]:<50}    ║")
                    report_lines.append(f"║      by {result['creator'][:20]:<20} ⭐{result['rating']:.1f} ⬇{result['download_count']:<8} ║")
                    report_lines.append(f"║      {result['url']:<56} ║")
                else:
                    report_lines.append(f"║ {lora_name[:30]:<30}                              ║")
                    report_lines.append(f"║   └─ Not found on CivitAI                               ║")
                    # Generate a search link instead
                    import urllib.parse
                    search_url = f"https://civitai.com/search/models?query={urllib.parse.quote(lora_name)}&type=LORA"
                    civitai_links.append(search_url)
                    report_lines.append(f"║      Search: {search_url[:45]:<45} ║")
        
        report_lines.append(f"╚══════════════════════════════════════════════════════════════╝")
        
        # Build outputs
        report = "\n".join(report_lines)
        missing_str = ", ".join(missing_loras)
        links_str = "\n".join(civitai_links)
        
        # Print report to console too
        print(report)
        
        return (report, missing_str, links_str, len(found_loras), len(missing_loras))
    
    @classmethod
    def IS_CHANGED(cls, json_file, **kwargs):
        """Check if the JSON file has been modified"""
        if not HAS_FOLDER_PATHS:
            return float("nan")
        
        try:
            file_path = folder_paths.get_annotated_filepath(json_file)
            if os.path.isfile(file_path):
                return os.path.getmtime(file_path)
        except Exception:
            pass
        
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaLoRAValidator": LunaLoRAValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaLoRAValidator": "Luna LoRA Validator",
}
