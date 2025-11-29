"""
Luna Civitai Metadata Scraper - Fetch and embed Civitai metadata for LoRAs/Embeddings

Inspired by SwarmUI's metadata handling:
- Fetches metadata from Civitai API (trigger words, tags, descriptions, weights)
- Embeds into safetensors file headers (modelspec.* format)
- Optionally writes .swarm.json sidecar for compatibility
- Can update SwarmUI's LiteDB cache for shared model folders

This allows keeping metadata current without relying on SwarmUI.
"""

import json
import os
import re
import struct
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import urllib.request
import urllib.error
import ssl

import folder_paths

# Try to import aiohttp for async requests, fall back to urllib
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Import Luna metadata database
try:
    from utils.luna_metadata_db import get_db, store_civitai_metadata
    HAS_METADATA_DB = True
except ImportError:
    HAS_METADATA_DB = False
    print("LunaCivitai: Metadata database not available")

# Try to import PromptServer for web endpoints
try:
    from server import PromptServer
    from aiohttp import web
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False

# =============================================================================
# CIVITAI API UTILITIES
# =============================================================================

CIVITAI_API_BASE = "https://civitai.com/api/v1"

# SSL context for HTTPS requests
SSL_CONTEXT = ssl.create_default_context()


def compute_tensor_hash(filepath: str) -> Optional[str]:
    """
    Compute SHA-256 tensor hash for a safetensors file.
    This matches the hash format used by Civitai for model lookup.
    
    The tensor hash is computed on the tensor DATA only (after the header),
    not the entire file.
    """
    try:
        with open(filepath, 'rb') as f:
            # Read header length (first 8 bytes, little-endian uint64)
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            
            # Skip past the header
            f.seek(8 + header_len)
            
            # Hash the remaining tensor data
            sha256 = hashlib.sha256()
            while chunk := f.read(8192):
                sha256.update(chunk)
            
            return f"0x{sha256.hexdigest().upper()}"
    except Exception as e:
        print(f"LunaCivitai: Error computing hash for {filepath}: {e}")
        return None


def fetch_civitai_by_hash(tensor_hash: str, api_key: str = "") -> Optional[Dict]:
    """
    Fetch Civitai metadata using the tensor hash.
    Returns the model version data if found.
    """
    # Civitai uses first 12 chars of hash (without 0x prefix)
    short_hash = tensor_hash.replace("0x", "")[:12].upper()
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{short_hash}"
    
    headers = {"User-Agent": "ComfyUI-Luna-Collection/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"LunaCivitai: Model not found on Civitai for hash {short_hash}")
        else:
            print(f"LunaCivitai: HTTP error {e.code} fetching {url}")
    except Exception as e:
        print(f"LunaCivitai: Error fetching from Civitai: {e}")
    
    return None


def fetch_civitai_model_info(model_id: int, api_key: str = "") -> Optional[Dict]:
    """
    Fetch full model info from Civitai by model ID.
    """
    url = f"{CIVITAI_API_BASE}/models/{model_id}"
    
    headers = {"User-Agent": "ComfyUI-Luna-Collection/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"LunaCivitai: Error fetching model {model_id}: {e}")
    
    return None


def parse_civitai_metadata(version_data: Dict, model_data: Optional[Dict] = None) -> Dict:
    """
    Parse Civitai API response into modelspec format.
    This matches SwarmUI's metadata format for compatibility.
    """
    metadata = {}
    
    # From version data
    if version_data:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        
        # Title: "Model Name - Version Name"
        model_name = version_data.get("model", {}).get("name", "")
        version_name = version_data.get("name", "")
        if model_name and version_name:
            metadata["modelspec.title"] = f"{model_name} - {version_name}"
        elif model_name:
            metadata["modelspec.title"] = model_name
        
        # Trigger words
        trained_words = version_data.get("trainedWords", [])
        if trained_words:
            metadata["modelspec.trigger_phrase"] = ", ".join(trained_words)
        
        # Base model (for usage hints)
        base_model = version_data.get("baseModel", "")
        if base_model in ["Illustrious", "Pony", "SDXL 1.0", "SD 1.5"]:
            metadata["modelspec.usage_hint"] = base_model
        
        # Date
        created_at = version_data.get("createdAt", "")
        if created_at:
            metadata["modelspec.date"] = created_at
        
        # Description (version-level)
        if version_data.get("description"):
            metadata["modelspec.description"] = version_data["description"]
    
    # From model data (if available)
    if model_data:
        # Author
        creator = model_data.get("creator", {})
        if creator.get("username"):
            metadata["modelspec.author"] = creator["username"]
        
        # Tags
        tags = model_data.get("tags", [])
        if tags:
            metadata["modelspec.tags"] = ", ".join(tags)
        
        # Full description (combine model + version)
        model_desc = model_data.get("description", "")
        if model_desc:
            existing_desc = metadata.get("modelspec.description", "")
            if existing_desc:
                metadata["modelspec.description"] = f"{existing_desc}\n\n{model_desc}"
            else:
                metadata["modelspec.description"] = model_desc
    
    return metadata


def read_safetensors_header(filepath: str) -> Tuple[int, Dict]:
    """
    Read the JSON header from a safetensors file.
    Returns (header_length, header_dict).
    """
    with open(filepath, 'rb') as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        
        return header_len, header


def write_safetensors_header(filepath: str, new_metadata: Dict, 
                              spacer_kb: int = 4) -> bool:
    """
    Update the __metadata__ section of a safetensors file header.
    
    Uses journaling (temp file swap) for safety, like SwarmUI does.
    Adds spacer padding to allow future in-place updates.
    
    Args:
        filepath: Path to the safetensors file
        new_metadata: Dict of modelspec.* keys to add/update
        spacer_kb: Kilobytes of spacer to add for future updates
        
    Returns:
        True if successful
    """
    try:
        # Read existing header
        with open(filepath, 'rb') as f:
            header_len_bytes = f.read(8)
            old_header_len = struct.unpack('<Q', header_len_bytes)[0]
            
            header_bytes = f.read(old_header_len)
            header = json.loads(header_bytes.decode('utf-8'))
            
            # Get or create __metadata__ section
            meta = header.get("__metadata__", {})
            
            # Update with new metadata
            for key, value in new_metadata.items():
                if value is not None and value != "":
                    meta[key] = str(value) if not isinstance(value, str) else value
            
            # Add spacer for future updates
            meta["__spacer"] = " " * (spacer_kb * 1024)
            
            header["__metadata__"] = meta
            
            # Encode new header
            new_header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
            new_header_len = len(new_header_bytes)
            
            # Check if we can do in-place update
            if new_header_len <= old_header_len:
                # Pad to match old length
                padding_needed = old_header_len - new_header_len
                meta["__spacer"] = " " * padding_needed
                header["__metadata__"] = meta
                new_header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
                
                # Direct in-place update
                f.seek(0)
                
        # For in-place update
        if len(new_header_bytes) <= old_header_len:
            with open(filepath, 'r+b') as f:
                f.seek(8)
                f.write(new_header_bytes)
                # Pad remainder with spaces if needed
                remaining = old_header_len - len(new_header_bytes)
                if remaining > 0:
                    f.write(b' ' * remaining)
            print(f"LunaCivitai: Updated metadata in-place for {Path(filepath).name}")
            return True
        
        # Need to rewrite entire file (header grew)
        temp_path = filepath + ".tmp"
        with open(filepath, 'rb') as src:
            # Skip old header
            src.seek(8 + old_header_len)
            
            with open(temp_path, 'wb') as dst:
                # Write new header length
                dst.write(struct.pack('<Q', new_header_len))
                # Write new header
                dst.write(new_header_bytes)
                # Copy tensor data
                while chunk := src.read(8192):
                    dst.write(chunk)
        
        # Journaling swap
        backup_path = filepath + ".tmp2"
        creation_time = os.path.getctime(filepath)
        
        os.rename(filepath, backup_path)
        os.rename(temp_path, filepath)
        os.remove(backup_path)
        
        # Preserve creation time
        os.utime(filepath, (creation_time, os.path.getmtime(filepath)))
        
        print(f"LunaCivitai: Rewrote {Path(filepath).name} with expanded header")
        return True
        
    except Exception as e:
        print(f"LunaCivitai: Error writing header to {filepath}: {e}")
        # Cleanup temp files
        for ext in [".tmp", ".tmp2"]:
            if os.path.exists(filepath + ext):
                try:
                    os.remove(filepath + ext)
                except:
                    pass
        return False


def write_swarm_json_sidecar(filepath: str, metadata: Dict) -> bool:
    """
    Write a .swarm.json sidecar file with the metadata.
    This is SwarmUI's alternative storage when not writing to safetensors.
    """
    try:
        sidecar_path = filepath.rsplit('.', 1)[0] + ".swarm.json"
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"LunaCivitai: Wrote sidecar {Path(sidecar_path).name}")
        return True
    except Exception as e:
        print(f"LunaCivitai: Error writing sidecar: {e}")
        return False


# =============================================================================
# COMFYUI NODE: METADATA SCRAPER
# =============================================================================

class LunaCivitaiScraper:
    """
    Scrape Civitai metadata for a LoRA or Embedding and embed it into the file.
    
    This node:
    1. Computes the tensor hash of the safetensors file
    2. Queries Civitai API to find the model
    3. Extracts trigger words, tags, descriptions, author info
    4. Writes metadata to the safetensors header (modelspec.* format)
    5. Optionally writes .swarm.json sidecar for compatibility
    
    The metadata format matches SwarmUI for cross-compatibility.
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("trigger_words", "tags", "status", "success")
    FUNCTION = "scrape_metadata"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRAs and embeddings
        loras = ["-- Select LoRA --"] + folder_paths.get_filename_list("loras")
        embeddings = ["-- Select Embedding --"] + folder_paths.get_filename_list("embeddings")
        
        return {
            "required": {
                "model_type": (["LoRA", "Embedding"], {
                    "default": "LoRA",
                    "tooltip": "Type of model to scrape metadata for"
                }),
                "write_to_file": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write metadata to the safetensors header"
                }),
                "write_sidecar": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also write .swarm.json sidecar file (SwarmUI compatible)"
                }),
            },
            "optional": {
                "lora_name": (loras, {
                    "default": loras[0],
                    "tooltip": "LoRA to fetch metadata for"
                }),
                "embedding_name": (embeddings, {
                    "default": embeddings[0],
                    "tooltip": "Embedding to fetch metadata for"
                }),
                "civitai_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional Civitai API key for gated content"
                }),
                "force_refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-fetch even if metadata already exists"
                }),
            }
        }
    
    def scrape_metadata(self, model_type: str, write_to_file: bool, 
                        write_sidecar: bool,
                        lora_name: str = "", embedding_name: str = "",
                        civitai_api_key: str = "", 
                        force_refresh: bool = False) -> Tuple[str, str, str, bool]:
        """Scrape metadata from Civitai for the selected model."""
        
        # Determine which model to process
        if model_type == "LoRA":
            if not lora_name or lora_name.startswith("--"):
                return ("", "", "Please select a LoRA", False)
            model_name = lora_name
            model_path = folder_paths.get_full_path("loras", lora_name)
        else:
            if not embedding_name or embedding_name.startswith("--"):
                return ("", "", "Please select an Embedding", False)
            model_name = embedding_name
            model_path = folder_paths.get_full_path("embeddings", embedding_name)
        
        if not model_path or not os.path.exists(model_path):
            return ("", "", f"Model file not found: {model_name}", False)
        
        if not model_path.endswith(".safetensors"):
            return ("", "", "Only .safetensors files are supported", False)
        
        # Check for existing metadata
        if not force_refresh:
            try:
                _, header = read_safetensors_header(model_path)
                meta = header.get("__metadata__", {})
                if meta.get("modelspec.trigger_phrase") or meta.get("modelspec.title"):
                    trigger = meta.get("modelspec.trigger_phrase", "")
                    tags = meta.get("modelspec.tags", "")
                    return (trigger, tags, "Metadata already exists (use force_refresh to update)", True)
            except:
                pass
        
        # Compute tensor hash
        print(f"LunaCivitai: Computing hash for {model_name}...")
        tensor_hash = compute_tensor_hash(model_path)
        
        if not tensor_hash:
            return ("", "", "Failed to compute tensor hash", False)
        
        print(f"LunaCivitai: Hash = {tensor_hash[:18]}...")
        
        # Query Civitai
        print(f"LunaCivitai: Querying Civitai API...")
        version_data = fetch_civitai_by_hash(tensor_hash, civitai_api_key)
        
        if not version_data:
            return ("", "", f"Model not found on Civitai (hash: {tensor_hash[:18]}...)", False)
        
        # Get full model info for additional data
        model_id = version_data.get("modelId")
        model_data = None
        if model_id:
            model_data = fetch_civitai_model_info(model_id, civitai_api_key)
        
        # Parse metadata
        metadata = parse_civitai_metadata(version_data, model_data)
        
        # Add hash to metadata
        metadata["modelspec.hash_sha256"] = tensor_hash
        
        # Extract key info for return
        trigger_words = metadata.get("modelspec.trigger_phrase", "")
        tags = metadata.get("modelspec.tags", "")
        title = metadata.get("modelspec.title", model_name)
        
        # Write to file
        if write_to_file:
            success = write_safetensors_header(model_path, metadata)
            if not success:
                return (trigger_words, tags, "Failed to write to file", False)
        
        # Write sidecar
        if write_sidecar:
            write_swarm_json_sidecar(model_path, metadata)
        
        # Store in Luna metadata database
        if HAS_METADATA_DB:
            try:
                db_model_type = "lora" if model_type == "LoRA" else "embedding"
                store_civitai_metadata(model_name, db_model_type, metadata, tensor_hash)
                print(f"LunaCivitai: Stored metadata in database for {model_name}")
            except Exception as e:
                print(f"LunaCivitai: Warning - failed to store in database: {e}")
        
        status = f"Success! Found: {title}"
        if trigger_words:
            status += f"\nTriggers: {trigger_words[:100]}{'...' if len(trigger_words) > 100 else ''}"
        
        return (trigger_words, tags, status, True)


# =============================================================================
# BATCH SCRAPER NODE
# =============================================================================

class LunaCivitaiBatchScraper:
    """
    Batch scrape Civitai metadata for multiple models.
    
    Processes all LoRAs or Embeddings in a folder, skipping those
    that already have metadata (unless force_refresh is enabled).
    """
    
    CATEGORY = "Luna/Utilities"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("report", "processed", "failed")
    FUNCTION = "batch_scrape"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["LoRA", "Embedding"], {
                    "default": "LoRA"
                }),
                "folder_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Only process models in folders matching this (e.g., 'Illustrious/')"
                }),
                "write_to_file": ("BOOLEAN", {
                    "default": True
                }),
                "write_sidecar": ("BOOLEAN", {
                    "default": False
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip models that already have metadata"
                }),
                "max_models": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Maximum number of models to process (rate limiting)"
                }),
            },
            "optional": {
                "civitai_api_key": ("STRING", {
                    "default": ""
                }),
            }
        }
    
    def batch_scrape(self, model_type: str, folder_filter: str,
                     write_to_file: bool, write_sidecar: bool,
                     skip_existing: bool, max_models: int,
                     civitai_api_key: str = "") -> Tuple[str, int, int]:
        """Batch process multiple models."""
        
        # Get model list
        if model_type == "LoRA":
            models = folder_paths.get_filename_list("loras")
            get_path = lambda m: folder_paths.get_full_path("loras", m)
        else:
            models = folder_paths.get_filename_list("embeddings")
            get_path = lambda m: folder_paths.get_full_path("embeddings", m)
        
        # Filter by folder
        if folder_filter:
            models = [m for m in models if folder_filter.lower() in m.lower()]
        
        # Filter to safetensors only
        models = [m for m in models if m.endswith(".safetensors")]
        
        report_lines = [f"Luna Civitai Batch Scraper - {model_type}s"]
        report_lines.append(f"Found {len(models)} models matching filter")
        report_lines.append("=" * 50)
        
        processed = 0
        failed = 0
        skipped = 0
        
        for model_name in models[:max_models]:
            model_path = get_path(model_name)
            if not model_path or not os.path.exists(model_path):
                continue
            
            # Check existing metadata
            if skip_existing:
                try:
                    _, header = read_safetensors_header(model_path)
                    meta = header.get("__metadata__", {})
                    if meta.get("modelspec.trigger_phrase") or meta.get("modelspec.title"):
                        skipped += 1
                        continue
                except:
                    pass
            
            # Compute hash and query
            tensor_hash = compute_tensor_hash(model_path)
            if not tensor_hash:
                report_lines.append(f"SKIP {model_name}: Hash failed")
                failed += 1
                continue
            
            version_data = fetch_civitai_by_hash(tensor_hash, civitai_api_key)
            if not version_data:
                report_lines.append(f"MISS {model_name}: Not on Civitai")
                failed += 1
                continue
            
            # Get full info
            model_id = version_data.get("modelId")
            model_data = fetch_civitai_model_info(model_id, civitai_api_key) if model_id else None
            
            metadata = parse_civitai_metadata(version_data, model_data)
            metadata["modelspec.hash_sha256"] = tensor_hash
            
            # Write
            success = True
            if write_to_file:
                success = write_safetensors_header(model_path, metadata)
            if write_sidecar:
                write_swarm_json_sidecar(model_path, metadata)
            
            # Store in database
            if HAS_METADATA_DB:
                try:
                    db_model_type = "lora" if model_type == "LoRA" else "embedding"
                    store_civitai_metadata(model_name, db_model_type, metadata, tensor_hash)
                except Exception as e:
                    print(f"LunaCivitai: DB write failed for {model_name}: {e}")
            
            if success:
                title = metadata.get("modelspec.title", model_name)
                triggers = metadata.get("modelspec.trigger_phrase", "")[:50]
                report_lines.append(f"OK   {model_name}: {title}")
                if triggers:
                    report_lines.append(f"     Triggers: {triggers}...")
                processed += 1
            else:
                report_lines.append(f"FAIL {model_name}: Write error")
                failed += 1
        
        report_lines.append("=" * 50)
        report_lines.append(f"Processed: {processed}, Failed: {failed}, Skipped: {skipped}")
        
        return ("\n".join(report_lines), processed, failed)


# =============================================================================
# WEB ENDPOINTS FOR FRONTEND INTEGRATION
# =============================================================================

if HAS_PROMPT_SERVER:
    
    @PromptServer.instance.routes.post("/luna/civitai/scrape")
    async def scrape_single_model(request):
        """API endpoint to scrape a single model"""
        try:
            data = await request.json()
            model_type = data.get("type", "lora")
            model_name = data.get("name", "")
            api_key = data.get("api_key", "")
            write_file = data.get("write_file", True)
            
            if not model_name:
                return web.Response(status=400, text="name required")
            
            # Get path
            folder_type = "loras" if model_type == "lora" else "embeddings"
            model_path = folder_paths.get_full_path(folder_type, model_name)
            
            if not model_path or not os.path.exists(model_path):
                return web.Response(status=404, text="Model not found")
            
            # Compute hash
            tensor_hash = compute_tensor_hash(model_path)
            if not tensor_hash:
                return web.json_response({"error": "Hash computation failed"})
            
            # Query Civitai
            version_data = fetch_civitai_by_hash(tensor_hash, api_key)
            if not version_data:
                return web.json_response({
                    "error": "Not found on Civitai",
                    "hash": tensor_hash
                })
            
            # Get full model data
            model_id = version_data.get("modelId")
            model_data = fetch_civitai_model_info(model_id, api_key) if model_id else None
            
            metadata = parse_civitai_metadata(version_data, model_data)
            metadata["modelspec.hash_sha256"] = tensor_hash
            
            # Write if requested
            if write_file:
                write_safetensors_header(model_path, metadata)
            
            return web.json_response({
                "success": True,
                "title": metadata.get("modelspec.title", ""),
                "trigger_phrase": metadata.get("modelspec.trigger_phrase", ""),
                "tags": metadata.get("modelspec.tags", ""),
                "author": metadata.get("modelspec.author", ""),
                "hash": tensor_hash
            })
            
        except Exception as e:
            return web.Response(status=500, text=str(e))
    
    @PromptServer.instance.routes.get("/luna/civitai/check")
    async def check_model_metadata(request):
        """Check if a model already has metadata"""
        model_type = request.query.get("type", "lora")
        model_name = request.query.get("name", "")
        
        if not model_name:
            return web.Response(status=400, text="name required")
        
        folder_type = "loras" if model_type == "lora" else "embeddings"
        model_path = folder_paths.get_full_path(folder_type, model_name)
        
        if not model_path or not os.path.exists(model_path):
            return web.Response(status=404, text="Model not found")
        
        try:
            _, header = read_safetensors_header(model_path)
            meta = header.get("__metadata__", {})
            
            return web.json_response({
                "has_metadata": bool(meta.get("modelspec.trigger_phrase") or meta.get("modelspec.title")),
                "title": meta.get("modelspec.title", ""),
                "trigger_phrase": meta.get("modelspec.trigger_phrase", ""),
                "tags": meta.get("modelspec.tags", ""),
                "hash": meta.get("modelspec.hash_sha256", "")
            })
        except Exception as e:
            return web.json_response({"error": str(e)})


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaCivitaiScraper": LunaCivitaiScraper,
    "LunaCivitaiBatchScraper": LunaCivitaiBatchScraper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaCivitaiScraper": "Luna Civitai Metadata Scraper",
    "LunaCivitaiBatchScraper": "Luna Civitai Batch Scraper",
}
