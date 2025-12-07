# luna_civitai_scraper.py

## Purpose
Fetch and embed Civitai metadata for LoRAs/Embeddings. Supports single/batch scraping, safetensors header embedding, SwarmUI compatibility.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
json, os, re, struct, hashlib, asyncio, base64, io, threading, concurrent.futures, pathlib, typing, datetime, urllib.request/error, ssl, time, folder_paths, PIL (optional), aiohttp (optional), utils.luna_metadata_db (optional), PromptServer (optional)

## ComfyUI Node Configuration
- LunaCivitaiScraper: CATEGORY="Luna/Utilities", RETURN_TYPES=("STRING", "STRING", "STRING", "BOOLEAN"), FUNCTION="scrape_metadata", OUTPUT_NODE=True
- LunaCivitaiBatchScraper: CATEGORY="Luna/Utilities", RETURN_TYPES=("STRING", "INT", "INT"), FUNCTION="batch_scrape", OUTPUT_NODE=True

## Input Schema
- LunaCivitaiScraper: model_type (["LoRA", "Embedding"]), write_to_file (BOOLEAN), write_sidecar (BOOLEAN), lora_name (from folder_paths), embedding_name (from folder_paths), civitai_api_key (STRING), force_refresh (BOOLEAN)
- LunaCivitaiBatchScraper: model_type (["LoRA", "Embedding"]), folder_filter (STRING), write_to_file (BOOLEAN), write_sidecar (BOOLEAN), skip_existing (BOOLEAN), max_models (INT), civitai_api_key (STRING, optional), parallel_workers (INT, optional)

## Key Methods
- LunaCivitaiScraper.scrape_metadata(model_type, write_to_file, write_sidecar, lora_name, embedding_name, civitai_api_key, force_refresh) -> Tuple[str, str, str, bool]
- LunaCivitaiBatchScraper.batch_scrape(model_type, folder_filter, write_to_file, write_sidecar, skip_existing, max_models, civitai_api_key, parallel_workers) -> Tuple[str, int, int]
- fetch_civitai_by_hash(tensor_hash, api_key, use_rate_limit) -> Optional[Dict]
- parse_civitai_metadata(version_data, model_data, api_key, download_thumbnail) -> Dict
- write_safetensors_header(filepath, new_metadata, spacer_kb) -> bool

## Dependencies
folder_paths, PIL (optional), aiohttp (optional), utils.luna_metadata_db (optional), PromptServer (optional)

## Integration Points
CivitAI API v1, folder_paths for model resolution, safetensors header embedding, SwarmUI metadata format, Luna metadata database, web endpoints for frontend

## Notes
Computes SHA256 tensor hashes for CivitAI lookup, supports parallel batch processing with API key, embeds thumbnails, thread-safe file operations, rate limiting, HTML sanitization</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_civitai_scraper.md