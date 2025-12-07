# utils/luna_metadata_db.py

## Purpose
SQLite-based metadata storage system for LoRAs and embeddings providing fast lookups without file parsing. Stores model metadata, trigger words, tags, and user customizations with full-text search capabilities.

## Exports
- `LunaMetadataDB`: Thread-safe singleton database class
- `get_db()`: Global database instance accessor
- `get_trigger_phrase(file_path, model_type)`: Quick trigger phrase lookup
- `get_model_metadata(file_path, model_type)`: Complete metadata lookup
- `store_civitai_metadata(file_path, model_type, civitai_data, tensor_hash)`: Store Civitai API metadata

## Key Imports
- `sqlite3`: Database operations
- `os`, `pathlib`: File system operations
- `json`: Data serialization
- `threading`: Thread-local connections
- `datetime`: Timestamp handling
- `contextlib`: Transaction context manager
- `folder_paths`: ComfyUI path resolution

## ComfyUI Node Configuration
N/A - Database utility, not a node.

## Input Schema
N/A - Database class with automatic schema initialization.

## Key Methods
- `LunaMetadataDB.upsert_model(data)`: Insert/update model record with conflict resolution
- `LunaMetadataDB.get_by_path(file_path, model_type)`: Retrieve model by file path
- `LunaMetadataDB.get_by_hash(tensor_hash)`: Retrieve model by Civitai tensor hash
- `LunaMetadataDB.search(query, model_type, limit)`: Full-text search across metadata
- `LunaMetadataDB.get_by_base_model(base_model, model_type)`: Filter by base model type
- `LunaMetadataDB.set_favorite(file_path, model_type, favorite)`: Mark/unmark as favorite
- `LunaMetadataDB.record_usage(file_path, model_type)`: Track model usage statistics
- `LunaMetadataDB.get_stats()`: Database statistics and counts

## Dependencies
- `sqlite3`: Built-in Python database
- `folder_paths`: ComfyUI path resolution (optional fallback)
- `pathlib`: File path operations

## Integration Points
- Database location: {ComfyUI}/user/default/ComfyUI-Luna-Collection/metadata.db
- Used by LoRA loader nodes for metadata display and search
- Provides offline access to Civitai metadata
- Full-text search across titles, triggers, tags, descriptions
- User customization support (favorites, ratings, notes)
- Schema versioning with automatic migrations

## Notes
- Thread-safe with connection pooling and WAL mode
- FTS5 virtual table for fast text search
- Automatic schema creation and migration (v1→v2)
- Independent of SwarmUI's LiteDB system
- Stores thumbnails, usage hints, and NSFW flags
- Tracks usage statistics and last-used timestamps