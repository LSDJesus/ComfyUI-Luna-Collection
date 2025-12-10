"""
Luna Metadata Database - SQLite-based metadata storage for LoRAs and Embeddings

Stores model metadata (trigger words, tags, descriptions, etc.) in a local SQLite database
for fast lookups without needing to read safetensors headers or query Civitai repeatedly.

Database location: {ComfyUI}/user/default/ComfyUI-Luna-Collection/metadata.db

This is independent of SwarmUI's LiteDB - Luna maintains its own cache for:
- Faster startup (no header parsing needed)
- Offline access to metadata
- Custom fields and tagging
- Cross-session persistence
"""

import sqlite3
import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

# Try to import folder_paths to find ComfyUI root
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


# =============================================================================
# DATABASE PATH RESOLUTION
# =============================================================================

def get_comfyui_root() -> Path:
    """
    Get the ComfyUI root directory.
    Uses folder_paths if available, otherwise walks up from this file.
    """
    if HAS_FOLDER_PATHS:
        # folder_paths.base_path is the ComfyUI root
        return Path(folder_paths.base_path)
    
    # Fallback: walk up from this file until we find main.py or ComfyUI markers
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").exists() or (parent / "comfy").is_dir():
            return parent
    
    # Last resort: assume standard custom_nodes structure
    # This file is in: ComfyUI/custom_nodes/ComfyUI-Luna-Collection/utils/
    return current.parent.parent.parent.parent


def get_database_path() -> Path:
    """
    Get the path to Luna's metadata database.
    Location: {ComfyUI}/user/default/ComfyUI-Luna-Collection/metadata.db
    """
    comfyui_root = get_comfyui_root()
    db_dir = comfyui_root / "user" / "default" / "ComfyUI-Luna-Collection"
    
    # Ensure directory exists
    db_dir.mkdir(parents=True, exist_ok=True)
    
    return db_dir / "metadata.db"


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Model metadata table
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Identification
    file_path TEXT NOT NULL,           -- Relative path from models folder
    file_name TEXT NOT NULL,           -- Just the filename
    model_type TEXT NOT NULL,          -- 'lora', 'embedding', 'checkpoint', etc.
    tensor_hash TEXT,                  -- SHA256 hash of tensor data (Civitai format)
    file_hash TEXT,                    -- SHA256 of entire file (for change detection)
    file_size INTEGER,                 -- File size in bytes
    file_mtime REAL,                   -- File modification time
    
    -- Civitai metadata
    civitai_model_id INTEGER,          -- Civitai model ID
    civitai_version_id INTEGER,        -- Civitai version ID
    
    -- Display info
    title TEXT,                        -- "Model Name - Version Name"
    author TEXT,                       -- Creator username
    description TEXT,                  -- Full description
    thumbnail TEXT,                    -- Base64 encoded JPEG thumbnail (max 256px)
    cover_image_url TEXT,              -- Original cover image URL from Civitai
    
    -- Usage info
    trigger_phrase TEXT,               -- Comma-separated trigger words
    tags TEXT,                         -- Comma-separated tags
    base_model TEXT,                   -- "SDXL 1.0", "Pony", "Illustrious", etc.
    usage_hint TEXT,                   -- Additional usage notes
    nsfw INTEGER DEFAULT 0,            -- NSFW flag from Civitai
    
    -- Weights and settings
    default_weight REAL DEFAULT 1.0,   -- Suggested model weight
    default_clip_weight REAL DEFAULT 1.0,  -- Suggested CLIP weight
    
    -- User customization
    user_tags TEXT,                    -- User-added tags (comma-separated)
    user_notes TEXT,                   -- User notes
    favorite INTEGER DEFAULT 0,        -- User favorite flag
    rating INTEGER,                    -- User rating 1-5
    use_count INTEGER DEFAULT 0,       -- Times used in workflows
    
    -- Timestamps
    created_at TEXT NOT NULL,          -- When record was created
    updated_at TEXT NOT NULL,          -- When record was last updated
    last_used_at TEXT,                 -- When model was last used
    
    -- Indexing
    UNIQUE(file_path, model_type)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_models_tensor_hash ON models(tensor_hash);
CREATE INDEX IF NOT EXISTS idx_models_file_name ON models(file_name);
CREATE INDEX IF NOT EXISTS idx_models_model_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_base_model ON models(base_model);
CREATE INDEX IF NOT EXISTS idx_models_favorite ON models(favorite);
CREATE INDEX IF NOT EXISTS idx_models_civitai_model_id ON models(civitai_model_id);

-- Full-text search on trigger phrases and tags
CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5(
    file_name,
    title,
    trigger_phrase,
    tags,
    user_tags,
    description,
    content='models',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS models_ai AFTER INSERT ON models BEGIN
    INSERT INTO models_fts(rowid, file_name, title, trigger_phrase, tags, user_tags, description)
    VALUES (NEW.id, NEW.file_name, NEW.title, NEW.trigger_phrase, NEW.tags, NEW.user_tags, NEW.description);
END;

CREATE TRIGGER IF NOT EXISTS models_ad AFTER DELETE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, file_name, title, trigger_phrase, tags, user_tags, description)
    VALUES ('delete', OLD.id, OLD.file_name, OLD.title, OLD.trigger_phrase, OLD.tags, OLD.user_tags, OLD.description);
END;

CREATE TRIGGER IF NOT EXISTS models_au AFTER UPDATE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, file_name, title, trigger_phrase, tags, user_tags, description)
    VALUES ('delete', OLD.id, OLD.file_name, OLD.title, OLD.trigger_phrase, OLD.tags, OLD.user_tags, OLD.description);
    INSERT INTO models_fts(rowid, file_name, title, trigger_phrase, tags, user_tags, description)
    VALUES (NEW.id, NEW.file_name, NEW.title, NEW.trigger_phrase, NEW.tags, NEW.user_tags, NEW.description);
END;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# =============================================================================
# DATABASE CONNECTION MANAGEMENT
# =============================================================================

class LunaMetadataDB:
    """
    Thread-safe SQLite database for Luna model metadata.
    
    Usage:
        db = LunaMetadataDB()
        
        # Store metadata
        db.upsert_model({
            'file_path': 'Illustrious/my_lora.safetensors',
            'model_type': 'lora',
            'trigger_phrase': 'my_trigger',
            'tags': 'anime, character',
            ...
        })
        
        # Query
        model = db.get_by_path('Illustrious/my_lora.safetensors', 'lora')
        model = db.get_by_hash('0xABCD1234...')
        
        # Search
        results = db.search('anime character')
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - only one database connection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = get_database_path()
        self._local = threading.local()
        self._initialized = True
        
        # Initialize database schema
        self._init_schema()
        
        print(f"LunaMetadataDB: Initialized at {self.db_path}")
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_schema(self):
        """Initialize database schema if needed."""
        with self._transaction() as conn:
            # Check if schema exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'"
            )
            
            if cursor.fetchone() is None:
                # Fresh database - create schema
                conn.executescript(SCHEMA_SQL)
                conn.execute(
                    "INSERT INTO schema_info (key, value) VALUES (?, ?)",
                    ('version', str(SCHEMA_VERSION))
                )
                print("LunaMetadataDB: Created new database schema")
            else:
                # Check version for migrations
                cursor = conn.execute(
                    "SELECT value FROM schema_info WHERE key = 'version'"
                )
                row = cursor.fetchone()
                current_version = int(row['value']) if row else 0
                
                if current_version < SCHEMA_VERSION:
                    self._migrate_schema(conn, current_version, SCHEMA_VERSION)
    
    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int, to_version: int):
        """Run schema migrations."""
        print(f"LunaMetadataDB: Migrating schema from v{from_version} to v{to_version}")
        
        # Migration v1 -> v2: Add thumbnail, cover_image_url, nsfw columns
        if from_version < 2:
            print("LunaMetadataDB: Adding thumbnail, cover_image_url, nsfw columns...")
            try:
                conn.execute("ALTER TABLE models ADD COLUMN thumbnail TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE models ADD COLUMN cover_image_url TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE models ADD COLUMN nsfw INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            print("LunaMetadataDB: Migration to v2 complete")
        
        # Update version
        conn.execute(
            "UPDATE schema_info SET value = ? WHERE key = 'version'",
            (str(to_version),)
        )
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def upsert_model(self, data: Dict[str, Any]) -> int:
        """
        Insert or update a model record.
        
        Required keys: file_path, model_type
        Optional keys: all other columns
        
        Returns the row ID.
        """
        now = datetime.utcnow().isoformat()
        
        # Ensure required fields
        if 'file_path' not in data or 'model_type' not in data:
            raise ValueError("file_path and model_type are required")
        
        # Extract file_name if not provided
        if 'file_name' not in data:
            data['file_name'] = Path(data['file_path']).name
        
        # Set timestamps
        data['updated_at'] = now
        if 'created_at' not in data:
            data['created_at'] = now
        
        # Build upsert query
        columns = list(data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        updates = ', '.join([f"{col} = excluded.{col}" for col in columns if col != 'created_at'])
        
        sql = f"""
            INSERT INTO models ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(file_path, model_type) DO UPDATE SET
                {updates}
        """
        
        with self._transaction() as conn:
            cursor = conn.execute(sql, list(data.values()))
            return cursor.lastrowid if cursor.lastrowid is not None else 0
    
    def get_by_path(self, file_path: str, model_type: str) -> Optional[Dict]:
        """Get model by file path and type."""
        cursor = self._conn.execute(
            "SELECT * FROM models WHERE file_path = ? AND model_type = ?",
            (file_path, model_type)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_by_hash(self, tensor_hash: str) -> Optional[Dict]:
        """Get model by tensor hash (Civitai format)."""
        cursor = self._conn.execute(
            "SELECT * FROM models WHERE tensor_hash = ?",
            (tensor_hash,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_by_filename(self, file_name: str, model_type: Optional[str] = None) -> List[Dict]:
        """Get models by filename (may return multiple if same name in different folders)."""
        if model_type:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE file_name = ? AND model_type = ?",
                (file_name, model_type)
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE file_name = ?",
                (file_name,)
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all(self, model_type: Optional[str] = None, 
                limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Get all models, optionally filtered by type."""
        if model_type:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE model_type = ? ORDER BY file_name LIMIT ? OFFSET ?",
                (model_type, limit, offset)
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM models ORDER BY model_type, file_name LIMIT ? OFFSET ?",
                (limit, offset)
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_by_path(self, file_path: str, model_type: str) -> bool:
        """Delete a model record."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM models WHERE file_path = ? AND model_type = ?",
                (file_path, model_type)
            )
            return cursor.rowcount > 0
    
    # =========================================================================
    # SEARCH AND QUERY
    # =========================================================================
    
    def search(self, query: str, model_type: Optional[str] = None, 
               limit: int = 50) -> List[Dict]:
        """
        Full-text search across model metadata.
        Searches: file_name, title, trigger_phrase, tags, user_tags, description
        """
        if model_type:
            cursor = self._conn.execute("""
                SELECT m.* FROM models m
                JOIN models_fts fts ON m.id = fts.rowid
                WHERE models_fts MATCH ? AND m.model_type = ?
                ORDER BY rank
                LIMIT ?
            """, (query, model_type, limit))
        else:
            cursor = self._conn.execute("""
                SELECT m.* FROM models m
                JOIN models_fts fts ON m.id = fts.rowid
                WHERE models_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_by_base_model(self, base_model: str, model_type: Optional[str] = None) -> List[Dict]:
        """Get all models for a specific base model (SDXL, Pony, Illustrious, etc.)."""
        if model_type:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE base_model = ? AND model_type = ? ORDER BY file_name",
                (base_model, model_type)
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE base_model = ? ORDER BY file_name",
                (base_model,)
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_favorites(self, model_type: Optional[str] = None) -> List[Dict]:
        """Get all favorited models."""
        if model_type:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE favorite = 1 AND model_type = ? ORDER BY file_name",
                (model_type,)
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM models WHERE favorite = 1 ORDER BY model_type, file_name"
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recently_used(self, limit: int = 20, model_type: Optional[str] = None) -> List[Dict]:
        """Get recently used models."""
        if model_type:
            cursor = self._conn.execute(
                """SELECT * FROM models 
                   WHERE last_used_at IS NOT NULL AND model_type = ?
                   ORDER BY last_used_at DESC LIMIT ?""",
                (model_type, limit)
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM models 
                   WHERE last_used_at IS NOT NULL
                   ORDER BY last_used_at DESC LIMIT ?""",
                (limit,)
            )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_most_used(self, limit: int = 20, model_type: Optional[str] = None) -> List[Dict]:
        """Get most frequently used models."""
        if model_type:
            cursor = self._conn.execute(
                """SELECT * FROM models 
                   WHERE use_count > 0 AND model_type = ?
                   ORDER BY use_count DESC LIMIT ?""",
                (model_type, limit)
            )
        else:
            cursor = self._conn.execute(
                """SELECT * FROM models 
                   WHERE use_count > 0
                   ORDER BY use_count DESC LIMIT ?""",
                (limit,)
            )
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # USER ACTIONS
    # =========================================================================
    
    def set_favorite(self, file_path: str, model_type: str, favorite: bool = True) -> bool:
        """Set or clear favorite flag."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE models SET favorite = ?, updated_at = ? WHERE file_path = ? AND model_type = ?",
                (1 if favorite else 0, datetime.utcnow().isoformat(), file_path, model_type)
            )
            return cursor.rowcount > 0
    
    def set_rating(self, file_path: str, model_type: str, rating: int) -> bool:
        """Set user rating (1-5)."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")
        
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE models SET rating = ?, updated_at = ? WHERE file_path = ? AND model_type = ?",
                (rating, datetime.utcnow().isoformat(), file_path, model_type)
            )
            return cursor.rowcount > 0
    
    def add_user_tags(self, file_path: str, model_type: str, tags: List[str]) -> bool:
        """Add user tags to a model."""
        model = self.get_by_path(file_path, model_type)
        if not model:
            return False
        
        existing = set(t.strip() for t in (model.get('user_tags') or '').split(',') if t.strip())
        existing.update(tags)
        
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE models SET user_tags = ?, updated_at = ? WHERE file_path = ? AND model_type = ?",
                (', '.join(sorted(existing)), datetime.utcnow().isoformat(), file_path, model_type)
            )
            return cursor.rowcount > 0
    
    def record_usage(self, file_path: str, model_type: str) -> bool:
        """Record that a model was used (increments count, updates last_used)."""
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            cursor = conn.execute(
                """UPDATE models 
                   SET use_count = use_count + 1, last_used_at = ?, updated_at = ?
                   WHERE file_path = ? AND model_type = ?""",
                (now, now, file_path, model_type)
            )
            return cursor.rowcount > 0
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self._conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN model_type = 'lora' THEN 1 ELSE 0 END) as loras,
                SUM(CASE WHEN model_type = 'embedding' THEN 1 ELSE 0 END) as embeddings,
                SUM(CASE WHEN trigger_phrase IS NOT NULL AND trigger_phrase != '' THEN 1 ELSE 0 END) as with_triggers,
                SUM(CASE WHEN favorite = 1 THEN 1 ELSE 0 END) as favorites,
                SUM(use_count) as total_uses
            FROM models
        """)
        row = cursor.fetchone()
        
        return {
            'total_models': row['total'],
            'loras': row['loras'],
            'embeddings': row['embeddings'],
            'with_triggers': row['with_triggers'],
            'favorites': row['favorites'],
            'total_uses': row['total_uses'] or 0,
            'database_path': str(self.db_path),
            'database_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0
        }
    
    def get_base_model_counts(self) -> Dict[str, int]:
        """Get count of models per base model."""
        cursor = self._conn.execute("""
            SELECT base_model, COUNT(*) as count 
            FROM models 
            WHERE base_model IS NOT NULL AND base_model != ''
            GROUP BY base_model
            ORDER BY count DESC
        """)
        return {row['base_model']: row['count'] for row in cursor.fetchall()}
    
    # =========================================================================
    # MAINTENANCE
    # =========================================================================
    
    def vacuum(self):
        """Optimize database (reclaim space, rebuild indexes)."""
        self._conn.execute("VACUUM")
        print("LunaMetadataDB: Database vacuumed")
    
    def rebuild_fts(self):
        """Rebuild full-text search index."""
        with self._transaction() as conn:
            conn.execute("INSERT INTO models_fts(models_fts) VALUES ('rebuild')")
        print("LunaMetadataDB: FTS index rebuilt")
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance
_db: Optional[LunaMetadataDB] = None


def get_db() -> LunaMetadataDB:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = LunaMetadataDB()
    return _db


def get_trigger_phrase(file_path: str, model_type: str = 'lora') -> str:
    """Quick lookup of trigger phrase for a model."""
    model = get_db().get_by_path(file_path, model_type)
    return model.get('trigger_phrase', '') if model else ''


def get_model_metadata(file_path: str, model_type: str = 'lora') -> Optional[Dict]:
    """Quick lookup of all metadata for a model."""
    return get_db().get_by_path(file_path, model_type)


def store_civitai_metadata(file_path: str, model_type: str, 
                           civitai_data: Dict, tensor_hash: Optional[str] = None) -> int:
    """
    Store metadata from Civitai API response.
    
    Args:
        file_path: Relative path to model file
        model_type: 'lora' or 'embedding'
        civitai_data: Parsed metadata dict with modelspec.* keys
        tensor_hash: Optional tensor hash
    
    Returns:
        Database row ID
    """
    db = get_db()
    
    record = {
        'file_path': file_path,
        'model_type': model_type,
        'tensor_hash': tensor_hash or civitai_data.get('modelspec.hash_sha256'),
        
        # IDs for linking back to Civitai
        'civitai_model_id': int(civitai_data['modelspec.civitai_model_id']) if civitai_data.get('modelspec.civitai_model_id') else None,
        'civitai_version_id': int(civitai_data['modelspec.civitai_version_id']) if civitai_data.get('modelspec.civitai_version_id') else None,
        
        # Display info
        'title': civitai_data.get('modelspec.title'),
        'author': civitai_data.get('modelspec.author'),
        'description': civitai_data.get('modelspec.description'),
        'thumbnail': civitai_data.get('modelspec.thumbnail'),
        'cover_image_url': civitai_data.get('modelspec.cover_image_url'),
        
        # Usage info
        'trigger_phrase': civitai_data.get('modelspec.trigger_phrase'),
        'tags': civitai_data.get('modelspec.tags'),
        'base_model': civitai_data.get('modelspec.usage_hint'),
        'nsfw': 1 if civitai_data.get('modelspec.nsfw') == 'true' else 0,
        
        # Weights
        'default_weight': float(civitai_data['modelspec.default_weight']) if civitai_data.get('modelspec.default_weight') else None,
        'default_clip_weight': float(civitai_data['modelspec.default_clip_weight']) if civitai_data.get('modelspec.default_clip_weight') else None,
    }
    
    # Remove None values
    record = {k: v for k, v in record.items() if v is not None}
    
    return db.upsert_model(record)
