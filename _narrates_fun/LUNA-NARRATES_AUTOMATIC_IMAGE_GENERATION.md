# Automatic Image Generation for LUNA-Narrates

**Created:** 2025-11-04
Multi-GPU local image generation system powered by the Dreamer for real-time story visualization.

## Overview

LUNA-narrates automatically generates images for each story turn using the **Dreamer** as the creative image generation worker. The Dreamer handles both content ideation AND visual generation through local ComfyUI or Automatic1111 APIs, creating a unified creative pipeline that produces images for realized content and pre-generates assets for future use.

### The Dreamer's Dual Role

The Dreamer serves as both:
1. **Content Generator**: Dreams up new scenarios, characters, locations, items (existing functionality)
2. **Visual Artist**: Generates images for all content, both realized and speculative

This unified approach means:
- When the Dreamer creates a new character idea, it **immediately generates their portrait**
- When dreaming up new locations, it **pre-renders the background images**
- When imagining new items/objects, it **creates visual references**
- All these images sit ready in a **speculative asset pool**
- If the Preprocessor or Main LLM uses a Dreamer idea, **the image is already waiting**
- No generation delays when incorporating Dreamer content into the story

### Idle-Time Generation & User Pacing

The system is designed for **natural gameplay pacing**:
- **Typical turn time**: 1-10+ minutes between turns (reading, planning, typing)
- **Background generation window**: Dreamer generates images during player think time
- **Async processing**: All image generation happens in background workers
- **Zero gameplay interruption**: Images generate while you read and decide next action
- **Example**: Player takes 5 minutes to read response and plan action → Dreamer can generate 5-10 images in that time across multiple GPUs

This means the system naturally leverages your gameplay rhythm - every moment you spend crafting your next action is time the Dreamer spends building the world visually.

## Image Types

### 1. Background/Location Images
- **Generated when**: Location changes detected by LLM
- **Cached**: Yes - reused until location changes
- **Prompt source**: Location description from turn context
- **Use case**: Scene setting, environment visualization

### 2. Turn-Specific Action Images
- **Generated when**: Every turn (or configurable interval)
- **Cached**: No - unique per turn
- **Prompt source**: Current turn summary + action description
- **Use case**: Key moment visualization, dramatic scenes

### 3. Character Portrait Images
- **Generated when**: 
  - Character first appears in story
  - Significant appearance change detected (outfit, body, transformation)
- **Cached**: Yes - reused until change trigger
- **Prompt source**: Character description from tracked_items or LLM analysis
- **Update triggers**:
  - "changed outfit", "transformed into", "now wearing"
  - "appearance changed", "looks different"
  - Explicit character description updates

### 4. Multiple Variants (Optional)
- Generate 2-4 variants per turn for selection
- A/B testing for best visual match
- Gallery view in web UI

## Architecture

### Dreamer-Centric Image Generation

```
┌───────────────────────────────────────────────────────────────────┐
│                         LUNA-Narrates                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    THE DREAMER                           │    │
│  │  ┌────────────────────┐      ┌─────────────────────┐    │    │
│  │  │ Content Ideation   │ ───→ │ Image Generation    │    │    │
│  │  │  • Characters      │      │  • Character Art    │    │    │
│  │  │  • Locations       │      │  • Location Renders │    │    │
│  │  │  • Items/Objects   │      │  • Item Concepts    │    │    │
│  │  │  • Plot Twists     │      │  • Scene Sketches   │    │    │
│  │  └────────────────────┘      └─────────────────────┘    │    │
│  │           │                            │                 │    │
│  │           │                            ▼                 │    │
│  │           │                  ┌──────────────────┐        │    │
│  │           │                  │ Speculative      │        │    │
│  │           │                  │ Asset Pool       │        │    │
│  │           │                  │ (Pre-rendered)   │        │    │
│  │           │                  └──────────────────┘        │    │
│  │           ▼                            │                 │    │
│  └───────────┼────────────────────────────┼─────────────────┘    │
│              │                            │                      │
│              ▼                            ▼                      │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │  Preprocessor    │          │   Main LLM       │             │
│  │  • Use Dreamer   │          │   • Use Dreamer  │             │
│  │    ideas?        │          │     ideas?       │             │
│  │  • Images ready! │          │   • Images ready!│             │
│  └──────────────────┘          └──────────────────┘             │
│              │                            │                      │
│              └────────────┬───────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│              ┌─────────────────────────┐                         │
│              │  Realized Content       │                         │
│              │  (with images attached) │                         │
│              └────────┬────────────────┘                         │
│                       │                                          │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Multi-GPU Pool            │
        │  (Background Workers)         │
        ├───────────────────────────────┤
        │ PC1: RTX 4090 (ComfyUI)      │
        │ PC2: RTX 3090 (A1111)        │
        │ PC3: RTX 4070 (ComfyUI)      │
        │                               │
        │ • Dreamer sends image jobs    │
        │ • Load-balanced distribution  │
        │ • Results stored in asset DB  │
        └───────────────────────────────┘
```

### Image Generation Workflow

**1. Continuous Background Generation (Dreamer)**
```
Dreamer is always running → Dreams up content → Generates images immediately
                                    ↓
                        Speculative Asset Pool
                  (Characters, Locations, Items, Scenes)
                                    ↓
                    Ready when Preprocessor/LLM needs them
```

**2. Per-Turn Generation (Dreamer)**
```
Turn begins → Dreamer generates action image for current events
                                    ↓
                        Generated in real-time
                                    ↓
                        Attached to turn narration
```

**3. Asset Realization**
```
Preprocessor/LLM selects Dreamer idea → Image already exists in pool
                                                    ↓
                                            Move to active assets
                                                    ↓
                                            Display immediately
```

## Configuration

### Database Schema

```sql
-- Image generation queue and history
CREATE TABLE luna.generated_images (
    image_id SERIAL PRIMARY KEY,
    turn_id INTEGER REFERENCES luna.turn_history(turn_id), -- NULL for speculative assets
    image_type VARCHAR(50) NOT NULL, -- 'background', 'action', 'character', 'item', 'scene'
    subject_key VARCHAR(255), -- location name, character name, item name, etc.
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    
    -- Generation settings
    model_name VARCHAR(100),
    sampler VARCHAR(50),
    steps INTEGER,
    cfg_scale DECIMAL(4,2),
    seed BIGINT,
    width INTEGER,
    height INTEGER,
    
    -- Dreamer integration
    dreamer_content_id INTEGER, -- Links to Dreamer's content generation
    is_speculative BOOLEAN DEFAULT FALSE, -- True if pre-generated, False if realized
    realized_at TIMESTAMPTZ, -- When speculative asset became used in story
    realized_by VARCHAR(50), -- 'preprocessor', 'main_llm', 'direct'
    
    -- Cache management
    cache_key VARCHAR(64), -- SHA-256 of (type + subject_key + prompt)
    is_cached BOOLEAN DEFAULT FALSE,
    cache_source INTEGER REFERENCES luna.generated_images(image_id),
    
    -- Multi-GPU routing
    gpu_worker VARCHAR(100), -- 'pc1-comfyui', 'pc2-a1111', etc.
    generation_time_ms INTEGER,
    
    -- Storage
    local_path TEXT NOT NULL,
    checksum VARCHAR(64),
    file_size INTEGER,
    
    -- Metadata
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending' -- pending, generating, complete, failed
);

CREATE INDEX idx_generated_images_turn ON luna.generated_images(turn_id);
CREATE INDEX idx_generated_images_cache_key ON luna.generated_images(cache_key);
CREATE INDEX idx_generated_images_type ON luna.generated_images(image_type);
CREATE INDEX idx_generated_images_subject ON luna.generated_images(subject_key);

-- Image generation workers/GPUs
CREATE TABLE luna.image_workers (
    worker_id SERIAL PRIMARY KEY,
    worker_name VARCHAR(100) UNIQUE NOT NULL, -- 'pc1-comfyui', 'pc2-a1111'
    api_type VARCHAR(20) NOT NULL, -- 'comfyui', 'automatic1111'
    api_url TEXT NOT NULL,
    
    -- Capabilities
    max_concurrent INTEGER DEFAULT 1,
    supports_sdxl BOOLEAN DEFAULT TRUE,
    supports_controlnet BOOLEAN DEFAULT FALSE,
    vram_gb INTEGER,
    
    -- Status
    is_online BOOLEAN DEFAULT TRUE,
    current_load INTEGER DEFAULT 0, -- number of active generations
    last_heartbeat TIMESTAMPTZ,
    
    -- Stats
    total_generated INTEGER DEFAULT 0,
    avg_generation_time_ms INTEGER,
    error_count INTEGER DEFAULT 0
);

-- Dreamer content tracking (links text ideas to generated images)
CREATE TABLE luna.dreamer_content (
    content_id SERIAL PRIMARY KEY,
    story_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'character', 'location', 'item', 'plot_twist', 'scene'
    
    -- Content description
    name VARCHAR(255), -- Character/location/item name
    description TEXT NOT NULL,
    metadata JSONB, -- Additional structured data
    
    -- Realization tracking
    is_realized BOOLEAN DEFAULT FALSE, -- Used in story yet?
    realized_at TIMESTAMPTZ,
    realized_in_turn INTEGER REFERENCES luna.turn_history(turn_id),
    realized_by VARCHAR(50), -- 'preprocessor', 'main_llm'
    
    -- Associated images
    primary_image_id INTEGER REFERENCES luna.generated_images(image_id),
    variant_image_ids INTEGER[], -- Additional generated variants
    
    -- Dreamer metadata
    dreamed_at TIMESTAMPTZ DEFAULT NOW(),
    dream_session_id VARCHAR(100), -- Groups content from same dreaming session
    confidence_score DECIMAL(3,2), -- How "good" Dreamer thinks this is (0-1)
    
    -- Expiry (cleanup unused speculative content)
    expires_at TIMESTAMPTZ, -- Auto-cleanup after N days if unrealized
    
    CONSTRAINT unique_content UNIQUE (story_id, content_type, name)
);

CREATE INDEX idx_dreamer_content_story ON luna.dreamer_content(story_id);
CREATE INDEX idx_dreamer_content_type ON luna.dreamer_content(content_type);
CREATE INDEX idx_dreamer_content_realized ON luna.dreamer_content(is_realized);
CREATE INDEX idx_dreamer_content_expires ON luna.dreamer_content(expires_at) WHERE is_realized = FALSE;

-- Character appearance tracking
CREATE TABLE luna.character_appearances (
    appearance_id SERIAL PRIMARY KEY,
    character_name VARCHAR(255) NOT NULL,
    turn_id INTEGER REFERENCES luna.turn_history(turn_id),
    dreamer_content_id INTEGER REFERENCES luna.dreamer_content(content_id), -- Link to dreamed character
    
    -- Appearance description
    description TEXT NOT NULL,
    outfit TEXT,
    physical_features TEXT,
    
    -- Change detection
    is_significant_change BOOLEAN DEFAULT FALSE,
    change_reason TEXT, -- 'new_outfit', 'transformation', 'initial_appearance'
    previous_appearance_id INTEGER REFERENCES luna.character_appearances(appearance_id),
    
    -- Associated image
    portrait_image_id INTEGER REFERENCES luna.generated_images(image_id),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_character_appearances_name ON luna.character_appearances(character_name);
CREATE INDEX idx_character_appearances_turn ON luna.character_appearances(turn_id);
CREATE INDEX idx_character_appearances_dreamer ON luna.character_appearances(dreamer_content_id);
```

### Settings Configuration

```json
{
  "image_generation": {
    "enabled": true,
    "mode": "dreamer_driven", // Dreamer manages all image generation
    
    "dreamer": {
      "enabled": true,
      "continuous_generation": true, // Always generate images for dreamed content
      "speculative_pool_size": 100, // Max unrealized assets before cleanup
      "asset_expiry_days": 30, // Delete unrealized assets after N days
      "variants_per_concept": 1, // Generate N image variants per dreamed concept
      "confidence_threshold": 0.6 // Only generate images for ideas above this score
    },
    
    "workers": [
      {
        "name": "pc1-comfyui",
        "api_type": "comfyui",
        "api_url": "http://192.168.1.100:8188",
        "max_concurrent": 2,
        "priority": 1
      },
      {
        "name": "pc2-automatic1111",
        "api_type": "automatic1111",
        "api_url": "http://192.168.1.101:7860",
        "max_concurrent": 1,
        "priority": 2
      },
      {
        "name": "pc3-comfyui",
        "api_type": "comfyui",
        "api_url": "http://192.168.1.102:8188",
        "max_concurrent": 2,
        "priority": 3
      }
    ],
    
    "background_images": {
      "enabled": true,
      "generate_on_location_change": true,
      "cache_duration_turns": 50, // reuse for N turns before regenerating
      "default_size": [1024, 768],
      "model": "sdxl_base_1.0",
      "steps": 30,
      "cfg_scale": 7.5
    },
    
    "turn_images": {
      "enabled": true,
      "generate_every_n_turns": 1, // 1 = every turn, 2 = every other turn
      "variants": 1, // number of variants per turn
      "default_size": [1024, 1024],
      "model": "sdxl_base_1.0",
      "steps": 40,
      "cfg_scale": 8.0
    },
    
    "character_portraits": {
      "enabled": true,
      "generate_on_first_appearance": true,
      "regenerate_on_change": true,
      "change_keywords": [
        "changed outfit", "wearing", "dressed in",
        "transformed into", "turned into",
        "appearance changed", "looks different",
        "now has", "gained", "lost"
      ],
      "default_size": [768, 1024],
      "model": "sdxl_base_1.0",
      "steps": 35,
      "cfg_scale": 7.0
    },
    
    "storage": {
      "base_path": "data/generated_images",
      "organize_by": "story_and_type", // 'story_and_type', 'turn', 'date'
      "format": "png",
      "quality": 95
    },
    
    "prompt_templates": {
      "background": "cinematic wide shot of {location}, {description}, {time_of_day}, {weather}, professional photography, highly detailed, 8k",
      "action": "{action_description}, {characters_present}, {location}, cinematic composition, dramatic lighting, high quality",
      "character": "portrait of {character_name}, {physical_description}, {outfit}, {expression}, professional character art, detailed face, 8k"
    },
    
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, signature, text"
  }
}
```

## Implementation

### 1. Dreamer Image Generation Module

```python
# server/narration/dreamer_image_generator.py

import asyncio
from typing import Optional, Dict, List
from enum import Enum

class DreamerImageGenerator:
    """
    Image generation module integrated into the Dreamer.
    Generates images for all dreamed content immediately upon creation.
    """
    
    def __init__(self, db_pool, gpu_workers, settings):
        self.db = db_pool
        self.workers = gpu_workers
        self.settings = settings['image_generation']['dreamer']
        self.queue = asyncio.Queue()
        self.running = False
        
    async def start(self):
        """Start background image generation worker."""
        self.running = True
        asyncio.create_task(self._generation_worker())
        
    async def stop(self):
        """Stop background worker."""
        self.running = False
        
    async def dream_with_image(
        self,
        story_id: str,
        content_type: str,
        name: str,
        description: str,
        metadata: Dict,
        confidence: float
    ) -> Dict:
        """
        Dream up new content AND generate its image immediately.
        This is called by the Dreamer during content generation.
        
        Returns:
            {
                'content_id': int,
                'content': {...},
                'image_path': str,
                'generation_time_ms': int
            }
        """
        
        # Only generate images for high-confidence ideas
        if confidence < self.settings['confidence_threshold']:
            return await self._store_content_without_image(
                story_id, content_type, name, description, metadata, confidence
            )
        
        # Store dreamed content first
        content_id = await self._store_dreamer_content(
            story_id, content_type, name, description, metadata, confidence
        )
        
        # Generate image immediately (queue it)
        image_job = {
            'content_id': content_id,
            'story_id': story_id,
            'content_type': content_type,
            'name': name,
            'description': description,
            'metadata': metadata
        }
        
        await self.queue.put(image_job)
        
        # Return immediately - image generates in background
        return {
            'content_id': content_id,
            'content': {
                'type': content_type,
                'name': name,
                'description': description,
                'metadata': metadata
            },
            'image_status': 'queued',
            'confidence': confidence
        }
    
    async def _generation_worker(self):
        """
        Background worker that continuously generates images
        for queued Dreamer content.
        """
        while self.running:
            try:
                # Get next job from queue (wait up to 1 second)
                job = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=1.0
                )
                
                # Generate image
                await self._generate_for_content(job)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in Dreamer image generation: {e}")
                continue
    
    async def _generate_for_content(self, job: Dict):
        """Generate image for a specific piece of dreamed content."""
        
        content_type = job['content_type']
        
        # Build prompt based on content type
        if content_type == 'character':
            prompt = self._build_character_prompt(
                job['name'], 
                job['description']
            )
            size = [768, 1024]
            
        elif content_type == 'location':
            prompt = self._build_location_prompt(
                job['name'],
                job['description']
            )
            size = [1024, 768]
            
        elif content_type == 'item':
            prompt = self._build_item_prompt(
                job['name'],
                job['description']
            )
            size = [768, 768]
            
        elif content_type == 'scene':
            prompt = self._build_scene_prompt(
                job['description']
            )
            size = [1024, 1024]
            
        else:
            # Generic prompt for other types
            prompt = f"{job['description']}, highly detailed, professional art"
            size = [1024, 1024]
        
        # Select worker
        worker = await self._select_available_worker()
        
        # Generate image
        import time
        start_time = time.time()
        
        image_path = await self._call_worker_api(
            worker, prompt, size[0], size[1]
        )
        
        generation_time = int((time.time() - start_time) * 1000)
        
        # Store in database
        async with self.db.acquire() as conn:
            image_id = await conn.fetchval("""
                INSERT INTO luna.generated_images (
                    turn_id, image_type, subject_key, prompt,
                    dreamer_content_id, is_speculative,
                    width, height, local_path,
                    gpu_worker, generation_time_ms, status
                )
                VALUES (
                    NULL, $1, $2, $3, $4, TRUE,
                    $5, $6, $7, $8, $9, 'complete'
                )
                RETURNING image_id
            """, content_type, job['name'], prompt, job['content_id'],
                size[0], size[1], image_path, worker['worker_name'], generation_time)
            
            # Link to dreamer content
            await conn.execute("""
                UPDATE luna.dreamer_content
                SET primary_image_id = $1
                WHERE content_id = $2
            """, image_id, job['content_id'])
        
        print(f"✨ Dreamer generated {content_type} image: {job['name']} ({generation_time}ms)")
    
    async def realize_content(
        self,
        content_id: int,
        turn_id: int,
        realized_by: str
    ):
        """
        Mark dreamed content as realized (used in story).
        Updates both content and image records.
        """
        async with self.db.acquire() as conn:
            # Update dreamer content
            await conn.execute("""
                UPDATE luna.dreamer_content
                SET is_realized = TRUE,
                    realized_at = NOW(),
                    realized_in_turn = $1,
                    realized_by = $2
                WHERE content_id = $3
            """, turn_id, realized_by, content_id)
            
            # Update associated image
            await conn.execute("""
                UPDATE luna.generated_images
                SET is_speculative = FALSE,
                    realized_at = NOW(),
                    realized_by = $1,
                    turn_id = $2
                WHERE dreamer_content_id = $3
            """, realized_by, turn_id, content_id)
        
        print(f"✅ Realized Dreamer content #{content_id} in turn {turn_id}")
    
    async def get_speculative_assets(
        self,
        story_id: str,
        content_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all unrealized (speculative) content with images
        available for the Preprocessor/LLM to use.
        """
        async with self.db.acquire() as conn:
            query = """
                SELECT 
                    dc.content_id,
                    dc.content_type,
                    dc.name,
                    dc.description,
                    dc.metadata,
                    dc.confidence_score,
                    dc.dreamed_at,
                    gi.image_id,
                    gi.local_path as image_path,
                    gi.width,
                    gi.height
                FROM luna.dreamer_content dc
                LEFT JOIN luna.generated_images gi 
                    ON dc.primary_image_id = gi.image_id
                WHERE dc.story_id = $1
                  AND dc.is_realized = FALSE
            """
            
            params = [story_id]
            
            if content_type:
                query += " AND dc.content_type = $2"
                params.append(content_type)
            
            query += " ORDER BY dc.confidence_score DESC, dc.dreamed_at DESC"
            
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]

    async def cleanup_expired_assets(self):
        """
        Remove old unrealized speculative content and images.
        Runs periodically to free up storage.
        """
        expiry_days = self.settings['asset_expiry_days']
        
        async with self.db.acquire() as conn:
            # Get expired content IDs
            expired = await conn.fetch("""
                SELECT content_id, primary_image_id
                FROM luna.dreamer_content
                WHERE is_realized = FALSE
                  AND dreamed_at < NOW() - INTERVAL '{} days'
            """.format(expiry_days))
            
            for row in expired:
                # Delete image file
                if row['primary_image_id']:
                    image_path = await conn.fetchval("""
                        SELECT local_path FROM luna.generated_images
                        WHERE image_id = $1
                    """, row['primary_image_id'])
                    
                    if image_path:
                        import os
                        try:
                            os.remove(image_path)
                        except:
                            pass
                    
                    # Delete image record
                    await conn.execute("""
                        DELETE FROM luna.generated_images
                        WHERE image_id = $1
                    """, row['primary_image_id'])
                
                # Delete content record
                await conn.execute("""
                    DELETE FROM luna.dreamer_content
                    WHERE content_id = $1
                """, row['content_id'])
            
            if expired:
                print(f"🧹 Cleaned up {len(expired)} expired speculative assets")
```

### 2. Image Generation Orchestrator (For Turn-Specific Images)

```python
# server/image_generation/orchestrator.py

import asyncio
import hashlib
from typing import Optional, List, Dict
from enum import Enum

class ImageType(Enum):
    BACKGROUND = "background"
    ACTION = "action"
    CHARACTER = "character"

class ImageGenerationOrchestrator:
    """
    Coordinates image generation across multiple GPU workers.
    Manages caching, change detection, and load balancing.
    """
    
    def __init__(self, db_pool, settings):
        self.db = db_pool
        self.settings = settings['image_generation']
        self.workers = []
        self.location_cache = {}
        self.character_cache = {}
        
    async def initialize(self):
        """Load workers and restore cache from database."""
        await self._load_workers()
        await self._restore_caches()
        
    async def generate_images_for_turn(
        self, 
        turn_id: int,
        turn_data: Dict,
        force_regenerate: bool = False
    ) -> Dict[str, List[str]]:
        """
        Generate all relevant images for a turn.
        
        Returns:
            Dict with image_type -> [image_paths]
        """
        tasks = []
        results = {}
        
        # 1. Background/Location image
        if self.settings['background_images']['enabled']:
            location = self._extract_location(turn_data)
            bg_task = self._generate_background_image(
                turn_id, location, force_regenerate
            )
            tasks.append(('background', bg_task))
        
        # 2. Turn action image(s)
        if self.settings['turn_images']['enabled']:
            turn_num = turn_data['turn_number']
            every_n = self.settings['turn_images']['generate_every_n_turns']
            
            if turn_num % every_n == 0 or force_regenerate:
                variants = self.settings['turn_images']['variants']
                for v in range(variants):
                    action_task = self._generate_action_image(
                        turn_id, turn_data, variant=v
                    )
                    tasks.append((f'action_{v}', action_task))
        
        # 3. Character portraits
        if self.settings['character_portraits']['enabled']:
            character_tasks = await self._check_character_changes(
                turn_id, turn_data, force_regenerate
            )
            tasks.extend(character_tasks)
        
        # Execute all generation tasks concurrently
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (key, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    print(f"Error generating {key}: {result}")
                    continue
                
                if key.startswith('action_'):
                    results.setdefault('action', []).append(result)
                else:
                    results[key] = result
        
        return results
    
    async def _generate_background_image(
        self,
        turn_id: int,
        location: str,
        force: bool = False
    ) -> str:
        """Generate or retrieve cached background image."""
        
        # Check cache
        cache_key = self._calculate_cache_key(
            ImageType.BACKGROUND, location, None
        )
        
        if not force and cache_key in self.location_cache:
            cached_image_id = self.location_cache[cache_key]
            # Verify cache is still valid
            if await self._is_cache_valid(cached_image_id):
                await self._link_cached_image(turn_id, cached_image_id)
                return await self._get_image_path(cached_image_id)
        
        # Generate new image
        prompt = self._build_background_prompt(location)
        settings = self.settings['background_images']
        
        image_path = await self._generate_image(
            turn_id=turn_id,
            image_type=ImageType.BACKGROUND,
            subject_key=location,
            prompt=prompt,
            width=settings['default_size'][0],
            height=settings['default_size'][1],
            model=settings['model'],
            steps=settings['steps'],
            cfg_scale=settings['cfg_scale']
        )
        
        # Update cache
        image_id = await self._get_image_id_by_path(image_path)
        self.location_cache[cache_key] = image_id
        
        return image_path
    
    async def _generate_action_image(
        self,
        turn_id: int,
        turn_data: Dict,
        variant: int = 0
    ) -> str:
        """Generate turn-specific action image."""
        
        prompt = self._build_action_prompt(turn_data)
        settings = self.settings['turn_images']
        
        # Use different seed for variants
        seed = self._generate_seed(turn_id, variant)
        
        image_path = await self._generate_image(
            turn_id=turn_id,
            image_type=ImageType.ACTION,
            subject_key=f"turn_{turn_data['turn_number']}_v{variant}",
            prompt=prompt,
            width=settings['default_size'][0],
            height=settings['default_size'][1],
            model=settings['model'],
            steps=settings['steps'],
            cfg_scale=settings['cfg_scale'],
            seed=seed
        )
        
        return image_path
    
    async def _check_character_changes(
        self,
        turn_id: int,
        turn_data: Dict,
        force: bool = False
    ) -> List[tuple]:
        """
        Detect character appearance changes and queue portrait generation.
        
        Returns:
            List of (key, task) tuples for character portrait generation
        """
        tasks = []
        characters = self._extract_characters(turn_data)
        change_keywords = self.settings['character_portraits']['change_keywords']
        
        for char_name, char_data in characters.items():
            description = char_data.get('description', '')
            
            # Check for first appearance
            if char_name not in self.character_cache:
                if self.settings['character_portraits']['generate_on_first_appearance']:
                    task = self._generate_character_portrait(
                        turn_id, char_name, description, 
                        change_reason='initial_appearance'
                    )
                    tasks.append((f'character_{char_name}', task))
                continue
            
            # Check for significant changes
            if force or self.settings['character_portraits']['regenerate_on_change']:
                has_change, reason = self._detect_character_change(
                    description, change_keywords
                )
                
                if has_change:
                    task = self._generate_character_portrait(
                        turn_id, char_name, description,
                        change_reason=reason
                    )
                    tasks.append((f'character_{char_name}', task))
        
        return tasks
    
    async def _generate_character_portrait(
        self,
        turn_id: int,
        character_name: str,
        description: str,
        change_reason: str
    ) -> str:
        """Generate character portrait image."""
        
        prompt = self._build_character_prompt(character_name, description)
        settings = self.settings['character_portraits']
        
        image_path = await self._generate_image(
            turn_id=turn_id,
            image_type=ImageType.CHARACTER,
            subject_key=character_name,
            prompt=prompt,
            width=settings['default_size'][0],
            height=settings['default_size'][1],
            model=settings['model'],
            steps=settings['steps'],
            cfg_scale=settings['cfg_scale']
        )
        
        # Record appearance change
        await self._record_character_appearance(
            character_name, turn_id, description,
            change_reason, image_path
        )
        
        # Update cache
        image_id = await self._get_image_id_by_path(image_path)
        self.character_cache[character_name] = {
            'image_id': image_id,
            'description': description,
            'turn_id': turn_id
        }
        
        return image_path
    
    async def _generate_image(
        self,
        turn_id: int,
        image_type: ImageType,
        subject_key: str,
        prompt: str,
        width: int,
        height: int,
        model: str,
        steps: int,
        cfg_scale: float,
        seed: Optional[int] = None
    ) -> str:
        """
        Core image generation method.
        Routes to best available worker and manages generation.
        """
        
        # Select worker
        worker = await self._select_worker()
        
        if not worker:
            raise RuntimeError("No available image generation workers")
        
        # Calculate cache key
        cache_key = self._calculate_cache_key(
            image_type, subject_key, prompt
        )
        
        # Create database record
        negative_prompt = self.settings.get('negative_prompt', '')
        
        async with self.db.acquire() as conn:
            image_id = await conn.fetchval("""
                INSERT INTO luna.generated_images (
                    turn_id, image_type, subject_key, prompt, negative_prompt,
                    model_name, steps, cfg_scale, seed, width, height,
                    cache_key, gpu_worker, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'generating')
                RETURNING image_id
            """, turn_id, image_type.value, subject_key, prompt, negative_prompt,
                model, steps, cfg_scale, seed, width, height,
                cache_key, worker['worker_name'])
        
        # Generate image
        try:
            import time
            start_time = time.time()
            
            image_path = await self._call_worker_api(
                worker, prompt, negative_prompt,
                model, steps, cfg_scale, seed, width, height
            )
            
            generation_time = int((time.time() - start_time) * 1000)
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(image_path)
            file_size = Path(image_path).stat().st_size
            
            # Update database record
            async with self.db.acquire() as conn:
                await conn.execute("""
                    UPDATE luna.generated_images
                    SET status = 'complete',
                        local_path = $1,
                        checksum = $2,
                        file_size = $3,
                        generation_time_ms = $4
                    WHERE image_id = $5
                """, image_path, checksum, file_size, generation_time, image_id)
            
            return image_path
            
        except Exception as e:
            # Mark as failed
            async with self.db.acquire() as conn:
                await conn.execute("""
                    UPDATE luna.generated_images
                    SET status = 'failed'
                    WHERE image_id = $1
                """, image_id)
            raise
    
    def _detect_character_change(
        self,
        description: str,
        keywords: List[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Detect if character description indicates significant appearance change.
        
        Returns:
            (has_change, reason)
        """
        description_lower = description.lower()
        
        for keyword in keywords:
            if keyword.lower() in description_lower:
                return True, keyword
        
        return False, None
    
    def _calculate_cache_key(
        self,
        image_type: ImageType,
        subject_key: str,
        prompt: Optional[str]
    ) -> str:
        """Calculate SHA-256 cache key."""
        content = f"{image_type.value}:{subject_key}:{prompt or ''}"
        return hashlib.sha256(content.encode()).hexdigest()
```

### 2. Worker API Adapters

```python
# server/image_generation/workers/comfyui_adapter.py

class ComfyUIAdapter:
    """Adapter for ComfyUI API."""
    
    async def generate(
        self,
        api_url: str,
        prompt: str,
        negative_prompt: str,
        model: str,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
        width: int,
        height: int
    ) -> str:
        """
        Call ComfyUI API to generate image.
        Returns path to saved image.
        """
        import aiohttp
        import json
        
        # Build ComfyUI workflow
        workflow = self._build_workflow(
            prompt, negative_prompt, model,
            steps, cfg_scale, seed, width, height
        )
        
        # Submit to queue
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/prompt",
                json={"prompt": workflow}
            ) as resp:
                result = await resp.json()
                prompt_id = result['prompt_id']
            
            # Poll for completion
            while True:
                async with session.get(
                    f"{api_url}/history/{prompt_id}"
                ) as resp:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        # Get output image
                        outputs = history[prompt_id]['outputs']
                        # Extract image data and save
                        # ... implementation details ...
                        return image_path
                
                await asyncio.sleep(1)
```

```python
# server/image_generation/workers/automatic1111_adapter.py

class Automatic1111Adapter:
    """Adapter for Automatic1111 API."""
    
    async def generate(
        self,
        api_url: str,
        prompt: str,
        negative_prompt: str,
        model: str,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
        width: int,
        height: int
    ) -> str:
        """
        Call Automatic1111 API to generate image.
        Returns path to saved image.
        """
        import aiohttp
        import base64
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed if seed else -1,
            "width": width,
            "height": height,
            "sampler_name": "DPM++ 2M Karras"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/sdapi/v1/txt2img",
                json=payload
            ) as resp:
                result = await resp.json()
                
                # Decode base64 image
                image_data = base64.b64decode(result['images'][0])
                
                # Save to disk
                # ... implementation details ...
                return image_path
```

### 3. Integration with LUNA-Narrates

```python
# server/narration/turn_processor.py

class TurnProcessor:
    """Main turn processing with image generation."""
    
    def __init__(self):
        self.image_orchestrator = ImageGenerationOrchestrator(db_pool, settings)
        
    async def process_turn(self, turn_data: Dict):
        """Process turn with narration and image generation."""
        
        # 1. Generate narration (existing logic)
        narration = await self.generate_narration(turn_data)
        
        # 2. Generate images (parallel with narration if desired)
        if settings['image_generation']['enabled']:
            images = await self.image_orchestrator.generate_images_for_turn(
                turn_id=turn_data['turn_id'],
                turn_data=turn_data
            )
            
            # Attach images to narration result
            narration['images'] = images
        
        return narration
```

## Usage Examples

### 1. Dreamer Creates Content with Images

```python
# Dreamer dreams up a new character WITH image generation
dreamer = DreamerImageGenerator(db_pool, gpu_workers, settings)
await dreamer.start()

result = await dreamer.dream_with_image(
    story_id='story_123',
    content_type='character',
    name='Shadow Assassin',
    description='A mysterious figure cloaked in darkness, dual daggers gleaming',
    metadata={'faction': 'rogue', 'threat_level': 8},
    confidence=0.85
)

# Result:
# {
#   'content_id': 456,
#   'content': {
#     'type': 'character',
#     'name': 'Shadow Assassin',
#     'description': '...',
#     'metadata': {...}
#   },
#   'image_status': 'queued',  # Generating in background
#   'confidence': 0.85
# }

# Image generates in background, stored in speculative asset pool
```

### 2. Preprocessor/LLM Uses Dreamer Content

```python
# Preprocessor checks available Dreamer content
speculative_characters = await dreamer.get_speculative_assets(
    story_id='story_123',
    content_type='character'
)

# Results:
# [
#   {
#     'content_id': 456,
#     'content_type': 'character',
#     'name': 'Shadow Assassin',
#     'description': '...',
#     'confidence_score': 0.85,
#     'image_id': 789,
#     'image_path': '/data/generated_images/speculative/character_456.png',
#     'width': 768,
#     'height': 1024
#   }
# ]

# Preprocessor decides to use this character
await dreamer.realize_content(
    content_id=456,
    turn_id=42,
    realized_by='preprocessor'
)

# Image is now linked to turn 42, ready to display immediately!
```

### 3. Dreamer Pre-Generates Multiple Assets

```python
# Dreamer continuously generates speculative content
async def dreamer_background_loop():
    while True:
        # Dream up new location
        await dreamer.dream_with_image(
            story_id='story_123',
            content_type='location',
            name='Abandoned Temple',
            description='Ancient stone ruins overgrown with vines, shafts of light through collapsed ceiling',
            metadata={'danger_level': 7, 'loot_potential': 9},
            confidence=0.78
        )
        
        # Dream up new item
        await dreamer.dream_with_image(
            story_id='story_123',
            content_type='item',
            name='Cursed Amulet',
            description='A dark jade pendant that pulses with malevolent energy',
            metadata={'type': 'accessory', 'rarity': 'legendary'},
            confidence=0.92
        )
        
        # Dream up plot twist scenario
        await dreamer.dream_with_image(
            story_id='story_123',
            content_type='scene',
            name='Betrayal at Dawn',
            description='Your trusted companion reveals their true allegiance as enemy forces surround you',
            metadata={'impact': 'high', 'emotional_weight': 8},
            confidence=0.88
        )
        
        await asyncio.sleep(60)  # Dream every minute

# All images generate in background, ready when needed!
```

### 4. Turn-Specific Action Images (Still Generated)

```python
# Orchestrator handles per-turn action images
orchestrator = ImageGenerationOrchestrator(db_pool, settings)

# Generate action image for current turn events
action_image = await orchestrator._generate_action_image(
    turn_id=42,
    turn_data={
        'turn_number': 42,
        'player_input': 'I charge the dragon with my sword raised',
        'ai_response': 'The dragon roars and spreads its wings...'
    }
)

# Result: '/data/generated_images/story_123/turn_0042/action.png'
```

### 5. Check Speculative Asset Pool Status

```python
# Get all unrealized Dreamer content with images
all_speculative = await dreamer.get_speculative_assets(story_id='story_123')

print(f"Speculative asset pool: {len(all_speculative)} items")
# Speculative asset pool: 47 items

# Breakdown by type
from collections import Counter
types = Counter(asset['content_type'] for asset in all_speculative)
print(types)
# Counter({'character': 15, 'location': 12, 'item': 10, 'scene': 8, 'plot_twist': 2})

# Top confidence assets
top_assets = sorted(all_speculative, key=lambda x: x['confidence_score'], reverse=True)[:5]
for asset in top_assets:
    print(f"{asset['name']} ({asset['content_type']}): {asset['confidence_score']}")
# Cursed Amulet (item): 0.92
# Betrayal at Dawn (scene): 0.88
# Shadow Assassin (character): 0.85
# Crystal Caverns (location): 0.83
# Ancient Tome (item): 0.81
```

### 6. Cleanup Old Unused Assets

```python
# Periodically cleanup expired speculative content
await dreamer.cleanup_expired_assets()
# 🧹 Cleaned up 12 expired speculative assets

# This removes:
# - Content that wasn't used after 30 days (configurable)
# - Associated image files
# - Database records
```

## Performance Considerations

### Multi-GPU Load Balancing

The orchestrator automatically distributes work across available GPUs:

1. **Worker Selection**: Chooses least-loaded worker with required capabilities
2. **Concurrent Limits**: Respects `max_concurrent` per worker
3. **Priority Routing**: Higher priority workers get jobs first
4. **Failover**: Automatically retries on different worker if generation fails

### Generation Times (Estimated)

**Single GPU (RTX 4090 + SDXL Base):**
- Character portrait: ~3.5-5 seconds (35 steps)
- Location background: ~3-5 seconds (30 steps)
- Item/object: ~3-4 seconds (30 steps)
- Turn action: ~4-6 seconds (40 steps)
- Scene sketch: ~4-6 seconds (35 steps)

**Dreamer Background Generation:**
- Continuous generation while idle
- No impact on turn processing speed
- Pre-generates 10-20 assets per hour per GPU
- Assets ready instantly when needed

**Per Turn (action image only):** ~4-6 seconds
- Background/character images already cached from Dreamer pool
- Only current action needs real-time generation

**100 Turns with Dreamer System:**
- Turn action images: 100 × 5s = ~8 minutes
- Speculative assets: Generated in background (free time)
- **Total wait time: ~8 minutes** (vs 25-40 minutes without Dreamer caching)

### Dreamer Caching Benefits

**Traditional approach (generate on-demand):**
- 100 turns × 3 images = 300 generations during gameplay
- Player waits 10-15 seconds per turn for images
- **Total wait time: 25-40 minutes**

**Dreamer approach (speculative pre-generation):**
- Background generation: Continuous while idle (no wait)
- Speculative pool: 20-50 pre-generated assets ready instantly
- Per-turn generation: Only action images (4-6 seconds)
- **Total wait time: ~8 minutes** (70% reduction)

**Speculative Asset Pool Efficiency:**
- Dreamer generates 10-20 assets per hour per GPU
- Typical story uses 30-40% of generated speculative content
- Unused assets expire after 30 days (configurable)
- Net storage overhead: ~2-5GB per long-running story
- Benefit: Zero-latency asset insertion when ideas are realized

**Example Pool Status After 3 Hours:**
- Characters: 15 dreamed, 4 realized (27% usage)
- Locations: 12 dreamed, 3 realized (25% usage)
- Items: 18 dreamed, 8 realized (44% usage)
- Scenes: 10 dreamed, 2 realized (20% usage)
- **Total: 55 pre-generated, 17 used** (31% realization rate)

## PNG Metadata Embedding & Asset Cards

### Concept: Self-Contained Asset Cards

Every generated image is a **complete, portable asset card** with all metadata embedded:

```
character_portrait.png
├─ Visual: Generated character portrait
└─ Embedded Metadata (PNG chunks):
   ├─ content_type: "character"
   ├─ name: "Shadow Assassin"
   ├─ description: "A mysterious figure cloaked in darkness..."
   ├─ attributes: {"faction": "rogue", "threat_level": 8, ...}
   ├─ generation_prompt: "portrait of Shadow Assassin, cloaked in darkness..."
   ├─ model: "sdxl_base_1.0"
   ├─ generation_params: {"steps": 35, "cfg": 7.0, "seed": 12345}
   ├─ dreamed_by: "user_12345"
   ├─ confidence_score: 0.85
   ├─ story_context: "Cyberpunk noir detective story"
   └─ version: "luna-narrates-v1.0"
```

### Implementation

```python
# server/image_generation/png_metadata.py

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

class AssetCardMetadata:
    """Embed and extract LUNA asset card metadata in PNG files."""
    
    @staticmethod
    def embed_metadata(
        image_path: str,
        content_type: str,
        name: str,
        description: str,
        attributes: dict,
        generation_data: dict,
        user_id: str = None
    ):
        """
        Embed complete asset metadata into PNG file.
        Compatible with ComfyUI/A1111 existing metadata.
        """
        img = Image.open(image_path)
        metadata = PngInfo()
        
        # LUNA-specific metadata (custom chunk)
        luna_data = {
            'format_version': '1.0',
            'content_type': content_type,
            'name': name,
            'description': description,
            'attributes': attributes,
            'generation': generation_data,
            'created_by': user_id,
            'created_at': datetime.now().isoformat(),
            'license': 'CC-BY-SA-4.0'  # Community sharing license
        }
        
        metadata.add_text('LUNA-Asset-Card', json.dumps(luna_data))
        
        # Also embed standard generation params for compatibility
        if 'prompt' in generation_data:
            metadata.add_text('prompt', generation_data['prompt'])
        if 'model' in generation_data:
            metadata.add_text('model', generation_data['model'])
        
        # Save with metadata
        img.save(image_path, pnginfo=metadata)
    
    @staticmethod
    def extract_metadata(image_path: str) -> dict:
        """Extract LUNA asset card metadata from PNG."""
        img = Image.open(image_path)
        
        if 'LUNA-Asset-Card' in img.text:
            return json.loads(img.text['LUNA-Asset-Card'])
        
        return None
    
    @staticmethod
    def validate_asset_card(metadata: dict) -> bool:
        """Validate asset card has required fields."""
        required = ['format_version', 'content_type', 'name', 'description']
        return all(field in metadata for field in required)
```

### Community Asset Sharing System

Users can **import/export** Dreamer-generated content as PNG files:

**Export Flow:**
1. User downloads character/location/item PNG from their story
2. PNG contains complete embedded metadata
3. Share file on community hub, Discord, Reddit, etc.
4. Other users download and import

**Import Flow:**
1. User drags PNG into LUNA-narrates web UI
2. System extracts metadata from PNG
3. Validates asset card format
4. Adds to user's personal asset library
5. Available for Preprocessor/LLM to use in stories

### Database Schema for Community Assets

```sql
-- Community asset library
CREATE TABLE luna.community_assets (
    asset_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES luna.users(user_id),
    
    -- Content metadata
    content_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    attributes JSONB,
    
    -- Image storage
    image_path TEXT NOT NULL,
    image_checksum VARCHAR(64),
    
    -- Community features
    is_public BOOLEAN DEFAULT FALSE,
    download_count INTEGER DEFAULT 0,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    tags TEXT[],
    
    -- Attribution
    original_creator_id INTEGER REFERENCES luna.users(user_id),
    created_from_story_id VARCHAR(255),
    
    -- Metadata
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_user_asset UNIQUE (user_id, content_type, name)
);

CREATE INDEX idx_community_assets_type ON luna.community_assets(content_type);
CREATE INDEX idx_community_assets_public ON luna.community_assets(is_public) WHERE is_public = TRUE;
CREATE INDEX idx_community_assets_tags ON luna.community_assets USING GIN(tags);
CREATE INDEX idx_community_assets_popularity ON luna.community_assets(upvotes DESC, download_count DESC);

-- Asset usage tracking
CREATE TABLE luna.asset_imports (
    import_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES luna.users(user_id),
    asset_id INTEGER REFERENCES luna.community_assets(asset_id),
    imported_from VARCHAR(50), -- 'community_hub', 'file_upload', 'share_link'
    imported_at TIMESTAMPTZ DEFAULT NOW()
);

-- Asset collections (themed packs)
CREATE TABLE luna.asset_collections (
    collection_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES luna.users(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    cover_image_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE luna.collection_assets (
    collection_id INTEGER REFERENCES luna.asset_collections(collection_id),
    asset_id INTEGER REFERENCES luna.community_assets(asset_id),
    sort_order INTEGER,
    PRIMARY KEY (collection_id, asset_id)
);
```

### Community Hub Features

**1. Asset Browser**
- Search by type (character, location, item, quest, etc.)
- Filter by tags, popularity, upload date
- Preview cards with metadata
- One-click import to personal library

**2. Asset Packs**
- Themed collections (e.g., "Cyberpunk Character Pack", "Fantasy Locations")
- Bulk download entire packs
- User-curated collections

**3. Attribution & Licensing**
- Track original creator
- CC-BY-SA-4.0 default license (attribution + share-alike)
- Derivative tracking (remixed assets)

**4. Quality Control**
- Community voting (upvote/downvote)
- Report inappropriate content
- Moderation queue

### Usage Example

```python
# User dreams up character and exports
character = await dreamer.dream_with_image(
    story_id='story_123',
    content_type='character',
    name='Neon Samurai',
    description='Cyberpunk warrior with glowing katana...',
    metadata={'faction': 'street_samurai', 'archetype': 'warrior'},
    confidence=0.92
)

# Embed full metadata in PNG
AssetCardMetadata.embed_metadata(
    image_path=character['image_path'],
    content_type='character',
    name='Neon Samurai',
    description='Cyberpunk warrior with glowing katana...',
    attributes={'faction': 'street_samurai', 'archetype': 'warrior'},
    generation_data={
        'prompt': 'portrait of cyberpunk samurai...',
        'model': 'sdxl_base_1.0',
        'steps': 35,
        'cfg': 7.0,
        'seed': 42069
    },
    user_id='user_12345'
)

# User downloads and shares PNG file
# Other user imports it:
imported_metadata = AssetCardMetadata.extract_metadata('neon_samurai.png')

if AssetCardMetadata.validate_asset_card(imported_metadata):
    # Add to community library
    await community_hub.import_asset(
        user_id='user_67890',
        image_path='neon_samurai.png',
        metadata=imported_metadata
    )
    
    print(f"✅ Imported {imported_metadata['name']} by {imported_metadata['created_by']}")
```

## Distributed Computing Network (LUNA-Compute Pool)

### Concept: Community GPU Sharing

A **peer-to-peer compute network** where users can:
- **Contribute**: Share idle GPU compute for credits
- **Consume**: Use network compute for image/LLM generation
- **Earn**: Build up credits by running background worker
- **Spend**: Use credits for faster generation or cloud LLM access

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LUNA-Compute Pool Coordinator              │
│                  (Central Matchmaking)                  │
├─────────────────────────────────────────────────────────┤
│  • Worker Registry (GPU specs, availability)            │
│  • Job Queue (image gen, LLM inference, summarization)  │
│  • Credit Ledger (earnings, spending, transactions)     │
│  • Performance Monitoring (uptime, speed, reliability)  │
│  • Security (sandboxing, content filtering)             │
└────────────┬────────────────────────────────┬───────────┘
             │                                │
    ┌────────┴────────┐              ┌────────┴─────────┐
    │  GPU Provider   │              │  GPU Consumer    │
    │  (Worker Node)  │              │  (Story Player)  │
    ├─────────────────┤              ├──────────────────┤
    │ RTX 4090 8GB    │◄────Job─────┤ Needs image gen  │
    │ Idle: 80%       │              │ Has: 0 credits   │
    │ Earns: Credits  │─────Result──►│ Pays: Credits    │
    └─────────────────┘              └──────────────────┘
```

### Worker Node Software

```python
# server/compute_pool/worker_node.py

class LUNAComputeWorker:
    """
    Background worker that contributes GPU compute to the pool.
    Runs on user's PC when GPU is idle.
    """
    
    def __init__(self, config):
        self.config = config
        self.coordinator_url = config['coordinator_url']
        self.worker_id = config['worker_id']
        self.gpu_info = self._detect_gpu()
        self.credits_earned = 0
        
    async def start(self):
        """Register with coordinator and start accepting jobs."""
        
        # Register worker
        await self._register_worker()
        
        # Start job processing loop
        while True:
            # Check if GPU is idle
            if self._is_gpu_idle():
                # Request job from coordinator
                job = await self._request_job()
                
                if job:
                    # Process job
                    result = await self._process_job(job)
                    
                    # Submit result and earn credits
                    credits = await self._submit_result(job['job_id'], result)
                    self.credits_earned += credits
                    
                    print(f"✅ Completed job {job['job_id']}, earned {credits} credits")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def _is_gpu_idle(self) -> bool:
        """Check if GPU usage is below threshold (e.g., <20%)."""
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu < 20  # Less than 20% usage
    
    async def _process_job(self, job: dict):
        """Process compute job (image gen, LLM inference, etc.)."""
        
        job_type = job['type']
        
        if job_type == 'image_generation':
            return await self._generate_image(job['params'])
        
        elif job_type == 'llm_inference':
            return await self._run_llm_inference(job['params'])
        
        elif job_type == 'turn_summarization':
            return await self._summarize_turn(job['params'])
        
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    async def _generate_image(self, params: dict):
        """Generate image using local ComfyUI/A1111."""
        # Use same image generation adapters as Dreamer
        # ... implementation ...
        return {
            'image_data': base64_encoded_image,
            'generation_time_ms': 4500
        }
```

### Credit System

```sql
-- User credit ledger
CREATE TABLE luna.compute_credits (
    credit_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES luna.users(user_id),
    
    -- Transaction
    transaction_type VARCHAR(50) NOT NULL, -- 'earned', 'spent', 'purchased', 'granted'
    amount DECIMAL(10,2) NOT NULL, -- Can be negative for spending
    balance_after DECIMAL(10,2) NOT NULL,
    
    -- Context
    job_id VARCHAR(100), -- If earned/spent via job
    description TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_compute_credits_user ON luna.compute_credits(user_id);
CREATE INDEX idx_compute_credits_type ON luna.compute_credits(transaction_type);

-- Worker performance stats
CREATE TABLE luna.compute_workers (
    worker_id VARCHAR(100) PRIMARY KEY,
    user_id INTEGER REFERENCES luna.users(user_id),
    
    -- GPU specs
    gpu_model VARCHAR(100),
    vram_gb INTEGER,
    compute_capability VARCHAR(20),
    
    -- Performance
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0,
    total_credits_earned DECIMAL(10,2) DEFAULT 0,
    avg_job_time_ms INTEGER,
    uptime_hours DECIMAL(10,2) DEFAULT 0,
    
    -- Status
    is_online BOOLEAN DEFAULT FALSE,
    last_heartbeat TIMESTAMPTZ,
    
    -- Reputation
    reliability_score DECIMAL(3,2) DEFAULT 1.0, -- 0-1, decreases with failures
    user_rating DECIMAL(3,2), -- Community ratings
    
    -- Metadata
    registered_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_compute_workers_online ON luna.compute_workers(is_online) WHERE is_online = TRUE;
CREATE INDEX idx_compute_workers_reliability ON luna.compute_workers(reliability_score DESC);
```

### Pricing Model

**Credit Costs:**
- **Image Generation** (SDXL, 35 steps): 1-2 credits (~$0.01)
- **LLM Inference** (Llama 70B, 500 tokens): 5-10 credits (~$0.05)
- **Turn Summarization** (Gemma 12B): 1 credit (~$0.01)

**Earning Rates:**
- **RTX 4090**: 1-2 credits per image gen (4-5 seconds)
- **RTX 3090**: 0.8-1.5 credits per image gen (5-7 seconds)
- **RTX 4070**: 0.5-1 credits per image gen (8-10 seconds)

**Bootstrap Strategy (Early Users):**
- **New users**: Start with 100 free credits
- **GPU contributors**: Earn 2x credits for first month
- **Referrals**: 50 credits per referred user who runs worker

### Job Distribution Algorithm

```python
# server/compute_pool/coordinator.py

class ComputePoolCoordinator:
    """
    Central coordinator for distributed compute network.
    Matches jobs with available workers.
    """
    
    async def assign_job(self, job: dict) -> dict:
        """
        Assign job to best available worker.
        
        Priority factors:
        1. GPU capability (VRAM, compute)
        2. Reliability score (history of success)
        3. Network latency (ping time)
        4. Current load (jobs in queue)
        5. User preference (pay more for priority)
        """
        
        # Get available workers
        workers = await self._get_available_workers(
            min_vram=job.get('min_vram', 8),
            job_type=job['type']
        )
        
        # Score workers
        scored_workers = []
        for worker in workers:
            score = (
                worker['reliability_score'] * 0.4 +
                (1 - worker['current_load'] / worker['max_jobs']) * 0.3 +
                (1 / max(worker['latency_ms'], 1)) * 0.2 +
                worker['compute_capability'] * 0.1
            )
            scored_workers.append((score, worker))
        
        # Select best worker
        best_worker = max(scored_workers, key=lambda x: x[0])[1]
        
        # Assign job
        await self._assign_job_to_worker(job['job_id'], best_worker['worker_id'])
        
        return best_worker
```

### Security & Content Safety

**Sandboxing:**
- Workers run jobs in isolated containers
- No access to local file system
- Rate limits on API calls

**Content Filtering:**
- Pre-screening of prompts (NSFW detection)
- Post-processing of generated images (safety classifier)
- Blacklist of prohibited content types

**Privacy:**
- Story context never shared with workers
- Only generation parameters sent (prompt, model, settings)
- Results encrypted in transit

### Future Enhancements

### 1. ControlNet Integration
- Pose consistency across character images
- Composition guidance for action scenes
- Depth maps from 3D scene reconstruction

### 2. LoRA Management
- Character-specific LoRAs for consistency
- Style LoRAs per story genre
- Automatic LoRA training from cached images
- **Community LoRA sharing** via asset cards

### 3. Image Quality Validation
- Automatic face detection and quality checks
- Reject and regenerate poor-quality images
- A/B testing with multiple variants
- **Community voting on best variants**

### 4. Real-time Generation
- Generate images during narration playback
- Progressive loading (low-res → high-res)
- Prefetch next turn images
- **Distributed pre-generation** across compute pool

### 5. Image Editing
- Manual touchups in UI
- Inpainting for corrections
- Style transfer between images
- **Community remix & derivative tracking**

### 6. Video Generation
- Animate transitions between turns
- Character expression animations
- Camera movements through scenes
- **Distributed video rendering** across pool

## Economic Model & Community Ecosystem

### The Vision: True Community Collaboration

LUNA-narrates creates a **self-sustaining creative economy** where:

**Content Creators** (all users):
- Generate unique characters, locations, items through gameplay
- Every Dreamer creation becomes shareable asset card
- Build reputation through quality contributions
- Earn credits by running background GPU worker

**GPU Contributors** (power users):
- Monetize idle compute time
- Help community members without powerful hardware
- Earn steady stream of credits
- Support network decentralization

**Players Without Hardware** (broader audience):
- Play LUNA-narrates on low-end devices
- Access distributed GPU compute pool
- Pay only for what they use (credits)
- Alternative to expensive cloud services

### Economic Benefits

**For Users:**
```
Traditional Cloud Path:
- Cloud LLM (Anthropic Sonnet): $3/million tokens = ~$0.15 per story turn
- Cloud Image Gen (Midjourney): $10/month = 200 images
- Monthly cost: $30-50 for regular play

LUNA-Compute Pool Path:
- Local/Distributed Image Gen: 1-2 credits per image = ~$0.01
- Distributed LLM: 5-10 credits per turn = ~$0.05
- Monthly cost: $5-10 for regular play
- **OR earn credits by contributing GPU → Play for free**
```

**For LUNA Project:**
```
Bootstrap Phase (No Capital):
- No infrastructure costs (P2P network)
- No cloud GPU bills
- Community provides compute
- Credits as virtual currency (not real money initially)
- Build user base without runway burn

Growth Phase (Post-Funding):
- Revenue share: Small % of credit transactions
- Premium features: Priority queue, cloud backup, advanced models
- Enterprise tier: Private compute pools for studios/teams
- LoRA marketplace: Sell custom style models
```

**For Community:**
```
Crowdsourced Content Library:
- Thousands of asset cards generated organically
- Curated collections by genre/theme
- Quality ratings and reviews
- Derivative works and remixes

Network Effects:
- More users → More compute available → Faster generation
- More creators → Better asset variety → Richer stories
- More GPU contributors → Lower costs → Broader access
```

### Real-World Comparison: Folding@Home Model

**Folding@Home** (scientific computing):
- 2.4 million users worldwide
- 1.5 exaFLOPS peak performance
- Volunteers contribute idle compute for research

**LUNA-Compute** (creative computing):
- Target: 10,000+ GPU contributors in Year 1
- Estimated: 50-100 petaFLOPS for image/LLM generation
- Users earn credits while contributing
- Circular economy: Earn → Spend → Create → Share

### Monetization Without Capital

**Phase 1: Virtual Credit System** (Months 0-6)
```
- Credits are virtual currency (not real money)
- Earned by: GPU contribution, content creation, moderation
- Spent on: Image gen, LLM inference, premium features
- Bootstrap network without payment processing
```

**Phase 2: Hybrid Model** (Months 6-12)
```
- Option to purchase credits with real money
- Option to cash out credits (revenue share model)
- Small transaction fee (5-10%) funds development
- Still free to play by earning credits
```

**Phase 3: Full Marketplace** (Year 2+)
```
- Asset card marketplace (buy/sell quality assets)
- LoRA marketplace (custom style models)
- Collection subscriptions (curated asset packs)
- Enterprise API access
- White-label licensing for game studios
```

### Community Governance

**Democratic Model:**
- Credit holders vote on features
- Content moderation by community
- Open-source core components
- Transparent roadmap and development

**Reputation System:**
- Asset creators: Downloads, ratings, usage stats
- GPU workers: Uptime, reliability, speed
- Moderators: Fair decisions, community trust
- Voters: Participation in governance

### Success Metrics

**Network Health:**
- Total GPU compute available (petaFLOPS)
- Average job wait time (<30 seconds)
- Worker/consumer ratio (target: 1:3)
- Credit circulation velocity

**Content Quality:**
- Assets in library (target: 100K+ by Year 1)
- Average asset rating (>4.0/5.0)
- Usage rate (% of assets used in stories)
- Derivative creation rate

**User Growth:**
- Monthly active users (MAU)
- GPU contributors (target: 5-10% of MAU)
- Asset creators (target: 30-40% of MAU)
- Credit-earning users (target: 50%+ play free)

### Risk Mitigation

**Content Safety:**
- Automated NSFW filtering
- Community moderation queue
- User reporting system
- Age-gated content categories

**Economic Stability:**
- Credit price floors/ceilings
- Anti-inflation mechanisms
- Reserve fund for rewards
- Graduated earn rates (prevent farming)

**Network Reliability:**
- Redundant job assignment (2-3 workers per job)
- Automatic failover on worker disconnect
- Quality checks on results
- Reputation-based worker selection

## Integration with Dreamer Workflow

### Dreamer Content Generation Flow

```python
# server/narration/dreamer.py (modified)

class Dreamer:
    """
    Enhanced Dreamer that generates both content AND images.
    """
    
    def __init__(self, llm_client, db_pool, gpu_workers, settings):
        self.llm = llm_client
        self.db = db_pool
        # NEW: Image generation module
        self.image_gen = DreamerImageGenerator(db_pool, gpu_workers, settings)
        
    async def start(self):
        """Start Dreamer and image generation worker."""
        await self.image_gen.start()
        asyncio.create_task(self.continuous_dreaming_loop())
        
    async def continuous_dreaming_loop(self):
        """
        Dreamer continuously generates content + images in background.
        """
        while True:
            # 1. Dream up new content (existing logic)
            dream_ideas = await self.generate_dream_ideas()
            
            # 2. For each idea, generate image immediately
            for idea in dream_ideas:
                await self.image_gen.dream_with_image(
                    story_id=self.current_story_id,
                    content_type=idea['type'],
                    name=idea['name'],
                    description=idea['description'],
                    metadata=idea['metadata'],
                    confidence=idea['confidence']
                )
            
            await asyncio.sleep(60)  # Dream every minute
    
    async def get_suggestions_for_preprocessor(self, context: Dict) -> List[Dict]:
        """
        Preprocessor requests suggestions from Dreamer.
        Returns content WITH images already generated.
        """
        # Get speculative content that fits current context
        relevant_content = await self.image_gen.get_speculative_assets(
            story_id=context['story_id'],
            content_type=None  # All types
        )
        
        # Filter by relevance to current context
        suggestions = self._filter_by_context(relevant_content, context)
        
        # Each suggestion includes:
        # - content_id
        # - name, description, metadata
        # - image_path (already generated!)
        # - confidence_score
        
        return suggestions
    
    async def on_content_used(self, content_id: int, turn_id: int, used_by: str):
        """
        Called when Preprocessor or Main LLM uses a Dreamer suggestion.
        Marks content and image as realized.
        """
        await self.image_gen.realize_content(
            content_id=content_id,
            turn_id=turn_id,
            realized_by=used_by
        )
        
        print(f"✨ Dreamer content #{content_id} realized in turn {turn_id}!")
```

### Preprocessor Integration

```python
# server/narration/preprocessor.py (modified)

class Preprocessor:
    """
    Preprocessor checks Dreamer speculative pool before making decisions.
    """
    
    def __init__(self, llm_client, dreamer, db_pool):
        self.llm = llm_client
        self.dreamer = dreamer
        self.db = db_pool
        
    async def analyze_turn(self, turn_data: Dict) -> Dict:
        """
        Analyze turn and optionally incorporate Dreamer content.
        """
        # Standard analysis
        analysis = await self.llm.analyze(turn_data)
        
        # Check if Dreamer has relevant content
        context = {
            'story_id': turn_data['story_id'],
            'current_location': turn_data['location'],
            'active_characters': turn_data['characters'],
            'plot_state': turn_data['plot_summary']
        }
        
        dreamer_suggestions = await self.dreamer.get_suggestions_for_preprocessor(context)
        
        # Decide if any suggestions fit
        for suggestion in dreamer_suggestions:
            if self._should_use_suggestion(suggestion, analysis):
                # Use it! Image already exists
                analysis['use_dreamer_content'] = {
                    'content_id': suggestion['content_id'],
                    'type': suggestion['content_type'],
                    'name': suggestion['name'],
                    'description': suggestion['description'],
                    'image_path': suggestion['image_path']  # Already generated!
                }
                
                # Mark as realized
                await self.dreamer.on_content_used(
                    content_id=suggestion['content_id'],
                    turn_id=turn_data['turn_id'],
                    used_by='preprocessor'
                )
                break
        
        return analysis
```

### Main LLM Integration

```python
# server/narration/main_llm.py (modified)

class MainLLM:
    """
    Main narration LLM can also use Dreamer content with pre-generated images.
    """
    
    async def generate_narration(
        self, 
        turn_data: Dict, 
        preprocessor_analysis: Dict,
        dreamer: Dreamer
    ) -> Dict:
        """
        Generate narration, optionally using Dreamer content.
        """
        # Check if preprocessor suggested Dreamer content
        if 'use_dreamer_content' in preprocessor_analysis:
            dreamer_content = preprocessor_analysis['use_dreamer_content']
            
            # Enhance narration with Dreamer idea
            prompt = f"""
            Incorporate this element into the story:
            {dreamer_content['description']}
            
            (Image already prepared and will be displayed)
            """
        else:
            # Check speculative pool directly
            suggestions = await dreamer.get_suggestions_for_preprocessor({
                'story_id': turn_data['story_id']
            })
            
            # Optionally use high-confidence suggestions
            # ...
        
        narration = await self.llm.generate(prompt)
        
        return {
            'narration': narration,
            'dreamer_content': dreamer_content if 'use_dreamer_content' in preprocessor_analysis else None
        }
```

## API Reference

### DreamerImageGenerator Methods

```python
# Start/stop
await dreamer_image_gen.start()
await dreamer_image_gen.stop()

# Dream with image generation
result = await dreamer_image_gen.dream_with_image(
    story_id: str,
    content_type: str,  # 'character', 'location', 'item', 'scene', 'plot_twist'
    name: str,
    description: str,
    metadata: Dict,
    confidence: float  # 0.0 - 1.0
) -> Dict

# Get speculative assets
assets = await dreamer_image_gen.get_speculative_assets(
    story_id: str,
    content_type: Optional[str] = None
) -> List[Dict]

# Realize content (mark as used)
await dreamer_image_gen.realize_content(
    content_id: int,
    turn_id: int,
    realized_by: str  # 'preprocessor', 'main_llm', 'direct'
)

# Cleanup
await dreamer_image_gen.cleanup_expired_assets()
```

### Orchestrator Methods

```python
# Initialize orchestrator
await orchestrator.initialize()

# Generate for turn (action images)
images = await orchestrator.generate_images_for_turn(
    turn_id: int,
    turn_data: Dict,
    force_regenerate: bool = False
) -> Dict[str, List[str]]

# Regenerate specific image
image = await orchestrator.regenerate_image(
    image_id: int,
    new_prompt: Optional[str] = None
) -> str

# Get worker status
workers = await orchestrator.get_worker_status() -> List[Dict]
```

### Database Queries

```sql
-- Get all speculative (unrealized) Dreamer content with images
SELECT 
    dc.content_id,
    dc.content_type,
    dc.name,
    dc.description,
    dc.confidence_score,
    dc.dreamed_at,
    gi.local_path as image_path,
    gi.width,
    gi.height
FROM luna.dreamer_content dc
LEFT JOIN luna.generated_images gi ON dc.primary_image_id = gi.image_id
WHERE dc.story_id = 'story_123'
  AND dc.is_realized = FALSE
ORDER BY dc.confidence_score DESC;

-- Get realized content usage stats
SELECT 
    dc.content_type,
    COUNT(*) as total_dreamed,
    SUM(CASE WHEN dc.is_realized THEN 1 ELSE 0 END) as realized_count,
    ROUND(100.0 * SUM(CASE WHEN dc.is_realized THEN 1 ELSE 0 END) / COUNT(*), 2) as realization_rate,
    AVG(EXTRACT(EPOCH FROM (dc.realized_at - dc.dreamed_at)) / 3600) as avg_hours_to_realize
FROM luna.dreamer_content dc
WHERE dc.story_id = 'story_123'
GROUP BY dc.content_type;

-- Get all images for a turn (including realized Dreamer content)
SELECT 
    gi.image_id,
    gi.image_type,
    gi.subject_key,
    gi.is_speculative,
    gi.realized_by,
    gi.local_path,
    dc.name as dreamer_content_name,
    dc.confidence_score
FROM luna.generated_images gi
LEFT JOIN luna.dreamer_content dc ON gi.dreamer_content_id = dc.content_id
WHERE gi.turn_id = 42
ORDER BY gi.image_type, gi.generated_at;

-- Worker performance stats
SELECT 
    worker_name,
    COUNT(*) as total_generated,
    AVG(generation_time_ms) as avg_time_ms,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
    SUM(CASE WHEN is_speculative THEN 1 ELSE 0 END) as speculative_count,
    SUM(CASE WHEN NOT is_speculative THEN 1 ELSE 0 END) as realized_count
FROM luna.generated_images
GROUP BY worker_name;

-- Storage usage by story (including speculative assets)
SELECT 
    dc.story_id,
    COUNT(DISTINCT gi.image_id) as total_images,
    SUM(CASE WHEN gi.is_speculative THEN 1 ELSE 0 END) as speculative_images,
    SUM(CASE WHEN NOT gi.is_speculative THEN 1 ELSE 0 END) as realized_images,
    SUM(gi.file_size) / 1024 / 1024 as total_mb,
    SUM(CASE WHEN gi.is_speculative THEN gi.file_size ELSE 0 END) / 1024 / 1024 as speculative_mb
FROM luna.generated_images gi
LEFT JOIN luna.dreamer_content dc ON gi.dreamer_content_id = dc.content_id
WHERE dc.story_id = 'story_123'
GROUP BY dc.story_id;

-- Find expired speculative content ready for cleanup
SELECT 
    dc.content_id,
    dc.content_type,
    dc.name,
    dc.dreamed_at,
    AGE(NOW(), dc.dreamed_at) as age,
    gi.local_path,
    gi.file_size / 1024 / 1024 as size_mb
FROM luna.dreamer_content dc
LEFT JOIN luna.generated_images gi ON dc.primary_image_id = gi.image_id
WHERE dc.is_realized = FALSE
  AND dc.dreamed_at < NOW() - INTERVAL '30 days'
ORDER BY dc.dreamed_at;
```

## Troubleshooting

### Worker Not Responding
- Check API URL and network connectivity
- Verify ComfyUI/A1111 is running on target PC
- Check worker heartbeat in `luna.image_workers` table
- Test worker directly: `curl http://192.168.1.100:8188` (ComfyUI) or `http://192.168.1.101:7860` (A1111)

### Slow Generation
- Reduce steps in settings (30 → 20)
- Use faster sampler (Euler a, DPM++ 2M)
- Ensure VRAM is sufficient (8GB+ for SDXL)
- Check if multiple workers are active and distributing load

### Poor Image Quality
- Increase steps (30 → 40)
- Adjust CFG scale (try 7-9 range)
- Improve prompt templates
- Use better base models (SDXL, Pony Diffusion, Juggernaut XL)
- Add negative prompts for common issues

### Dreamer Not Generating Images
- Check `dreamer.continuous_generation = true` in settings
- Verify confidence threshold isn't too high (default 0.6)
- Check Dreamer background worker is running: `await dreamer.image_gen.running`
- Look for errors in generation queue

### Speculative Pool Growing Too Large
- Reduce `speculative_pool_size` in settings (default 100)
- Lower `asset_expiry_days` to cleanup faster (default 30)
- Run cleanup manually: `await dreamer.cleanup_expired_assets()`
- Check realization rate (should be 20-40%)

### Images Not Appearing When Content Realized
- Verify `realize_content()` is called when Preprocessor/LLM uses idea
- Check `is_speculative` flag flips to FALSE
- Ensure `realized_by` and `turn_id` are set correctly
- Query database to verify image record is linked to turn

### Storage Issues
- Check disk space on image storage path
- Cleanup old speculative assets: `await dreamer.cleanup_expired_assets()`
- Compress old images: Convert PNG to JPEG for storage savings
- Archive completed story images to backup storage

## Phased Growth Strategy: From Bootstrap to Platform

### Overview: Scalable Evolution Path

A **financially sustainable roadmap** from zero-capital launch to community-powered platform:

```
Phase 0: Bootstrap       Phase 1: Cloud          Phase 2: Hybrid         Phase 3: Community Platform
(Months 0-3)            (Months 3-12)           (Months 12-24)          (Year 2+)
─────────────────       ─────────────────       ─────────────────       ───────────────────────────

Local Only              Cloud + Local           Self-Hosted + Pool      Decentralized
• Your GPU              • Rented compute        • H100 cluster          • P2P compute network
• User's GPU            • Pay-per-use           • Community GPUs        • Asset marketplace
• Free/Freemium         • Per-turn fee          • Credit system         • Custom model training
• Build product         • Validate market       • Scale operations      • Self-sustaining economy

Investment: $0          Investment: $500-2K/mo  Investment: $50-100K    Revenue: $10K-100K+/mo
Revenue: $0             Revenue: $1K-10K/mo     Revenue: $10K-50K/mo    Profit: 40-60% margins
Users: 10-100           Users: 100-1,000        Users: 1K-10K           Users: 10K-100K+
```

### Phase 0: Bootstrap (Months 0-3)
**Goal:** Prove product-market fit with zero capital

**Architecture:**
- Users run everything locally (ComfyUI/A1111 + Ollama/LM Studio)
- Your GPU for testing/demos
- PostgreSQL on user's machine or shared free tier
- No cloud costs

**Monetization:**
- Freemium model: Free basic, $5-10/mo premium
- Premium features: Cloud backup, advanced models, priority support
- Alternative: Donation/tip jar for early adopters

**Success Metrics:**
- 50-100 early adopters
- 10+ active daily users
- Positive feedback on core gameplay loop
- Build community on Discord

**Costs:** ~$0-50/mo (domain, email, Discord Nitro)

---

### Phase 1: Cloud Hybrid (Months 3-12)
**Goal:** Scale to paying users without massive infrastructure

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│              LUNA-Narrates Cloud                │
├─────────────────────────────────────────────────┤
│ User's Local:           Cloud Services:         │
│ • Browser UI            • PostgreSQL (Supabase) │
│ • Optional local GPU    • Redis (Upstash)       │
│                         • File storage (S3)     │
│ Generation:             • GPU compute:          │
│ • Local: Free           • Runpod/Vast.ai        │
│ • Cloud: Credits        • Pay-per-second        │
│                         • Auto-scale            │
└─────────────────────────────────────────────────┘
```

**Monetization - InfiniteWorlds Model:**
```
Per-Turn Pricing:
• Standard turn (no image): 5 credits = $0.05
• With image (1x): 10 credits = $0.10
• With images (3x): 15 credits = $0.15

Credit Packs:
• 100 credits: $9.99 (10% bonus)
• 500 credits: $45 (15% bonus)
• 1000 credits: $80 (20% bonus)

Monthly Subscription Alternative:
• Basic: $19.99/mo (250 credits)
• Pro: $39.99/mo (600 credits + priority)
• Unlimited: $79.99/mo (no credit limits)
```

**Cloud Costs (at scale):**
```
For 1,000 Users @ Avg 50 Turns/User/Month:

LLM Generation:
• 50K turns × $0.003/turn = $150/mo
• Provider: Gemini Flash, Groq Llama, or Claude Haiku

Image Generation:
• 25K images × $0.02/image = $500/mo
• Provider: Runpod (RTX 4090 @ $0.50/hr, 50% utilization)

Database & Storage:
• Supabase Pro: $25/mo
• S3 storage (100GB): $3/mo
• Redis: $10/mo

Total Costs: ~$700/mo
Total Revenue: $1,000 users × $10/mo avg = $10,000/mo
Gross Margin: ~93% ($9,300 profit)
```

**Growth Tactics:**
- Free tier: 50 credits/mo (10 turns)
- Referral program: 100 credits per referral
- Content creator program: Sponsored stories, asset packs
- Early adopter lifetime discounts

**Success Metrics:**
- 500-1,000 paying users
- $5K-10K MRR (Monthly Recurring Revenue)
- <10% churn rate
- 4.5+ star rating

---

### Phase 2: Self-Hosted Infrastructure (Months 12-24)
**Goal:** Own core infrastructure, reduce cloud dependency

**Investment Strategy:**
```
Hardware Purchase:
• 2-4x H100 80GB: $30K-60K
  (Used/refurb market, or Nvidia startup program)
• Or rent H100 cluster: $2-3/hr × 24/7 = $1.5-2K/mo
• Server rack, networking, cooling: $5-10K
• Colocate at data center: $500-1K/mo

Why H100s:
• 3-4x faster than RTX 4090 for LLM inference
• 80GB VRAM → Run 70B models at high throughput
• FP8 precision → 2x faster image generation
• ROI: Pay off in 12-18 months vs cloud costs
```

**Hybrid Architecture:**
```
┌────────────────────────────────────────────────────────┐
│              Self-Hosted Core + Community              │
├────────────────────────────────────────────────────────┤
│ Your H100 Cluster:        Community GPU Pool:         │
│ • Dreamer (pre-gen)       • User-contributed          │
│ • Preprocessor            • Per-job earnings          │
│ • Priority queue          • Distributed image gen     │
│ • Fine-tuned models       • Overflow capacity         │
│                                                        │
│ Cost Structure:                                        │
│ • Fixed: $2-3K/mo (hosting, power)                    │
│ • Variable: Community credits (self-funding)          │
└────────────────────────────────────────────────────────┘
```

**Monetization Evolution:**
```
Tier Structure:
• Free: 50 credits/mo + earn via GPU contribution
• Hobby: $14.99/mo (200 credits)
• Creator: $29.99/mo (500 credits + custom LoRAs)
• Pro: $59.99/mo (1,500 credits + priority + API access)
• Team: $199/mo (5,000 credits + multi-user + white-label)

New Revenue Streams:
• Custom LoRA training: $50-200 per model
• Asset pack marketplace: 70/30 revenue share
• API access: $0.001/token (for game studios)
• Enterprise hosting: $500-5K/mo
```

**Financial Projections:**
```
5,000 Users:
• 3,000 free (earn credits)
• 1,500 paid ($20/mo avg)
• 500 pro/enterprise ($60/mo avg)

Revenue: $30K + $30K = $60K/mo

Costs:
• Infrastructure: $3K/mo
• Community credits: $5K/mo (offset by contributions)
• Team salaries: $15K/mo (2 developers, 1 support)
• Marketing: $5K/mo
• Total: $28K/mo

Profit: $32K/mo (~53% margin)
Annual: $384K profit
```

**Development Focus:**
- Fine-tune SDXL on community assets
- Train LUNA-specific language model (Llama 70B base)
- Build custom LoRA training pipeline
- Automated asset quality curation

---

### Phase 3: Community Platform (Year 2+)
**Goal:** Decentralized, self-sustaining creative economy

**Full Platform Vision:**
```
┌─────────────────────────────────────────────────────────────┐
│                  LUNA Creative Platform                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Story Engine   │  │Asset Library │  │Model Training│   │
│  │  • Narration    │  │• 100K+ cards │  │• Fine-tuning │   │
│  │  • Dreamer      │  │• Collections │  │• LoRA studio │   │
│  │  • Multi-user   │  │• Marketplace │  │• Style blend │   │
│  └─────────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Distributed Compute Network               │   │
│  │  • 10,000+ GPU contributors                          │   │
│  │  • Auto-scaling based on demand                      │   │
│  │  • Reputation-based routing                          │   │
│  │  • 99.9% uptime SLA for pro users                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Data Flywheel                        │   │
│  │                                                      │   │
│  │  User Stories  →  Asset Generation  →  Community    │   │
│  │      ↑                                      ↓        │   │
│  │  Model Training  ←  Data Collection  ←  Sharing     │   │
│  │                                                      │   │
│  │  Better Models → Better Stories → More Users → More │   │
│  │  Data → Better Models (virtuous cycle)              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Monetization at Scale:**
```
Revenue Streams:
1. Subscriptions: $50-100K/mo (10K paid users)
2. Credit sales: $20-40K/mo (one-time purchases)
3. Marketplace: $10-30K/mo (70/30 split on $50K GMV)
4. Enterprise/API: $20-50K/mo (10-20 customers)
5. Model licensing: $10-20K/mo (studios, indie devs)
6. Advertising: $5-10K/mo (ethical, non-intrusive)

Total Revenue: $115-250K/mo
Annual: $1.4M - $3M

Costs:
• Infrastructure: $10K/mo (H100 cluster + bandwidth)
• Team: $80K/mo (6 engineers, 2 support, 1 community)
• Community credits: $15K/mo (revenue share)
• Marketing: $20K/mo
• Operations: $10K/mo
• Total: $135K/mo

Profit: $25-115K/mo (15-45% margin at scale)
Annual: $300K - $1.4M profit
```

**The Data Moat:**

This is where you build **true competitive advantage**:

```
Proprietary Dataset After 2 Years:
─────────────────────────────────
• 100K+ user stories (500M+ words of narrative)
• 10M+ AI-generated images with quality ratings
• 500K+ community asset cards (curated)
• 50K+ dreamer-generated characters/locations/items
• 1M+ turn-level interactions (player choices + outcomes)

Value:
• Train LUNA-specific models (better than GPT-4 at storytelling)
• Fine-tune image models (unique style, character consistency)
• Behavioral data (what makes engaging stories)
• Community preferences (genre trends, popular archetypes)

Competitive Moat:
• Impossible to replicate without years of user data
• Increasing returns: Better models → Better stories → More users
• Network effects: More assets → More creativity → More value
```

**Exit Opportunities:**
1. **Acquisition** by gaming company (Unity, Epic, Roblox): $20-50M
2. **Acquisition** by AI company (Anthropic, OpenAI, Stability): $50-100M+
3. **IPO** (if hitting $10M+ ARR): $100-500M valuation
4. **Stay Independent**: Build $5-10M ARR lifestyle business

---

### Phase 4: Platform Ecosystem (Year 3+)
**Goal:** LUNA becomes the creative infrastructure layer

**Expansion Vectors:**

**1. Game Studios Integration**
```
White-Label LUNA for Game Development:
• Procedural quest generation
• NPC dialogue systems
• Dynamic worldbuilding
• Asset generation pipeline

Pricing: $5K-50K/mo per studio + rev share
Target: 50-100 indie studios = $250K-2M/mo
```

**2. Educational Sector**
```
LUNA for Creative Writing Education:
• Interactive storytelling curriculum
• AI writing partner for students
• Automated feedback and grading
• Portfolio building

Pricing: $5/student/semester
Target: 10K students = $50K per semester
```

**3. Publishing Industry**
```
LUNA for Authors:
• Plot development assistant
• Character consistency checking
• World bible maintenance
• Cover art generation

Pricing: $49-99/mo per author
Target: 1,000 authors = $50-100K/mo
```

**4. Licensing & Partnerships**
```
Technology Licensing:
• Dreamer system to other AI companies
• Distributed compute protocol (open source → paid support)
• Asset card standard (become the PNG metadata format)

Revenue: $500K-2M/year in licensing deals
```

---

### Key Success Factors

**Technical Moat:**
✅ Proprietary dataset (user stories, generations, preferences)
✅ Fine-tuned models (LUNA-storytelling, LUNA-vision)
✅ Distributed compute protocol (open source → ecosystem)
✅ Asset card standard (interoperability → network effects)

**Business Moat:**
✅ Community network effects (more users → more value)
✅ Creator ecosystem (asset marketplace, LoRAs, collections)
✅ Switching costs (users' stories, libraries, custom models)
✅ Brand & reputation (quality, ethical AI, community-first)

**Growth Levers:**
✅ Viral loops (invite friends for credits)
✅ Content creator program (YouTube, Twitch integration)
✅ Education partnerships (schools, bootcamps)
✅ Open source community (developers, researchers)

**Risk Mitigation:**
✅ Diversified revenue (subscriptions, marketplace, API, licensing)
✅ Low customer concentration (many small users vs few whales)
✅ Defensible technology (data moat, custom models)
✅ Community ownership (avoid rug pull, sustainable growth)

---

### Investment Timeline (If Raising Capital)

**Pre-Seed: $500K-1M** (Months 6-12)
- **Valuation:** $4-8M pre-money
- **Use:** 2 engineers, H100 cluster, 12-month runway
- **Investors:** Angel investors, pre-seed funds, AI-focused VCs

**Seed: $3-5M** (Months 18-24)
- **Valuation:** $15-25M pre-money
- **Use:** 6-person team, expand infrastructure, marketing
- **Investors:** Tier-1 VCs (a16z, Sequoia, Benchmark)

**Series A: $10-20M** (Months 30-36)
- **Valuation:** $50-100M pre-money
- **Use:** 20-person team, international expansion, enterprise sales
- **Investors:** Growth-stage VCs

**Or Bootstrap to Profitability:**
- Phase 1: Self-fund from revenue
- Phase 2: Reinvest profits into infrastructure
- Phase 3: Consider outside capital for faster scaling
- **Advantage:** Maintain control, avoid dilution, sustainable growth

---

### The Ultimate Vision

**LUNA becomes the creative infrastructure layer for interactive storytelling:**

```
Just as:
• AWS powers the internet
• Unity powers game development
• Shopify powers e-commerce

LUNA powers:
• Interactive narrative experiences
• Procedural content generation
• Community-driven worldbuilding
• Ethical AI creativity
```

**Impact Metrics (5-Year Goal):**
- **100K+ creators** using LUNA for storytelling
- **1M+ stories** generated and shared
- **10M+ assets** in community library
- **$10M+ ARR** with sustainable margins
- **Open source** core components (Dreamer, asset format)
- **Industry standard** for AI-assisted narrative

This isn't just a product - it's a **movement** toward democratized creative AI, community ownership, and sustainable technology.

---

## See Also

- [LUNA-Narrates Architecture](./LUNA-NARRATES.md)
- [LUNA-Narrates Action Suggestion System](./LUNA-NARRATES_ACTION_SUGGESTION_SYSTEM.md)
- [LUNA-Narrates Future Enhancements](./LUNA-NARRATES_FUTURE_ENHANCEMENTS_VISION.md)
- [Database Schema](../database/README.md)
- [Business Plan](./BUSINESS-PLAN.md)
