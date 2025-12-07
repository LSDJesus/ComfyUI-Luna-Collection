# LUNA-Narrates Future Features Vision

**Document Version:** 1.0  
**Date:** November 18, 2025  
**Status:** Planning Phase - Core Implementation Required First

---

## Overview

This document outlines three major feature systems designed for LUNA-Narrates after core functionality is complete:

1. **Chapter-Based Guided Branching System** - Structured narrative progression with self-healing adaptation
2. **Multi-User Database Architecture** - Scalable user management with Row-Level Security
3. **Content Marketplace with Fork Management** - Community asset sharing with Copy-on-Write efficiency

**IMPORTANT:** These features require stable core systems (multi-agent pipeline, database pooling, orchestrator, WebUI integration) before implementation.

---

## 1. Chapter-Based Guided Branching System

### Inspiration & Goals

**Inspired by:** High-quality character cards (Eternal Limbo, Aidonis, premium lorebooks) that provide structured progression through predefined narrative beats while maintaining player agency.

**Core Problem Solved:** Players often don't know what to do next in open-ended stories. Chapters provide guidance without railroading.

**Design Philosophy:**
- **Guided, not forced**: Chapters suggest direction but don't block alternative paths
- **Self-healing**: Adapts when player actions diverge from expected beats
- **Author-friendly**: Easy to create and import from existing lorebooks
- **Transparent**: Players see available chapters and progression status

### Architecture: Preprocessor-Based Detection

**Decision Rationale:** Chapter management belongs in the Preprocessor because:
- Preprocessor is the **context curator** - already evaluates story state
- Keeps Writer focused on prose generation (separation of concerns)
- Strategist receives clean chapter context in briefing
- Dreamer can react to chapter events for thematic enrichment

**Data Flow:**
```
User Action
    ↓
Preprocessor
    ├─→ Query available chapters for story
    ├─→ Evaluate trigger conditions (location, items, characters present)
    ├─→ Check chapter prerequisites (previous chapters completed)
    ├─→ Calculate divergence score if chapter active (0.0-1.0)
    ├─→ Adapt content if divergence detected
    └─→ Build StrategistBriefing with chapter context
    ↓
Strategist (receives clean chapter instructions in briefing)
    └─→ Creates ResponseTemplate incorporating chapter beats
    ↓
Writer (executes template, unaware of chapter mechanics)
    └─→ Generates prose
    ↓
Dreamer (enriches based on chapter themes)
    └─→ Updates chapter progress, generates thematic assets
```

### Database Schema (Phase 1)

```sql
-- Migration: Add chapter system tables

-- Main chapter definitions (can be story-specific or from marketplace)
CREATE TABLE luna.story_chapters (
    chapter_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID REFERENCES luna.stories(story_id) ON DELETE CASCADE,
    source_type VARCHAR(20) NOT NULL DEFAULT 'manual', -- 'manual', 'imported', 'marketplace'
    source_chapter_id UUID, -- Reference to community_chapters if from marketplace
    
    -- Chapter metadata
    chapter_number INTEGER NOT NULL,
    chapter_title VARCHAR(200) NOT NULL,
    chapter_description TEXT,
    chapter_summary TEXT, -- Author's intended narrative
    
    -- Prerequisites
    required_previous_chapters INTEGER[], -- Chapter numbers that must be complete
    
    -- Activation conditions (JSON for flexibility)
    trigger_conditions JSONB NOT NULL, -- {location: "castle", has_item: "sword", character_present: "king"}
    
    -- Content
    narrative_instructions TEXT NOT NULL, -- Instructions for Strategist
    suggested_beats JSONB, -- [{beat: "Confront the king", priority: "high"}, ...]
    
    -- Self-healing configuration
    allow_divergence BOOLEAN DEFAULT true,
    min_similarity_threshold FLOAT DEFAULT 0.4, -- Below this = create alternate path
    
    -- State
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(story_id, chapter_number)
);

-- Track chapter state per story
CREATE TABLE luna.chapter_progress (
    progress_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID REFERENCES luna.stories(story_id) ON DELETE CASCADE,
    chapter_id UUID REFERENCES luna.story_chapters(chapter_id) ON DELETE CASCADE,
    
    -- Activation tracking
    activated_at TIMESTAMPTZ,
    activated_turn_number INTEGER,
    
    -- Completion tracking
    completed_at TIMESTAMPTZ,
    completed_turn_number INTEGER,
    completion_method VARCHAR(50), -- 'natural', 'adapted', 'diverged_alternate'
    
    -- Divergence metrics
    divergence_score FLOAT, -- 0.0 = perfect match, 1.0 = complete divergence
    adaptation_count INTEGER DEFAULT 0, -- How many times content was adapted
    
    UNIQUE(story_id, chapter_id)
);

-- Track adaptation history for debugging/analytics
CREATE TABLE luna.chapter_divergence_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID REFERENCES luna.stories(story_id) ON DELETE CASCADE,
    chapter_id UUID REFERENCES luna.story_chapters(chapter_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    
    divergence_score FLOAT NOT NULL,
    divergence_reason TEXT, -- Why story diverged
    adaptation_strategy VARCHAR(50), -- 'relax_conditions', 'adapt_content', 'alternate_path'
    adapted_content JSONB, -- What changed
    
    logged_at TIMESTAMPTZ DEFAULT NOW()
);

-- Store adapted chapter versions (self-healing generates these)
CREATE TABLE luna.chapter_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chapter_id UUID REFERENCES luna.story_chapters(chapter_id) ON DELETE CASCADE,
    story_id UUID REFERENCES luna.stories(story_id) ON DELETE CASCADE,
    
    version_number INTEGER NOT NULL,
    version_description TEXT, -- "Adapted for player chose diplomacy over combat"
    
    -- Modified content
    adapted_trigger_conditions JSONB,
    adapted_narrative_instructions TEXT,
    adapted_suggested_beats JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(50), -- 'dreamer_agent', 'manual', 'import'
    
    UNIQUE(chapter_id, story_id, version_number)
);

-- Index for fast chapter queries
CREATE INDEX idx_chapters_story_active ON luna.story_chapters(story_id, is_active);
CREATE INDEX idx_chapter_progress_story ON luna.chapter_progress(story_id, chapter_id);
CREATE INDEX idx_divergence_log_chapter ON luna.chapter_divergence_log(chapter_id, turn_number);
```

### Self-Healing Adaptation System

**Problem:** User imports a chapter from a premium lorebook that expects:
- Location: "Ancient Throne Room"  
- Character: "King Aldric"  
- Item: "Royal Decree"

But the player took the story in a different direction:
- Location: "Forest Camp"  
- Character: "King Aldric" ✓ (present)  
- Item: "Stolen Crown" (different item)

**Solution: Three-Tier Adaptation Strategy**

#### Tier 1: Relax Trigger Conditions (Divergence 0.4-0.7)
*Minor deviations - adjust requirements without changing chapter intent*

```python
# Preprocessor logic
if chapter.is_active and divergence_score > 0.4:
    # Example: Location doesn't match but character is present
    relaxed_conditions = {
        "character_present": ["King Aldric"],  # Keep critical condition
        "location": None,  # Remove location requirement
        "has_item": ["Royal Decree", "Stolen Crown"]  # Add alternative items
    }
    
    await db_manager.store_chapter_version(
        chapter_id=chapter.chapter_id,
        adapted_conditions=relaxed_conditions,
        reason="Player in unexpected location but chapter-critical character present"
    )
```

**Result:** Chapter proceeds in forest camp instead of throne room. Narrative adjusts: "The king's presence lends gravitas even in this makeshift camp..."

#### Tier 2: Adapt Content (Divergence 0.7-1.0)
*Moderate divergence - rewrite chapter content to match story state*

```python
# Preprocessor detects divergence, signals Dreamer
if divergence_score > 0.7:
    await dreamer_agent.adapt_chapter_content(
        chapter_id=chapter.chapter_id,
        current_story_state=preprocessor_output.story_context,
        original_chapter=chapter.narrative_instructions
    )
```

**Dreamer generates adapted version:**
```
Original: "Confront King Aldric in his throne room. Demand answers about the Royal Decree."

Adapted: "Track down King Aldric's camp. Confront him about the stolen crown and his role in the conspiracy."
```

**Result:** Chapter intent preserved (confront king), but execution matches player's path.

#### Tier 3: Create Alternate Path (Divergence >1.0 or impossible conditions)
*High divergence - original chapter impossible, generate new branch*

```python
# Preprocessor determines chapter cannot proceed
if divergence_score > 1.0 or critical_condition_impossible():
    # Create alternate chapter that achieves similar narrative goal
    alternate_chapter = await dreamer_agent.generate_alternate_chapter(
        original_chapter=chapter,
        story_state=current_state,
        narrative_goal="Resolve king's betrayal subplot"
    )
    
    await db_manager.create_chapter_branch(
        original_chapter_id=chapter.chapter_id,
        alternate_chapter=alternate_chapter,
        divergence_reason="Player killed king before confrontation chapter"
    )
```

**Result:** New chapter created: "Investigate King's Private Chambers" (achieve closure through discovery, not confrontation).

### Divergence Scoring Algorithm

**Method:** Embedding similarity + condition matching

```python
async def calculate_divergence_score(
    chapter: StoryChapter,
    current_state: StoryContext
) -> float:
    """
    Calculate 0.0-1.0 divergence score.
    0.0 = perfect match (all conditions met)
    1.0 = complete divergence (no conditions met)
    """
    
    # 1. Check hard conditions (70% weight)
    condition_score = 0.0
    total_conditions = len(chapter.trigger_conditions)
    
    for condition, expected_value in chapter.trigger_conditions.items():
        if condition == "location":
            if current_state.location != expected_value:
                condition_score += 1.0 / total_conditions
        elif condition == "character_present":
            if expected_value not in current_state.present_characters:
                condition_score += 1.0 / total_conditions
        elif condition == "has_item":
            if expected_value not in current_state.inventory:
                condition_score += 1.0 / total_conditions
    
    hard_divergence = condition_score * 0.7
    
    # 2. Check narrative embedding similarity (30% weight)
    expected_embedding = chapter.expected_narrative_embedding  # Pre-computed
    current_embedding = await get_story_state_embedding(current_state)
    
    cosine_similarity = compute_cosine_similarity(
        expected_embedding, 
        current_embedding
    )
    semantic_divergence = (1.0 - cosine_similarity) * 0.3
    
    return hard_divergence + semantic_divergence
```

### Implementation Phases

**Phase 1: Database + Basic Detection** (1-2 weeks)
- ✅ Create migration with all tables
- ✅ Add chapter query methods to `db_manager.py`
- ✅ Implement Preprocessor chapter evaluation logic
- ✅ Update `StrategistBriefing` model to include chapter context
- ✅ Test with manually-created chapters

**Phase 2: Self-Healing Adaptation** (2-3 weeks)
- ✅ Implement divergence scoring algorithm
- ✅ Add adaptation logic to Preprocessor
- ✅ Extend Dreamer agent with chapter adaptation capabilities
- ✅ Create `chapter_versions` and `chapter_divergence_log` writers
- ✅ Test with stories that diverge from chapter expectations

**Phase 3: Import & Authoring Tools** (2-3 weeks)
- ✅ Build chapter importer for `‹chapter-X›` format (InfiniteWorlds)
- ✅ Create chapter editor API endpoints
- ✅ WebUI chapter management interface
- ✅ Chapter preview/testing system

**Phase 4: Dreamer Auto-Generation** (3-4 weeks)
- ✅ Train Dreamer to generate chapters from story progression
- ✅ Implement chapter suggestion system ("Story seems ready for chapter about X")
- ✅ Auto-chapter generation from completed story patterns

### Open Design Questions

1. **Trigger Format:** Keep `‹chapter-X›` for compatibility or migrate to pure JSON?
2. **Chapter Visibility:** Should players see inactive chapters (spoilers) or only active ones?
3. **Manual Override:** Allow players to force-activate chapters out of order?
4. **Chapter Editor:** API-first or build WebUI editor simultaneously?
5. **Auto-Generation Priority:** Start with manual authoring or jump to Dreamer generation?

---

## 2. Multi-User Database Architecture

### Requirements & Scale Planning

**Target:** Support 5,000 users, each with 10 stories (50,000 total stories)

**Storage Estimates (per user, 10 stories):**
- **Story metadata:** ~50 KB (10 stories × 5KB)
- **Turns (150 turns/story avg):** ~1.5 MB (10 stories × 150 turns × 1KB compressed)
- **Turn summaries:** ~300 KB (10 stories × 150 × 200 bytes)
- **Arc summaries:** ~50 KB (10 stories × 15 arcs × 333 bytes)
- **Images (5 per story):** ~2.5 MB (10 stories × 5 images × 50KB WebP)
- **Vector embeddings:** ~600 KB (10 stories × 150 turns × 400 bytes)
- **Total per user:** ~5 MB

**5,000 users:** ~25 GB raw data, **~5 TB realistic** (with backups, indexes, overhead)

**Cost Estimate (AWS RDS PostgreSQL):**
- db.t3.large (8GB RAM, 2 vCPUs): ~$150/month
- 5TB storage (gp3): ~$500/month
- Backups (3TB): ~$270/month
- Network transfer: ~$50/month
- **Total: ~$970/month** ($0.19/user/month)

### Recommended Approach: Row-Level Security (RLS)

**Why RLS?**
- ✅ Best for <10,000 users (performance stays good)
- ✅ Simple queries (no schema prefixes needed)
- ✅ Automatic security (PostgreSQL enforces policies)
- ✅ Easy to implement (one migration)
- ✅ Shared indexes benefit all users
- ✅ Centralized backups and maintenance

**Alternative Approaches (Not Recommended for Current Scale):**
- ❌ **Schema-per-user:** Overkill for 5,000 users, complex migrations, harder backups
- ❌ **Database-per-user:** Only for 10,000+ users, connection overhead, infrastructure complexity

### Database Schema Changes

```sql
-- Migration: Add multi-user support with RLS

-- User management table
CREATE TABLE luna.users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Quotas
    max_stories INTEGER DEFAULT 10,
    max_images_per_story INTEGER DEFAULT 100,
    storage_limit_mb INTEGER DEFAULT 100, -- 100MB per user
    
    -- Subscription tier
    subscription_tier VARCHAR(50) DEFAULT 'free', -- 'free', 'pro', 'premium'
    subscription_expires_at TIMESTAMPTZ,
    
    -- State
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

-- Add user_id to all user-owned tables
ALTER TABLE luna.stories ADD COLUMN user_id UUID REFERENCES luna.users(user_id) ON DELETE CASCADE;
ALTER TABLE luna.character_evolution_snapshots ADD COLUMN user_id UUID REFERENCES luna.users(user_id) ON DELETE CASCADE;

-- Create indexes for RLS performance
CREATE INDEX idx_stories_user ON luna.stories(user_id);
CREATE INDEX idx_characters_user ON luna.character_evolution_snapshots(user_id);

-- Enable Row-Level Security
ALTER TABLE luna.stories ENABLE ROW LEVEL SECURITY;
ALTER TABLE luna.turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE luna.turn_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE luna.story_arcs ENABLE ROW LEVEL SECURITY;
ALTER TABLE luna.turn_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE luna.character_evolution_snapshots ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Users can only access their own data
CREATE POLICY stories_user_isolation ON luna.stories
    USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY turns_user_isolation ON luna.turns
    USING (story_id IN (SELECT story_id FROM luna.stories WHERE user_id = current_setting('app.current_user_id')::uuid));

CREATE POLICY turn_summaries_user_isolation ON luna.turn_summaries
    USING (story_id IN (SELECT story_id FROM luna.stories WHERE user_id = current_setting('app.current_user_id')::uuid));

CREATE POLICY arcs_user_isolation ON luna.story_arcs
    USING (story_id IN (SELECT story_id FROM luna.stories WHERE user_id = current_setting('app.current_user_id')::uuid));

CREATE POLICY images_user_isolation ON luna.turn_images
    USING (story_id IN (SELECT story_id FROM luna.stories WHERE user_id = current_setting('app.current_user_id')::uuid));

CREATE POLICY characters_user_isolation ON luna.character_evolution_snapshots
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Admin bypass policy (for system operations)
CREATE POLICY admin_bypass ON luna.stories
    USING (current_setting('app.current_user_id', true) = 'admin');
```

### Application Integration

**FastAPI Middleware:**
```python
# core/middleware/auth_middleware.py

from fastapi import Request
from core.database.db_manager import DatabaseManager

async def set_user_context(request: Request, user_id: str, db_manager: DatabaseManager):
    """Set PostgreSQL session variable for RLS."""
    async with db_manager.pool.acquire() as conn:
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, false)",
            str(user_id)
        )
        request.state.user_id = user_id
        request.state.db_conn = conn
        return conn

# Usage in routes
@router.post("/stories/create")
async def create_story(
    story_data: StoryCreate,
    request: Request,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    # Middleware already set app.current_user_id
    # RLS automatically filters queries
    story = await db_manager.create_story(
        user_id=request.state.user_id,
        title=story_data.title,
        # ... RLS ensures this user can only see their stories
    )
```

### Storage Optimization Strategies

**Lazy Image Loading:**
```python
# Only load image URLs, not actual image data
SELECT image_id, local_path, image_url 
FROM luna.turn_images 
WHERE story_id = $1
-- Actual images loaded on-demand via CDN
```

**Turn Pruning (Optional):**
```python
# For long stories (>500 turns), keep:
# - Full text for last 100 turns
# - Summaries only for older turns
# - Embeddings for all (for semantic search)
```

**Tiered Storage:**
- **Hot data** (last 30 days): SSD (fast, expensive)
- **Warm data** (30-180 days): Standard storage
- **Cold data** (>180 days): Glacier (slow, cheap)

### Quota Enforcement

```python
# core/services/quota_manager.py

class QuotaManager:
    async def check_story_limit(self, user_id: str) -> bool:
        """Check if user can create another story."""
        user = await db_manager.get_user(user_id)
        story_count = await db_manager.count_user_stories(user_id)
        return story_count < user.max_stories
    
    async def check_storage_limit(self, user_id: str, additional_mb: float) -> bool:
        """Check if user has storage quota available."""
        user = await db_manager.get_user(user_id)
        current_usage = await db_manager.calculate_user_storage(user_id)
        return (current_usage + additional_mb) <= user.storage_limit_mb
```

### Migration Timeline

**Phase 1: User Management** (1 week)
- ✅ Create `luna.users` table
- ✅ Add user registration/login endpoints
- ✅ Implement JWT authentication
- ✅ Create user dashboard API

**Phase 2: RLS Implementation** (1 week)
- ✅ Add `user_id` columns to all tables
- ✅ Enable RLS and create policies
- ✅ Update middleware to set session variables
- ✅ Test data isolation

**Phase 3: Quota System** (1 week)
- ✅ Implement `QuotaManager` service
- ✅ Add quota checks to story/image creation
- ✅ Build quota monitoring dashboard
- ✅ Create upgrade prompts for quota limits

**Phase 4: Storage Optimization** (2 weeks)
- ✅ Implement lazy image loading
- ✅ Set up CDN for image delivery
- ✅ Build turn pruning system (optional)
- ✅ Configure tiered storage policies

---

## 3. Content Marketplace with Fork Management

### Vision: Community Asset Sharing

**Core Concept:** Users can share stories, chapters, characters, and prompts as community assets. Other users can "fork" these assets (copy to their account) and modify them without affecting the original.

**Key Features:**
- **Browse marketplace:** Discover popular stories, chapters, character templates
- **Fork efficiency:** Copy-on-Write system (only store differences, not full copies)
- **Source updates:** Pull updates from original when author improves content
- **Remix culture:** Users can publish their forks back to marketplace
- **Attribution:** Automatic tracking of asset lineage

### Architecture: Copy-on-Write with Delta Tracking

**Storage Comparison (1,000 users fork same story):**

| Approach | Storage per Fork | Total (1000 forks) |
|----------|------------------|-------------------|
| Full Copy | 75 KB | 75 MB |
| Copy-on-Write (0% modified) | 0 KB (ref only) | 0 KB |
| Copy-on-Write (10% modified) | 7.5 KB (deltas) | 7.5 MB |
| Copy-on-Write (100% modified) | 75 KB (full copy) | 75 MB |

**Savings:** 88% storage reduction for typical use (10% modification rate)

### Database Schema

```sql
-- Shared public schema for community content
CREATE SCHEMA IF NOT EXISTS public;

-- Community stories (source of truth)
CREATE TABLE public.community_stories (
    community_story_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_user_id UUID REFERENCES luna.users(user_id) ON DELETE SET NULL,
    
    -- Content metadata
    title VARCHAR(255) NOT NULL,
    description TEXT,
    genre VARCHAR(100),
    tags TEXT[], -- ['fantasy', 'romance', 'character-driven']
    
    -- Original story reference
    source_story_id UUID REFERENCES luna.stories(story_id) ON DELETE SET NULL,
    
    -- Marketplace stats
    fork_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    
    -- Versioning
    version_number INTEGER DEFAULT 1,
    changelog TEXT,
    
    -- State
    is_public BOOLEAN DEFAULT true,
    is_featured BOOLEAN DEFAULT false,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Community chapters (reusable across stories)
CREATE TABLE public.community_chapters (
    community_chapter_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_user_id UUID REFERENCES luna.users(user_id) ON DELETE SET NULL,
    
    -- Chapter content
    title VARCHAR(255) NOT NULL,
    description TEXT,
    trigger_conditions JSONB,
    narrative_instructions TEXT,
    suggested_beats JSONB,
    
    -- Marketplace
    fork_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    
    published_at TIMESTAMPTZ DEFAULT NOW()
);

-- User's forked stories (in user's own schema or luna schema with user_id)
CREATE TABLE luna.story_forks (
    fork_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES luna.users(user_id) ON DELETE CASCADE,
    story_id UUID REFERENCES luna.stories(story_id) ON DELETE CASCADE, -- User's story
    
    -- Source tracking
    source_community_story_id UUID REFERENCES public.community_stories(community_story_id),
    source_version_number INTEGER, -- Which version was forked
    
    -- Copy-on-Write tracking
    fork_strategy VARCHAR(50) NOT NULL, -- 'reference', 'delta', 'full_copy'
    modification_percentage FLOAT DEFAULT 0.0, -- 0.0-100.0
    
    -- Delta storage (only changed fields)
    overridden_turns JSONB, -- {turn_5: {turn_text: "..."}, turn_12: {...}}
    overridden_metadata JSONB, -- {title: "My Version", genre: "horror"}
    
    -- Update syncing
    last_synced_version INTEGER, -- Last version pulled from source
    has_unsynced_updates BOOLEAN DEFAULT false,
    
    forked_at TIMESTAMPTZ DEFAULT NOW(),
    last_modified_at TIMESTAMPTZ DEFAULT NOW()
);

-- Track which community assets user has liked/downloaded
CREATE TABLE luna.user_marketplace_activity (
    activity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES luna.users(user_id) ON DELETE CASCADE,
    
    community_story_id UUID REFERENCES public.community_stories(community_story_id),
    community_chapter_id UUID REFERENCES public.community_chapters(community_chapter_id),
    
    activity_type VARCHAR(50) NOT NULL, -- 'like', 'fork', 'download'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, community_story_id, activity_type),
    UNIQUE(user_id, community_chapter_id, activity_type)
);

-- Indexes
CREATE INDEX idx_community_stories_author ON public.community_stories(author_user_id);
CREATE INDEX idx_community_stories_tags ON public.community_stories USING GIN(tags);
CREATE INDEX idx_story_forks_user ON luna.story_forks(user_id);
CREATE INDEX idx_story_forks_source ON luna.story_forks(source_community_story_id);
```

### Copy-on-Write Implementation

**Step 1: User forks a community story**
```python
# core/services/marketplace_service.py

async def fork_story(
    user_id: str,
    community_story_id: str,
    db_manager: DatabaseManager
) -> str:
    """
    Fork a community story using Copy-on-Write.
    Initially stores only reference, no actual data copied.
    """
    
    # Get community story metadata
    community_story = await db_manager.get_community_story(community_story_id)
    
    # Create user's story record (minimal data)
    user_story_id = await db_manager.create_story(
        user_id=user_id,
        title=f"{community_story.title} (My Fork)",
        # ... other metadata
    )
    
    # Create fork tracking record (Copy-on-Write starts here)
    fork_id = await db_manager.create_fork_record(
        user_id=user_id,
        story_id=user_story_id,
        source_community_story_id=community_story_id,
        source_version_number=community_story.version_number,
        fork_strategy='reference',  # No data copied yet
        modification_percentage=0.0
    )
    
    # Increment fork count in marketplace
    await db_manager.increment_fork_count(community_story_id)
    
    return user_story_id
```

**Step 2: User reads forked story (lazy materialization)**
```python
async def get_story_turns(user_id: str, story_id: str) -> List[Turn]:
    """
    Fetch turns for a forked story.
    Merges source data + user overrides transparently.
    """
    
    # Check if this is a fork
    fork_record = await db_manager.get_fork_record(story_id)
    
    if not fork_record:
        # Regular story, return directly
        return await db_manager.get_turns(story_id)
    
    # Forked story: Merge source + overrides
    if fork_record.fork_strategy == 'reference':
        # Pure reference: fetch from community story
        source_turns = await db_manager.get_community_story_turns(
            fork_record.source_community_story_id
        )
        
        # Apply any overrides
        if fork_record.overridden_turns:
            for turn_num, override_data in fork_record.overridden_turns.items():
                source_turns[turn_num].update(override_data)
        
        return source_turns
    
    elif fork_record.fork_strategy == 'delta':
        # Fetch source + merge deltas
        source_turns = await db_manager.get_community_story_turns(
            fork_record.source_community_story_id
        )
        user_deltas = await db_manager.get_fork_deltas(fork_record.fork_id)
        
        # Merge deltas into source
        merged_turns = merge_deltas(source_turns, user_deltas)
        return merged_turns
    
    else:  # 'full_copy'
        # User has heavily modified, stored as separate data
        return await db_manager.get_turns(story_id)
```

**Step 3: User modifies forked story (Copy-on-Write trigger)**
```python
async def update_turn(
    user_id: str,
    story_id: str,
    turn_number: int,
    new_text: str
) -> None:
    """
    User edits a turn in their fork.
    Triggers Copy-on-Write to store override.
    """
    
    fork_record = await db_manager.get_fork_record(story_id)
    
    if not fork_record:
        # Regular story, update normally
        await db_manager.update_turn(story_id, turn_number, new_text)
        return
    
    # Forked story: Store override in delta tracking
    if fork_record.fork_strategy == 'reference':
        # First modification: create override record
        await db_manager.store_fork_override(
            fork_id=fork_record.fork_id,
            turn_number=turn_number,
            override_data={'turn_text': new_text}
        )
        
        # Update modification percentage
        total_turns = await db_manager.count_story_turns(
            fork_record.source_community_story_id
        )
        modified_turns = len(fork_record.overridden_turns or {}) + 1
        modification_pct = (modified_turns / total_turns) * 100
        
        await db_manager.update_fork_record(
            fork_id=fork_record.fork_id,
            modification_percentage=modification_pct
        )
        
        # Auto-upgrade fork strategy if heavily modified
        if modification_pct > 60:
            await upgrade_fork_to_full_copy(fork_record.fork_id)
```

### Fork Strategy Decision Tree

```python
def determine_fork_strategy(modification_percentage: float) -> str:
    """
    Automatically choose optimal storage strategy.
    """
    if modification_percentage < 0.1:
        return 'reference'  # <0.1% modified: pure reference (0% storage)
    elif modification_percentage < 60:
        return 'delta'  # 0.1-60% modified: store deltas (5-15% storage)
    else:
        return 'full_copy'  # >60% modified: store full copy (100% storage)
```

### Source Update Syncing

**Problem:** Community story author publishes v2 with improvements. How do forks get updates?

**Solution: Pull-based sync with conflict resolution**

```python
async def sync_fork_with_source(fork_id: str) -> SyncResult:
    """
    Pull latest version from source, merge with user modifications.
    """
    
    fork_record = await db_manager.get_fork_record(fork_id)
    community_story = await db_manager.get_community_story(
        fork_record.source_community_story_id
    )
    
    # Check if new version available
    if community_story.version_number <= fork_record.last_synced_version:
        return SyncResult(status='up_to_date')
    
    # Fetch changes from source
    source_changes = await db_manager.get_story_changes(
        community_story_id=fork_record.source_community_story_id,
        from_version=fork_record.last_synced_version,
        to_version=community_story.version_number
    )
    
    # Merge with user's modifications
    conflicts = []
    for change in source_changes:
        if change.turn_number in fork_record.overridden_turns:
            # User modified same turn: conflict!
            conflicts.append({
                'turn_number': change.turn_number,
                'source_version': change.new_text,
                'user_version': fork_record.overridden_turns[change.turn_number]
            })
        else:
            # No conflict: apply source change
            await db_manager.apply_source_change(fork_id, change)
    
    # Update sync metadata
    await db_manager.update_fork_record(
        fork_id=fork_id,
        last_synced_version=community_story.version_number,
        has_unsynced_updates=False
    )
    
    return SyncResult(
        status='synced',
        conflicts=conflicts,
        applied_changes=len(source_changes) - len(conflicts)
    )
```

**User Experience:**
```
Notification: "Update available for forked story: 'Epic Quest v2'"
User clicks: Show changes
UI shows:
  ✅ 12 changes applied automatically
  ⚠️  3 conflicts detected (you modified these turns)
  
  Conflict 1: Turn 45
    [Source version] [Your version] [Keep both]
```

### Marketplace API Endpoints

```python
# core/routes/marketplace.py

@router.get("/marketplace/stories")
async def browse_stories(
    genre: Optional[str] = None,
    tags: Optional[List[str]] = None,
    sort_by: str = 'popular',  # 'popular', 'recent', 'trending'
    page: int = 1,
    limit: int = 20
) -> List[CommunityStory]:
    """Browse community stories."""
    pass

@router.post("/marketplace/stories/{story_id}/fork")
async def fork_story(
    story_id: str,
    user_id: str = Depends(get_current_user)
) -> ForkResult:
    """Fork a community story to user's account."""
    pass

@router.post("/marketplace/stories/{story_id}/publish")
async def publish_story(
    story_id: str,
    metadata: PublishMetadata,
    user_id: str = Depends(get_current_user)
) -> CommunityStory:
    """Publish user's story to marketplace."""
    pass

@router.post("/forks/{fork_id}/sync")
async def sync_fork(
    fork_id: str,
    user_id: str = Depends(get_current_user)
) -> SyncResult:
    """Pull updates from source story."""
    pass

@router.get("/forks/{fork_id}/conflicts")
async def get_sync_conflicts(
    fork_id: str,
    user_id: str = Depends(get_current_user)
) -> List[SyncConflict]:
    """Get unresolved merge conflicts."""
    pass
```

### Implementation Phases

**Phase 1: Basic Marketplace** (2-3 weeks)
- ✅ Create public schema tables
- ✅ Build publish story endpoint (full copy, no CoW yet)
- ✅ Build browse/search endpoints
- ✅ Implement basic fork (full copy)
- ✅ WebUI marketplace browser

**Phase 2: Copy-on-Write** (2-3 weeks)
- ✅ Implement fork tracking with delta storage
- ✅ Build lazy materialization queries
- ✅ Add automatic strategy upgrades
- ✅ Test storage savings

**Phase 3: Source Sync** (2-3 weeks)
- ✅ Implement version tracking for community stories
- ✅ Build sync service with conflict detection
- ✅ Create conflict resolution UI
- ✅ Add update notifications

**Phase 4: Chapter Marketplace** (1-2 weeks)
- ✅ Extend marketplace to chapters
- ✅ Build chapter browser
- ✅ Implement chapter fork (simpler than stories)
- ✅ Chapter import/export tools

---

## Implementation Priority & Dependencies

### Prerequisites (Must Complete First)

**Before ANY of these features:**
1. ✅ Stable multi-agent pipeline (Preprocessor → Strategist → Writer → Dreamer)
2. ✅ Robust `DatabaseManager` connection pooling
3. ✅ Working `Orchestrator` with cost tracking
4. ✅ Basic WebUI integration for story creation/viewing
5. ✅ Error handling and logging infrastructure

**Chapter System Requires:**
- Stable Preprocessor agent
- Working story context queries
- Dreamer agent (for adaptation)

**Multi-User Requires:**
- Authentication system
- User registration/login
- JWT token management

**Marketplace Requires:**
- Multi-user system (users needed to share)
- Stable storage infrastructure
- CDN or image hosting solution

### Suggested Implementation Order

**Year 1 (Post-Core):**
1. **Multi-User Architecture** (Month 1-2) - Foundation for everything else
2. **Chapter System Phase 1** (Month 3-4) - High-value feature, author-friendly
3. **Chapter System Phase 2** (Month 5-6) - Self-healing adaptation
4. **Marketplace Phase 1** (Month 7-8) - Basic sharing, builds community

**Year 2 (Advanced Features):**
5. **Marketplace Phase 2** (Month 9-10) - Copy-on-Write efficiency
6. **Chapter System Phase 3** (Month 11-12) - Import tools, WebUI editor
7. **Marketplace Phase 3** (Month 13-14) - Source syncing, conflicts
8. **Chapter System Phase 4** (Month 15-16) - Dreamer auto-generation

---

## Key Design Principles

### 1. Progressive Enhancement
Start simple, add complexity as needed. Users shouldn't need to understand Copy-on-Write to fork a story.

### 2. Fail-Safe Defaults
- Chapters default to `allow_divergence=true` (don't break stories)
- Forks default to `reference` strategy (cheapest)
- RLS blocks by default (secure by default)

### 3. Transparent Costs
Show users storage usage, quota limits, marketplace attribution. No hidden surprises.

### 4. Preserve User Agency
- Chapters guide but don't force
- Forks can diverge completely
- Users control sync decisions

### 5. Community-First
Design for remixing, attribution, and collaboration. Make sharing easy and rewarding.

---

## Open Questions & Future Exploration

### Chapter System
1. Should chapters have "difficulty" ratings? (easy/medium/hard trigger conditions)
2. How to handle multi-path chapters? (Chapter 5A vs 5B based on previous choices)
3. Chapter analytics: Which chapters are most popular? Most abandoned?
4. Integration with Dreamer: Should it suggest chapters mid-story?

### Multi-User
1. Team stories? (multiple users collaborating on one story)
2. Story sharing without marketplace? (private links)
3. Data export tools? (GDPR compliance)
4. Admin moderation tools for marketplace?

### Marketplace
1. Monetization? (premium stories, tip authors)
2. Content moderation? (NSFW tags, reporting system)
3. License system? (CC-BY, CC-BY-NC, proprietary)
4. Remix chains? (show full lineage of forks)

---

## Appendix: Code Examples

### Example: Preprocessor Chapter Detection

```python
# core/agents/preprocessor.py

async def _evaluate_chapter_triggers(
    self,
    story_id: str,
    current_context: StoryContext
) -> Optional[ChapterActivation]:
    """
    Check if any chapters should activate based on current story state.
    """
    
    # Get available chapters for this story
    available_chapters = await self.db_manager.get_available_chapters(
        story_id=story_id,
        exclude_completed=True
    )
    
    for chapter in available_chapters:
        # Check prerequisites
        if chapter.required_previous_chapters:
            completed = await self.db_manager.get_completed_chapter_numbers(story_id)
            if not all(req in completed for req in chapter.required_previous_chapters):
                continue  # Prerequisites not met
        
        # Evaluate trigger conditions
        conditions_met = self._check_trigger_conditions(
            chapter.trigger_conditions,
            current_context
        )
        
        if conditions_met:
            # Activate chapter!
            await self.db_manager.activate_chapter(
                story_id=story_id,
                chapter_id=chapter.chapter_id,
                turn_number=current_context.turn_number
            )
            
            return ChapterActivation(
                chapter_id=chapter.chapter_id,
                chapter_title=chapter.chapter_title,
                narrative_instructions=chapter.narrative_instructions,
                suggested_beats=chapter.suggested_beats
            )
    
    # Check active chapters for divergence
    active_chapters = await self.db_manager.get_active_chapters(story_id)
    for chapter in active_chapters:
        divergence_score = await self.calculate_divergence_score(
            chapter, current_context
        )
        
        if divergence_score > chapter.min_similarity_threshold:
            # Story diverged from chapter expectations
            await self._handle_chapter_divergence(
                chapter, divergence_score, current_context
            )
    
    return None
```

### Example: Marketplace Browse Query

```sql
-- Efficient marketplace browse with RLS
-- (RLS ensures users only see public stories)

SELECT 
    cs.community_story_id,
    cs.title,
    cs.description,
    cs.genre,
    cs.tags,
    cs.fork_count,
    cs.like_count,
    cs.download_count,
    cs.published_at,
    u.username AS author_username,
    
    -- Check if current user has liked/forked
    EXISTS(
        SELECT 1 FROM luna.user_marketplace_activity uma
        WHERE uma.user_id = current_setting('app.current_user_id')::uuid
        AND uma.community_story_id = cs.community_story_id
        AND uma.activity_type = 'like'
    ) AS user_has_liked,
    
    EXISTS(
        SELECT 1 FROM luna.story_forks sf
        WHERE sf.user_id = current_setting('app.current_user_id')::uuid
        AND sf.source_community_story_id = cs.community_story_id
    ) AS user_has_forked

FROM public.community_stories cs
JOIN luna.users u ON cs.author_user_id = u.user_id

WHERE cs.is_public = true
    AND ($1::text IS NULL OR cs.genre = $1)  -- Genre filter
    AND ($2::text[] IS NULL OR cs.tags && $2)  -- Tags filter (array overlap)

ORDER BY 
    CASE WHEN $3 = 'popular' THEN cs.fork_count + cs.like_count END DESC,
    CASE WHEN $3 = 'recent' THEN cs.published_at END DESC,
    CASE WHEN $3 = 'trending' THEN cs.download_count END DESC

LIMIT $4 OFFSET $5;
```

---

## Document Maintenance

**Owner:** Development Team  
**Review Frequency:** Quarterly or after major architecture changes  
**Related Documents:**
- `LUNA-NARRATES_MULTI_AGENT_ARCHITECTURE.md` - Current system design
- `LUNA-NARRATES_STANDALONE_ARCHITECTURE.md` - Deployment architecture
- `TODO.md` - Current sprint priorities

**Change Log:**
- v1.0 (2025-11-18): Initial vision document capturing chapter system, multi-user, and marketplace designs
