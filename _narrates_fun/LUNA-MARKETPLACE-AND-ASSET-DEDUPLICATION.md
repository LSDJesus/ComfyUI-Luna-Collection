# LUNA Asset Marketplace & Anti-Duplication System

**Created:** 2025-11-08
## Overview

**Concept:** Users create and perfect assets (characters, locations, items) through iterative prompting and metadata refinement. They retain ownership (NFT-like) and earn micropayment revenue when others use their creations. The system prevents duplication and plagiarism using dual-vector comparison (metadata semantics + visual similarity).

---

## Database Schema

### Asset Ownership & Marketplace

```sql
-- Core asset table (central database)
CREATE TABLE luna_assets.marketplace_assets (
    asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Asset Content
    asset_type VARCHAR(50) NOT NULL,  -- 'character', 'location', 'item', 'quest_hook'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    content_metadata JSONB,  -- Full metadata (see Asset Card format below)
    
    -- Visual Content
    image_data bytea NOT NULL,  -- WebP compressed
    image_vector VECTOR(1024),  -- BGE embeddings of visual features
    visual_fingerprint VARCHAR(256),  -- CLIP-based visual hash
    
    -- Metadata Vector (NEW)
    metadata_text TEXT,  -- Concatenated searchable text
    metadata_vector VECTOR(1024),  -- BGE embeddings of metadata/description
    metadata_hash VARCHAR(256),  -- Quick similarity check
    
    -- Ownership & Licensing
    creator_user_id UUID NOT NULL REFERENCES core.users(user_id),
    ownership_type VARCHAR(50),  -- 'user_created', 'derived', 'collaboration'
    license_type VARCHAR(50),  -- 'exclusive', 'non_exclusive', 'cc_by_sa'
    
    -- Monetization
    price_usd DECIMAL(10, 4),  -- Usage fee (0.01 - 1.00 typical)
    is_for_sale BOOLEAN DEFAULT TRUE,
    lifetime_revenue_usd DECIMAL(12, 2) DEFAULT 0.00,
    
    -- Derivation Tracking (for NFT-like ownership)
    parent_asset_id UUID REFERENCES luna_assets.marketplace_assets(asset_id),
    derivation_depth INTEGER DEFAULT 0,  -- 0=original, 1=remix, 2=remix of remix
    is_derivative BOOLEAN DEFAULT FALSE,
    
    -- Quality & Curation
    quality_score FLOAT,  -- 1-5 stars from usage
    usage_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    violation_reports INTEGER DEFAULT 0,
    
    -- Metadata
    tags JSONB,  -- searchable tags
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP,
    
    -- Indexes for marketplace operations
    INDEX idx_asset_type (asset_type),
    INDEX idx_creator (creator_user_id),
    INDEX idx_for_sale (is_for_sale),
    INDEX idx_quality (quality_score DESC),
    INDEX idx_usage (usage_count DESC),
    
    -- Vector indexes for similarity search
    INDEX idx_image_vector USING ivfflat (image_vector) WITH (lists=100),
    INDEX idx_metadata_vector USING ivfflat (metadata_vector) WITH (lists=100),
    
    -- Uniqueness constraint per creator (can't upload identical twice)
    UNIQUE(creator_user_id, metadata_hash)
);

-- Track all usage of marketplace assets
CREATE TABLE luna_assets.marketplace_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES luna_assets.marketplace_assets(asset_id),
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    story_id VARCHAR(255),
    turn_id INTEGER,
    
    -- Payment tracking
    price_paid_usd DECIMAL(10, 4),
    revenue_to_creator_usd DECIMAL(10, 4),  -- After platform cut
    status VARCHAR(50),  -- 'completed', 'refunded', 'disputed'
    
    used_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_asset (asset_id),
    INDEX idx_user (user_id),
    INDEX idx_used_at (used_at DESC)
);

-- Duplicate/plagiarism detection results
CREATE TABLE luna_assets.similarity_analysis (
    analysis_id BIGSERIAL PRIMARY KEY,
    
    -- Compared assets
    asset_a_id UUID NOT NULL REFERENCES luna_assets.marketplace_assets(asset_id),
    asset_b_id UUID NOT NULL REFERENCES luna_assets.marketplace_assets(asset_id),
    
    -- Similarity scores (0.0 - 1.0)
    metadata_similarity FLOAT,  -- Metadata vector cosine similarity
    visual_similarity FLOAT,    -- Image vector cosine similarity
    content_hash_similarity FLOAT,  -- Metadata hash overlap
    combined_similarity FLOAT,  -- Weighted average
    
    -- Flags
    is_duplicate BOOLEAN,  -- combined_similarity > 0.95
    is_plagiarism BOOLEAN,  -- High similarity + same creator pattern
    is_derivative BOOLEAN,  -- Moderate similarity + acknowledged parent
    
    -- Analysis details
    flagged_differences JSONB,  -- What was actually different
    analysis_details JSONB,  -- Full comparison breakdown
    
    analyzed_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_asset_a (asset_a_id),
    INDEX idx_asset_b (asset_b_id),
    INDEX idx_similarity (combined_similarity DESC),
    INDEX idx_duplicate (is_duplicate) WHERE is_duplicate = TRUE
);

-- Derivation relationships (explicit parent-child for remix culture)
CREATE TABLE luna_assets.asset_derivations (
    derivation_id BIGSERIAL PRIMARY KEY,
    parent_asset_id UUID NOT NULL REFERENCES luna_assets.marketplace_assets(asset_id),
    child_asset_id UUID NOT NULL REFERENCES luna_assets.marketplace_assets(asset_id),
    
    -- Relationship type
    derivation_type VARCHAR(50),  -- 'remix', 'recolor', 'reimagine', 'mashup'
    description TEXT,  -- What was changed?
    
    -- Revenue sharing
    parent_royalty_percent FLOAT DEFAULT 10.0,  -- Parent gets 10% of child sales
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(parent_asset_id, child_asset_id),
    INDEX idx_parent (parent_asset_id),
    INDEX idx_child (child_asset_id)
);

-- Creator reputation & verification
CREATE TABLE luna_assets.creator_reputation (
    creator_id UUID PRIMARY KEY REFERENCES core.users(user_id),
    
    -- Stats
    total_assets INTEGER DEFAULT 0,
    total_sales INTEGER DEFAULT 0,
    total_revenue_usd DECIMAL(12, 2) DEFAULT 0.00,
    average_quality_score FLOAT DEFAULT 0.0,
    
    -- Trust metrics
    plagiarism_strikes INTEGER DEFAULT 0,  -- 3 strikes = banned
    dispute_rate FLOAT DEFAULT 0.0,  -- % of sales with refunds
    verified_creator BOOLEAN DEFAULT FALSE,  -- Manual verification
    
    -- Badges
    badges JSONB,  -- ['trending', 'quality', 'verified', 'community_favorite']
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_verified (verified_creator),
    INDEX idx_revenue (total_revenue_usd DESC)
);
```

---

## Asset Card Format (Complete Metadata)

### Structure with Embeddable Content

```json
{
  "luna_asset_card": {
    "version": "1.0",
    "created_at": "2025-11-08T12:00:00Z",
    
    "asset_id": "uuid-123",
    "asset_type": "character",
    "name": "Shadowveil, the Drow Rogue",
    
    "creator": {
      "user_id": "uuid-creator",
      "username": "fantasy_artist_2024",
      "verified": true,
      "profile_url": "https://luna-narrates.com/creators/fantasy_artist_2024"
    },
    
    "description": "A mysterious drow rogue with silver eyes and midnight black skin...",
    "tags": ["drow", "rogue", "female", "mysterious", "dark_fantasy"],
    
    "character": {
      "race": "Drow",
      "class": "Rogue",
      "level": 5,
      "alignment": "Chaotic Neutral",
      "personality_traits": ["sarcastic", "cunning", "honorable_thieves_code"],
      "backstory_summary": "Escaped from Underdark matriarchy...",
      "equipment": ["shortswords", "leather_armor", "lockpicks"],
      "special_abilities": ["shadow_stepping", "poison_craft"]
    },
    
    "visual_generation": {
      "model": "SDXL",
      "prompt": "A stunning drow female rogue, silver eyes, obsidian skin, leather armor...",
      "seed": 42957381,
      "steps": 40,
      "guidance_scale": 7.5,
      "style_loras": ["dnd_character_style", "fantasy_portrait_v2"]
    },
    
    "licensing": {
      "ownership_type": "exclusive",
      "license": "proprietary",
      "price_usd": 0.25,
      "royalty_percent": 100.0
    },
    
    "fingerprints": {
      "visual_hash": "sha256:xyz789...",
      "metadata_hash": "sha256:meta456...",
      "perceptual_hash": "phash:..."
    }
  }
}
```

---

## Anti-Duplication System: Multi-Layer Vector Comparison

### Layer 1: Metadata Vector Similarity

```python
class MetadataDeduplicator:
    """Detect duplicate/plagiarized metadata using semantic vectors."""
    
    def __init__(self, embedding_service, db):
        self.embeddings = embedding_service
        self.db = db
    
    async def extract_metadata_text(self, asset_metadata):
        """Convert metadata JSON to concatenated searchable text."""
        texts = [
            asset_metadata.get('name', ''),
            asset_metadata.get('description', ''),
            ' '.join(asset_metadata.get('tags', [])),
        ]
        
        if asset_metadata.get('asset_type') == 'character':
            char = asset_metadata.get('character', {})
            texts.extend([
                char.get('race', ''),
                char.get('class', ''),
                ' '.join(char.get('personality_traits', [])),
                char.get('backstory_summary', ''),
            ])
        
        return ' '.join(filter(None, texts))
    
    async def analyze_metadata_similarity(
        self,
        new_asset_id,
        new_metadata,
        new_image,
        creator_id
    ):
        """Check for similar metadata across marketplace."""
        
        new_text = await self.extract_metadata_text(new_metadata)
        new_metadata_vector = await self.embeddings.embed_text(new_text)
        new_visual_fingerprint = await self._generate_visual_fingerprint(new_image)
        
        # Search similar in database
        similar_metadata = await self.db.query("""
            SELECT 
                asset_id, name, creator_user_id, metadata_vector,
                image_vector, visual_fingerprint, metadata_hash, created_at
            FROM luna_assets.marketplace_assets
            WHERE asset_type = $1
            ORDER BY metadata_vector <=> $2::vector
            LIMIT 20
        """, new_metadata['asset_type'], new_metadata_vector)
        
        results = []
        for row in similar_metadata:
            metadata_sim = 1 - await self._cosine_distance(
                new_metadata_vector, row['metadata_vector']
            )
            visual_sim = 1 - await self._cosine_distance(
                new_visual_fingerprint, row['visual_fingerprint']
            )
            combined = (metadata_sim * 0.6) + (visual_sim * 0.4)
            
            results.append({
                'similar_asset_id': row['asset_id'],
                'name': row['name'],
                'creator_user_id': row['creator_user_id'],
                'metadata_similarity': metadata_sim,
                'visual_similarity': visual_sim,
                'combined_similarity': combined,
                'is_duplicate': combined > 0.95,
                'is_plagiarism': combined > 0.85 and row['creator_user_id'] != creator_id,
                'created_at': row['created_at']
            })
        
        return {
            'new_asset_id': new_asset_id,
            'total_matches': len(results),
            'similarity_results': sorted(results, key=lambda x: x['combined_similarity'], reverse=True),
            'flagged_as_duplicate': any(r['is_duplicate'] for r in results),
            'recommendation': self._generate_recommendation(results, creator_id)
        }
    
    def _generate_recommendation(self, results, creator_id):
        """Generate action recommendation based on similarity."""
        
        if not results:
            return {
                'status': 'approved',
                'reason': 'No similar assets found',
                'action': 'Publish immediately'
            }
        
        top_match = results[0]
        
        if top_match['combined_similarity'] > 0.95:
            if top_match['creator_user_id'] == creator_id:
                return {
                    'status': 'duplicate_self',
                    'reason': f"95%+ similar to your existing asset '{top_match['name']}'",
                    'action': 'Update existing asset instead of creating duplicate'
                }
            else:
                return {
                    'status': 'flagged_plagiarism',
                    'reason': f"95%+ similar to '{top_match['name']}' by another creator",
                    'action': 'Manual review required before publishing'
                }
        
        elif top_match['combined_similarity'] > 0.85:
            if top_match['creator_user_id'] != creator_id:
                return {
                    'status': 'derivative_potential',
                    'reason': f"85%+ similar to existing asset",
                    'action': 'Publish as derivative (parent gets 10% royalty) or modify'
                }
        
        return {
            'status': 'approved',
            'reason': 'Sufficiently unique from existing assets',
            'action': 'Publish immediately'
        }
```

### Layer 2: Visual Similarity (CLIP Embeddings)

```python
class VisualDeduplicator:
    """Detect duplicate/similar images using CLIP embeddings."""
    
    async def compare_images(self, image_bytes_a, image_bytes_b):
        """Compare two images for visual similarity (0.0 - 1.0)."""
        
        embedding_a = await self._get_image_embedding(image_bytes_a)
        embedding_b = await self._get_image_embedding(image_bytes_b)
        
        import numpy as np
        similarity = np.dot(embedding_a, embedding_b) / (
            np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        )
        
        return float(similarity)
    
    async def _get_image_embedding(self, image_bytes):
        """Get CLIP image embedding."""
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        
        if not hasattr(self, '_clip_model'):
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        with torch.no_grad():
            inputs = self._clip_processor(images=image, return_tensors="pt")
            image_features = self._clip_model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().flatten()
```

### Layer 3: Content Hash Fingerprinting

```python
class ContentFingerprinter:
    """Generate deterministic fingerprints for quick duplicate detection."""
    
    def generate_metadata_hash(self, metadata):
        """Generate hash of metadata content (ignores creator/timestamps)."""
        import json
        import hashlib
        
        normalized = {
            'asset_type': metadata.get('asset_type'),
            'name': metadata.get('name'),
            'description': metadata.get('description'),
            'tags': sorted(metadata.get('tags', [])),
        }
        
        if metadata.get('asset_type') == 'character':
            char = metadata.get('character', {})
            normalized['character'] = {
                'race': char.get('race'),
                'class': char.get('class'),
                'personality_traits': sorted(char.get('personality_traits', [])),
                'backstory_summary': char.get('backstory_summary'),
            }
        
        content_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def generate_visual_hash(self, image_bytes):
        """Generate perceptual hash of image."""
        from PIL import Image
        import imagehash
        import hashlib
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        
        phash = imagehash.phash(image)
        dhash = imagehash.dhash(image)
        
        combined = f"{str(phash)}_{str(dhash)}"
        return hashlib.sha256(combined.encode()).hexdigest()
```

---

## Marketplace Publishing Workflow

```python
async def publish_asset(
    user_id,
    asset_metadata,
    image_bytes,
    db,
    deduplicator,
    embeddings
):
    """Publish asset to marketplace with anti-duplication checks."""
    
    # Step 1: Run deduplication analysis
    dup_analysis = await deduplicator.analyze_metadata_similarity(
        new_asset_id=None,
        new_metadata=asset_metadata,
        new_image=image_bytes,
        creator_id=user_id
    )
    
    print(f"Duplication Analysis:")
    print(f"  Similar Assets Found: {dup_analysis['total_matches']}")
    print(f"  Flagged as Duplicate: {dup_analysis['flagged_as_duplicate']}")
    print(f"  Recommendation: {dup_analysis['recommendation']}")
    
    # Step 2: Handle different scenarios
    if dup_analysis['flagged_as_duplicate']:
        return {
            'status': 'rejected',
            'reason': 'Duplicate of existing asset',
            'similar_asset': dup_analysis['similarity_results'][0],
            'action': 'Modify your asset to make it more unique'
        }
    
    elif dup_analysis['recommendation']['status'] == 'derivative_potential':
        parent_asset = dup_analysis['similarity_results'][0]
        return {
            'status': 'derivative_option',
            'parent_asset': parent_asset,
            'action': 'Publish as derivative (parent gets 10% royalty)?'
        }
    
    # Step 3: Generate content fingerprints
    fingerprinter = ContentFingerprinter()
    metadata_hash = fingerprinter.generate_metadata_hash(asset_metadata)
    visual_hash = fingerprinter.generate_visual_hash(image_bytes)
    
    # Step 4: Extract metadata text and generate vectors
    metadata_text = await deduplicator.extract_metadata_text(asset_metadata)
    metadata_vector = await embeddings.embed_text(metadata_text)
    image_vector = await embeddings.embed_image(image_bytes)
    
    # Step 5: Store in database
    asset_id = await db.execute("""
        INSERT INTO luna_assets.marketplace_assets (
            asset_type, name, description, content_metadata, image_data,
            image_vector, visual_fingerprint, metadata_text, metadata_vector,
            metadata_hash, creator_user_id, ownership_type, price_usd, tags,
            published_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
        RETURNING asset_id
    """,
        asset_metadata['asset_type'],
        asset_metadata['name'],
        asset_metadata['description'],
        json.dumps(asset_metadata),
        image_bytes,
        metadata_vector,
        visual_hash,
        metadata_text,
        metadata_vector,
        metadata_hash,
        user_id,
        'user_created',
        asset_metadata.get('price_usd', 0.25),
        asset_metadata.get('tags', [])
    )
    
    # Step 6: Store similarity analysis for top matches
    for similar in dup_analysis['similarity_results'][:5]:
        await db.execute("""
            INSERT INTO luna_assets.similarity_analysis (
                asset_a_id, asset_b_id, metadata_similarity,
                visual_similarity, combined_similarity, is_duplicate, is_plagiarism
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            asset_id,
            similar['similar_asset_id'],
            similar['metadata_similarity'],
            similar['visual_similarity'],
            similar['combined_similarity'],
            similar['is_duplicate'],
            similar['is_plagiarism']
        )
    
    return {
        'status': 'published',
        'asset_id': asset_id,
        'uniqueness_score': 1.0 - (dup_analysis['similarity_results'][0]['combined_similarity'] if dup_analysis['similarity_results'] else 0),
        'similar_assets': dup_analysis['similarity_results'][:3]
    }
```

---

## Usage & Revenue System

```python
async def use_marketplace_asset(
    user_id,
    asset_id,
    story_id,
    turn_id,
    db
):
    """User includes marketplace asset in their story."""
    
    # Fetch asset
    asset = await db.fetchrow("""
        SELECT creator_user_id, price_usd, image_data,
               content_metadata, parent_asset_id
        FROM luna_assets.marketplace_assets
        WHERE asset_id = $1 AND is_for_sale = TRUE
    """, asset_id)
    
    if not asset:
        raise ValueError("Asset not available for purchase")
    
    price = float(asset['price_usd'])
    platform_cut = price * 0.15
    creator_revenue = price * 0.85
    parent_royalty = 0
    
    if asset['parent_asset_id']:
        parent_royalty = creator_revenue * 0.10
        creator_revenue -= parent_royalty
    
    # Record usage
    await db.execute("""
        INSERT INTO luna_assets.marketplace_usage (
            asset_id, user_id, story_id, turn_id,
            price_paid_usd, revenue_to_creator_usd, status
        ) VALUES ($1, $2, $3, $4, $5, $6, 'completed')
    """, asset_id, user_id, story_id, turn_id, price, creator_revenue)
    
    # Update asset stats
    await db.execute("""
        UPDATE luna_assets.marketplace_assets
        SET download_count = download_count + 1,
            usage_count = usage_count + 1,
            lifetime_revenue_usd = lifetime_revenue_usd + $1
        WHERE asset_id = $2
    """, price, asset_id)
    
    # Update creator reputation
    await db.execute("""
        UPDATE luna_assets.creator_reputation
        SET total_sales = total_sales + 1,
            total_revenue_usd = total_revenue_usd + $1
        WHERE creator_id = $2
    """, creator_revenue, asset['creator_user_id'])
    
    # Update parent if derivative
    if asset['parent_asset_id'] and parent_royalty > 0:
        parent = await db.fetchval("""
            SELECT creator_user_id FROM luna_assets.marketplace_assets
            WHERE asset_id = $1
        """, asset['parent_asset_id'])
        
        if parent:
            await db.execute("""
                UPDATE luna_assets.creator_reputation
                SET total_revenue_usd = total_revenue_usd + $1
                WHERE creator_id = $2
            """, parent_royalty, parent)
    
    return {
        'status': 'purchased',
        'asset_id': asset_id,
        'price_paid': price,
        'creator_received': creator_revenue,
        'parent_royalty': parent_royalty,
        'asset_data': {
            'image': asset['image_data'],
            'metadata': json.loads(asset['content_metadata'])
        }
    }
```

---

## Creator Dashboard & Analytics

```python
async def get_creator_analytics(user_id, time_period='30d', db=None):
    """Get comprehensive analytics for asset creator."""
    from datetime import datetime, timedelta
    from collections import Counter
    
    days = {'7d': 7, '30d': 30, '90d': 90, 'all': 999999}[time_period]
    start_date = datetime.now() - timedelta(days=days)
    
    # Asset stats
    assets = await db.fetch("""
        SELECT asset_id, name, asset_type, quality_score,
               download_count, usage_count, lifetime_revenue_usd, created_at
        FROM luna_assets.marketplace_assets
        WHERE creator_user_id = $1
        ORDER BY lifetime_revenue_usd DESC
    """, user_id)
    
    # Usage stats
    usage_stats = await db.fetchrow("""
        SELECT 
            COUNT(*) as total_downloads,
            SUM(price_paid_usd) as total_revenue,
            AVG(price_paid_usd) as avg_price_paid,
            COUNT(DISTINCT user_id) as unique_buyers
        FROM luna_assets.marketplace_usage mu
        JOIN luna_assets.marketplace_assets ma ON mu.asset_id = ma.asset_id
        WHERE ma.creator_user_id = $1
    """, user_id)
    
    # Derivative tracking
    derivatives = await db.fetch("""
        SELECT ma.asset_id, ma.name, ma.creator_user_id,
               COUNT(mu.usage_id) as times_used,
               SUM(mu.price_paid_usd) * 0.10 as my_royalties
        FROM luna_assets.asset_derivations ad
        JOIN luna_assets.marketplace_assets ma ON ad.child_asset_id = ma.asset_id
        LEFT JOIN luna_assets.marketplace_usage mu ON ma.asset_id = mu.asset_id
        WHERE ad.parent_asset_id IN (
            SELECT asset_id FROM luna_assets.marketplace_assets 
            WHERE creator_user_id = $1
        )
        GROUP BY ma.asset_id, ma.name, ma.creator_user_id
        ORDER BY my_royalties DESC
    """, user_id)
    
    # Quality analysis
    quality = await db.fetchrow("""
        SELECT 
            AVG(quality_score) as avg_quality,
            MIN(quality_score) as min_quality,
            MAX(quality_score) as max_quality,
            COUNT(CASE WHEN quality_score >= 4.5 THEN 1 END) as top_tier_assets
        FROM luna_assets.marketplace_assets
        WHERE creator_user_id = $1
    """, user_id)
    
    return {
        'assets': {
            'total': len(assets),
            'by_type': dict(Counter(a['asset_type'] for a in assets)),
            'top_performers': assets[:5],
            'avg_quality': quality['avg_quality']
        },
        'revenue': {
            'total_revenue': float(usage_stats['total_revenue'] or 0),
            'total_downloads': usage_stats['total_downloads'],
            'unique_buyers': usage_stats['unique_buyers'],
            'avg_price_paid': float(usage_stats['avg_price_paid'] or 0)
        },
        'derivatives': {
            'total_derivative_assets': len(derivatives),
            'derivative_royalties': sum(d['my_royalties'] or 0 for d in derivatives),
            'top_derivatives': derivatives[:5]
        }
    }
```

---

## Marketplace Search with Semantic Similarity

```python
async def search_marketplace(
    query,
    asset_type=None,
    sort_by='relevance',
    db=None,
    embeddings=None
):
    """Search marketplace with semantic + traditional search."""
    
    query_vector = await embeddings.embed_text(query)
    
    sql = """
        SELECT 
            asset_id, name, description, asset_type, price_usd,
            creator_user_id, quality_score, usage_count, image_data,
            metadata_vector <=> $1::vector as metadata_distance
        FROM luna_assets.marketplace_assets
        WHERE is_for_sale = TRUE
            AND (metadata_vector <=> $1::vector) < 0.5
    """
    
    params = [query_vector]
    param_count = 2
    
    if asset_type:
        sql += f" AND asset_type = ${param_count}"
        params.append(asset_type)
        param_count += 1
    
    if sort_by == 'relevance':
        sql += " ORDER BY (metadata_vector <=> $1::vector) ASC"
    elif sort_by == 'popularity':
        sql += " ORDER BY usage_count DESC"
    elif sort_by == 'price_low':
        sql += " ORDER BY price_usd ASC"
    elif sort_by == 'price_high':
        sql += " ORDER BY price_usd DESC"
    elif sort_by == 'newest':
        sql += " ORDER BY created_at DESC"
    
    sql += " LIMIT 50"
    
    results = await db.fetch(sql, *params)
    return [dict(r) for r in results]
```

---

## Key Anti-Duplication Features

### ✅ Layer 1: Semantic Metadata Matching (60%)
- BGE embeddings of name, description, tags, personality
- Cosine similarity threshold: >0.85 = similar, >0.95 = duplicate
- Catches: "Shadow Rogue" vs "Shadowveil, Drow Rogue"

### ✅ Layer 2: Visual Similarity (40%)
- CLIP embeddings of generated images
- Detects: Color swaps, outfit changes, pose variations
- Catches: "Blue Wizard" vs "Red Wizard" (95% similarity)

### ✅ Layer 3: Content Hash Fingerprinting
- Deterministic SHA256 of normalized metadata
- Quick uniqueness check (prevents self-duplication)
- One constraint: `UNIQUE(creator_user_id, metadata_hash)`

### ✅ Layer 4: Creator Reputation System
- 3 strikes = plagiarism ban
- Verified creators get badge
- Dispute rate tracked for trust

### ✅ Layer 5: Explicit Derivatives
- Users can acknowledge "remix of X"
- Parent gets 10% of child sales
- Transparent derivation chain (NFT-like ownership proof)

---

## Revenue Model Example

**Asset: "Shadowveil, Drow Rogue"**
- Creator: fantasy_artist_2024
- Price: $0.25 per use

**100 downloads in Month 1:**
- Total revenue: $25.00
- Platform takes 15%: $3.75
- Creator receives 85%: $21.25

**If derivative created ("Shadowveil, Dark Assassin"):**
- Derived asset price: $0.15
- 50 downloads in Month 1: $7.50 total
- Platform: $1.13
- Derivative creator: $6.38
- **Original creator gets 10% of derivative's revenue: $0.64**

**Monthly scalability (100 assets, 10K downloads):**
- Total marketplace revenue: $2,500
- Platform revenue (15%): $375
- Creator revenue (85%): $2,125 distributed

---

## Summary

This system enables:
1. **True Digital Ownership** - Assets are NFT-like (without blockchain overhead)
2. **Micropayment Revenue** - Creators earn $0.01-$1.00 per use
3. **Remix Culture** - Derivatives with transparent royalty sharing
4. **Anti-Duplication** - 5-layer vector + hash comparison prevents plagiarism
5. **Creator Reputation** - Verified badges, strike system, analytics dashboard

**Perfect for:** Asset marketplace with sustainable monetization where creators retain ownership and earn passive revenue.