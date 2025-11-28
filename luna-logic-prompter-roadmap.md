# 🌙 Luna Logic Prompter: Refined Development Roadmap

## Project Vision

**Core Goal:** Build an intelligent random prompt generator that produces coherent, contextually-consistent outputs by tracking semantic compatibility between prompt elements. No LLMs at runtime—just smart tag-based logic that prevents nonsensical combinations.

**Philosophy:** Random, but not chaotic. Let wildcards reference each other, maintain context state, and enforce compatibility rules to avoid "medieval knight in a cyberpunk neon cityscape wearing a swimsuit" type disasters.

**LLM Role:** Use AI tools *during wildcard authoring* to generate and classify items, but the resolver itself is deterministic and lightweight.

---

## Architecture Overview

```
wildcards/
├── outfit.yaml              # Character clothing
├── location.yaml            # Settings/environments
├── accessory.yaml           # Props and details
├── pose.yaml                # Character poses
└── lighting.yaml            # Lighting conditions

utils/
└── logic_engine.py          # Core resolution engine (no AI dependencies)

nodes/
└── preprocessing/
    ├── luna_logic_resolver.py      # Main resolver node
    └── luna_wildcard_generator.py  # AI-assisted YAML authoring (optional)
```

---

## Core Features

### 1. Context-Aware Resolution
**Problem:** Random wildcards often combine incompatible elements.

**Solution:** Maintain a running context (set of active tags) and filter candidates based on:
- **Blacklist:** "medieval" and "scifi" can't coexist
- **Whitelist:** Some items require specific tags to be present
- **Requires_tags:** Dependencies (e.g., "ninja_outfit" requires "stealth" tag)
- **Weights:** Higher-priority items get selected more often

**Example Flow:**
```
1. Resolve __location__ → "neon cityscape" (adds tags: scifi, urban, night)
2. Resolve __outfit__ → Filters out "plate_armor" (blacklist: scifi)
              → Selects "cyber suit" (whitelist: tech, compatible with scifi)
3. Context evolves: {scifi, urban, night, tech, tight}
4. Resolve __accessory__ → Only considers items compatible with current context
```

### 2. Wildcard Composition
**Problem:** Redundant definitions across multiple wildcard files.

**Solution:** Wildcards can reference other wildcards using `composition` field.

**Example:**
```yaml
# wildcards/character.yaml
items:
  - id: "cyberpunk_mercenary"
    text: "__scifi_outfit__, __tactical_accessory__"
    tags: ["scifi", "combat"]
    composition: ["scifi_outfit", "tactical_accessory"]
```

**Benefits:**
- Define "tactical_accessory" once, reuse everywhere
- Semantic coherence (outfit + accessory auto-match context)
- Easier maintenance

### 3. Weighted Random Selection
**Problem:** All items have equal probability, but some are better defaults.

**Solution:** Add `weight: float` field (default 1.0).

```yaml
items:
  - id: "generic_outfit"
    weight: 0.5  # Less likely
  - id: "hero_outfit"
    weight: 1.5  # More likely
```

### 4. LoRA Payload Integration
**Problem:** Certain prompt elements need specific LoRAs but managing them manually is tedious.

**Solution:** Bundle LoRA syntax with wildcard items.

```yaml
items:
  - id: "cyber_suit"
    text: "high tech bodysuit, neon trim"
    payload: "<lora:Cyberpunk_v3:0.8> embedding:BadDream"
```

**Output:** `"high tech bodysuit, neon trim <lora:Cyberpunk_v3:0.8> embedding:BadDream"`

---

## YAML Schema

```yaml
name: "wildcard_category"  # Used as __wildcard_category__ in prompts
items:
  - id: "unique_identifier"
    text: "the actual prompt text to insert"
    tags: ["tag1", "tag2"]          # Categories for compatibility
    whitelist: ["required_tag"]     # Only select if context has these
    blacklist: ["forbidden_tag"]    # Never select if context has these
    requires_tags: ["dependency"]   # Must have these in context
    weight: 1.0                     # Selection probability multiplier
    payload: "<lora:...>"           # Optional LoRA/embedding syntax
    composition: ["other_wildcard"] # Reference to other wildcards
```

**Validation Rules:**
- `id` must be unique within a wildcard file
- `tags`, `whitelist`, `blacklist` are case-insensitive
- `composition` items must exist (checked at load time)
- Circular dependencies are detected and rejected

---

## Development Phases

### Phase 1: Core Engine (Week 1)
**Deliverable:** Functional logic resolver with no UI polish.

**Tasks:**
1. Create `LogicItem` dataclass with all fields
2. Implement `LunaLogicEngine` class:
   - `load_definitions(wildcards_dir)` - Scan and parse YAML files
   - `resolve_prompt(template, seed, initial_context)` - Main resolver
   - `is_compatible(item, context)` - Blacklist/whitelist checker
   - Weighted random selection with `random.choices()`
3. Handle `__wildcard__` regex replacement
4. Detect circular dependencies in `composition` fields
5. Unit tests for compatibility logic

**No AI dependencies yet.** Pure Python + PyYAML.

### Phase 2: ComfyUI Integration (Week 2)
**Deliverable:** Working node in ComfyUI with basic I/O.

**Tasks:**
1. Create `LunaLogicResolver` node:
   - Input: `text` (multiline STRING with `__wildcards__`)
   - Input: `seed` (INT for reproducibility)
   - Input: `initial_context` (STRING, comma-separated tags)
   - Output: `STRING` (resolved prompt)
2. Auto-detect `wildcards/` folder relative to node file
3. Register node in `__init__.py`
4. Error handling: Graceful failure if YAML is malformed
5. Test with 3+ example wildcard files (outfit, location, pose)

### Phase 3: AI-Assisted Authoring (Week 3-4)
**Deliverable:** Separate node to generate wildcard YAML using LLMs.

**Purpose:** Use AI *once* during wildcard creation, not at runtime.

**Tasks:**
1. Create `LunaWildcardGenerator` node:
   - Input: `source_text` (multiline STRING - user-provided examples)
   - Input: `wildcard_name` (STRING - e.g., "outfit")
   - Input: `num_variations` (INT, default 10)
   - Input: `tag_vocabulary` (STRING - comma-separated allowed tags)
   - Output: `YAML_STRING` (valid YAML to save manually)
2. Use `sentence-transformers` to:
   - Cluster `source_text` into semantic groups
   - Generate variations (paraphrase-MiniLM or similar)
   - Auto-classify tags by similarity to `tag_vocabulary`
3. Use lightweight generative model (e.g., GPT-2, TinyLlama, or Llama 3.2 1B):
   - Generate 5-10 prompt variations per cluster
   - Fine-tune on Stable Diffusion prompt datasets if available
4. Output human-readable YAML with comments
5. User reviews and saves to `wildcards/` folder

**Why separate node?**
- Main resolver stays lightweight (no AI overhead)
- You control when AI is invoked (only during authoring)
- Can run on CPU since it's not time-critical

### Phase 4: Refinements (Week 5+)
**Optional enhancements based on usage:**

1. **Wildcard Validator Node:**
   - Input: `wildcards_directory` (auto-detect or manual)
   - Output: `VALIDATION_REPORT` (markdown with errors/warnings)
   - Checks: YAML syntax, duplicate IDs, circular dependencies, orphaned tags

2. **Template Library:**
   - Create `wildcards/_templates/` folder with common prompt structures
   - Example: `character_portrait.yaml` with pre-configured wildcards
   - Node input: `template` dropdown auto-populates `text` field

3. **Context Presets:**
   - Store common tag combinations in `utils/presets.json`
   - Node input: `context_preset` dropdown (e.g., "scifi", "fantasy", "portrait")
   - Auto-sets `initial_context` to predefined tags

4. **Resolution Trace:**
   - Optional output: `DEBUG_INFO` (STRING showing step-by-step decisions)
   - Useful for understanding why an item was/wasn't selected

5. **Lazy Loading:**
   - Don't parse all YAMLs at startup
   - Only load referenced wildcards on-demand
   - Improves performance with 100+ wildcard files

---

## Example Workflow

### Creating Wildcards (One-Time Setup)
```
1. Write example prompts in a text file:
   "a warrior in heavy armor"
   "a knight with ornate plate mail"
   "a soldier in tactical gear"

2. Add Luna Wildcard Generator node
   - source_text: [paste examples]
   - wildcard_name: "outfit_armor"
   - num_variations: 15
   - tag_vocabulary: "medieval, modern, scifi, fantasy, heavy, light"

3. Node outputs YAML with:
   - 15 armor variations
   - Auto-classified tags
   - Inferred blacklists (e.g., medieval ↔ scifi)

4. Review YAML, tweak as needed
5. Save to wildcards/outfit_armor.yaml
```

### Using Wildcards (Every Generation)
```
1. Add Luna Logic Resolver node
   - text: "__character__ wearing __outfit__ in __location__, __lighting__"
   - seed: 12345
   - initial_context: "fantasy, female"

2. Connect to CLIP Text Encode / Impact Wildcard Encode

3. Node resolves:
   - __character__ → "elven mage" (adds: magic, elegant)
   - __outfit__ → Filters out "cyber suit" (blacklist: scifi)
             → Selects "flowing robes" (compatible with magic, elegant)
   - __location__ → "ancient library" (adds: indoors, scholarly)
   - __lighting__ → "soft candlelight" (compatible with indoors)

4. Output: "elven mage wearing flowing robes in ancient library, soft candlelight <lora:FantasyElf:0.9>"
```

---

## Technical Specifications

### Dependencies
```txt
# Core (required for resolver)
PyYAML>=6.0
jsonschema>=4.17.0  # For YAML validation

# AI Authoring (optional, only for generator node)
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
```

### Performance Targets
- Resolution time: <50ms for STRICT mode (1000 items)
- Memory: <100MB for resolver (no AI loaded)
- Startup: <200ms (lazy loading enabled)

### File Sizes
- Typical wildcard file: 5-20KB
- Full wildcard library (50 files): ~500KB
- AI model (if used): 80-400MB (one-time download)

---

## Design Decisions

### 1. Why YAML over JSON?
- **Human-friendly:** Comments, multi-line strings, no quote hell
- **Easier to edit:** Non-programmers can maintain wildcards
- **Validation:** jsonschema works with YAML too

### 2. Why no runtime LLMs?
- **Speed:** Context-aware filtering is instant, LLM inference is slow
- **Control:** Deterministic results (same seed = same output)
- **Resources:** Works on potato PCs, no VRAM required
- **Reliability:** No API calls, no rate limits, no hallucinations

### 3. Why separate authoring node?
- **Separation of concerns:** Core resolver stays simple
- **Optional complexity:** Users who don't want AI can ignore it
- **Performance:** AI overhead only during authoring, not every generation

### 4. Context as Set vs List?
- **Set:** Order-independent, no duplicates, fast lookup
- **Immutable during resolution:** Prevents race conditions
- **Merged across items:** Each selected item adds its tags to context

---

## Potential Challenges

### 1. YAML Authoring Learning Curve
**Solution:** Provide 5+ complete example wildcard packs covering common use cases (fantasy, scifi, portrait, landscape, abstract).

### 2. Tag Vocabulary Consistency
**Solution:** Include a `recommended_tags.txt` with common categories. Validator warns about typos (e.g., "sci-fi" vs "scifi").

### 3. Circular Dependencies
**Solution:** Detect at load time using topological sort. Fail fast with clear error message.

### 4. Performance with Large Libraries
**Solution:** Lazy loading (only parse referenced wildcards). Profile and optimize if needed.

### 5. LoRA Syntax Compatibility
**Solution:** Test with Impact Pack, ComfyUI-Manager, and standard LoRA loaders. Document supported formats.

---

## Success Criteria

**For Personal Use:**
- [ ] Generates coherent prompts 95%+ of the time
- [ ] No more manual blacklist management
- [ ] Wildcard authoring takes <30 minutes per category
- [ ] Resolution is fast enough to not slow down workflow (<100ms)
- [ ] Can create 100+ prompts from 10 wildcard files

**For Potential Sharing:**
- [ ] Someone else can use it without asking questions
- [ ] README explains YAML schema clearly
- [ ] Example wildcard packs demonstrate capabilities
- [ ] Validator catches 90%+ of authoring errors

---

## Implementation Priority

### Must-Have (MVP)
1. Core resolution engine with blacklist/whitelist
2. Weighted random selection
3. Basic ComfyUI node (text in, text out)
4. 3 example wildcard files (outfit, location, lighting)
5. YAML validation with clear errors

### Should-Have (Polish)
1. Wildcard composition (referencing other wildcards)
2. LoRA payload integration
3. AI-assisted generator node
4. Context presets
5. Template library

### Nice-to-Have (Future)
1. Validator node
2. Resolution trace / debug output
3. Lazy loading
4. Statistics/analytics
5. Import/export wildcard packs

---

## Next Steps

1. **Set up project structure:** Create `wildcards/`, `utils/`, `nodes/preprocessing/` folders
2. **Implement `LogicItem` dataclass:** Start with basic fields (text, tags, blacklist)
3. **Build core resolver:** Get `__wildcard__` replacement working
4. **Test with manual YAMLs:** Create 2-3 outfit.yaml entries, verify compatibility logic
5. **Integrate with ComfyUI:** Create basic node, test in workflow
6. **Iterate:** Add features based on what's actually needed during use

---

*This roadmap focuses on building a lightweight, deterministic, context-aware prompt generator that uses AI for authoring assistance, not runtime generation. Built with ❤️ for personal workflow optimization.*
