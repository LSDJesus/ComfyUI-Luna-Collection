# 🔗 LoRA & Embedding Connections Guide

The Luna Connection system allows you to intelligently link wildcards, prompts, and categories to specific LoRAs and embeddings. When a wildcard resolves or a prompt contains certain keywords, the appropriate LoRAs are automatically suggested or applied.

---

## 📁 The connections.json File

Located at `models/wildcards/connections.json`, this file defines all the mappings:

```json
{
  "version": "2.0",
  "loras": {
    "lora_filename.safetensors": {
      "type": "character",
      "base_model": "pony",
      "triggers": ["trigger word", "another trigger"],
      "activation_text": "full activation prompt",
      "training_tags": ["tag1", "tag2"],
      "civitai_tags": ["style", "character"],
      "yaml_paths": ["path.to.wildcard"],
      "categories": ["category_name"],
      "default_weight": 0.8,
      "clip_weight": 1.0
    }
  },
  "embeddings": {
    "embedding_name.pt": {
      "type": "style",
      "triggers": ["embedding trigger"],
      "yaml_paths": ["style.artistic"],
      "polarity": "positive"
    }
  }
}
```

---

## 📝 Connection Properties

### LoRA Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Civitai type: `character`, `style`, `concept`, `clothing`, `poses`, etc. |
| `base_model` | string | Target model: `pony`, `sdxl`, `illustrious`, `sd15` |
| `triggers` | array | Keywords that activate this LoRA |
| `activation_text` | string | Full prompt from Civitai training |
| `training_tags` | array | Tags from LoRA training data |
| `civitai_tags` | array | Civitai-assigned category tags |
| `yaml_paths` | array | Wildcard paths that should trigger this LoRA |
| `categories` | array | Custom category groupings |
| `default_weight` | float | Default model strength (0.0-2.0) |
| `clip_weight` | float | Default CLIP strength (0.0-2.0) |

### Embedding Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Category: `style`, `character`, `concept`, `quality` |
| `triggers` | array | Keywords that activate this embedding |
| `yaml_paths` | array | Wildcard paths linked to this embedding |
| `polarity` | string | `positive` or `negative` - which prompt to add to |
| `weight` | float | Default embedding weight |

---

## 🔧 Using Connection Nodes

### Luna Smart LoRA Linker

Automatically matches LoRAs based on your prompt.

**Inputs:**
- `prompt`: Your positive prompt text
- `model`, `clip`: Model/CLIP to apply LoRAs to
- `match_mode`: How to match (`triggers`, `training_tags`, `both`)
- `civitai_type_filter`: Filter by type (e.g., only `character` LoRAs)
- `base_model_filter`: Filter by base model
- `max_loras`: Maximum LoRAs to apply

**How it works:**
1. Parses your prompt for keywords
2. Matches against `triggers` and `training_tags` in connections.json
3. Filters by type and base model
4. Returns matched LoRAs as a stack

**Example:**
```
Prompt: "a beautiful elf girl with detailed eyes, fantasy style"

Matches:
- elf_character.safetensors (trigger: "elf")
- detailed_eyes.safetensors (trigger: "detailed eyes")
- fantasy_style.safetensors (trigger: "fantasy style")
```

### Luna Connection Matcher

Lower-level node for custom matching logic.

**Inputs:**
- `text`: Text to match against
- `connection_type`: `loras` or `embeddings`
- `match_fields`: Which fields to check

**Output:** List of matched connection names

### Luna Civitai Metadata Scraper

Fetches metadata from Civitai and embeds it into your models.

**Inputs:**
- `model_path`: Path to LoRA/model file
- `write_to_model`: Whether to embed metadata in the file

**Fetches:**
- Trigger words
- Training tags
- Base model info
- Description and usage tips

---

## 📋 Example Workflow

### 1. Set Up connections.json

```json
{
  "version": "2.0",
  "loras": {
    "detailed_face_v2.safetensors": {
      "type": "concept",
      "base_model": "pony",
      "triggers": ["detailed face", "beautiful face", "perfect face"],
      "training_tags": ["face", "portrait", "closeup"],
      "default_weight": 0.7
    },
    "anime_style_v1.safetensors": {
      "type": "style",
      "base_model": "pony",
      "triggers": ["anime style", "anime art"],
      "yaml_paths": ["style.anime"],
      "default_weight": 0.8
    },
    "elf_character.safetensors": {
      "type": "character",
      "base_model": "pony",
      "triggers": ["elf", "elven"],
      "yaml_paths": ["species.fantasy.elf"],
      "activation_text": "elf, pointed ears, ethereal beauty",
      "default_weight": 0.9
    }
  }
}
```

### 2. Create Matching Wildcards

```yaml
# species.yaml
fantasy:
  elf:
    - elf
    - elven princess
    - high elf
    - wood elf

# style.yaml
anime:
  - anime style
  - anime art
  - cel shaded
```

### 3. Use in Workflow

```
Prompt: "a beautiful {species:fantasy.elf} with detailed face, anime style"

After wildcard resolution:
"a beautiful wood elf with detailed face, anime style"

Luna Smart LoRA Linker matches:
1. elf_character.safetensors (trigger: "elf" from "wood elf")
2. detailed_face_v2.safetensors (trigger: "detailed face")
3. anime_style_v1.safetensors (trigger: "anime style")

All three LoRAs applied automatically!
```

---

## 🔄 Building connections.json

### Automatic with Civitai Scraper

1. Use **Luna Civitai Batch Scraper** to fetch metadata for all your LoRAs
2. Run `scripts/parse_metadata_connections.py` to generate connections.json

```powershell
python scripts/parse_metadata_connections.py
```

### Manual Entry

Add entries by hand for custom mappings:

```json
{
  "loras": {
    "my_custom_lora.safetensors": {
      "triggers": ["my keyword"],
      "default_weight": 0.75
    }
  }
}
```

### From SwarmUI Metadata

If you use SwarmUI, Luna can read its metadata:

```powershell
python scripts/extract_lora_metadata_v2.py --source swarmui
```

---

## 💡 Advanced Features

### Training Tag Matching

Match against the actual training data tags:

```json
{
  "loras": {
    "portrait_lora.safetensors": {
      "training_tags": ["1girl", "portrait", "face focus", "upper body"]
    }
  }
}
```

When your prompt contains any of these tags, the LoRA is suggested.

### Civitai Type Filtering

Filter LoRAs by their Civitai category:

```
Luna Smart LoRA Linker:
  civitai_type_filter: "character"

Only matches LoRAs with "type": "character"
```

Available types:
- `character` - Character LoRAs
- `style` - Art style LoRAs
- `concept` - Concept/object LoRAs
- `clothing` - Outfit LoRAs
- `poses` - Pose LoRAs
- `celebrity` - Real person LoRAs (use responsibly)

### Wildcard Path Linking

Link wildcards directly to LoRAs:

```json
{
  "loras": {
    "school_uniform.safetensors": {
      "yaml_paths": ["clothing.uniforms.school", "outfits.school"]
    }
  }
}
```

When `{clothing:uniforms.school}` resolves, this LoRA is automatically matched.

---

## 🔍 Troubleshooting

### LoRA Not Matching

1. **Check triggers**: Are the exact words in your prompt?
2. **Check base_model**: Does it match your current model?
3. **Check type filter**: Is it being filtered out?

### Too Many Matches

1. **Increase specificity**: Use more specific triggers
2. **Use type filtering**: Only match character or style LoRAs
3. **Reduce max_loras**: Limit the number applied

### Weights Too Strong/Weak

Adjust in connections.json:
```json
{
  "default_weight": 0.5,
  "clip_weight": 0.7
}
```

---

## 📚 Related Guides

- [YAML Wildcards Guide](yaml_wildcards.md) - Wildcard syntax
- [Node Reference](node_reference.md) - Complete parameters
- [Daemon Setup](../luna_daemon/README.md) - Shared model setup
