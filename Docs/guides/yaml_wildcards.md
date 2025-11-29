# 🌿 YAML Wildcard System Guide

The Luna YAML Wildcard system provides a powerful, hierarchical alternative to traditional .txt wildcards. It supports nested categories, templates, numeric ranges, and intelligent path resolution.

---

## 📁 File Structure

YAML wildcard files are stored in your ComfyUI `models/wildcards/` directory:

```
models/wildcards/
├── clothing.yaml
├── hair.yaml
├── colors.yaml
├── character_prompts.yaml
└── subdirectory/
    └── special.yaml
```

---

## 📝 YAML File Format

### Basic Structure

```yaml
# clothing.yaml

# Simple lists
tops:
  casual:
    - t-shirt
    - tank top
    - hoodie
  formal:
    - blazer
    - dress shirt
    - vest

bottoms:
  - jeans
  - skirt
  - shorts
  - pants

# Templates section (optional)
templates:
  full_outfit:
    - "{tops.casual}, {bottoms}"
    - "wearing [tops.formal] with [bottoms]"
  minimal:
    - "[tops.casual]"
```

### Nested Hierarchies

```yaml
# hair.yaml
styles:
  long:
    straight:
      - flowing straight hair
      - sleek long hair
    wavy:
      - cascading waves
      - beachy waves
  short:
    - pixie cut
    - bob cut
    - buzzcut

colors:
  natural:
    - blonde
    - brunette
    - black
    - red
  fantasy:
    - pink
    - blue
    - purple
    - silver
```

---

## 🎯 Syntax Reference

### Basic Wildcard Resolution

| Syntax | Description | Example |
|--------|-------------|---------|
| `{file}` | Random from file's `templates` section | `{clothing}` |
| `{file:path}` | Random from specific path | `{clothing:tops.casual}` |
| `{file:path.to.items}` | Nested path resolution | `{hair:styles.long.straight}` |

### Inline Templates

Use `[path]` syntax inside templates for inline substitution:

```yaml
templates:
  full:
    - "a person with [colors.natural] hair wearing [tops.casual]"
```

When you use `{file: some text with [path]}`, the bracketed paths are resolved:
```
{clothing: wearing [tops.casual] and [bottoms]}
→ "wearing hoodie and jeans"
```

### Numeric Ranges

| Syntax | Description | Example Result |
|--------|-------------|----------------|
| `{1-10}` | Random integer | `7` |
| `{5-100}` | Integer range | `42` |
| `{0.5-1.5:0.1}` | Float with step | `0.8` |
| `{0.0-1.0:0.05}` | Fine-grained float | `0.35` |

### Legacy .txt Wildcards

The system maintains backward compatibility with traditional wildcards:
```
__location__           → Resolves location.txt
__hair/color__         → Resolves hair/color.txt
__subdirectory/file__  → Resolves subdirectory/file.txt
```

---

## 🔧 Using the Nodes

### Luna YAML Wildcard

The main node for wildcard resolution.

**Inputs:**
- `text`: Your prompt with wildcard syntax
- `seed`: Random seed for reproducibility
- `wildcards_dir`: Optional custom directory

**Example:**
```
Input:  "a {hair:colors.fantasy}-haired girl wearing {clothing:tops.casual}"
Output: "a purple-haired girl wearing hoodie"
```

### Luna YAML Wildcard Batch

Generate multiple variations at once.

**Inputs:**
- `text`: Prompt template
- `count`: Number of variations (1-100)
- `seed`: Starting seed

**Output:** Newline-separated variations

### Luna Wildcard Builder

Visual node for constructing wildcard expressions without typing syntax.

**Inputs:**
- `file`: Select from available YAML files
- `path`: Browse nested paths
- `template`: Optional template name

### Luna Random Int Range / Float Range

Standalone numeric generation nodes.

**Luna Random Int Range:**
- `min_value`, `max_value`: Range bounds
- `seed`: For reproducibility

**Luna Random Float Range:**
- `min_value`, `max_value`: Range bounds
- `step`: Resolution (0.1, 0.05, etc.)
- `seed`: For reproducibility

---

## 📋 Complete Example

### YAML File: `character.yaml`

```yaml
# Character generation wildcards

species:
  human:
    - woman
    - man
    - girl
    - boy
  fantasy:
    - elf
    - fairy
    - vampire
    - demon

attributes:
  positive:
    - beautiful
    - elegant
    - mysterious
    - powerful
  age:
    - young
    - mature
    - ancient

poses:
  standing:
    - standing confidently
    - leaning against wall
    - arms crossed
  sitting:
    - sitting elegantly
    - lounging casually

templates:
  basic:
    - "[attributes.positive] [species.human]"
    - "[attributes.positive] [species.fantasy] [attributes.age]"
  full:
    - "[attributes.positive] [species.human], [poses.standing], detailed"
    - "a [attributes.age] [species.fantasy], [poses.sitting], fantasy art"
```

### Workflow Usage

```
Prompt: "{character:templates.full}, {hair:colors.fantasy} hair, {clothing:tops.formal}"

Possible outputs:
- "beautiful woman, standing confidently, detailed, purple hair, blazer"
- "a ancient vampire, lounging casually, fantasy art, silver hair, dress shirt"
```

---

## 💡 Tips & Best Practices

### 1. Organize by Category
Keep related items in the same file:
- `clothing.yaml` - All clothing items
- `hair.yaml` - Hair styles and colors
- `poses.yaml` - Character poses
- `settings.yaml` - Locations and backgrounds

### 2. Use Templates for Common Patterns
Instead of repeating complex patterns, define templates:

```yaml
templates:
  portrait:
    - "portrait of [subject], [lighting], [style]"
  full_body:
    - "full body shot of [subject], [pose], [setting]"
```

### 3. Combine with LoRA Connections
Link wildcards to LoRAs in `connections.json`:

```json
{
  "loras": {
    "detailed_eyes_v1.safetensors": {
      "triggers": ["detailed eyes", "beautiful eyes"],
      "yaml_paths": ["eyes.detailed", "attributes.eye_quality"]
    }
  }
}
```

### 4. Use Numeric Ranges for Variation
Add subtle variation with float ranges:

```
Quality: {0.6-1.0:0.1} detailed, {0.5-0.9:0.1} realistic
```

### 5. Recursive Resolution
Wildcards can contain other wildcards:

```yaml
# In one file
outfit:
  - "{tops.casual} with {bottoms}"
  - "{clothing:full_outfit}"  # References another file
```

---

## 🔍 Troubleshooting

### Wildcard Not Resolving

1. **Check file exists**: Ensure the YAML file is in `models/wildcards/`
2. **Check path**: Verify the path matches your YAML structure
3. **Check syntax**: Use `{file:path}` not `{file.path}`

### Empty Results

1. **Check for empty arrays**: Ensure lists have items
2. **Check nesting**: Make sure path leads to a list, not a dict

### Seed Not Working

- Each wildcard in a prompt uses incrementing seeds
- Use the same base seed for reproducible results
- Batch nodes increment seed per variation

---

## 📚 Related Guides

- [LoRA Connections Guide](lora_connections.md) - Link wildcards to LoRAs
- [Performance Optimization](performance.md) - Caching and speed tips
- [Node Reference](node_reference.md) - Complete parameter documentation
