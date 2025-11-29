# 🌙 Luna Collection Quick Reference Card

## 📋 Node Categories at a Glance

| Category | Nodes | Key Use Case |
|----------|-------|--------------|
| `Luna/Wildcards` | 7 | YAML-based dynamic prompts |
| `Luna/Loaders` | 8 | LoRA stacking, embeddings, checkpoints |
| `Luna/Detailing` | 4 | MediaPipe face/body inpainting |
| `Luna/Upscaling` | 3 | Simple to Ultimate SD upscale |
| `Luna/Shared` | 7 | Daemon VAE/CLIP sharing |
| `Luna/Preprocessing` | 10 | Prompt caching & optimization |
| `Luna/Connections` | 4 | Smart LoRA/embedding linking |
| `Luna/Utils` | 6 | Saving, captioning, parameters |

---

## 🌿 YAML Wildcard Syntax

```
{file}                    → Random from templates section
{file:path.to.items}      → Random from nested path
{file: text [path]}       → Inline template substitution
{1-10}                    → Random integer
{0.5-1.5:0.1}             → Random float with step
__legacy/wildcard__       → .txt wildcard fallback
```

**Example:**
```
a {character:species.fantasy} with {hair:colors.natural} hair
→ "a elf with blonde hair"
```

---

## 🔌 Key Node Connections

### Standard Workflow
```
[Checkpoint Loader] → model, clip, vae
                         ↓
[Luna LoRA Stacker] → lora_stack
                         ↓
[Apply LoRA Stack] → model, clip
                         ↓
[Luna YAML Wildcard] → text
                         ↓
[CLIP Text Encode] → conditioning
                         ↓
[KSampler] → latent → [VAE Decode] → image
```

### With Daemon (Multi-Instance)
```
[Checkpoint Loader] → model (UNet only)
                         ↓
[Luna YAML Wildcard] → text
                         ↓
[Luna Shared CLIP Encode] → pos, neg (via daemon)
                         ↓
[KSampler] → latent
                         ↓
[Luna Shared VAE Decode] → image (via daemon)
```

### Smart LoRA Matching
```
[Luna YAML Wildcard] → resolved_text
                         ↓
[Luna Smart LoRA Linker] ← model, clip
  ↓                      ↓
  → model, clip (with matched LoRAs applied)
```

---

## ⚡ Performance Tips

| Goal | Solution |
|------|----------|
| Share VAE/CLIP across instances | Start Luna Daemon |
| Faster prompt loading | Preprocess to safetensors |
| Reduce VRAM | Use quantized embeddings |
| Generate variations | Use batch nodes |
| Debug performance | Luna Performance Monitor |

---

## 🎯 Common Patterns

### Pattern 1: Randomized Character Generation
```
Luna YAML Wildcard:
  text: "a {appearance:age} {species:fantasy}, {hair:colors.fantasy} hair, {outfit:casual}"

Luna LoRA Randomizer:
  category: "character"
  count: 2

→ Combine with Luna Smart LoRA Linker
```

### Pattern 2: Batch Prompt Processing
```
Luna Wildcard Prompt Generator:
  pattern: "{subject}, {style}, {quality}"
  num_variations: 1000
  ↓
Luna Prompt Preprocessor:
  prompt_list_path: [output]
  quantize: True
  ↓
Luna Optimized Preprocessed Loader:
  enable_caching: True
```

### Pattern 3: Face Detailing Workflow
```
[Image] → Luna MediaPipe Detailer:
            detect_face: True
            confidence: 0.7
            ↓
         → mask, segs
            ↓
         [Inpaint with higher detail]
```

---

## 📁 File Locations

| File | Location |
|------|----------|
| YAML Wildcards | `models/wildcards/*.yaml` |
| connections.json | `models/wildcards/connections.json` |
| Preprocessed Prompts | `output/luna_prompts/` |
| Daemon Config | `luna_daemon/config.py` |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Wildcard not resolving | Check YAML file exists in `models/wildcards/` |
| Daemon not connecting | Run `.\scripts\start_daemon.ps1` |
| LoRA not matching | Check triggers in `connections.json` |
| Cache hit rate low | Increase `max_cache_size` |
| Out of VRAM | Enable `quantize_embeddings`, use daemon |

---

## 📚 Full Documentation

- [README](../README.md) - Overview
- [YAML Wildcards Guide](yaml_wildcards.md) - Complete syntax
- [LoRA Connections Guide](lora_connections.md) - Smart linking
- [Node Reference](node_reference.md) - All parameters
- [Performance Guide](performance.md) - Optimization
- [Daemon Setup](../luna_daemon/README.md) - Multi-instance
