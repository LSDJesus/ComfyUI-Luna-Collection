# ⚡ Performance Optimization Guide

Tips and techniques for getting the best performance from Luna Collection nodes.

---

## 🚀 Luna Daemon (Multi-Instance VRAM Sharing)

The Luna Daemon is the most impactful performance optimization for multi-GPU or multi-instance setups.

### What It Does

Instead of each ComfyUI instance loading its own VAE and CLIP models (~4-6GB VRAM each), the daemon loads them once on a dedicated GPU and serves all instances via socket.

**Without Daemon:**
```
Instance 1 (cuda:0): UNet + VAE + CLIP = ~16GB
Instance 2 (cuda:0): UNet + VAE + CLIP = ~16GB  ❌ Won't fit!
```

**With Daemon:**
```
Daemon (cuda:1): VAE + CLIP = ~6GB
Instance 1 (cuda:0): UNet only = ~10GB
Instance 2 (cuda:0): UNet only = ~10GB
Instance 3 (cuda:0): UNet only = ~10GB  ✅ All fit!
```

### Setup

1. **Configure daemon** (`luna_daemon/config.py`):
```python
SHARED_DEVICE = "cuda:1"  # GPU for shared models
VAE_PATH = "models/vae/sdxl_vae.safetensors"
CLIP_L_PATH = "models/clip/clip_l.safetensors"
CLIP_G_PATH = "models/clip/clip_g.safetensors"
```

2. **Start daemon**:
```powershell
.\scripts\start_daemon.ps1
```

3. **Use in workflow**:
- Add **Luna Daemon Config** node (optional, for dynamic model switching)
- Replace standard VAE Encode/Decode with **Luna Shared VAE Encode/Decode**
- Replace CLIP Text Encode with **Luna Shared CLIP Encode**

### Dynamic Model Switching

Use **Luna Daemon Config** to change models without restarting:

```
Luna Daemon Config:
  vae: "sdxl-vae-fp16-fix.safetensors"
  clip_l: "clip_l.safetensors"
  clip_g: "clip_g.safetensors"
  device: "cuda:1"
```

The daemon only reloads if the model actually changed.

### Presets

Use **Luna Daemon Model Switch** for quick changes:
- SDXL
- SDXL (FP16 VAE)
- Pony
- Illustrious
- SD 1.5
- Flux

---

## 💾 Embedding Cache

The Luna preprocessing system includes an LRU cache for CLIP embeddings.

### How It Works

1. First time a prompt is processed → encoded and cached
2. Same prompt again → retrieved from cache instantly
3. Cache evicts least-recently-used when full

### Configuration

```python
# In workflow, use Luna Cache Manager node
action: "set_max_size"
max_cache_size: 200  # Increase for larger prompt libraries
```

### Best Practices

1. **Enable caching** in Luna Optimized Preprocessed Loader
2. **Use preloading** for sequential workflows:
   ```
   preload_batch: 5  # Preload 5 adjacent prompts
   ```
3. **Monitor hit rate**:
   ```
   Luna Cache Manager → action: "get_stats"
   ```
   Aim for >70% hit rate

---

## 🔄 Prompt Preprocessing

For large prompt libraries, preprocess to safetensors for instant loading.

### Preprocessing Workflow

1. **Create prompt list** (`prompts.txt`):
   ```
   a beautiful woman, detailed face, realistic
   a handsome man, portrait, professional
   ...
   ```

2. **Run preprocessor**:
   ```
   Luna Prompt Preprocessor:
     clip: [your CLIP]
     prompt_list_path: "prompts.txt"
     batch_size: 50
     quantize_embeddings: True  # Saves VRAM
   ```

3. **Load in workflow**:
   ```
   Luna Optimized Preprocessed Loader:
     enable_caching: True
     preload_batch: 3
   ```

### Storage Optimization

| Option | Size | Quality |
|--------|------|---------|
| FP32 (default) | ~20KB/prompt | Full |
| FP16 (quantized) | ~10KB/prompt | Minimal loss |
| Compressed | ~5KB/prompt | Minimal loss |

Enable with:
```
quantize_embeddings: True
compression_level: 3  # gzip
```

---

## 🎯 YAML Wildcard Optimization

### File Organization

Keep files small and focused:
```
# Good: Small, focused files
clothing/
  tops.yaml      # 50 items
  bottoms.yaml   # 40 items
  shoes.yaml     # 30 items

# Bad: One massive file
clothing.yaml    # 500+ items, slow to parse
```

### Caching

YAML files are cached after first load. The cache persists for the session.

### Batch Processing

Use **Luna YAML Wildcard Batch** for generating variations:
```
text: "{character} wearing {clothing:tops}"
count: 100
```

This is faster than calling Luna YAML Wildcard 100 times.

---

## 🖼️ Upscaling Performance

### Tile-Based Processing

For large images, use tiled upscaling:

```
Luna Ultimate SD Upscale:
  tile_size: 512    # Smaller = less VRAM, slower
  tile_overlap: 64  # More overlap = better seams, slower
```

**Recommendations:**
| VRAM | Tile Size | Overlap |
|------|-----------|---------|
| 8GB  | 384       | 48      |
| 12GB | 512       | 64      |
| 24GB | 768       | 96      |

### Simple vs Advanced

- **Luna Simple Upscaler**: Fastest, no diffusion
- **Luna Advanced Upscaler**: Supersampling, moderate speed
- **Luna Ultimate SD Upscale**: Slowest, best quality

---

## 🎨 MediaPipe Optimization

### Detection Efficiency

Only enable detections you need:
```
detect_face: True
detect_hands: False   # Disable if not needed
detect_eyes: False
detect_mouth: False
```

Each additional target adds processing time.

### Confidence Threshold

Higher confidence = fewer false positives, faster:
```
confidence: 0.7   # Strict, fast
confidence: 0.3   # Lenient, slower
```

### Max Objects

Limit detections:
```
max_objects: 3  # Stop after 3 detections
```

---

## 📊 Performance Monitoring

### Luna Performance Monitor

Use to identify bottlenecks:

```
action: "get_report"

Output:
📊 Cache Performance:
   - Hit Rate: 78%
   - Cache Size: 85/100

⚡ Loading Performance:
   - Average Load Time: 0.023s
   - Fastest Load: 0.001s
   - Slowest Load: 0.156s

💾 Memory Usage:
   - Current Usage: 1847 MB
```

### Bottleneck Analysis

```
action: "analyze_bottlenecks"

Checks:
- Cache hit rate < 30% → Increase cache size
- Load time > 2s → Use preprocessing
- Memory > 3GB → Use quantized embeddings
```

---

## 🔧 General Tips

### 1. Reduce Redundant Encoding

❌ Bad:
```
[CLIP Text Encode] → positive
[CLIP Text Encode] → negative  # Same CLIP loaded twice
```

✅ Good:
```
[Luna Shared CLIP Encode] → positive + negative
```

### 2. Batch Operations

❌ Bad: Running 100 separate workflows

✅ Good: Using batch nodes:
- Luna YAML Wildcard Batch
- Luna Optimized Preprocessed Loader with preloading

### 3. Model Management

- Use Luna Daemon for shared VAE/CLIP
- Use Luna Checkpoint Loader to see model info
- Use Luna LoRA Stacker instead of multiple Apply LoRA nodes

### 4. Memory Management

```
# Clear caches periodically
Luna Cache Manager → action: "optimize_cache"

# Monitor memory
Luna Performance Monitor → action: "get_report"
```

---

## 📈 Benchmarks

Typical performance improvements:

| Optimization | Speed Improvement |
|--------------|-------------------|
| Luna Daemon (vs local VAE) | 10-20% faster |
| Preprocessed prompts (vs live) | 50-100x faster |
| Embedding cache (hit) | 100x faster |
| Batch wildcards (vs single) | 10-50x faster |
| Quantized embeddings | 50% less VRAM |

---

## 📚 Related Guides

- [Daemon Setup](../luna_daemon/README.md)
- [Node Reference](node_reference.md)
- [YAML Wildcards](yaml_wildcards.md)
