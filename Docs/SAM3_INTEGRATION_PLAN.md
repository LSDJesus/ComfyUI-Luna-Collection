# SAM3 Integration Plan

## Problem
Currently using the `sam3` Python package via `build_sam3_image_model()` which:
- Tries to download config.json from HuggingFace even with local checkpoints
- Requires authentication for gated models
- Has unclear parameter support (`config_path` doesn't exist)
- Not integrated with ComfyUI's model management

## Solution
Vendor the SAM3 model loading code from `comfyui-sam3` into Luna Collection for:
- ✅ Local-first loading (no HuggingFace dependency)
- ✅ Safetensors support
- ✅ ComfyUI model_management integration
- ✅ Simpler, cleaner API

---

## Code to Vendor

From `d:\AI\ComfyUI\custom_nodes\comfyui-sam3`:

### Core Files (Required)
```
nodes/sam3_lib/model_builder.py         # Model architecture + checkpoint loading
nodes/sam3_lib/sam3_video_predictor.py  # Main predictor class
nodes/sam3_lib/model/sam3_image.py      # Image segmentation interface
nodes/sam3_lib/model/sam3_image_processor.py  # Sam3Processor class
```

### Supporting Files (Required)
```
nodes/sam3_lib/model/
  ├── decoder.py          # Transformer decoder
  ├── encoder.py          # Transformer encoder  
  ├── vitdet.py           # Vision backbone
  ├── text_encoder_ve.py  # Text encoder
  ├── tokenizer_ve.py     # Tokenizer
  ├── maskformer_segmentation.py  # Segmentation head
  ├── memory.py           # Memory modules
  ├── geometry_encoders.py
  ├── position_encoding.py
  ├── model_misc.py
  └── utils/
      ├── misc.py
      └── sam2_utils.py
```

### Assets
```
nodes/sam3_lib/bpe_simple_vocab_16e6.txt.gz  # Tokenizer vocabulary
```

---

## What We Don't Need

### Exclude (Video/Interactive Features)
- `sam3_video_base.py` - Video tracking
- `sam3_tracker_*` - Video tracking
- `sam3_video_inference.py` - Video inference
- `perflib/` - Performance optimizations (optional)
- `sam/` - SAM1/SAM2 compatibility (we only need SAM3)

### Exclude (ComfyUI Node UI)
- `nodes/load_model.py` - We'll write our own
- `nodes/segmentation.py` - UI node
- `nodes/sam3_interactive.py` - Interactive UI
- `nodes/sam3_video_nodes.py` - Video nodes
- `web/*.js` - Frontend widgets

---

## Integration Points

### Luna Daemon Server (`luna_daemon/daemon_server.py`)

Replace current SAM3 loading:
```python
# OLD (using sam3 package)
from sam3.model_builder import build_sam3_image_model
sam3_model = build_sam3_image_model(
    bpe_path=str(bpe_path),
    checkpoint_path=checkpoint_path,
    device=device,
    load_from_HF=(checkpoint_path is None)
)
```

With:
```python
# NEW (using vendored code)
from luna_sam3.sam3_video_predictor import Sam3VideoPredictor
from luna_sam3.model.sam3_image_processor import Sam3Processor

# Build model with local checkpoint
video_predictor = Sam3VideoPredictor(
    checkpoint_path=checkpoint_path,
    bpe_path=bpe_path,
    enable_inst_interactivity=True,  # For point/box prompts
)

# Extract image interface
sam3_model = video_predictor.model.detector  # Sam3Image instance
processor = Sam3Processor(sam3_model)
```

### Directory Structure
```
ComfyUI-Luna-Collection/
├── luna_sam3/              # NEW: Vendored SAM3 code
│   ├── __init__.py
│   ├── model_builder.py
│   ├── sam3_video_predictor.py
│   ├── bpe_simple_vocab_16e6.txt.gz
│   └── model/
│       ├── __init__.py
│       ├── sam3_image.py
│       ├── sam3_image_processor.py
│       ├── decoder.py
│       ├── encoder.py
│       └── ... (supporting modules)
```

---

## Benefits

1. **Local-First**: No HuggingFace authentication issues
2. **Safetensors**: Native support (comfyui-sam3 already handles it)
3. **Clean API**: `Sam3VideoPredictor` + `Sam3Processor` pattern
4. **ComfyUI Integration**: Already designed for ComfyUI's model management
5. **Proven**: comfyui-sam3 is actively maintained and works

---

## Implementation Steps

1. ☐ Create `luna_sam3/` directory
2. ☐ Copy core model files from comfyui-sam3
3. ☐ Update imports (remove video/interactive dependencies)
4. ☐ Copy BPE vocabulary file
5. ☐ Update `luna_daemon/daemon_server.py` to use vendored code
6. ☐ Test SAM3 loading with local safetensors checkpoint
7. ☐ Remove dependency on `sam3` pip package

---

## Key Functions

### From `model_builder.py`
```python
def _load_checkpoint_file(checkpoint_path: str) -> dict:
    """Loads .pt or .safetensors, validates format"""
    
def build_sam3_video_model(
    checkpoint_path=None,
    bpe_path=None,
    has_presence_token=True,
    enable_inst_interactivity=False,
    ...
) -> Sam3VideoOnMultiGPU:
    """Main model builder function"""
```

### From `sam3_video_predictor.py`
```python
class Sam3VideoPredictor:
    def __init__(self, checkpoint_path, bpe_path, enable_inst_interactivity=False):
        self.model = build_sam3_video_model(...)
        
    # We only need the detector for image segmentation
    # video_predictor.model.detector -> Sam3Image instance
```

### From `sam3_image_processor.py`
```python
class Sam3Processor:
    def __init__(self, model):
        self.model = model  # Sam3Image instance
        
    def generate_masks_for_text(self, images, text, ...):
        """Text-based segmentation (our use case)"""
```

---

## Next Steps

Should I proceed with vendoring the code? This will:
1. Remove the unreliable `sam3` pip package dependency
2. Give us full control over SAM3 loading
3. Use proven ComfyUI-compatible code
4. Fix the config.json HuggingFace issue permanently
