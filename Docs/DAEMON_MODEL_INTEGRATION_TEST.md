# DaemonModel Integration Testing Guide

## Overview
This document guides testing of the complete DaemonModel architecture, which routes UNet inference through Luna Daemon with frozen weights for 60-70% VRAM reduction.

## Architecture Summary

```
ComfyUI (8-9 instances on 32GB GPU)
    ↓
Luna Model Router (load checkpoint)
    ↓
DaemonModel proxy (self-reference compatible)
    ↓
Luna Daemon (frozen UNet + transient LoRAs)
    ↓
Inference with per-request LoRA stack
```

## Components

### 1. Luna Model Router (`nodes/loaders/luna_model_router.py`)
- **load()** method: Returns DaemonModel proxy when daemon is running and enabled
- **_wrap_model_as_daemon_proxy()**: Creates proxy and registers model with daemon
- **_register_checkpoint_with_daemon()**: Tracks loaded checkpoints

### 2. DaemonModel Proxy (`luna_daemon/proxy.py`)
- **__call__(x, timesteps, context)**: Routes inference to daemon_client.model_forward()
- **add_lora(lora_name, model_strength, clip_strength)**: Appends to lora_stack
- **lora_stack**: List of (name, model_str, clip_str) tuples
- **ModelPatcher compatibility**: clone(), patch_model(), unpatch_model(), etc.

### 3. Luna Daemon Server (`luna_daemon/server.py`)
- **ModelRegistry.get_model()**: Load + freeze (eval + requires_grad=False)
- **ModelRegistry.model_forward()**: Execute with transient LoRA context
- **_apply_lora_transient()**: Per-request LoRA hooks (load → apply → restore)
- **Command handlers**: register_model_by_path, register_model, model_forward

### 4. Config Gateway (`nodes/workflow/luna_config_gateway.py`)
- **load_loras()**: Intelligent routing based on model type
- **DaemonModel.add_lora()** → Adds to proxy's lora_stack
- **DaemonCLIP.add_lora_by_name()** → Daemon loads CLIP LoRAs
- **Standard local** → comfy.sd.load_lora_for_models()

## Pre-Test Checklist

- [ ] Luna Daemon running: `python -m luna_daemon.server`
- [ ] ComfyUI running with daemon enabled (Luna Model Router available)
- [ ] VRAM monitor running: `python scripts/performance_monitor.py`
- [ ] Test model available (e.g., sdxl-1.0.safetensors)
- [ ] Test LoRAs available in models/loras/

## Test 1: Basic Model Loading with DaemonModel Proxy

### Steps
1. Start Luna Daemon in separate terminal
2. Open ComfyUI UI
3. Add Luna Model Router node
4. Select:
   - Model Source: "Diffusion Models" or "Workflow LoRA"
   - Model Name: Select a model (e.g., "SDXL 1.0")
   - Model Type: "SDXL"
   - Daemon Mode: "auto"
5. Connect output to Luna KSampler (or other sampler)
6. Check console logs

### Expected Results
```
[LunaModelRouter] ✓ Model registered with daemon: sdxl (7200.5 MB)
[LunaModelRouter] ✓ Using DaemonModel proxy - inference routes through daemon
[DaemonModel] Added LoRA '...' - stack size: 1
```

### Verify
- [ ] No AttributeError on _wrap_model_as_daemon_proxy
- [ ] Model registered message appears
- [ ] "DaemonModel proxy" message appears
- [ ] No VRAM spike during model loading

---

## Test 2: VRAM Usage Verification (Main Goal)

### Expected Baseline
- **Without daemon** (local): ~10.089 GB per instance
- **With daemon** (frozen): ~3.7 GB per instance
- **Savings**: 7.4 GB per instance × 8-9 instances = 59-67 GB capacity

### Setup
1. Run VRAM monitor in another terminal
2. Load model with daemon enabled
3. Run a quick workflow (5 steps)

### Monitor Output
```
[VRAM Monitor] Total: 32GB
[VRAM Monitor] Reserved: 9.2GB (DaemonModel proxy - no gradient overhead)
[VRAM Monitor] Allocated: 3.7GB (UNet working memory only)
[VRAM Monitor] Free: 19.1GB
```

### Compare Results
```
Local Model:
  Reserved: 15.3GB (includes gradient tracking)
  Allocated: 10.089GB
  
DaemonModel:
  Reserved: 9.2GB (frozen, no gradients)
  Allocated: 3.7GB
  
Savings: 5.1GB reserved + 6.4GB allocated = 11.5GB per instance
```

### Verify
- [ ] Reserved VRAM is < 10GB (shows freezing works)
- [ ] Allocated is ~3.7GB during inference
- [ ] No memory leaks between steps
- [ ] Can load 8-9 instances on 32GB GPU

---

## Test 3: LoRA Application with Transient Stack

### Setup
1. Add multiple LoRAs to workflow
2. Use Config Gateway to apply them
3. Monitor daemon console for transient LoRA hooks

### Workflow
```
Luna Model Router (DaemonModel proxy)
    ↓
Luna Config Gateway (extracts LoRAs from prompt or lora_stack)
    ↓
Add LoRA: "character_lora:0.8"
Add LoRA: "style_lora:0.9"
    ↓
Luna KSampler (calls model() with lora_stack in memory)
    ↓
Daemon receives: model_forward(x, timesteps, context, lora_stack=[...])
```

### Console Output Expected
```
[LunaConfigGateway] DaemonModel detected - UNet LoRAs via daemon
[LunaConfigGateway] LoRA 'character_lora' → daemon (model=0.8, clip=0.0)
[LunaConfigGateway] LoRA 'style_lora' → daemon (model=0.9, clip=0.0)
[DaemonModel] Added LoRA 'character_lora' (model_str=0.8), stack size: 1
[DaemonModel] Added LoRA 'style_lora' (model_str=0.9), stack size: 2

[Daemon] Applied LoRA 'character_lora' (strength=0.8)
[Daemon] Applied LoRA 'style_lora' (strength=0.9)
[Daemon] Restored original weights (transient LoRA context exit)
```

### Verify
- [ ] LoRAs appear in console logs
- [ ] Stack size increases correctly
- [ ] "Restored original weights" message appears
- [ ] Different LoRA stacks work without interference

---

## Test 4: Mixed Proxy Types (Advanced)

### Goal
Test Config Gateway's intelligent routing with mixed proxies.

### Scenario 1: DaemonModel + DaemonCLIP
```
Model: DaemonModel (frozen UNet on daemon)
CLIP: DaemonCLIP (frozen CLIP on daemon)
LoRA: Both handled by daemon
```

Expected:
```
[LunaConfigGateway] DaemonCLIP detected - CLIP LoRAs via daemon
[LunaConfigGateway] DaemonModel detected - UNet LoRAs via daemon
[LunaConfigGateway] LoRA 'example' → daemon (model=0.8, clip=0.8)
```

### Scenario 2: DaemonModel + Local CLIP
```
Model: DaemonModel (frozen UNet)
CLIP: Standard local (gradients enabled)
LoRA: UNet to daemon, CLIP to local
```

Expected:
```
[LunaConfigGateway] DaemonModel detected - UNet LoRAs via daemon
[LunaConfigGateway] LoRA 'example' - CLIP via local, UNet via daemon
```

### Verify
- [ ] Correct proxy detected in each case
- [ ] LoRAs routed to correct location
- [ ] No cross-contamination between proxy types
- [ ] Config Gateway handles all combinations

---

## Test 5: Concurrent Workflows (Integration)

### Setup
1. Start 3-4 ComfyUI instances on same 32GB GPU
2. Each connects to same Luna Daemon
3. Each loads same model (shared)
4. Each applies different LoRA stacks

### Command (PowerShell)
```powershell
# Terminal 1: Daemon
python -m luna_daemon.server

# Terminal 2-5: ComfyUI instances
$port = 8188; python main.py --listen 127.0.0.1 --port $port; $port += 1
# Repeat for ports 8189, 8190, 8191
```

### Workflow
- Each instance loads model via Luna Model Router
- Each uses different LoRA stack in Config Gateway
- All run workflows simultaneously
- Monitor VRAM and stability

### Expected Results
```
VRAM per instance:
  Instance 1 (8188): 3.7GB + 1.2GB (sampler working memory) = 4.9GB
  Instance 2 (8189): 3.7GB + 1.2GB = 4.9GB
  Instance 3 (8190): 3.7GB + 1.2GB = 4.9GB
  Instance 4 (8191): 3.7GB + 1.2GB = 4.9GB
  Daemon (shared model): 10.5GB (frozen UNet + VAE + CLIP)
  Total: ~32GB (4 × 4.9GB + 10.5GB)

No slowdown from concurrent requests
No LoRA cross-contamination between instances
No memory leaks over 100+ steps
```

### Verify
- [ ] All 4 instances load without errors
- [ ] VRAM usage below 32GB total
- [ ] No out-of-memory errors
- [ ] LoRAs don't leak between instances
- [ ] Image quality matches local-only setup

---

## Test 6: Error Handling & Fallbacks

### Test 6a: Daemon Not Running
```
Daemon Mode: "force_daemon"
Luna Daemon: Not running

Expected: Error message
[LunaModelRouter] Daemon mode is 'force_daemon' but Luna Daemon is not running!
[LunaModelRouter] Expected daemon at 127.0.0.1:19283
```

### Test 6b: Model Registration Fails
```
Daemon running but model path invalid

Expected: Fallback to local
[LunaModelRouter] ⚠ Model registration with daemon failed: ..., falling back to local
[LunaModelRouter] ⚠ Using local model
```

### Test 6c: LoRA Not Found
```
Config Gateway tries to apply non-existent LoRA

Expected: Skip with warning
[LunaConfigGateway] Warning: LoRA 'nonexistent' not found, skipping
```

### Verify
- [ ] Proper error messages appear
- [ ] Fallbacks work without crashes
- [ ] Daemon mode "auto" gracefully downgrades to local

---

## Test 7: Memory Leak Detection

### Setup
1. Run 50 steps in a loop
2. Monitor VRAM every 10 steps
3. Check for growth pattern

### Script
```python
import time
from server import PromptServer

for step in range(50):
    # Run workflow here
    prompt = {"1": {...}}
    PromptServer.instance.send("execution_start", {"value": 1.0})
    # Execute
    time.sleep(2)
    
    if step % 10 == 0:
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss / 1e9
        print(f"Step {step}: {rss:.2f} GB")
```

### Expected Pattern
```
Step 0: 3.7 GB
Step 10: 3.7 GB (stable)
Step 20: 3.7 GB (stable)
Step 30: 3.7 GB (stable)
Step 40: 3.7 GB (stable)
Step 50: 3.7 GB (stable - no growth)
```

### Verify
- [ ] VRAM stays constant
- [ ] No 0.1GB increases per step
- [ ] No unbounded growth

---

## Test 8: LoRA Weight Application (Correctness)

### Goal
Verify that LoRA weights actually affect output.

### Setup
1. Generate image with LoRA strength 0.0 (no effect)
2. Generate same image with LoRA strength 1.0 (full effect)
3. Visual inspection should show difference

### Expected
- LoRA at 0.0: Image matches base model
- LoRA at 0.5: Image shows some influence
- LoRA at 1.0: Image heavily influenced by LoRA
- LoRA stacking: Multiple LoRAs visible in result

### Verify
- [ ] LoRA strength 0.0 → baseline image
- [ ] LoRA strength 1.0 → noticeably different
- [ ] Strength values between 0-1 show proportional effect
- [ ] Multiple LoRAs combine visibly

---

## Known Issues & Debugging

### Issue 1: "DaemonModel has no source model and use_existing=False"
**Cause**: _wrap_model_as_daemon_proxy created proxy but didn't set use_existing properly

**Fix**:
```python
def _wrap_model_as_daemon_proxy(self, model, model_type, model_path):
    proxy = DaemonModel(source_model=model, model_type=model_type, use_existing=False)
    # ↑ source_model MUST be provided
```

### Issue 2: LoRAs not appearing in daemon console
**Cause**: lora_stack not passed to model_forward

**Fix**: Verify Config Gateway calls model.add_lora() for DaemonModel:
```python
if use_daemon_model:
    model.add_lora(lora_file, model_strength, clip_strength)  # ← Essential
```

### Issue 3: VRAM doesn't decrease with daemon
**Cause**: Model not frozen (requires_grad still True)

**Fix**: Verify server's get_model() freezes:
```python
model.eval()  # ← Required
for param in model.parameters():
    param.requires_grad = False  # ← Required
```

### Issue 4: Concurrent requests interfere
**Cause**: Transient LoRA context not properly isolated

**Fix**: Verify _apply_lora_transient restores ALL weights:
```python
finally:
    for layer_key, original_weight in original_weights.items():
        param.data = original_weight  # ← Must restore EVERY weight
```

---

## Test Execution Order

1. **Basic Loading** (Test 1) - Verify architecture works at all
2. **VRAM Verification** (Test 2) - Main goal, check if 10GB → 3.7GB
3. **LoRA Application** (Test 3) - Verify transient stacks work
4. **Mixed Proxies** (Test 4) - Edge cases in Config Gateway
5. **Concurrent Workflows** (Test 5) - Full integration test
6. **Error Handling** (Test 6) - Robustness
7. **Memory Leaks** (Test 7) - Long-term stability
8. **LoRA Correctness** (Test 8) - Visual verification

---

## Success Criteria

✅ **MUST PASS**:
- [ ] Model Router returns DaemonModel proxy (not source model)
- [ ] Config Gateway routes LoRAs to daemon
- [ ] VRAM per instance < 4GB (proves freezing works)
- [ ] Transient LoRAs don't leak between requests
- [ ] Can run 4 instances on 32GB without OOM

✅ **SHOULD PASS**:
- [ ] LoRA weights visually affect output
- [ ] No memory leaks over 50 steps
- [ ] Concurrent requests stable
- [ ] Mixed proxy scenarios work

⚠️ **KNOWN LIMITATIONS**:
- LoRA application currently simplified (TODO: proper low-rank decomposition)
- LoRA strength might not match exact local values (approximation)
- No distributed multi-GPU daemon yet (single daemon on single GPU)

---

## Running Tests

```bash
# Test 1: Basic loading
cd ComfyUI-Luna-Collection
python main.py  # ComfyUI
# Manually load model in UI

# Test 2: VRAM check
python scripts/performance_monitor.py &
# Load model via UI, check output

# Test 3-8: Workflow-based
# Create test workflows in assets/workflows/
# Load and execute via ComfyUI UI

# Integration test
pwsh scripts/start_server_workflow.ps1 -Port 8188  # Instance 1
pwsh scripts/start_server_workflow.ps1 -Port 8189  # Instance 2
pwsh scripts/start_server_workflow.ps1 -Port 8190  # Instance 3
# Load same model on all 3, check VRAM
```

---

## Continuation After Testing

Once all tests pass:

1. **Optimization**: Profile transient LoRA application, optimize weight restoration
2. **Quality**: Compare LoRA outputs with local-only version, fine-tune strength factors
3. **Documentation**: Add to NODES_DOCUMENTATION.md with usage guide
4. **Examples**: Create example workflows in assets/workflows/
5. **Production**: Mark as stable, announce in README

