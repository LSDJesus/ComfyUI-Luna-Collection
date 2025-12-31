# Missing Methods Analysis - Old vs New Daemon

## Client Methods Comparison

### ✅ Already Implemented
- `is_running()` ✓
- `get_info()` ✓
- `set_attention_mode()` ✓
- `negotiate_ipc()` ✓
- `shutdown()` ✓
- `unload_models()` ✓ (as `unload_daemon_models()`)
- `register_checkpoint()` ✓ (stub)
- `unregister_checkpoint()` ✓ (stub)
- `register_vae()` ✓
- `register_clip()` ✓
- `register_vae_by_path()` ✓
- `register_clip_by_path()` ✓
- `vae_encode()` ✓
- `vae_decode()` ✓
- `has_lora()` ✓
- `upload_lora()` ✓
- `zimage_encode()` ✓
- `submit_async()` ✓

### ❌ Missing Client Methods

**LoRA Management:**
- `register_lora(lora_name, clip_strength)` - Register LoRA for CLIP application
- `get_lora_stats()` - Get LoRA cache statistics (we have `lora_cache_stats()`)
- `clear_lora_cache()` - Clear all cached LoRAs

**Z-IMAGE/Qwen3 Extended:**
- `zimage_encode_batch(texts)` - Batch text encoding for Z-IMAGE
- `register_qwen3(model_path, mmproj_path)` - Register Qwen3-VL model

**VLM/Vision Extended:**
- `describe_image(image, prompt)` - Image description via VLM
- `extract_style(image)` - Extract artistic style from image
- `llm_generate(prompt, max_tokens, temperature, top_p)` - LLM text generation

**IPC Extended:**
- `ipc_enabled()` - Check if IPC mode is active
- `_vae_encode_ipc()` - VAE encode via CUDA IPC
- `_vae_decode_ipc()` - VAE decode via CUDA IPC

**Model Proxy (REMOVED - Intentionally):**
- ~~`register_model()` - Register diffusion model~~ (using InferenceModeWrapper instead)
- ~~`register_model_by_path()` - Register model by path~~ (using InferenceModeWrapper instead)
- ~~`model_forward()` - Forward pass through model~~ (using InferenceModeWrapper instead)

## Daemon Server Handlers Comparison

### ✅ Already Implemented
- `ping` ✓
- `get_info` / `get_status` ✓
- `set_attention_mode` ✓
- `negotiate_ipc` ✓
- `shutdown` ✓
- `unload_models` ✓
- `vae_encode` ✓
- `vae_decode` ✓
- `clip_encode` ✓
- `clip_encode_sdxl` ✓
- `clip_tokenize` ✓
- `clip_encode_from_tokens` ✓
- `lora_cache_get` / `lora_cache_put` / `lora_cache_check` / `lora_cache_stats` ✓
- `zimage_encode` ✓
- `vlm_generate` ✓ (generic)
- `encode_vision` ✓
- `submit_async` ✓

### ❌ Missing Daemon Handlers

**LoRA Extended:**
- `register_lora` - Apply LoRA to CLIP with strength
- `lora_stats` - LoRA cache statistics (we have `lora_cache_stats`)
- `clear_loras` - Clear all LoRAs

**Z-IMAGE/Qwen3 Extended:**
- `register_qwen3` - Load Qwen3-VL model
- `qwen3_status` - Get Qwen3 model status

**VLM Extended:**
- `llm_generate` - LLM text generation
- `vlm_describe` - Image description (we have generic `vlm_generate`)

**IPC Extended:**
- `vae_encode_ipc` - VAE encode via CUDA IPC
- `vae_decode_ipc` - VAE decode via CUDA IPC

**Registration Handlers:**
- `register_vae` - Register VAE object
- `register_clip` - Register CLIP object  
- `register_vae_by_path` - Register VAE by path
- `register_clip_by_path` - Register CLIP by path
- `register_checkpoint` - Register checkpoint metadata
- `unregister_checkpoint` - Unregister checkpoint
- `has_lora` - Check LoRA cache (we have `lora_cache_check`)
- `upload_lora` - Upload LoRA weights (we have `lora_cache_put`)

**Model Proxy (REMOVED - Intentionally):**
- ~~`register_model` - Register diffusion model~~
- ~~`register_model_by_path` - Register model by path~~
- ~~`model_forward` - Model inference~~

**Async Tasks:**
- `save_images_async` - Async image saving (we handle in `submit_async`)

## Analysis

### Critical Missing Functionality

**1. LoRA CLIP Application:**
The old daemon had `register_lora()` which would apply LoRA to CLIP with a strength parameter. This is different from just caching LoRA weights.

**Use Case:** Apply LoRA transformations to CLIP for style/character conditioning
**Status:** Missing from new implementation
**Priority:** HIGH (if using LoRAs with daemon CLIP)

**2. Qwen3 Model Management:**
- `register_qwen3()` - Load Qwen3-VL model into daemon
- `qwen3_status()` - Check if Qwen3 is loaded

**Use Case:** Z-IMAGE and VLM nodes need Qwen3 model loaded
**Status:** Missing - worker pools don't handle model loading
**Priority:** HIGH (Z-IMAGE won't work without this)

**3. VLM Specific Methods:**
- `describe_image()` - Image captioning
- `extract_style()` - Style extraction
- `llm_generate()` - Text generation

**Use Case:** VLM nodes for image description and style analysis
**Status:** Partially implemented (generic `vlm_generate` exists)
**Priority:** MEDIUM (can wrap with vlm_generate)

### Non-Critical Missing

**4. CUDA IPC Methods:**
- `ipc_enabled()`, `_vae_encode_ipc()`, `_vae_decode_ipc()`

**Status:** IPC disabled in simplified daemon
**Priority:** LOW (performance optimization, not functional requirement)

**5. LoRA Cache Management:**
- `clear_lora_cache()` - Clear all LoRAs

**Status:** Missing convenience method
**Priority:** LOW (can restart daemon)

## Recommendations

### HIGH Priority - Must Fix

1. **Add LoRA CLIP Application:**
   ```python
   # Client
   def register_lora(self, lora_name: str, clip_strength: float = 1.0) -> dict
   
   # Daemon Handler
   elif cmd == "register_lora":
       # Apply LoRA to CLIP with strength
       # Store in active LoRAs list
   ```

2. **Add Qwen3 Model Loading:**
   ```python
   # Client  
   def register_qwen3(self, model_path: str, mmproj_path: Optional[str] = None) -> dict
   
   # Daemon Handler
   elif cmd == "register_qwen3":
       # Load Qwen3-VL model
       # Initialize Z-IMAGE CLIP
   ```

3. **Add Qwen3 Status Check:**
   ```python
   # Client
   def get_qwen3_status(self) -> dict
   
   # Daemon Handler
   elif cmd == "qwen3_status":
       # Return Qwen3 loaded status
   ```

### MEDIUM Priority - Nice to Have

4. **Add Image Description:**
   ```python
   # Client
   def describe_image(self, image: torch.Tensor, prompt: str = "Describe this image.") -> str
   
   # Can wrap vlm_generate with specific parameters
   ```

5. **Add Style Extraction:**
   ```python
   # Client  
   def extract_style(self, image: torch.Tensor) -> str
   
   # Can wrap vlm_generate with style-specific prompt
   ```

### LOW Priority - Optional

6. **Add LoRA Cache Clear:**
   ```python
   # Client
   def clear_lora_cache(self) -> dict
   
   # Daemon Handler
   elif cmd == "clear_loras":
       self.lora_cache.clear()
   ```

## Summary

**Must Implement:**
- LoRA CLIP application (`register_lora`)
- Qwen3 model loading (`register_qwen3`)
- Qwen3 status check (`qwen3_status`)

**Can Implement as Wrappers:**
- Image description (wrap `vlm_generate`)
- Style extraction (wrap `vlm_generate`)
- Batch Z-IMAGE encoding (loop `zimage_encode`)

**Intentionally Removed:**
- Model proxy methods (using InferenceModeWrapper instead)
- CUDA IPC methods (simplified daemon doesn't use IPC)

**Already Covered:**
- Registration methods exist as stubs (daemon loads from config)
- LoRA caching works (`lora_cache_get/put/check/stats`)
- Basic VLM/Vision support exists
