# Luna Daemon v1.3 Architecture: Distributed LoRA System

## 1. Problem Definition: The "Silent LoRA"
**Current State:** `LunaCheckpointTunnel` registers the base CLIP model with the Daemon. When downstream nodes (like `LoraLoader`) attempt to patch the `DaemonCLIP` proxy, the operation fails or is ignored because the Daemon holds a frozen, shared model.
**Goal:** Enable transient, per-request LoRA application on the shared Daemon CLIP model without permanently mutating the weights, supporting high concurrency (7+ instances).

---

## 2. Common Infrastructure (Client & Proxy)
*These changes are required for BOTH methods.*

### A. Client Side (`proxy.py`: `DaemonCLIP`)
The Proxy object must act as a **State Container** rather than a Model Patcher.

1.  **Intercept `add_patches`:** Instead of raising `RuntimeError`:
    *   Clone the `DaemonCLIP` object.
    *   Extract the LoRA weights specific to CLIP (keys usually containing `te_` or `text_model`).
    *   Calculate a deterministic hash of these weights.
    *   **Upload:** Check if `daemon_client.has_lora(hash)` is true. If not, upload weights to Daemon.
    *   **Stack:** Append `{'hash': hash, 'strength': strength}` to `new_clip.lora_stack`.

2.  **Modify `encode` request:**
    *   When sending `clip_encode` to the Daemon, include the `lora_stack` from the proxy instance.

### B. Daemon Side (`server_v2.py`: Storage)
1.  **`LoRARegistry` Class:**
    *   **Storage:** A dictionary mapping `hash` -> ` {layer_key: tensor} `.
    *   **LRU Policy:** Implement a cleanup mechanism (TTL or Max Size) to evict unused LoRA tensors from VRAM.
    *   **Endpoints:** `upload_lora` and `check_lora_existence`.

---

## 3. Execution Strategy A: "The Ford F-150" (Stateful Locking)
*Reliable, standard PyTorch API, easier to debug. Uses Forward Hooks.*

**Concept:**
Treats the shared model as a "Typewriter." Workers take turns inserting their specific "LoRA Ribbon," typing, and removing it.

**Implementation Logic (`ModelWorker`):**
1.  **Acquire Lock:** `with self.lock:` (Serialized execution).
2.  **Context Manager (`TransientLoRAContext`):**
    *   Iterate through `request.lora_stack`.
    *   Retrieve weights from `LoRARegistry`.
    *   **Inject Hooks:** Use `module.register_forward_hook()` to inject logic: `output = (input @ W) + (input @ A @ B * strength)`.
3.  **Inference:** Run `self.model.encode(text)`.
4.  **Cleanup:** **CRITICAL.** Remove all hooks immediately in `finally` block.
5.  **Release Lock.**

**Pros:** Safe, works with all standard ComfyUI/PyTorch objects.
**Cons:** Serialized (one request at a time per GPU), potential state bleed if cleanup fails.

---

## 4. Execution Strategy B: "The Lamborghini" (Functional PyTorch)
*High-performance, stateless, true parallelism. Uses `torch.func`.*

**Concept:**
Separates Code from Data. Workers run the same code simultaneously using different "Virtual Parameter" sets.

**Implementation Logic (`ModelWorker`):**
1.  **Preparation (On Model Load):**
    *   Extract `base_params = dict(model.named_parameters())`.
    *   Extract `base_buffers = dict(model.named_buffers())`.
    *   (Optional) Move actual `model` container to `meta` device to act as a pure graph template.

2.  **Request Handling (No Lock Required):**
    *   **Virtual Weights:** Create a new dictionary `request_params`.
        *   If layer has LoRA: `request_params[key] = base_params[key] + (LoRA_A @ LoRA_B * strength)`.
        *   If no LoRA: `request_params[key] = base_params[key]` (Pointer copy, zero memory cost).
    
3.  **Inference:**
    *   Use `torch.func.functional_call(model, (request_params, base_buffers), input_tokens)`.
    *   This executes the model graph using the temporary `request_params` dictionary.

**Pros:** True concurrency (multiple workers encode simultaneously on the same GPU), zero side-effects.
**Cons:** Complex to implement with HuggingFace/Comfy objects (buffer management), requires `torch >= 2.0`.

---

## 5. Integration: The Toggle
**File:** `server_v2.py` -> `ModelWorker.process_clip_encode`

```python
def process_clip_encode(self, ...):
    mode = self.config.execution_mode # 'LOCKING' or 'FUNCTIONAL'
    
    if mode == 'LOCKING':
        with self.lock:
            with TransientLoRAContext(self.model, lora_stack, registry):
                return self.model.encode(...)
                
    elif mode == 'FUNCTIONAL':
        # Advanced Functional Path
        # 1. Compose virtual weights (W + Delta)
        # 2. functional_call(self.model, params, args)
        return self._functional_encode(lora_stack, ...)
```

## 6. Action Plan for Copilot
1.  **Update `proxy.py`** to implement `add_patches` and hash calculation.
2.  **Update `client.py`** to add `upload_lora` command.
3.  **Update `server_v2.py`** to add `LoRARegistry`.
4.  **Implement Strategy A** (Context Manager) first to ensure stability.
5.  **Implement Strategy B** (Functional) as an experimental path.