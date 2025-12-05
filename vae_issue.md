Here is the comprehensive specification document for the v1.3 architecture. Save this as a markdown file (e.g., `_!LUNA_V1.3_SPEC.md`) and open it in VS Code to context-load your Copilot.

--- START OF FILE _!LUNA_V1.3_SPEC.md ---

# Luna Collection v1.3: "Split Daemon" & IPC Optimization Specification

---
### **CONTEXT FOR AI ASSISTANT**
**Project:** ComfyUI-Luna-Collection
**Current Version:** v1.2.0 (Socket-based VAE/CLIP offloading)
**Target Version:** v1.3.0
**Goal:** Maximize throughput and minimize latency for high-density generation on multi-GPU setups.
**Primary User Hardware:**
*   **Main PC:** RTX 5090 (Worker + VAE Daemon), RTX 3080 Ti (CLIP Daemon).
*   **Server PC:** RTX 3090 (Worker + VAE Daemon), RTX 3080 (CLIP Daemon).

---

### **1. Architectural Shift: The "Split Daemon" Strategy**

We are moving from a generic "Offload Everything" model to a resource-aware "Split Service" model to eliminate bottlenecks.

#### **A. Daemon A (The "Brain") - Secondary GPU**
*   **Role:** Dedicated **Shared CLIP Service**.
*   **Hosting:** Secondary GPU (e.g., RTX 3080 Ti).
*   **Logic:**
    *   Loads CLIP Text Encoders (approx. 2-3GB VRAM).
    *   Serves embeddings to all Workers.
    *   **Protocol:** Standard Socket (Pickle). Text embeddings are small; network latency is negligible.
*   **Benefit:** Frees ~3GB VRAM per worker instance on the Primary GPU.

#### **B. Daemon B (The "Pixel Factory") - Primary GPU**
*   **Role:** Dedicated **Shared VAE Service**.
*   **Hosting:** Primary GPU (Same silicon as ComfyUI Workers, e.g., RTX 5090).
*   **Logic:**
    *   Loads VAE (approx. 300MB + decoding overhead).
    *   Acts as a **Mutex/Queue** to prevent concurrent VAE decoding spikes.
    *   **Protocol:** **CUDA IPC (Zero-Copy)**.
*   **Benefit:** Eliminates the "Ping-Pong" serialization latency tax (~4s) observed in v1.2 while maintaining VRAM efficiency.

---

### **2. Technical Implementation: CUDA IPC (Inter-Process Communication)**

The critical upgrade is modifying `luna_daemon` to detect when the Client and Server share the same GPU index and switch from TCP serialization to Shared Memory handles.

#### **Core Logic flow:**

1.  **Handshake:** Client sends `get_gpu_id` request to Server.
2.  **Negotiation:**
    *   If `Client_GPU_ID != Server_GPU_ID`: Use **TCP Mode** (Current v1.2 logic).
    *   If `Client_GPU_ID == Server_GPU_ID`: Use **IPC Mode**.

#### **IPC Mode Protocol (Conceptual):**

**Server Side (`server.py`):**
```python
import torch.multiprocessing as mp

def handle_vae_decode(latent_tensor):
    # 1. Perform Decode
    pixel_tensor = vae.decode(latent_tensor)
    
    # 2. Prepare for IPC
    # Only works if tensor is in shared memory or CUDA memory
    pixel_tensor.share_memory_() 
    
    # 3. Get Handle (The "Note")
    # This is a lightweight reference, not the data
    # Note: Implementation details vary by OS/PyTorch version for CUDA handles
    # For CUDA tensors, we pass the device pointer/storage ref? 
    # Actually, for standard PyTorch multiprocessing, passing the tensor object 
    # through a specific reducer might be required, OR strictly using shared_memory logic.
    
    return pixel_tensor # The framework needs to handle the serialization of the handle, not the data.
```

*Refinement:* Since standard Python sockets cannot inherently pickle a CUDA IPC handle directly without `torch.multiprocessing`'s specific reducer, we may need to use `torch.multiprocessing.Queue` or a specific handle wrapper if we stick to raw sockets. 
*Alternative:* Since we are on Windows, we verify if `cudart` handles are exposed correctly.

**Refined Approach for Windows/ComfyUI Environment:**
Since raw socket pickling of CUDA tensors works but invokes the physical copy, we must ensure `torch.multiprocessing.reductions` are active, or manually pass the storage pointer and rebuild the tensor on the client side (Advanced).

**Simpler Optimization (Step 1):** 
Ensure the socket connection is NOT loopback TCP but a **Named Pipe** (Windows) or Unix Domain Socket (Linux) for lower overhead, though this doesn't solve the copy cost.

**Target Optimization (Step 2 - The Real Fix):**
Use `torch.distributed` or specific `SharedMemory` buffers.
*However, given the complexity constraint:*
If IPC proves too unstable on Windows, implement **Process Queuing** (The "Mutex" aspect) as the priority. Even with the copy cost, if the VAE is on the Main GPU, the copy is PCIe (Fast) vs Network Stack (Slow).

---

### **3. Required Code Changes**

#### **A. `luna_daemon/server.py`**
*   Add configuration flag: `--service-type` (`[full, clip_only, vae_only]`).
*   Implement a `RequestQueue` to serialize VAE requests (FIFO).
*   Add logic to check `Client-ID` headers to identify local vs remote requests.

#### **B. `nodes/luna_shared_vae.py`**
*   Update `LunaSharedVAEDecode`:
    *   Add input: `daemon_address` (to allow pointing to `127.0.0.1:8190` vs `:8191`).
    *   Logic to handle the "Handle vs Data" response if we achieve IPC.

#### **C. `luna_daemon/config.py`**
*   Allow definition of multiple Daemon instances on different ports (e.g., Port 19283 for CLIP, Port 19284 for VAE).

---

### **4. Immediate Action Plan for Copilot**

1.  **Analyze** `luna_daemon/server.py` to identify where the request processing loop exists.
2.  **Draft** a `Queue` system to wrap the `vae.decode` calls so that only one request processes at a time (The Mutex).
3.  **Prototype** a modification to `config.py` to allow launching two distinct server instances with different model loading behaviors (CLIP-only vs VAE-only).

--- END OF FILE _!LUNA_V1.3_SPEC.md ---