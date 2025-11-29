# LUNA-ImageGen: Standalone Image Generation Service

**Created:** 2025-11-11
**Version:** 1.0  
**Date:** November 11, 2025  
**Status:** Architecture Specification - Separate Service

---

## Table of Contents

1. [Overview](#overview)
2. [Service Architecture](#service-architecture)
3. [API Design](#api-design)
4. [ComfyUI → Custom Python Migration Strategy](#comfyui--custom-python-migration-strategy)
5. [LUNA-Narrates Integration](#luna-narrates-integration)
6. [Performance Optimization](#performance-optimization)
7. [Monetization Strategy](#monetization-strategy)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

### Architectural Decision: Separation of Concerns

**LUNA-ImageGen** is a **standalone microservice** for AI image generation, completely independent of LUNA-Narrates narrative engine.

**Why Separate Service:**

1. **Independent Scaling**
   - Image generation is GPU-intensive, narrative generation is CPU-intensive
   - Scale GPU workers independently from narrative workers
   - Different hardware requirements (A6000 vs CPU-only servers)

2. **Technology Isolation**
   - ComfyUI backend evolves rapidly, shouldn't affect narrative stability
   - Python image processing vs TypeScript/Python narrative service
   - Different dependency stacks (torch, diffusers, PIL vs asyncpg, FastAPI)

3. **Reusability**
   - Other projects can use LUNA-ImageGen API (marketplace opportunity)
   - Character Creator can be standalone product
   - Image generation as separate SaaS offering

4. **Development Velocity**
   - Experiment with ComfyUI workflows without touching narrative code
   - Migrate to custom Python modules incrementally
   - Hot-reload image generation without restarting narrative service

5. **Monetization Flexibility**
   - Charge per image generation separately from narrative credits
   - Different pricing tiers for image quality/speed
   - API access for third-party developers

---

## Service Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        LUNA-Narrates                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Narrative Engine (FastAPI)                               │ │
│  │  - Multi-agent pipeline                                   │ │
│  │  - Story orchestration                                    │ │
│  │  - Database management                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ▲                                    │
│                            │ HTTP API calls                     │
│                            │ (image generation requests)        │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LUNA-ImageGen API Gateway                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Service (Port 8002)                              │ │
│  │  - Request routing                                        │ │
│  │  - Queue management                                       │ │
│  │  - Cost tracking                                          │ │
│  │  - Authentication                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ▲                                    │
│                            │                                    │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐             │
│  │ ComfyUI   │    │  Custom   │    │  Hybrid   │             │
│  │ Backend   │    │  Python   │    │  Pipeline │             │
│  │ (Phase 1) │    │  Modules  │    │           │             │
│  │           │    │ (Phase 3) │    │           │             │
│  └───────────┘    └───────────┘    └───────────┘             │
│       │                 │                 │                    │
│       └─────────────────┼─────────────────┘                    │
│                         │                                      │
│                         ▼                                      │
│              ┌───────────────────────┐                         │
│              │   GPU Worker Pool     │                         │
│              │   - A6000 GPUs        │                         │
│              │   - Load balancing    │                         │
│              │   - Priority queues   │                         │
│              └───────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Technology Stack

#### Phase 1: ComfyUI Backend (Rapid Prototyping)

**Purpose:** Fast implementation, workflow experimentation

**Stack:**
- **ComfyUI Server:** Workflow execution engine
- **Python 3.10+:** API server
- **FastAPI:** REST API framework
- **Redis:** Job queue + caching
- **PostgreSQL:** Generation tracking, cost accounting
- **WebSocket:** Real-time progress updates

**Advantages:**
- ✅ Drag-and-drop workflow design
- ✅ 1000+ community nodes available
- ✅ Rapid prototyping (hours vs weeks)
- ✅ Visual debugging
- ✅ Easy to test new models/techniques

**Disadvantages:**
- ❌ ComfyUI overhead (~500ms latency per workflow)
- ❌ JSON workflow serialization inefficiency
- ❌ Memory overhead (node graph in memory)
- ❌ Limited error handling
- ❌ Harder to optimize (black-box nodes)

---

#### Phase 3: Custom Python Modules (Production Optimization)

**Purpose:** Maximum performance, full control

**Stack:**
- **PyTorch:** Direct model inference
- **Diffusers:** Hugging Face pipeline library
- **ControlNet:** Multi-conditioning (direct implementation)
- **Triton Inference Server:** GPU-optimized serving (optional)
- **ONNX Runtime:** Model optimization (optional)
- **Custom CUDA kernels:** Ultimate optimization (optional)

**Advantages:**
- ✅ 5-10x faster (no ComfyUI overhead)
- ✅ Direct model control (custom schedulers, attention mechanisms)
- ✅ Better error handling
- ✅ Memory optimization
- ✅ Profiling and optimization visibility

**Disadvantages:**
- ❌ Longer development time (weeks vs hours)
- ❌ Harder to prototype new techniques
- ❌ Manual workflow implementation

---

#### Hybrid Approach (Best of Both Worlds)

**Strategy:** Use ComfyUI for R&D, migrate proven workflows to custom Python

```python
# core/services/generation_router.py
class GenerationRouter:
    """Routes requests to ComfyUI or custom modules based on workflow maturity."""
    
    def __init__(self):
        self.comfyui_client = ComfyUIClient()
        self.custom_modules = {
            'character_expression': CustomExpressionGenerator(),
            'full_body_pose': CustomPoseGenerator(),
            'outfit_change': CustomOutfitChanger(),
            # More modules as they're migrated...
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Route to appropriate backend."""
        
        workflow_type = request.workflow_type
        
        # Check if custom module exists
        if workflow_type in self.custom_modules:
            # Use optimized custom module
            generator = self.custom_modules[workflow_type]
            return await generator.generate(request)
        
        # Fall back to ComfyUI for new/experimental workflows
        return await self.comfyui_client.generate(request)
```

**Migration Priority:**
1. **Character expression generation** (most common, highest volume)
2. **Full-body pose generation** (second most common)
3. **Outfit changes** (img2img workflows)
4. **Scene/location backgrounds** (less critical, can stay ComfyUI)
5. **Experimental features** (always ComfyUI until proven)

---

## API Design

### REST API Endpoints

#### 1. Character Pack Generation

```http
POST /v1/character-packs/prototypes
Content-Type: application/json
Authorization: Bearer {api_key}

{
  "character_design": {
    "gender": "female",
    "age_appearance": "young_adult",
    "face_shape": "oval",
    "eye_size": "large",
    "eye_color": "blue",
    "hair_length": "long",
    "hair_color": "blonde",
    // ... all design parameters
  },
  "art_style": "anime",
  "variant_count": 4,
  "resolution": "1536x1536"
}

Response 200 OK:
{
  "task_id": "proto_abc123",
  "status": "queued",
  "estimated_time_seconds": 30,
  "cost_usd": 0.02,
  "webhook_url": null  // Optional callback
}
```

**Progress Tracking:**
```http
GET /v1/tasks/{task_id}

Response 200 OK:
{
  "task_id": "proto_abc123",
  "status": "processing",  // queued, processing, completed, failed
  "progress": 0.75,  // 0.0 - 1.0
  "current_step": "Generating variant 3/4",
  "images_completed": [
    {"url": "https://cdn.luna.ai/proto_abc123_1.png", "index": 0},
    {"url": "https://cdn.luna.ai/proto_abc123_2.png", "index": 1},
    {"url": "https://cdn.luna.ai/proto_abc123_3.png", "index": 2}
  ],
  "eta_seconds": 8
}
```

---

```http
POST /v1/character-packs/expressions
Content-Type: application/json
Authorization: Bearer {api_key}

{
  "character_lora_id": "lora_abc123",  // From previous training
  "expression_tags": [
    "neutral_resting",
    "smile_gentle",
    "smirk_confident",
    // ... up to 24 expressions
  ],
  "art_style": "anime",
  "resolution": "1536x1536",
  "priority": "standard"  // standard, priority, urgent
}

Response 200 OK:
{
  "task_id": "expr_xyz789",
  "status": "queued",
  "estimated_time_seconds": 180,  // ~3 minutes for 20 expressions
  "cost_usd": 0.05,
  "expression_count": 20
}
```

---

#### 2. Full-Body Generation

```http
POST /v1/character-packs/full-body
Content-Type: application/json
Authorization: Bearer {api_key}

{
  "character_lora_id": "lora_abc123",
  "character_design": { /* same as prototypes */ },
  "pose_data": {
    "source": "preset",  // preset, custom_json, custom_3d
    "pose_id": "standing_confident",  // If preset
    "pose_json": null,  // If custom_json (3D bone data)
    "pose_image": null  // If custom_3d (base64 rendered image)
  },
  "outfit_prompt": "leather jacket, ripped jeans, combat boots, punk style",
  "art_style": "semi_realistic",
  "resolution": "1024x1536",
  "apply_detailing": true,
  "upscale_to_high_res": false
}

Response 200 OK:
{
  "task_id": "fullbody_def456",
  "status": "queued",
  "estimated_time_seconds": 12,
  "cost_usd": 0.008
}
```

---

#### 3. Outfit Change (Img2Img)

```http
POST /v1/character-packs/outfit-change
Content-Type: application/json
Authorization: Bearer {api_key}

{
  "base_image_url": "https://cdn.luna.ai/char_abc123_pose_1.png",
  "character_lora_id": "lora_abc123",
  "new_outfit_prompt": "elegant red dress, high heels, formal event",
  "denoise_strength": 0.4,
  "preserve_pose": true,
  "art_style": "anime"
}

Response 200 OK:
{
  "task_id": "outfit_ghi789",
  "status": "processing",
  "estimated_time_seconds": 8,
  "cost_usd": 0.004
}
```

---

#### 4. Scene/Location Background

```http
POST /v1/scenes/generate
Content-Type: application/json
Authorization: Bearer {api_key}

{
  "scene_description": "medieval tavern interior, wooden tables, fireplace, dim lighting, fantasy setting",
  "art_style": "anime",
  "resolution": "1920x1080",
  "camera_angle": "wide_shot",
  "mood": "cozy_warm"
}

Response 200 OK:
{
  "task_id": "scene_jkl012",
  "status": "queued",
  "estimated_time_seconds": 15,
  "cost_usd": 0.012
}
```

---

#### 5. LoRA Training

```http
POST /v1/loras/train
Content-Type: multipart/form-data
Authorization: Bearer {api_key}

training_images[]: [file1.png, file2.png, file3.png, file4.png]
lora_type: "character_identity"
lora_name: "Elara_Moonwhisper"
training_steps: 500  // Auto-calculated based on image count
base_model: "animagine_xl_v3"

Response 200 OK:
{
  "task_id": "lora_train_mno345",
  "status": "queued",
  "estimated_time_seconds": 60,
  "cost_usd": 0.0,  // Free (local training)
  "lora_id": "lora_abc123"  // Will be ready when task completes
}
```

---

### WebSocket API (Real-Time Progress)

```javascript
// Client-side WebSocket connection
const ws = new WebSocket('wss://imagegen.luna.ai/v1/tasks/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    task_id: 'proto_abc123',
    auth_token: 'Bearer xyz...'
  }));
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  
  // {
  //   "task_id": "proto_abc123",
  //   "status": "processing",
  //   "progress": 0.5,
  //   "current_step": "Generating variant 2/4",
  //   "image_url": "https://cdn.luna.ai/proto_abc123_2.png"  // New image completed
  // }
  
  updateProgressBar(update.progress);
  if (update.image_url) {
    displayImage(update.image_url);
  }
};
```

---

## ComfyUI → Custom Python Migration Strategy

### Phase 1: ComfyUI Foundation (Weeks 1-8)

**Goal:** Get all workflows functional in ComfyUI

**Deliverables:**
1. Character prototype generation workflow
2. Expression pack generation workflow
3. Full-body pose generation workflow (with ControlNets)
4. Outfit change workflow (img2img)
5. Scene background generation workflow
6. LoRA training workflow (HyperLoRA)

**Workflow Storage:**
```
luna-imagegen/
  comfyui_workflows/
    character_prototypes.json
    expression_pack.json
    full_body_pose.json
    outfit_change.json
    scene_background.json
    lora_training.json
```

**ComfyUI API Client:**
```python
# core/backends/comfyui_client.py
class ComfyUIClient:
    """Client for ComfyUI API with queue management."""
    
    def __init__(self, comfyui_url: str = "http://localhost:8188"):
        self.base_url = comfyui_url
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.ws_client = None
    
    async def queue_workflow(
        self,
        workflow_template: str,
        parameters: Dict
    ) -> str:
        """
        Queue a ComfyUI workflow with parameter substitution.
        
        Args:
            workflow_template: Path to workflow JSON (e.g., "expression_pack.json")
            parameters: Values to substitute (e.g., {"lora_path": "...", "expression_tag": "smile"})
        
        Returns:
            prompt_id: ComfyUI task ID
        """
        # Load workflow template
        workflow = self._load_workflow(workflow_template)
        
        # Substitute parameters
        workflow = self._substitute_parameters(workflow, parameters)
        
        # Queue in ComfyUI
        response = await self.http_client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow}
        )
        
        data = response.json()
        return data["prompt_id"]
    
    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 300
    ) -> Dict:
        """
        Wait for workflow completion and retrieve outputs.
        
        Returns:
            {
                "status": "success" | "failed",
                "images": [{"filename": "...", "url": "..."}],
                "execution_time_seconds": 12.5
            }
        """
        start_time = time.time()
        
        # Connect to WebSocket for progress
        async with websockets.connect(f"ws://{self.base_url}/ws") as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                if data["type"] == "execution_complete" and data["prompt_id"] == prompt_id:
                    # Retrieve output images
                    images = await self._get_output_images(prompt_id)
                    return {
                        "status": "success",
                        "images": images,
                        "execution_time_seconds": time.time() - start_time
                    }
                
                elif data["type"] == "execution_error" and data["prompt_id"] == prompt_id:
                    return {
                        "status": "failed",
                        "error": data["error"],
                        "execution_time_seconds": time.time() - start_time
                    }
                
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Workflow {prompt_id} exceeded timeout")
    
    async def _get_output_images(self, prompt_id: str) -> List[Dict]:
        """Retrieve generated images from ComfyUI."""
        response = await self.http_client.get(f"{self.base_url}/history/{prompt_id}")
        history = response.json()
        
        images = []
        for node_output in history[prompt_id]["outputs"].values():
            if "images" in node_output:
                for img in node_output["images"]:
                    filename = img["filename"]
                    
                    # Download image
                    img_response = await self.http_client.get(
                        f"{self.base_url}/view?filename={filename}"
                    )
                    
                    # Upload to CDN/storage
                    cdn_url = await self._upload_to_cdn(img_response.content)
                    
                    images.append({
                        "filename": filename,
                        "url": cdn_url
                    })
        
        return images
```

---

### Phase 2: Performance Profiling (Weeks 9-10)

**Goal:** Identify bottlenecks in ComfyUI workflows

**Metrics to Track:**
- Total execution time per workflow
- Time spent per node type
- Memory usage peaks
- GPU utilization
- Queue wait times

**Profiling Tool:**
```python
# core/utils/workflow_profiler.py
class WorkflowProfiler:
    """Profile ComfyUI workflows to identify optimization targets."""
    
    async def profile_workflow(self, workflow_type: str, iterations: int = 10):
        """Run workflow multiple times and collect metrics."""
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = self._get_gpu_memory()
            
            # Execute workflow
            await self.comfyui_client.queue_workflow(workflow_type, sample_params)
            
            end_time = time.time()
            end_memory = self._get_gpu_memory()
            
            results.append({
                "iteration": i,
                "execution_time": end_time - start_time,
                "peak_memory_mb": end_memory - start_memory,
                "timestamp": datetime.now()
            })
        
        # Analyze results
        avg_time = sum(r["execution_time"] for r in results) / len(results)
        p95_time = sorted([r["execution_time"] for r in results])[int(0.95 * len(results))]
        
        return {
            "workflow_type": workflow_type,
            "iterations": iterations,
            "avg_execution_time": avg_time,
            "p95_execution_time": p95_time,
            "results": results
        }
```

**Expected Findings:**
- ComfyUI overhead: 300-500ms per workflow
- CLIP text encoding: 50-100ms
- VAE decode: 200-300ms
- Model loading (if not cached): 2-5 seconds
- Actual inference: 5-8 seconds

**Optimization Targets:**
1. **Model caching** (eliminate 2-5s loading time)
2. **CLIP encoding** (pre-compute common prompts)
3. **VAE batching** (decode multiple images together)
4. **Workflow overhead** (custom Python eliminates 500ms)

---

### Phase 3: Custom Python Migration (Weeks 11-20)

**Goal:** Reimplement high-volume workflows in pure Python

**Migration Order:**
1. **Character Expression Generation** (highest volume, biggest impact)
2. **Full-Body Pose Generation** (second highest, complex ControlNet logic)
3. **Outfit Changes** (img2img, moderate complexity)
4. **Scene Backgrounds** (lower priority, can stay ComfyUI)

---

#### Example: Custom Expression Generator

```python
# core/generators/custom_expression_generator.py
import torch
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image
import numpy as np

class CustomExpressionGenerator:
    """
    Custom Python implementation of expression generation.
    Replaces ComfyUI workflow for 5-10x speed improvement.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load models once (cached in memory)
        self._load_models()
    
    def _load_models(self):
        """Load SDXL + ControlNet + LoRA (cached)."""
        
        # SDXL base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "animagine-xl-3.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        
        # Optimized scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # Enable optimizations
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_vae_tiling()
        
        # ControlNet (Canny)
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        print("✅ Models loaded and cached in GPU memory")
    
    async def generate(self, request: ExpressionRequest) -> Image:
        """
        Generate single expression image.
        
        Args:
            request: {
                "character_lora_path": "loras/char_abc123.safetensors",
                "expression_tag": "smile_gentle",
                "character_prompt": "1girl, blue eyes, blonde hair, ...",
                "expression_template_path": "templates/smile_gentle.png",
                "resolution": (1536, 1536),
                "seed": 12345
            }
        
        Returns:
            PIL Image (1536x1536)
        """
        
        # Load character LoRA
        self.pipe.load_lora_weights(request.character_lora_path)
        
        # Load expression template and extract Canny edges
        template_image = Image.open(request.expression_template_path)
        canny_image = self._extract_canny_edges(template_image)
        
        # Build prompt
        full_prompt = self._build_expression_prompt(
            character_prompt=request.character_prompt,
            expression_tag=request.expression_tag
        )
        
        # Generate with ControlNet
        output = self.pipe(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, worst quality, deformed",
            image=canny_image,
            controlnet_conditioning_scale=0.8,
            width=request.resolution[0],
            height=request.resolution[1],
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.Generator(device=self.device).manual_seed(request.seed)
        )
        
        generated_image = output.images[0]
        
        # Apply face detailing (optional but recommended)
        detailed_image = await self._apply_face_detailing(generated_image)
        
        return detailed_image
    
    def _extract_canny_edges(self, image: Image, low_threshold: int = 100, high_threshold: int = 200) -> Image:
        """Extract Canny edges from template image."""
        import cv2
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL
        edges_image = Image.fromarray(edges).convert("RGB")
        
        return edges_image
    
    def _build_expression_prompt(self, character_prompt: str, expression_tag: str) -> str:
        """Add expression-specific tags to character prompt."""
        
        expression_descriptions = {
            "neutral_resting": "neutral expression, calm, resting face",
            "smile_gentle": "gentle smile, soft expression, warm eyes",
            "smirk_confident": "confident smirk, cocky expression, one-sided smile",
            # ... more expressions
        }
        
        expression_desc = expression_descriptions.get(expression_tag, "neutral expression")
        
        return f"{character_prompt}, {expression_desc}, portrait, high quality, detailed face, masterpiece"
    
    async def _apply_face_detailing(self, image: Image) -> Image:
        """
        Two-pass face enhancement (optional).
        Uses img2img with low denoise to enhance facial details.
        """
        # Optional: implement face detection + cropping + enhancement
        # For now, return original (can add later)
        return image
    
    async def generate_batch(self, requests: List[ExpressionRequest]) -> List[Image]:
        """
        Generate multiple expressions in parallel (GPU batch processing).
        
        Speed: ~2-3x faster than sequential generation
        """
        # Diffusers supports batch generation natively
        batch_size = min(len(requests), 4)  # Limit by VRAM
        
        images = []
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            
            # Process batch (Diffusers handles parallelization internally)
            batch_images = await asyncio.gather(*[
                self.generate(req) for req in batch
            ])
            
            images.extend(batch_images)
        
        return images
```

**Performance Comparison:**

| Implementation | Time per Expression | Time for 20 Expressions | Overhead |
|----------------|---------------------|-------------------------|----------|
| ComfyUI Workflow | 8-10 seconds | 160-200 seconds (sequential) | ~500ms per call |
| Custom Python Module | 6-7 seconds | 120-140 seconds (sequential) | ~0ms |
| Custom Python + Batching | 6-7 seconds | 40-50 seconds (parallel batch) | ~0ms |

**Speed Improvement:** 3-4x faster with custom Python + batching

---

#### Example: Custom ControlNet Multi-Pipeline

```python
# core/generators/custom_pose_generator.py
class CustomPoseGenerator:
    """
    Custom implementation for full-body generation with multi-ControlNet.
    Handles OpenPose + Depth + Canny simultaneously.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._load_models()
    
    def _load_models(self):
        """Load SDXL + 3 ControlNets."""
        
        # SDXL base
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "animagine-xl-3.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load 3 ControlNets
        self.controlnet_openpose = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        print("✅ Multi-ControlNet pipeline ready")
    
    async def generate(self, request: PoseRequest) -> Image:
        """
        Generate full-body character with multi-ControlNet.
        
        Args:
            request: {
                "character_lora_path": "loras/char_abc123.safetensors",
                "character_prompt": "1girl, blue eyes, ...",
                "outfit_prompt": "leather jacket, jeans, combat boots",
                "openpose_map": Image (skeleton),
                "depth_map": Image (depth),
                "canny_map": Image (edges),
                "resolution": (1024, 1536),
                "seed": 12345
            }
        
        Returns:
            PIL Image (1024x1536 full-body)
        """
        
        # Load character LoRA
        self.pipe.load_lora_weights(request.character_lora_path)
        
        # Build full prompt
        full_prompt = f"{request.character_prompt}, {request.outfit_prompt}, full body, standing, entire body visible, detailed anatomy, masterpiece, best quality"
        
        # Apply multi-ControlNet
        # Note: Diffusers supports this via MultiControlNetModel
        from diffusers import MultiControlNetModel
        
        multi_controlnet = MultiControlNetModel([
            self.controlnet_openpose,
            self.controlnet_depth,
            self.controlnet_canny
        ])
        
        # Generate with all 3 ControlNets
        output = self.pipe(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, deformed, bad anatomy, bad hands",
            image=[request.openpose_map, request.depth_map, request.canny_map],
            controlnet_conditioning_scale=[0.8, 0.6, 0.4],  # Different weights
            width=request.resolution[0],
            height=request.resolution[1],
            num_inference_steps=35,
            guidance_scale=7.5,
            generator=torch.Generator(device=self.device).manual_seed(request.seed)
        )
        
        generated_image = output.images[0]
        
        # Optional: Apply detailing pass (face, hands, feet)
        detailed_image = await self._apply_detailing(generated_image)
        
        return detailed_image
```

---

### Phase 4: Benchmarking & Validation (Weeks 21-22)

**Goal:** Verify custom Python modules match/exceed ComfyUI quality

**Validation Tests:**
1. **Visual Quality Comparison**
   - Generate same prompt with ComfyUI vs Custom Python
   - Side-by-side comparison (A/B test with users)
   - CLIP similarity score (should be >0.95)

2. **Performance Benchmarks**
   - 100 generation test suite
   - Measure: avg time, p95 time, peak memory, GPU utilization
   - Target: 3-5x faster than ComfyUI

3. **Edge Case Testing**
   - Unusual character designs
   - Complex poses
   - Multi-character scenes
   - Error handling (invalid inputs)

**Success Criteria:**
- ✅ Visual quality: 95%+ match to ComfyUI
- ✅ Speed: 3x+ faster
- ✅ Memory: 20%+ lower peak usage
- ✅ Error rate: <1% failures

---

## LUNA-Narrates Integration

### WebUI Plugin Architecture

```typescript
// luna-narrates/webui/plugins/LUNAImageGenPlugin.tsx
import React from 'react';

interface LUNAImageGenConfig {
  apiUrl: string;  // "https://imagegen.luna.ai"
  apiKey: string;
  defaultArtStyle: string;
  enableAutoGeneration: boolean;
}

export class LUNAImageGenPlugin {
  private config: LUNAImageGenConfig;
  private httpClient: AxiosInstance;
  
  constructor(config: LUNAImageGenConfig) {
    this.config = config;
    this.httpClient = axios.create({
      baseURL: config.apiUrl,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  }
  
  /**
   * Generate character expression for conversational mode.
   * Called automatically by Dreamer agent after each turn.
   */
  async generateExpression(
    characterId: string,
    expressionTag: string,
    storyId: string
  ): Promise<string> {
    
    // Check expression bank first (cache)
    const cached = await this.checkExpressionBank(characterId, expressionTag);
    if (cached) {
      return cached.imageUrl;  // Instant return (0.1s)
    }
    
    // Generate new expression
    const response = await this.httpClient.post('/v1/character-packs/expressions', {
      character_lora_id: await this.getCharacterLoRA(characterId),
      expression_tags: [expressionTag],
      art_style: this.config.defaultArtStyle,
      resolution: "1536x1536"
    });
    
    const taskId = response.data.task_id;
    
    // Poll for completion (or use WebSocket)
    const result = await this.pollTask(taskId);
    
    // Store in expression bank
    await this.storeExpression(characterId, expressionTag, result.images[0].url);
    
    return result.images[0].url;
  }
  
  /**
   * Generate scene background for narrative mode.
   * Called when scene changes.
   */
  async generateScene(sceneDescription: string, storyId: string): Promise<string> {
    const response = await this.httpClient.post('/v1/scenes/generate', {
      scene_description: sceneDescription,
      art_style: this.config.defaultArtStyle,
      resolution: "1920x1080",
      mood: "auto"  // Derive from scene description
    });
    
    const taskId = response.data.task_id;
    const result = await this.pollTask(taskId);
    
    return result.images[0].url;
  }
  
  /**
   * Character Creator integration.
   * Opens Character Creator UI in modal/new tab.
   */
  async openCharacterCreator(storyId: string): Promise<CharacterPack> {
    // Open Character Creator UI (hosted by LUNA-ImageGen)
    const creatorUrl = `${this.config.apiUrl}/character-creator?story_id=${storyId}`;
    
    // Open in modal or new window
    const characterPack = await this.openModal(creatorUrl);
    
    // Once user completes creation, return character pack ID
    return characterPack;
  }
  
  /**
   * Poll task for completion.
   */
  private async pollTask(taskId: string, timeout: number = 120): Promise<any> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout * 1000) {
      const response = await this.httpClient.get(`/v1/tasks/${taskId}`);
      const task = response.data;
      
      if (task.status === 'completed') {
        return task;
      } else if (task.status === 'failed') {
        throw new Error(`Generation failed: ${task.error}`);
      }
      
      // Wait 1 second before polling again
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error(`Task ${taskId} timed out`);
  }
}
```

---

### Dreamer Agent Integration

```python
# luna-narrates/core/agents/dreamer.py
class DreamerAgent:
    """
    Async enrichment agent.
    Now calls LUNA-ImageGen API instead of local ComfyUI.
    """
    
    def __init__(self, imagegen_client: LUNAImageGenClient):
        self.imagegen_client = imagegen_client
    
    async def process_turn_enrichment(
        self,
        turn_data: TurnData,
        story_context: StoryContext
    ):
        """
        Async enrichment after turn generation.
        Non-blocking - runs in background.
        """
        
        # Task 1: Expression generation (if conversational mode)
        if story_context.format == "conversational":
            expression_tag = turn_data.expression_tag
            character_id = turn_data.character_id
            
            # Check expression bank
            cached = await self.check_expression_bank(character_id, expression_tag)
            if not cached:
                # Queue expression generation (async)
                await self.imagegen_client.generate_expression(
                    character_id=character_id,
                    expression_tag=expression_tag,
                    callback_webhook=f"https://narrates.luna.ai/webhooks/expression-ready/{turn_data.turn_id}"
                )
        
        # Task 2: Scene background (if scene changed)
        if turn_data.scene_changed:
            await self.imagegen_client.generate_scene(
                scene_description=turn_data.scene_description,
                callback_webhook=f"https://narrates.luna.ai/webhooks/scene-ready/{turn_data.turn_id}"
            )
        
        # Task 3: Character state update (always)
        await self.update_character_matrix(turn_data)
```

---

## Performance Optimization

### GPU Worker Pool

**Strategy:** Multiple GPU workers for parallel processing

```python
# core/workers/gpu_worker_pool.py
class GPUWorkerPool:
    """
    Manage multiple GPU workers for parallel generation.
    
    Architecture:
    - 4 GPU workers (A6000 GPUs)
    - Redis queue for task distribution
    - Priority queues (urgent > priority > standard)
    """
    
    def __init__(self, num_workers: int = 4):
        self.workers = [
            GPUWorker(worker_id=i, device=f"cuda:{i}")
            for i in range(num_workers)
        ]
        self.redis_queue = redis.Redis(host='localhost', port=6379)
    
    async def submit_task(self, task: GenerationTask, priority: str = "standard"):
        """Submit task to appropriate priority queue."""
        
        queue_name = f"imagegen:{priority}"  # imagegen:urgent, imagegen:priority, imagegen:standard
        
        task_json = task.json()
        await self.redis_queue.lpush(queue_name, task_json)
        
        return task.task_id
    
    async def start_workers(self):
        """Start all GPU workers."""
        await asyncio.gather(*[worker.start() for worker in self.workers])

class GPUWorker:
    """Single GPU worker that processes tasks from queue."""
    
    def __init__(self, worker_id: int, device: str):
        self.worker_id = worker_id
        self.device = device
        self.redis_queue = redis.Redis(host='localhost', port=6379)
        
        # Load models once
        self.generators = {
            'expression': CustomExpressionGenerator(device=device),
            'full_body': CustomPoseGenerator(device=device),
            'outfit_change': CustomOutfitChanger(device=device)
        }
    
    async def start(self):
        """Worker loop: poll Redis queue and process tasks."""
        
        print(f"🚀 GPU Worker {self.worker_id} started on {self.device}")
        
        while True:
            # Check priority queues in order
            for priority in ['urgent', 'priority', 'standard']:
                queue_name = f"imagegen:{priority}"
                
                # Pop task from queue (blocking with 1s timeout)
                task_json = await self.redis_queue.brpop(queue_name, timeout=1)
                
                if task_json:
                    task = GenerationTask.parse_raw(task_json[1])
                    
                    # Process task
                    try:
                        result = await self.process_task(task)
                        await self.mark_task_complete(task.task_id, result)
                    except Exception as e:
                        await self.mark_task_failed(task.task_id, str(e))
                    
                    break  # Go back to priority queue checking
            
            # Small sleep to prevent tight loop
            await asyncio.sleep(0.1)
    
    async def process_task(self, task: GenerationTask) -> GenerationResult:
        """Execute generation task."""
        
        generator = self.generators[task.workflow_type]
        
        if task.workflow_type == 'expression':
            image = await generator.generate(task.parameters)
        elif task.workflow_type == 'full_body':
            image = await generator.generate(task.parameters)
        elif task.workflow_type == 'outfit_change':
            image = await generator.generate(task.parameters)
        
        # Upload to CDN
        cdn_url = await self.upload_to_cdn(image)
        
        return GenerationResult(
            task_id=task.task_id,
            status="completed",
            images=[{"url": cdn_url}],
            execution_time_seconds=task.execution_time
        )
```

**Scaling Strategy:**
- **Single GPU:** 1 worker (baseline)
- **4 GPUs:** 4 workers (4x throughput)
- **8 GPUs:** 8 workers (8x throughput)
- **Cloud scaling:** Add/remove workers based on queue depth

---

### Caching Strategy

**Multi-Level Cache:**

1. **Expression Bank Cache** (PostgreSQL + CDN)
   - 85-90% hit rate for expressions
   - Instant retrieval (0.1s)

2. **Model Cache** (GPU VRAM)
   - Keep models loaded in memory
   - No reload latency (2-5s saved per generation)

3. **CLIP Embedding Cache** (Redis)
   - Pre-compute common character prompts
   - 50-100ms saved per generation

4. **ControlNet Map Cache** (Redis)
   - Cache pose → ControlNet maps
   - Reuse across outfit changes

```python
# core/caching/cache_manager.py
class CacheManager:
    """Multi-level caching for image generation."""
    
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
        self.db = DatabaseManager()
    
    async def get_expression(self, character_id: str, expression_tag: str) -> Optional[str]:
        """Check expression bank cache."""
        
        # Level 1: Redis (fastest, 0.01s)
        redis_key = f"expr:{character_id}:{expression_tag}"
        cached_url = await self.redis.get(redis_key)
        if cached_url:
            return cached_url.decode()
        
        # Level 2: PostgreSQL (fast, 0.05s)
        db_result = await self.db.query_expression_bank(character_id, expression_tag)
        if db_result:
            # Cache in Redis for next time
            await self.redis.setex(redis_key, 3600, db_result.image_url)  # 1 hour TTL
            return db_result.image_url
        
        # Cache miss - need to generate
        return None
    
    async def get_clip_embedding(self, prompt: str) -> Optional[torch.Tensor]:
        """Check CLIP embedding cache."""
        
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        redis_key = f"clip:{prompt_hash}"
        
        cached_embedding = await self.redis.get(redis_key)
        if cached_embedding:
            # Deserialize tensor
            embedding = torch.frombuffer(cached_embedding, dtype=torch.float16)
            return embedding
        
        return None
    
    async def cache_clip_embedding(self, prompt: str, embedding: torch.Tensor):
        """Store CLIP embedding in cache."""
        
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        redis_key = f"clip:{prompt_hash}"
        
        # Serialize tensor
        embedding_bytes = embedding.cpu().numpy().tobytes()
        
        await self.redis.setex(redis_key, 7200, embedding_bytes)  # 2 hour TTL
```

---

## Monetization Strategy

### Pricing Model

**API Credits System:**

| Generation Type | Credits Cost | USD Cost | Generation Time |
|-----------------|--------------|----------|-----------------|
| Character Prototype (4 images) | 20 credits | $0.02 | 30 seconds |
| Expression Pack (20 images) | 50 credits | $0.05 | 3 minutes |
| Full-Body Standard | 8 credits | $0.008 | 12 seconds |
| Full-Body Premium (4K) | 15 credits | $0.015 | 20 seconds |
| Outfit Change | 4 credits | $0.004 | 8 seconds |
| Scene Background | 12 credits | $0.012 | 15 seconds |
| LoRA Training | FREE | $0 | 60 seconds |

**Credit Packages:**
- **Starter:** 1,000 credits = $1.00 (50 expressions)
- **Creator:** 10,000 credits = $9.00 (10% discount)
- **Pro:** 50,000 credits = $40.00 (20% discount)
- **Studio:** 200,000 credits = $140.00 (30% discount)

**Character Pack Bundles** (one-time purchase):
- **Expression Pack:** $7.99 (includes generation + storage)
- **Full-Body Standard:** $24.99
- **Full-Body Premium:** $39.99
- **Ultimate Studio:** $69.99

**Subscription Plans** (optional):
- **Creator Pro:** $14.99/month - 15,000 credits/month + 3 character packs
- **Studio Unlimited:** $29.99/month - 40,000 credits/month + 5 character packs + commercial license

---

### B2B API Access

**Third-Party Developer Pricing:**
- **Indie:** $49/month - 50,000 credits + API access
- **Startup:** $199/month - 250,000 credits + priority queue + webhooks
- **Enterprise:** Custom pricing - unlimited credits + dedicated GPU workers + SLA

**Revenue Potential:**
- 1,000 B2B developers × $49/month = $49,000/month
- 100 startups × $199/month = $19,900/month
- 10 enterprise clients × $2,000/month = $20,000/month

**Total B2B Revenue:** ~$90,000/month = $1.08M/year

---

## Implementation Roadmap

### Phase 1: ComfyUI Foundation (Weeks 1-8)

**Weeks 1-2: Infrastructure Setup**
- [ ] Create separate luna-imagegen repository
- [ ] Set up FastAPI server (port 8002)
- [ ] Configure ComfyUI server integration
- [ ] Implement Redis job queue
- [ ] Set up PostgreSQL for cost tracking

**Weeks 3-4: Core Workflows**
- [ ] Character prototype generation workflow
- [ ] Expression pack generation workflow
- [ ] LoRA training workflow (HyperLoRA)

**Weeks 5-6: Advanced Workflows**
- [ ] Full-body pose generation (multi-ControlNet)
- [ ] Outfit change (img2img)
- [ ] Scene background generation

**Weeks 7-8: API & Integration**
- [ ] REST API endpoints (all 6 core endpoints)
- [ ] WebSocket progress tracking
- [ ] LUNA-Narrates plugin development
- [ ] Testing & debugging

**Deliverable:** Functional LUNA-ImageGen service with ComfyUI backend

---

### Phase 2: LUNA-Narrates Integration (Weeks 9-10)

**Week 9: Dreamer Agent Integration**
- [ ] Update Dreamer agent to call ImageGen API
- [ ] Implement expression bank caching
- [ ] Scene generation on scene change

**Week 10: WebUI Plugin**
- [ ] Character Creator modal integration
- [ ] Expression display in conversational mode
- [ ] Scene backgrounds in narrative mode

**Deliverable:** LUNA-Narrates fully integrated with ImageGen service

---

### Phase 3: Custom Python Migration (Weeks 11-20)

**Weeks 11-13: Expression Generator**
- [ ] Custom Python expression generator
- [ ] Batching support
- [ ] Performance benchmarking (3-5x faster)

**Weeks 14-16: Pose Generator**
- [ ] Custom multi-ControlNet implementation
- [ ] Full-body generation
- [ ] Detailing pass (face, hands, feet)

**Weeks 17-18: Outfit Changer**
- [ ] Custom img2img pipeline
- [ ] Outfit variation system

**Weeks 19-20: Validation & Optimization**
- [ ] Visual quality validation (95%+ match)
- [ ] Performance benchmarks (3x+ faster)
- [ ] Memory optimization (20%+ reduction)

**Deliverable:** Custom Python modules for high-volume workflows

---

### Phase 4: Production Launch (Weeks 21-24)

**Week 21: GPU Worker Pool**
- [ ] Implement multi-GPU worker pool
- [ ] Priority queue system
- [ ] Load balancing

**Week 22: Monitoring & Analytics**
- [ ] Grafana dashboards (generation metrics)
- [ ] Cost tracking per request
- [ ] Error rate monitoring

**Week 23: Documentation & Marketing**
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Developer guides
- [ ] Marketing materials

**Week 24: Launch**
- [ ] Beta testing with users
- [ ] Performance tuning
- [ ] Public launch

**Deliverable:** Production-ready LUNA-ImageGen service

---

## Conclusion

**LUNA-ImageGen as Standalone Service** is the optimal architecture:

### ✅ Technical Advantages
- **Independent scaling** (GPU workers vs CPU narrative engine)
- **Technology isolation** (ComfyUI updates don't break narrative service)
- **Performance optimization** (custom Python modules 3-5x faster)
- **Development velocity** (experiment freely without affecting core service)

### ✅ Business Advantages
- **Reusability** (other projects can use ImageGen API)
- **Monetization** (B2B API access, credit system)
- **Defensible moat** (custom ComfyUI nodes + optimized pipelines)
- **Scalability** (add GPU workers as demand grows)

### ✅ Migration Strategy
- **Phase 1:** ComfyUI backend (fast implementation, 8 weeks)
- **Phase 3:** Custom Python migration (3-5x performance boost, 10 weeks)
- **Hybrid approach:** Keep ComfyUI for R&D, use custom modules for production

### 📊 Expected Performance
- **Generation speed:** 6-8 seconds per image (ComfyUI) → 2-3 seconds (custom Python)
- **Expression pack:** 3 minutes (ComfyUI) → 40-50 seconds (custom Python + batching)
- **Cost per image:** $0.001-0.015 (98%+ profit margin)
- **Cache hit rate:** 85-90% for expressions (instant retrieval)

### 💰 Revenue Potential
- **B2C Character Packs:** $370K/year (conservative)
- **B2B API Access:** $1.08M/year (1,000+ developers)
- **Total:** $1.45M/year revenue, ~$15K compute cost = **$1.435M profit** (99% margin)

---

**Document Status:** Complete - Standalone Service Architecture  
**Awaiting:** Repository setup + Phase 1 implementation start

**Next Steps:**
1. Create `luna-imagegen` repository
2. Set up FastAPI server skeleton
3. Configure ComfyUI integration
4. Implement first workflow (character prototypes)
