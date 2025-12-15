"""
Luna Z-IMAGE Encoder - Unified prompt input, AI enhancement, and conditioning encoder

This node serves as the all-in-one prompt handler for Z-IMAGE workflows:
- Manual prompt input with CLIP encoding
- Optional AI-powered prompt enhancement using Qwen3-VL
- Optional vision-guided prompt generation (describe/refine from image)
- Optional conditioning noise injection for seed variability
- Outputs both CONDITIONING and prompt text for metadata

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Luna Z-IMAGE Encoder                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUTS:                                                                    │
│    clip: CLIP (Qwen3-VL from Model Router - full model for encoding)       │
│    prompt: STRING (multiline text input)                                   │
│    ─────────────────────────────────────────────────────────────────────── │
│    enable_ai_enhancement: BOOLEAN (toggle AI prompt refinement)            │
│    enhancement_mode: [refine | expand | style_boost | custom]              │
│    ─────────────────────────────────────────────────────────────────────── │
│    enable_vision: BOOLEAN (toggle image-based prompt generation)           │
│    image: IMAGE (optional, for vision tasks)                               │
│    vision_mode: [describe | extract_style | blend_with_prompt]             │
│    ─────────────────────────────────────────────────────────────────────── │
│    enable_noise_injection: BOOLEAN (toggle conditioning noise)             │
│    noise_threshold: FLOAT (0.0-1.0, when to apply noise)                   │
│    noise_strength: FLOAT (0-100, noise magnitude)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  GENERATION SETTINGS (when AI enhancement enabled):                         │
│    max_tokens: INT, temperature: FLOAT, seed: INT                          │
│    keep_model_loaded: BOOLEAN                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS:                                                                   │
│    positive: CONDITIONING (encoded prompt, optionally with noise)          │
│    negative: CONDITIONING (empty, for KSampler compatibility)              │
│    prompt_text: STRING (final prompt text for metadata/display)            │
│    status: STRING (processing status message)                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict, List

import torch
import numpy as np

# Import centralized path constants from Luna Collection
try:
    from __init__ import COMFY_PATH, LUNA_PATH
except (ImportError, ModuleNotFoundError, AttributeError):
    # Fallback: construct paths if Luna constants aren't available
    COMFY_PATH = None
    LUNA_PATH = None

try:
    import folder_paths
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None

try:
    import node_helpers
    HAS_NODE_HELPERS = True
except ImportError:
    HAS_NODE_HELPERS = False
    node_helpers = None

# llama-cpp-python for GGUF model support
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None
    Llava15ChatHandler = None
    HAS_NODE_HELPERS = False
    node_helpers = None


# =============================================================================
# AI ENHANCEMENT PROMPTS
# =============================================================================

ENHANCEMENT_PROMPTS = {
    "refine": """You are an expert prompt engineer for AI image generation.
Take the following prompt and refine it for better image generation results.
Keep the core concept but improve clarity, add relevant details, and optimize for visual quality.
Return ONLY the refined prompt, no explanations.

Original prompt: {prompt}""",

    "expand": """You are an expert prompt engineer for AI image generation.
Expand the following brief prompt into a detailed, rich description suitable for high-quality image generation.
Add specific details about lighting, composition, style, and atmosphere while preserving the original intent.
Return ONLY the expanded prompt, no explanations.

Original prompt: {prompt}""",

    "style_boost": """You are an expert prompt engineer for AI image generation.
Enhance the following prompt by adding artistic style descriptors, quality tags, and aesthetic improvements.
Focus on visual style, artistic techniques, and quality modifiers.
Return ONLY the enhanced prompt, no explanations.

Original prompt: {prompt}""",

    "custom": "{prompt}"  # Placeholder for custom instructions
}

VISION_PROMPTS = {
    "describe": """Describe this image in detail for AI image generation.
Include: subject, appearance, pose, setting, lighting, colors, mood, and style.
Format as a comma-separated prompt suitable for image generation.""",

    "extract_style": """Analyze the artistic style of this image.
Extract: art style, color palette, lighting, mood, and techniques.
Format as style tags and descriptors for image generation.""",

    "blend_with_prompt": """Analyze this image and combine its visual elements with the following prompt.
Create a unified description that merges the image's style/mood with the prompt's content.

User prompt: {prompt}

Return ONLY the blended prompt, no explanations."""
}


# =============================================================================
# MODEL CACHE
# =============================================================================

_generation_models: Dict[str, Any] = {}


# =============================================================================
# LUNA Z-IMAGE ENCODER NODE
# =============================================================================

class LunaZImageEncoder:
    """
    Unified Z-IMAGE prompt input, AI enhancement, and CLIP encoding node.
    
    Combines manual prompting, AI-powered enhancement, vision analysis,
    and conditioning noise injection into a single streamlined node.
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt_text", "status")
    FUNCTION = "encode"
    
    ENHANCEMENT_MODES = ["refine", "expand", "style_boost", "custom"]
    VISION_MODES = ["describe", "extract_style", "blend_with_prompt"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model from Luna Model Router (Qwen3-VL for Z-IMAGE)"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your prompt text. Used directly or as base for AI enhancement."
                }),
            },
            "optional": {
                # === AI Enhancement Section ===
                "enable_ai_enhancement": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable AI-powered prompt refinement using Qwen3-VL"
                }),
                "enhancement_mode": (cls.ENHANCEMENT_MODES, {
                    "default": "refine",
                    "tooltip": "How to enhance the prompt: refine, expand, or style_boost"
                }),
                "custom_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom AI instruction when enhancement_mode is 'custom'"
                }),
                
                # === Vision Section ===
                "enable_vision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable image-based prompt generation"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image for vision analysis (optional)"
                }),
                "vision_mode": (cls.VISION_MODES, {
                    "default": "describe",
                    "tooltip": "How to use the image: describe it, extract style, or blend with prompt"
                }),
                
                # === Noise Injection Section ===
                "enable_noise_injection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add noise to conditioning for seed variability (useful for batches)"
                }),
                "noise_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising timestep threshold: noise applied from 0 to threshold, clean from threshold to 1.0"
                }),
                "noise_strength": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Noise magnitude to add to conditioning tensors"
                }),
                
                # === Generation Settings ===
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Maximum tokens for AI generation"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (higher = more creative)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for AI generation and noise injection (-1 = random)"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep generation model in VRAM after use"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "batch_size_from_js": ("INT", {"default": 1}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, seed=-1, batch_size_from_js=1, **kwargs):
        """Ensure re-execution when seed or batch size changes."""
        return f"{seed}_{batch_size_from_js}"
    
    def encode(
        self,
        clip,
        prompt: str,
        enable_ai_enhancement: bool = False,
        enhancement_mode: str = "refine",
        custom_instruction: str = "",
        enable_vision: bool = False,
        image: Optional[torch.Tensor] = None,
        vision_mode: str = "describe",
        enable_noise_injection: bool = False,
        noise_threshold: float = 0.2,
        noise_strength: float = 10.0,
        max_tokens: int = 256,
        temperature: float = 0.7,
        seed: int = -1,
        keep_model_loaded: bool = True,
        unique_id: str = "",
        batch_size_from_js: int = 1,
    ) -> Tuple[List, List, str, str]:
        """
        Process prompt through optional AI enhancement, encode with CLIP,
        and optionally inject noise for seed variability.
        """
        
        status_parts = []
        final_prompt = prompt.strip()
        
        # Handle seed
        if seed == -1:
            seed = int(torch.randint(0, 0xffffffff, (1,)).item())
        
        # =================================================================
        # STEP 1: AI Enhancement (if enabled)
        # =================================================================
        if enable_ai_enhancement or (enable_vision and image is not None):
            try:
                final_prompt = self._ai_enhance(
                    prompt=final_prompt,
                    clip=clip,
                    enable_enhancement=enable_ai_enhancement,
                    enhancement_mode=enhancement_mode,
                    custom_instruction=custom_instruction,
                    enable_vision=enable_vision,
                    image=image,
                    vision_mode=vision_mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    keep_model_loaded=keep_model_loaded,
                )
                if enable_ai_enhancement:
                    status_parts.append(f"AI enhanced ({enhancement_mode})")
                if enable_vision and image is not None:
                    status_parts.append(f"Vision ({vision_mode})")
            except Exception as e:
                status_parts.append(f"AI enhancement failed: {str(e)[:50]}")
                print(f"[LunaZImageEncoder] AI enhancement error: {e}")
        
        # =================================================================
        # STEP 2: CLIP Encoding
        # =================================================================
        if not final_prompt:
            final_prompt = " "  # Prevent empty prompt issues
        
        # Use ComfyUI's standard CLIP encoding
        # For Z-IMAGE with Qwen3-VL, this uses the full model's text encoder
        tokens = clip.tokenize(final_prompt)
        positive_cond = clip.encode_from_tokens_scheduled(tokens)
        
        # Create empty negative conditioning (Z-IMAGE doesn't use negative prompts traditionally)
        empty_tokens = clip.tokenize("")
        negative_cond = clip.encode_from_tokens_scheduled(empty_tokens)
        
        status_parts.append("Encoded")
        
        # =================================================================
        # STEP 3: Noise Injection (if enabled)
        # =================================================================
        if enable_noise_injection and noise_strength > 0:
            positive_cond = self._inject_noise(
                conditioning=positive_cond,
                threshold=noise_threshold,
                strength=noise_strength,
                seed=seed,
                batch_size=batch_size_from_js,
            )
            status_parts.append(f"Noise injected (t={noise_threshold}, s={noise_strength})")
        
        status = " | ".join(status_parts)
        
        return (positive_cond, negative_cond, final_prompt, status)
    
    def _ai_enhance(
        self,
        prompt: str,
        clip: Any,
        enable_enhancement: bool,
        enhancement_mode: str,
        custom_instruction: str,
        enable_vision: bool,
        image: Optional[torch.Tensor],
        vision_mode: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        keep_model_loaded: bool,
    ) -> str:
        """
        Use Qwen3-VL for AI prompt enhancement or vision-based generation.
        Supports both transformers (safetensors) and llama-cpp-python (GGUF).
        """
        from PIL import Image as PILImage
        
        # Get or load model - returns different types based on model format
        model_or_llm, processor, tokenizer = self._get_generation_model(clip, keep_model_loaded)
        
        # Detect if this is GGUF (processor/tokenizer will be None)
        is_gguf = processor is None
        
        # Set seed
        torch.manual_seed(seed)
        
        # Build the prompt text
        if enable_vision and image is not None:
            if vision_mode == "blend_with_prompt" and prompt:
                prompt_text = VISION_PROMPTS["blend_with_prompt"].format(prompt=prompt)
            else:
                prompt_text = VISION_PROMPTS.get(vision_mode, VISION_PROMPTS["describe"])
                if prompt and vision_mode != "describe":
                    prompt_text += f"\n\nAdditional context: {prompt}"
        elif enable_enhancement and prompt:
            if enhancement_mode == "custom" and custom_instruction:
                prompt_text = custom_instruction.replace("{prompt}", prompt)
            else:
                template = ENHANCEMENT_PROMPTS.get(enhancement_mode, ENHANCEMENT_PROMPTS["refine"])
                prompt_text = template.format(prompt=prompt)
        else:
            prompt_text = prompt
        
        # Get image as PIL if needed
        pil_image = None
        if enable_vision and image is not None:
            pil_image = self._tensor_to_pil(image)
        
        if is_gguf:
            # GGUF generation via llama-cpp-python
            result = self._generate_gguf(
                llm=model_or_llm,
                prompt=prompt_text,
                image=pil_image,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            # Transformers generation
            result = self._generate_transformers(
                model=model_or_llm,
                processor=processor,
                tokenizer=tokenizer,
                prompt=prompt_text,
                image=pil_image,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        # Cleanup if not keeping loaded
        if not keep_model_loaded:
            self._clear_generation_model()
        
        return result.strip()
    
    def _generate_gguf(
        self,
        llm: Any,
        prompt: str,
        image: Optional[Any],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text using GGUF model via llama-cpp-python."""
        import base64
        import io
        
        # Build system prompt
        system_prompt = """You are an expert prompt engineer for AI image generation.
Respond only with the refined/generated prompt text, no explanations."""
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        if image is not None:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_url = f"data:image/png;base64,{img_base64}"
            
            messages.append({
                "role": "user",
                "content": [  # type: ignore
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Generate
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens if max_tokens > 0 else None,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9,
            stop=["</s>", "Human:", "Assistant:", "\n\n"],
        )
        
        return response['choices'][0]['message']['content'].strip()
    
    def _generate_transformers(
        self,
        model: Any,
        processor: Any,
        tokenizer: Any,
        prompt: str,
        image: Optional[Any],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text using transformers model."""
        # Build conversation
        conversation = [{"role": "user", "content": []}]
        
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": image})
        
        conversation[0]["content"].append({"type": "text", "text": prompt})
        
        # Process and generate
        chat = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        images = [image] if image else None
        processed = processor(text=chat, images=images, return_tensors="pt")
        
        model_device = next(model.parameters()).device
        model_inputs = {
            k: v.to(model_device) if torch.is_tensor(v) else v
            for k, v in processed.items()
        }
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        input_len = model_inputs["input_ids"].shape[-1]
        return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
    
    def _get_generation_model(self, clip: Any, keep_loaded: bool):
        """Get or load the Qwen3-VL generation model (supports both transformers and GGUF)."""
        global _generation_models
        
        # Find model path from CLIP object if available (set by LunaModelRouter)
        if hasattr(clip, "model_path") and clip.model_path:
            model_path = clip.model_path
            print(f"[LunaZImageEncoder] Using model from router: {model_path}")
        else:
            model_path = self._find_qwen_model()
            print(f"[LunaZImageEncoder] Using auto-detected model: {model_path}")
        
        # Get mmproj path if available
        mmproj_path = getattr(clip, "mmproj_path", None)
        
        # Check if this is a GGUF model
        is_gguf = model_path.lower().endswith(".gguf")
        
        if is_gguf:
            return self._get_gguf_generation_model(model_path, mmproj_path, keep_loaded)
        else:
            return self._get_transformers_generation_model(model_path, keep_loaded)
    
    def _get_gguf_generation_model(self, model_path: str, mmproj_path: Optional[str], keep_loaded: bool):
        """Load GGUF model with llama-cpp-python for generation."""
        global _generation_models
        
        if not HAS_LLAMA_CPP:
            raise RuntimeError(
                "llama-cpp-python is required for GGUF model generation.\n"
                "Install with: pip install llama-cpp-python\n"
                "For GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
        
        cache_key = f"gguf_generation_{model_path}"
        
        if cache_key in _generation_models:
            cached = _generation_models[cache_key]
            return cached["llm"], None, None  # GGUF uses different API
        
        print(f"[LunaZImageEncoder] Loading GGUF model: {model_path}")
        
        # Set up chat handler with mmproj for vision support
        chat_handler = None
        if mmproj_path and os.path.exists(mmproj_path):
            print(f"[LunaZImageEncoder] Loading mmproj: {mmproj_path}")
            chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
        else:
            print("[LunaZImageEncoder] No mmproj found - vision features disabled")
        
        # Determine GPU layers
        n_gpu_layers = -1  # All layers on GPU by default
        if torch.cuda.is_available():
            # Check available VRAM and adjust
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb < 8:
                n_gpu_layers = 20  # Partial offload for low VRAM
        
        # Load the model
        llm = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=4096,  # Context window
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=False,
        )
        
        if keep_loaded:
            _generation_models[cache_key] = {
                "llm": llm,
                "chat_handler": chat_handler,
                "model_path": model_path,
            }
        
        print(f"[LunaZImageEncoder] GGUF model loaded successfully")
        return llm, None, None
    
    def _get_transformers_generation_model(self, model_path: str, keep_loaded: bool):
        """Load model with transformers for generation."""
        global _generation_models
        
        cache_key = "qwen3_vl_generation"
        
        if cache_key in _generation_models:
            cached = _generation_models[cache_key]
            return cached["model"], cached["processor"], cached["tokenizer"]
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        except ImportError:
            raise RuntimeError("transformers package required")
        
        print(f"[LunaZImageEncoder] Loading transformers model: {model_path}")
        
        # Load with optimal settings
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        
        # Try flash attention if available
        try:
            import flash_attn
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            load_kwargs["attn_implementation"] = "sdpa"
        
        model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if keep_loaded:
            _generation_models[cache_key] = {
                "model": model,
                "processor": processor,
                "tokenizer": tokenizer,
            }
        
        return model, processor, tokenizer
    
    def _find_qwen_model(self) -> str:
        """Find Qwen3-VL model in standard locations."""
        if not HAS_COMFY or folder_paths is None:
            raise RuntimeError("ComfyUI folder_paths not available")
        
        # Build search paths with fallback to COMFY_PATH if available
        search_paths = []
        
        # Try COMFY_PATH if available (from centralized __init__.py)
        if COMFY_PATH:
            models_base = os.path.join(COMFY_PATH, "models")
        else:
            models_base = folder_paths.models_dir
        
        search_paths = [
            os.path.join(models_base, "LLM", "Qwen-VL"),
            os.path.join(models_base, "LLM"),
            os.path.join(models_base, "text_encoders"),
        ]
        
        qwen_patterns = ["Qwen3-VL", "Qwen2.5-VL", "Qwen2-VL"]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for item in os.listdir(search_path):
                for pattern in qwen_patterns:
                    if pattern.lower() in item.lower():
                        full_path = os.path.join(search_path, item)
                        if os.path.isdir(full_path):
                            # Verify it has model files
                            if any(f.endswith(('.safetensors', '.bin', 'config.json')) 
                                   for f in os.listdir(full_path)):
                                return full_path
        
        raise FileNotFoundError(
            "Qwen3-VL model not found. Please download to models/LLM/Qwen-VL/\n"
            "Recommended: Qwen3-VL-4B-Instruct from HuggingFace"
        )
    
    def _clear_generation_model(self):
        """Clear cached generation model to free VRAM."""
        global _generation_models
        
        # Clear transformers model
        cache_key = "qwen3_vl_generation"
        if cache_key in _generation_models:
            del _generation_models[cache_key]
            print("[LunaZImageEncoder] Cleared transformers model from VRAM")
        
        # Clear any GGUF models
        gguf_keys = [k for k in _generation_models.keys() if k.startswith("gguf_generation_")]
        for key in gguf_keys:
            cached = _generation_models[key]
            # Close the chat handler exit stack if present
            if "chat_handler" in cached and cached["chat_handler"] is not None:
                try:
                    if hasattr(cached["chat_handler"], '_exit_stack'):
                        cached["chat_handler"]._exit_stack.close()
                except Exception as e:
                    print(f"[LunaZImageEncoder] Error closing chat_handler: {e}")
            # Delete the LLM
            if "llm" in cached and cached["llm"] is not None:
                del cached["llm"]
            del _generation_models[key]
            print(f"[LunaZImageEncoder] Cleared GGUF model from VRAM")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor):
        """Convert ComfyUI image tensor to PIL Image."""
        from PIL import Image as PILImage
        
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return PILImage.fromarray(array)
    
    def _inject_noise(
        self,
        conditioning: List,
        threshold: float,
        strength: float,
        seed: int,
        batch_size: int,
    ) -> List:
        """
        Inject noise into conditioning for seed variability.
        Based on ConditioningNoiseInjection pattern.
        
        Noise is applied from timestep 0 to threshold,
        clean conditioning from threshold to 1.0.
        """
        
        def get_time_intersection(params, limit_start, limit_end):
            old_start = params.get("start_percent", 0.0)
            old_end = params.get("end_percent", 1.0)
            new_start = max(old_start, limit_start)
            new_end = min(old_end, limit_end)
            if new_start >= new_end:
                return 1.0, 0.0
            return new_start, new_end
        
        # CPU generator for reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        
        c_out = []
        
        for i, t in enumerate(conditioning):
            original_tensor = t[0]
            original_dict = t[1].copy()
            
            # Handle batch expansion
            current_batch = original_tensor.shape[0]
            target_batch = max(current_batch, batch_size)
            
            processing_tensor = original_tensor
            if current_batch == 1 and target_batch > 1:
                processing_tensor = original_tensor.repeat(target_batch, 1, 1)
            
            # Generate noise
            noise = torch.randn(
                processing_tensor.size(),
                generator=g,
                device="cpu"
            ).to(
                processing_tensor.device,
                dtype=processing_tensor.dtype
            )
            
            # Apply noise
            noisy_tensor = processing_tensor + (noise * strength)
            
            # Noisy part (0 -> threshold)
            s_noise, e_noise = get_time_intersection(original_dict, 0.0, threshold)
            if s_noise < e_noise:
                n_noisy = [noisy_tensor, original_dict.copy()]
                n_noisy[1]["start_percent"] = s_noise
                n_noisy[1]["end_percent"] = e_noise
                c_out.append(n_noisy)
            
            # Clean part (threshold -> 1.0)
            s_clean, e_clean = get_time_intersection(original_dict, threshold, 1.0)
            if s_clean < e_clean:
                n_clean = [processing_tensor, original_dict.copy()]
                n_clean[1]["start_percent"] = s_clean
                n_clean[1]["end_percent"] = e_clean
                c_out.append(n_clean)
        
        return c_out


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaZImageEncoder": LunaZImageEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaZImageEncoder": "Luna Z-IMAGE Encoder 🌙",
}
