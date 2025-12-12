"""
Luna VLM Prompt Generator - Vision-Language Model Text Generation

This node uses a VLM (like Qwen3-VL) to generate prompts from images,
extract style information, or create training captions.

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Luna VLM Prompt Generator                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUTS:                                                                    │
│    llm: LLM (from Model Router - Qwen3-VL reference)                       │
│    image: IMAGE (optional, for vision tasks)                               │
│    mode: [describe | extract_style | caption | custom]                     │
│    custom_prompt: STRING                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  SETTINGS:                                                                  │
│    max_tokens: INT (default: 256)                                          │
│    temperature: FLOAT (default: 0.7)                                       │
│    seed: INT                                                                │
│    keep_model_loaded: BOOLEAN (default: True)                              │
│         ↳ If False, unloads decoder weights after generation               │
│         ↳ Keeps text encoder loaded for CLIP conditioning                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUTPUTS:                                                                   │
│    prompt: STRING (generated text)                                         │
│    style_tags: STRING (extracted style elements, comma-separated)          │
└─────────────────────────────────────────────────────────────────────────────┘

Preset Modes:
=============
  describe:      Detailed description of image content
  extract_style: Extract artistic style, lighting, colors, mood
  caption:       Training-ready caption (booru-style tags + natural language)
  custom:        Use your own prompt template
"""

from __future__ import annotations

import os
import gc
from typing import TYPE_CHECKING, Tuple, Optional, Any, Dict

import torch
import numpy as np

try:
    import folder_paths
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False
    folder_paths = None

# Daemon support
try:
    from ...luna_daemon import client as daemon_client
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    daemon_client = None


# =============================================================================
# PRESET PROMPTS
# =============================================================================

PRESET_PROMPTS = {
    "describe": """Describe this image in detail. Include:
- Main subject and their appearance
- Actions or poses
- Setting and environment
- Notable objects
- Overall mood and atmosphere

Be specific and detailed but concise.""",

    "extract_style": """Analyze the artistic style of this image. Extract:
- Art style (realistic, anime, painterly, etc.)
- Color palette and dominant colors
- Lighting style and direction
- Mood and atmosphere
- Any notable techniques or effects

Format your response as comma-separated tags followed by a brief description.""",

    "caption": """Create a training caption for this image suitable for AI image generation.

Format: Start with the most important subject/character tags, then add descriptive elements.
Use a mix of booru-style tags (underscores for multi-word concepts) and natural language.
Be specific about: character features, clothing, pose, expression, setting, lighting, style.

Example format:
1girl, long_blonde_hair, blue_eyes, school_uniform, sitting, window, soft_lighting, anime style, detailed background""",

    "custom": ""  # Placeholder, will be replaced with user input
}


# =============================================================================
# Model Cache for keep_model_loaded
# =============================================================================

_loaded_models: Dict[str, Any] = {}


class LunaVLMPromptGenerator:
    """
    Generate and enhance text prompts using a Vision-Language Model.
    
    Uses the LLM output from LunaModelRouter (Qwen3-VL or similar) to:
    - Generate descriptive prompts from images
    - Extract artistic styles
    - Create training captions
    - Enhance simple text prompts with LLM refinement
    """
    
    CATEGORY = "Luna"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "style_tags")
    FUNCTION = "generate"
    
    MODES = ["describe", "extract_style", "caption", "enhance_prompt", "custom"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("LLM", {
                    "tooltip": "LLM reference from Luna Model Router"
                }),
                "mode": (cls.MODES, {
                    "default": "describe",
                    "tooltip": "Preset prompt mode: describe (image), extract_style (image), caption (image), enhance_prompt (text), or 'custom'"
                }),
            },
            "optional": {
                # === Vision Inputs ===
                "image": ("IMAGE", {
                    "tooltip": "Image for vision tasks (required for describe/extract_style/caption)"
                }),
                
                # === Text Inputs ===
                "simple_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Simple prompt to enhance. Used when mode is 'enhance_prompt' or with vision modes"
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt template when mode is 'custom'. Use {image} placeholder for image context."
                }),
                
                # === Generation Settings ===
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 16,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Maximum tokens to generate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (higher = more creative)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffff,
                    "tooltip": "Random seed (-1 for random)"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep LLM decoder in memory after generation. False = unload to save VRAM (encoder stays for CLIP)"
                }),
            }
        }
    
    def generate(
        self,
        llm: Dict[str, Any],
        mode: str,
        image: Optional[torch.Tensor] = None,
        simple_prompt: str = "",
        custom_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        seed: int = -1,
        keep_model_loaded: bool = True
    ) -> Tuple[str, str]:
        """
        Generate or enhance text using the VLM.
        
        Returns:
            (prompt, style_tags): Generated text and extracted style elements
        """
        
        # Validate LLM reference
        if not isinstance(llm, dict) or "model_path" not in llm:
            raise ValueError("Invalid LLM reference. Use the 'llm' output from Luna Model Router.")
        
        # Build prompt template
        if mode == "custom":
            if not custom_prompt:
                raise ValueError("mode='custom' requires custom_prompt to be set")
            prompt_template = custom_prompt
        elif mode == "enhance_prompt":
            # Text-only enhancement mode
            if not simple_prompt:
                raise ValueError("mode='enhance_prompt' requires simple_prompt to be set")
            prompt_template = self._get_enhancement_prompt(simple_prompt)
            image = None  # Force no image for text-only mode
        else:
            # Vision modes (describe, extract_style, caption)
            if image is None and mode != "custom":
                raise ValueError(f"Mode '{mode}' requires an image input.")
            
            prompt_template = PRESET_PROMPTS[mode]
            
            # If simple_prompt is provided, blend it with the template
            if simple_prompt and mode in ["describe", "extract_style", "caption"]:
                prompt_template = f"{prompt_template}\n\nBase concept: {simple_prompt}"
        
        # Generate using appropriate backend
        use_daemon = llm.get("use_daemon", False)
        
        if use_daemon and DAEMON_AVAILABLE and daemon_client is not None:
            if daemon_client.is_daemon_running():
                try:
                    result = self._generate_daemon(
                        llm, prompt_template, image, max_tokens, temperature, seed
                    )
                    return self._parse_result(result, mode)
                except Exception as e:
                    print(f"[LunaVLMPromptGenerator] Daemon generation failed: {e}")
                    # Fall through to local
        
        # Local generation
        result = self._generate_local(
            llm, prompt_template, image, max_tokens, temperature, seed, keep_model_loaded
        )
        
        return self._parse_result(result, mode)
    
    @staticmethod
    def _get_enhancement_prompt(simple_prompt: str) -> str:
        """Generate prompt enhancement template for text refinement."""
        return f"""You are an expert prompt engineer for AI image generation.
Take the following simple prompt and enhance it into a detailed, rich description.
Improve clarity, add relevant details, and optimize for visual quality.
Preserve the original intent but make it more vivid and specific.
Return ONLY the enhanced prompt, no explanations.

Original prompt: {simple_prompt}"""
    
    def _generate_daemon(
        self,
        llm: Dict[str, Any],
        prompt: str,
        image: Optional[torch.Tensor],
        max_tokens: int,
        temperature: float,
        seed: int
    ) -> str:
        """Generate using Luna Daemon's Qwen3-VL."""
        
        # Convert image to numpy if provided
        img_data = None
        if image is not None:
            if len(image.shape) == 4:
                image = image[0]
            img_data = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # Call daemon
        result = daemon_client.vlm_generate(  # type: ignore
            prompt=prompt,
            image=img_data,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed if seed >= 0 else None
        )
        
        return result.get("text", "")
    
    def _generate_local(
        self,
        llm: Dict[str, Any],
        prompt: str,
        image: Optional[torch.Tensor],
        max_tokens: int,
        temperature: float,
        seed: int,
        keep_model_loaded: bool
    ) -> str:
        """Generate using local model loading."""
        
        model_path = llm["model_path"]
        mmproj_path = llm.get("mmproj_path")
        
        # Check cache
        cache_key = model_path
        
        if cache_key in _loaded_models:
            model_data = _loaded_models[cache_key]
            model = model_data["model"]
            processor = model_data["processor"]
        else:
            # Load model
            model, processor = self._load_model(model_path, mmproj_path)
            _loaded_models[cache_key] = {"model": model, "processor": processor}
        
        # Set seed
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Prepare inputs
        if image is not None:
            result = self._generate_with_image(model, processor, prompt, image, max_tokens, temperature)
        else:
            result = self._generate_text_only(model, processor, prompt, max_tokens, temperature)
        
        # Handle keep_model_loaded
        if not keep_model_loaded:
            self._unload_decoder(cache_key)
        
        return result
    
    def _load_model(self, model_path: str, mmproj_path: Optional[str]) -> Tuple[Any, Any]:
        """Load Qwen3-VL model and processor."""
        
        # Try GGUF first (llama-cpp-python)
        if model_path.endswith('.gguf'):
            return self._load_gguf_model(model_path, mmproj_path)
        
        # Safetensors via transformers
        return self._load_transformers_model(model_path)
    
    def _load_gguf_model(self, model_path: str, mmproj_path: Optional[str]) -> Tuple[Any, Any]:
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            chat_handler = None
            if mmproj_path and os.path.exists(mmproj_path):
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            
            model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=-1,  # Use all GPU layers
                chat_handler=chat_handler,
                verbose=False
            )
            
            print(f"[LunaVLMPromptGenerator] Loaded GGUF model: {os.path.basename(model_path)}")
            
            return model, None  # No separate processor for llama-cpp
            
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python required for GGUF models.\n"
                "Install with: pip install llama-cpp-python"
            )
    
    def _load_transformers_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load model using transformers."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
            
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print(f"[LunaVLMPromptGenerator] Loaded transformers model from: {model_dir}")
            
            return model, processor
            
        except ImportError:
            raise RuntimeError(
                "transformers required for safetensors models.\n"
                "Install with: pip install transformers"
            )
    
    def _generate_with_image(
        self,
        model: Any,
        processor: Any,
        prompt: str,
        image: torch.Tensor,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate text conditioned on an image."""
        from PIL import Image
        
        # Convert tensor to PIL
        if len(image.shape) == 4:
            image = image[0]
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # Check if GGUF model (llama-cpp)
        if hasattr(model, 'create_chat_completion'):
            # llama-cpp-python style
            import base64
            from io import BytesIO
            
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response["choices"][0]["message"]["content"]
        
        else:
            # Transformers style
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            
            # Decode only the generated tokens
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return processor.decode(generated, skip_special_tokens=True)
    
    def _generate_text_only(
        self,
        model: Any,
        processor: Any,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate text without an image."""
        
        if hasattr(model, 'create_chat_completion'):
            # llama-cpp
            response = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        
        else:
            # Transformers
            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return processor.decode(generated, skip_special_tokens=True)
    
    def _unload_decoder(self, cache_key: str) -> None:
        """
        Unload decoder weights to save VRAM.
        
        For Z-IMAGE, we keep the encoder loaded since it's needed for CLIP,
        but unload the LM head and decoder layers.
        """
        if cache_key not in _loaded_models:
            return
        
        model_data = _loaded_models[cache_key]
        model = model_data["model"]
        
        if hasattr(model, 'create_chat_completion'):
            # llama-cpp - full unload for now
            del _loaded_models[cache_key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[LunaVLMPromptGenerator] Unloaded GGUF model")
        
        else:
            # Transformers - could selectively unload decoder
            # For now, full unload
            del _loaded_models[cache_key]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[LunaVLMPromptGenerator] Unloaded transformers model")
    
    def _parse_result(self, result: str, mode: str) -> Tuple[str, str]:
        """
        Parse the generated text into prompt and style_tags.
        
        For extract_style mode, tries to separate tags from description.
        """
        result = result.strip()
        
        if mode == "extract_style":
            # Try to extract comma-separated tags from the beginning
            lines = result.split('\n')
            if lines:
                first_line = lines[0]
                # Check if first line looks like tags
                if ',' in first_line and len(first_line.split(',')) >= 3:
                    tags = first_line
                    description = '\n'.join(lines[1:]).strip()
                    return (description if description else result, tags)
            
            # Fallback: return full result as both
            return (result, result)
        
        else:
            # For other modes, tags are extracted from the result
            # Simple extraction: find comma-separated portions
            words = result.split(',')
            if len(words) >= 3:
                # Looks like tags are present
                tags = ', '.join(w.strip() for w in words[:10])  # First 10 tags
                return (result, tags)
            
            return (result, "")


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LunaVLMPromptGenerator": LunaVLMPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaVLMPromptGenerator": "Luna VLM Prompt Generator 🤖",
}
