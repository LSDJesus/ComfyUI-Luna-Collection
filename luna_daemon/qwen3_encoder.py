"""
Luna Daemon - Qwen3-VL Unified Text Encoder

A unified text encoder service that uses Qwen3-VL-4B to serve:
1. Z-IMAGE CLIP encoding (text embeddings only)
2. Standard CLIP text encoding for workflows 
3. Vision-Language captioning and style extraction

The key insight: Qwen3-VL-4B has the EXACT same text encoder architecture as
Qwen3-4B (vocab_size=151936, hidden_size=2560), so it can serve as Z-IMAGE's
CLIP encoder while ALSO providing vision-language capabilities.

Model Architecture Compatibility:
┌────────────────────────────────────────────────────────────────────┐
│                      Qwen3-VL-4B Architecture                      │
├────────────────────────────────────────────────────────────────────┤
│  text_config:                  │  vision_config:                   │
│    vocab_size: 151936 ✓        │    depth: 24                      │
│    hidden_size: 2560 ✓         │    hidden_size: 1024              │
│    num_layers: 36 ✓            │    out_hidden_size: 2560          │
│    (matches Qwen3-4B!)         │    (projects to text dim)         │
├────────────────────────────────┴───────────────────────────────────┤
│                   ▲                            ▲                    │
│                   │                            │                    │
│  encode_text_zimage()         describe_image() / caption()         │
│  (uses text layers only)      (uses full VL model)                 │
└────────────────────────────────────────────────────────────────────┘

Usage:
    encoder = Qwen3VLEncoder(model_path, device="cuda:1")
    
    # For Z-IMAGE CLIP encoding
    embeddings = encoder.encode_text("a beautiful sunset")
    
    # For VLM captioning
    caption = encoder.describe_image(image_tensor, "Describe this image")
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Qwen3VLConfig:
    """Configuration for Qwen3-VL encoder service"""
    # Model paths
    model_path: str = ""  # Path to Qwen3-VL model (HF or GGUF)
    
    # Device settings
    device: str = "cuda:1"
    dtype: str = "bfloat16"  # bfloat16, float16, float32, or "8bit" for quantization
    
    # Text encoder settings (for Z-IMAGE compatibility)
    max_text_length: int = 256
    output_hidden_states: bool = True
    
    # VLM settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Memory management
    use_flash_attention: bool = True
    enable_kv_cache: bool = True
    load_in_8bit: bool = False  # Use bitsandbytes 8-bit quantization


class Qwen3VLEncoder:
    """
    Unified Qwen3-VL encoder for both CLIP-style text encoding and VLM operations.
    
    This class loads a Qwen3-VL model and provides:
    1. Text embedding extraction (compatible with Z-IMAGE's Qwen3-4B CLIP)
    2. Full VLM inference for captioning and analysis
    
    The magic: The text encoder portion is architecturally identical to Qwen3-4B,
    so we can use the same embeddings for Z-IMAGE while having vision capabilities.
    """
    
    # Expected dimensions for Z-IMAGE compatibility
    ZIMAGE_VOCAB_SIZE = 151936
    ZIMAGE_HIDDEN_SIZE = 2560
    
    def __init__(self, config: Optional[Qwen3VLConfig] = None):
        """
        Initialize the Qwen3-VL encoder.
        
        Args:
            config: Configuration for the encoder. If None, uses defaults.
        """
        self.config = config or Qwen3VLConfig()
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = torch.device(self.config.device)
        self._dtype = self._get_dtype()
        self._loaded = False
        self._zimage_compatible = False
        self._is_gguf = False  # Track if model is GGUF format
        self._has_vision = False  # Track if vision is available
        self._mmproj_path = None  # Path to mmproj if available
        
        # Stats
        self._encode_count = 0
        self._vlm_count = 0
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the Qwen3-VL model.
        
        Args:
            model_path: Path to model. Can be:
                - HuggingFace model ID (e.g., "Qwen/Qwen3-VL-4B-Instruct")
                - Local directory with HF format
                - Path to GGUF file
        
        Returns:
            True if loaded successfully
        """
        model_path = model_path or self.config.model_path
        
        if not model_path:
            raise ValueError("No model path provided")
        
        logger.info(f"[Qwen3VLEncoder] Loading model from: {model_path}")
        
        # Detect model format
        if model_path.endswith('.gguf'):
            return self._load_gguf(model_path)
        else:
            return self._load_huggingface(model_path)
    
    def _load_huggingface(self, model_path: str) -> bool:
        """Load model from HuggingFace format."""
        try:
            from transformers import (
                Qwen3VLForConditionalGeneration,
                AutoProcessor,
                AutoTokenizer
            )
            
            logger.info(f"[Qwen3VLEncoder] Loading HuggingFace model: {model_path}")
            
            # Load processor (handles both text and images)
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load tokenizer separately for text-only encoding
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with optional quantization
            load_kwargs = {
                'device_map': self.config.device,
                'trust_remote_code': True,
            }
            
            # Handle quantization
            if self.config.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    logger.info("[Qwen3VLEncoder] Using 8-bit quantization (bitsandbytes)")
                except ImportError:
                    logger.warning("[Qwen3VLEncoder] bitsandbytes not available, falling back to full precision")
                    load_kwargs['dtype'] = self._dtype
            else:
                load_kwargs['dtype'] = self._dtype
            
            # Enable flash attention if available and not quantized
            if self.config.use_flash_attention and not self.config.load_in_8bit:
                try:
                    load_kwargs['attn_implementation'] = 'flash_attention_2'
                except Exception:
                    logger.warning("[Qwen3VLEncoder] Flash attention not available")
            
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            self._model.eval()
            self._loaded = True
            
            # Verify Z-IMAGE compatibility
            self._verify_zimage_compatibility()
            
            logger.info(f"[Qwen3VLEncoder] Model loaded successfully on {self.config.device}")
            logger.info(f"[Qwen3VLEncoder] Z-IMAGE compatible: {self._zimage_compatible}")
            
            return True
            
        except ImportError as e:
            logger.error(f"[Qwen3VLEncoder] Missing transformers package: {e}")
            logger.error("  Install with: pip install transformers>=4.57.0")
            return False
        except Exception as e:
            logger.error(f"[Qwen3VLEncoder] Failed to load model: {e}")
            return False
    
    def _load_gguf(self, gguf_path: str) -> bool:
        """
        Load model from GGUF format using llama-cpp-python.
        
        This loads the FULL model including lm_head weights, enabling both:
        1. Text embedding extraction (for CLIP-style encoding)
        2. Text generation (for VLM/LLM operations)
        
        The same model instance serves both purposes - no duplicate loading!
        """
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            logger.info(f"[Qwen3VLEncoder] Loading GGUF model: {gguf_path}")
            
            # Check for mmproj in same directory (for vision support)
            model_dir = os.path.dirname(gguf_path)
            mmproj_path = None
            for filename in os.listdir(model_dir):
                if 'mmproj' in filename.lower() and filename.lower().endswith(('.gguf', '.bin')):
                    mmproj_path = os.path.join(model_dir, filename)
                    logger.info(f"[Qwen3VLEncoder] Found mmproj: {filename}")
                    break
            
            # Create chat handler if mmproj found (enables vision)
            chat_handler = None
            if mmproj_path:
                try:
                    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                    logger.info("[Qwen3VLEncoder] Vision support enabled via mmproj")
                except Exception as e:
                    logger.warning(f"[Qwen3VLEncoder] Could not load mmproj: {e}")
            
            # Load with llama.cpp - enable embeddings extraction
            self._model = Llama(
                model_path=gguf_path,
                n_ctx=8192,
                n_gpu_layers=-1,  # Use all GPU layers
                chat_handler=chat_handler,
                embedding=True,  # Enable embedding extraction!
                verbose=False,
            )
            
            # Store mmproj path for reference
            self._mmproj_path = mmproj_path
            self._has_vision = chat_handler is not None
            
            self._loaded = True
            self._zimage_compatible = True  # GGUF Qwen3 should be compatible
            self._is_gguf = True
            
            logger.info(f"[Qwen3VLEncoder] GGUF model loaded successfully")
            logger.info(f"[Qwen3VLEncoder] - Embedding extraction: enabled")
            logger.info(f"[Qwen3VLEncoder] - Text generation: enabled")
            logger.info(f"[Qwen3VLEncoder] - Vision support: {'enabled' if self._has_vision else 'disabled (no mmproj)'}")
            
            return True
            
        except ImportError:
            logger.error("[Qwen3VLEncoder] llama-cpp-python not installed")
            logger.error("  Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"[Qwen3VLEncoder] Failed to load GGUF: {e}")
            return False
    
    def _verify_zimage_compatibility(self) -> None:
        """
        Verify the loaded model is compatible with Z-IMAGE's Qwen3-4B encoder.
        
        Checks:
        - vocab_size == 151936
        - hidden_size == 2560
        """
        if not self._loaded or self._model is None:
            self._zimage_compatible = False
            return
        
        try:
            if self._is_gguf:
                # GGUF models: get metadata from llama-cpp
                # Qwen3-4B GGUF should have n_vocab and n_embd in metadata
                metadata = getattr(self._model, 'metadata', {})
                n_ctx_train = getattr(self._model, 'n_ctx_train', lambda: 0)()
                
                # Try to get from model context
                vocab_size = getattr(self._model, 'n_vocab', lambda: 0)()
                hidden_size = getattr(self._model, 'n_embd', lambda: 0)()
                
                # Fallback to metadata keys if methods don't exist
                if vocab_size == 0:
                    vocab_size = metadata.get('llama.vocab_size', 
                                 metadata.get('qwen2.vocab_size', 0))
                if hidden_size == 0:
                    hidden_size = metadata.get('llama.embedding_length',
                                  metadata.get('qwen2.embedding_length', 0))
                    
            else:
                # HuggingFace models: get from config
                config = self._model.config
                
                # Check text config (Qwen3-VL has nested config)
                text_config = getattr(config, 'text_config', config)
                
                vocab_size = getattr(text_config, 'vocab_size', 0)
                hidden_size = getattr(text_config, 'hidden_size', 0)
            
            self._zimage_compatible = (
                vocab_size == self.ZIMAGE_VOCAB_SIZE and
                hidden_size == self.ZIMAGE_HIDDEN_SIZE
            )
            
            if self._zimage_compatible:
                logger.info(f"[Qwen3VLEncoder] ✓ Z-IMAGE compatible:")
                logger.info(f"  vocab_size: {vocab_size}")
                logger.info(f"  hidden_size: {hidden_size}")
            else:
                logger.warning(f"[Qwen3VLEncoder] ✗ Z-IMAGE incompatible:")
                logger.warning(f"  vocab_size: {vocab_size} (expected {self.ZIMAGE_VOCAB_SIZE})")
                logger.warning(f"  hidden_size: {hidden_size} (expected {self.ZIMAGE_HIDDEN_SIZE})")
                
        except Exception as e:
            logger.error(f"[Qwen3VLEncoder] Error checking compatibility: {e}")
            self._zimage_compatible = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def is_zimage_compatible(self) -> bool:
        """Check if model is compatible with Z-IMAGE."""
        return self._zimage_compatible
    
    @torch.inference_mode()
    def encode_text(
        self,
        text: Union[str, List[str]],
        output_type: str = "last_hidden_state",
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode text to embeddings (Z-IMAGE CLIP compatible).
        
        This extracts text embeddings from the Qwen3-VL model's text encoder,
        which is architecturally identical to Z-IMAGE's Qwen3-4B CLIP.
        
        Args:
            text: Input text or list of texts
            output_type: What to return:
                - "last_hidden_state": Full sequence embeddings [B, seq_len, 2560]
                - "pooled": Pooled output (mean over sequence) [B, 2560]
                - "embeddings_only": Just the token embeddings (no transformer) [B, seq_len, 2560]
            normalize: Whether to L2 normalize the output
        
        Returns:
            Tensor of embeddings in the requested format
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle single text
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt"
        ).to(self._device)
        
        self._encode_count += len(text)
        
        # For HuggingFace model, extract text embeddings
        if hasattr(self._model, 'model') and not self._is_gguf:
            # Get embedding layer
            embed_layer = self._model.model.get_input_embeddings() if hasattr(self._model.model, 'get_input_embeddings') else None
            
            # Get just the embedding layer output (for embeddings_only)
            if output_type == "embeddings_only" and embed_layer is not None:
                embeddings = embed_layer(inputs.input_ids)
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                return embeddings
            
            # Full forward through text encoder
            # We need to run through the transformer layers but not the vision encoder
            outputs = self._model.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            hidden_states = outputs.last_hidden_state
            
        elif self._is_gguf and hasattr(self._model, 'embed'):
            # GGUF model - use llama-cpp's embed() method
            # This extracts embeddings from the model (using embedding=True on load)
            embeddings = []
            for t in text:
                # llama-cpp's embed returns a list of floats for the pooled embedding
                # or we can get per-token embeddings
                emb = self._model.embed(t)
                if isinstance(emb, list):
                    emb = torch.tensor(emb, dtype=torch.float32)
                if emb.dim() == 1:
                    # Single embedding, expand to sequence format
                    emb = emb.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
                embeddings.append(emb)
            
            hidden_states = torch.cat(embeddings, dim=0).to(self._device)
            logger.debug(f"[Qwen3VLEncoder] GGUF embeddings shape: {hidden_states.shape}")
        else:
            raise RuntimeError("Unknown model type - cannot extract embeddings")
        
        # Format output
        if output_type == "pooled":
            # Mean pooling over sequence (masked)
            mask = inputs.attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            output = pooled
        else:
            # Return full sequence
            output = hidden_states
        
        if normalize:
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        
        return output
    
    @torch.inference_mode()
    def encode_text_for_zimage(
        self,
        text: str,
        negative_text: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text specifically for Z-IMAGE conditioning.
        
        Returns embeddings in the format Z-IMAGE expects:
        - Positive conditioning embeddings
        - Negative conditioning embeddings
        
        Args:
            text: Positive prompt
            negative_text: Negative prompt (empty string for uncond)
        
        Returns:
            Tuple of (positive_cond, negative_cond) tensors
        """
        if not self._zimage_compatible:
            raise RuntimeError(
                "Model is not Z-IMAGE compatible. "
                f"Expected vocab_size={self.ZIMAGE_VOCAB_SIZE}, hidden_size={self.ZIMAGE_HIDDEN_SIZE}"
            )
        
        pos_emb = self.encode_text(text, output_type="last_hidden_state")
        neg_emb = self.encode_text(negative_text or "", output_type="last_hidden_state")
        
        return pos_emb, neg_emb
    
    @torch.inference_mode()
    def describe_image(
        self,
        image: torch.Tensor,
        prompt: str = "Describe this image in detail.",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a description of an image using the VLM.
        
        Args:
            image: Image tensor in ComfyUI format [B, H, W, C] or [H, W, C], values 0-1
            prompt: The question or instruction to guide the description
            max_tokens: Maximum tokens to generate (default from config)
        
        Returns:
            Generated description text
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        max_new_tokens = max_tokens or self.config.max_new_tokens
        
        # Handle GGUF model with vision (via Llava15ChatHandler)
        if self._is_gguf and self._has_vision:
            return self._describe_image_gguf(image, prompt, max_new_tokens)
        
        # Handle HuggingFace model
        if self._processor is not None:
            return self._describe_image_hf(image, prompt, max_new_tokens)
        
        raise RuntimeError("VLM functions require either HuggingFace model or GGUF with mmproj")
    
    def _describe_image_gguf(
        self,
        image: torch.Tensor,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Generate description using GGUF model with Llava chat handler."""
        from PIL import Image
        import numpy as np
        import base64
        import io
        
        # Handle batch dimension
        if image.dim() == 4:
            image = image[0]  # Take first image
        
        # Convert to PIL image
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Encode image to base64 for llama-cpp
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{img_base64}"
        
        # Create chat completion with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
        )
        
        result = response['choices'][0]['message']['content']
        self._vlm_count += 1
        
        return result.strip()
    
    def _describe_image_hf(
        self,
        image: torch.Tensor,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Generate description using HuggingFace model."""
        # Convert image tensor to PIL for processor
        from PIL import Image
        import numpy as np
        
        # Handle batch dimension
        if image.dim() == 4:
            image = image[0]  # Take first image
        
        # Convert to numpy and PIL
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Format as chat message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            do_sample=self.config.temperature > 0,
        )
        
        # Decode output (skip input tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        self._vlm_count += 1
        
        return response.strip()
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text completion using the LLM.
        
        This method works for both GGUF and HuggingFace models.
        Uses the same model weights as CLIP encoding - no duplicate loading!
        
        Args:
            prompt: The input prompt/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        max_tokens = max_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        
        if self._is_gguf:
            return self._generate_text_gguf(prompt, max_tokens, temperature, system_prompt)
        else:
            return self._generate_text_hf(prompt, max_tokens, temperature, system_prompt)
    
    def _generate_text_gguf(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate text using GGUF model via llama-cpp."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
        )
        
        return response['choices'][0]['message']['content'].strip()
    
    def _generate_text_hf(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate text using HuggingFace model."""
        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return response.strip()
    
    @torch.inference_mode()
    def extract_style(
        self,
        image: torch.Tensor,
    ) -> str:
        """
        Extract style descriptors from an image.
        
        This generates a prompt-style description focusing on:
        - Artistic style and techniques
        - Lighting and mood
        - Color palette
        - Composition elements
        
        Args:
            image: Image tensor
        
        Returns:
            Style description suitable for use in prompts
        """
        style_prompt = """Analyze this image and extract its visual style. 
Focus on:
- Artistic style (photorealistic, anime, oil painting, etc.)
- Lighting (dramatic, soft, natural, studio, etc.)
- Color palette and mood
- Composition and framing

Respond with ONLY style keywords and phrases suitable for an AI image prompt.
Do not describe the subject matter, only the style and aesthetics.
Format as a comma-separated list."""

        return self.describe_image(image, style_prompt, max_tokens=150)
    
    @torch.inference_mode()
    def caption_for_training(
        self,
        image: torch.Tensor,
        style: str = "detailed",
    ) -> str:
        """
        Generate a caption suitable for training data.
        
        Args:
            image: Image tensor
            style: Caption style - "detailed", "brief", "booru", "natural"
        
        Returns:
            Caption text
        """
        style_prompts = {
            "detailed": "Describe this image in detail, including the subject, setting, style, lighting, and mood. Be thorough but concise.",
            "brief": "Write a brief, one-sentence caption for this image.",
            "booru": "Write tags for this image in booru/danbooru style. Use lowercase, underscores, commas between tags. Include: subject, pose, clothing, setting, style, colors.",
            "natural": "Describe this image as if telling someone about a photo you're looking at. Be natural and conversational.",
        }
        
        prompt = style_prompts.get(style, style_prompts["detailed"])
        return self.describe_image(image, prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        stats = {
            "loaded": self._loaded,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "zimage_compatible": self._zimage_compatible,
            "encode_count": self._encode_count,
            "vlm_count": self._vlm_count,
        }
        
        if self._loaded and torch.cuda.is_available():
            device_idx = self._device.index if self._device.type == "cuda" else 0
            stats["vram_used_gb"] = torch.cuda.memory_allocated(device_idx) / 1024**3
            stats["vram_reserved_gb"] = torch.cuda.memory_reserved(device_idx) / 1024**3
        
        return stats
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        self._loaded = False
        self._zimage_compatible = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("[Qwen3VLEncoder] Model unloaded")


# =============================================================================
# Factory function for daemon integration
# =============================================================================

_encoder_instance: Optional[Qwen3VLEncoder] = None


def get_encoder(
    model_path: Optional[str] = None,
    device: str = "cuda:1",
    force_reload: bool = False,
) -> Qwen3VLEncoder:
    """
    Get or create the global Qwen3-VL encoder instance.
    
    Args:
        model_path: Path to model (required on first call)
        device: Device to load model on
        force_reload: Force reload even if already loaded
    
    Returns:
        The encoder instance
    """
    global _encoder_instance
    
    if _encoder_instance is None or force_reload:
        config = Qwen3VLConfig(
            model_path=model_path or "",
            device=device,
        )
        _encoder_instance = Qwen3VLEncoder(config)
        
        if model_path:
            _encoder_instance.load_model(model_path)
    
    return _encoder_instance
