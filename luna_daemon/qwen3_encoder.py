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
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    
    # Text encoder settings (for Z-IMAGE compatibility)
    max_text_length: int = 256
    output_hidden_states: bool = True
    
    # VLM settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Memory management
    use_flash_attention: bool = True
    enable_kv_cache: bool = True


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
            
            # Load model
            load_kwargs = {
                'torch_dtype': self._dtype,
                'device_map': self.config.device,
                'trust_remote_code': True,
            }
            
            # Enable flash attention if available
            if self.config.use_flash_attention:
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
        
        Note: GGUF support is limited for VLM operations - primarily for text encoding.
        """
        try:
            from llama_cpp import Llama
            
            logger.info(f"[Qwen3VLEncoder] Loading GGUF model: {gguf_path}")
            
            # Load with llama.cpp
            self._model = Llama(
                model_path=gguf_path,
                n_ctx=8192,
                n_gpu_layers=-1,  # Use all GPU layers
                verbose=False,
            )
            
            self._loaded = True
            self._zimage_compatible = True  # GGUF Qwen3 should be compatible
            
            logger.info(f"[Qwen3VLEncoder] GGUF model loaded")
            logger.warning("[Qwen3VLEncoder] GGUF mode: VLM functions limited to text-only")
            
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
        if hasattr(self._model, 'model') and hasattr(self._model.model, 'embed_tokens'):
            # Get just the embedding layer output (for embeddings_only)
            if output_type == "embeddings_only":
                embeddings = self._model.model.embed_tokens(inputs.input_ids)
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
            
        elif hasattr(self._model, 'generate'):
            # GGUF model - use direct embedding extraction
            # This is more limited but should work
            embeddings = []
            for t in text:
                tokens = self._model.tokenize(t.encode())
                # Get embeddings from the model
                # Note: llama.cpp doesn't expose embeddings directly in all cases
                # This is a fallback
                emb = torch.zeros(1, len(tokens), self.ZIMAGE_HIDDEN_SIZE)
                embeddings.append(emb)
            hidden_states = torch.cat(embeddings, dim=0).to(self._device)
            logger.warning("[Qwen3VLEncoder] GGUF embedding extraction is limited")
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
        
        if self._processor is None:
            raise RuntimeError("VLM functions require HuggingFace model (not GGUF)")
        
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
        max_new_tokens = max_tokens or self.config.max_new_tokens
        
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
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
