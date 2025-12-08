"""
Convert Qwen3-VL model to FP8_E4M3FN format for efficient VRAM usage.

This script converts a BF16 Qwen3-VL model to FP8_E4M3FN precision, reducing
VRAM usage by ~50% while maintaining acceptable quality for CLIP text encoding.

Usage:
    python scripts/convert_qwen3vl_to_fp8.py

Input: D:\AI\SD Models\LLM\Huihui-Qwen3-VL-4B-Instruct-abliterated (BF16, 8.3 GB)
Output: D:\AI\SD Models\LLM\Huihui-Qwen3-VL-4B-Instruct-abliterated-FP8 (FP8, 4.1 GB)
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_MODEL = r"D:\AI\SD Models\LLM\Huihui-Qwen3-VL-4B-Instruct-abliterated"
OUTPUT_MODEL = r"D:\AI\SD Models\LLM\Huihui-Qwen3-VL-4B-Instruct-abliterated-FP8"

def convert_to_fp8():
    """Convert Qwen3-VL model to FP8_E4M3FN precision."""
    
    logger.info("=" * 70)
    logger.info("Qwen3-VL BF16 → FP8_E4M3FN Conversion")
    logger.info("=" * 70)
    logger.info(f"Input:  {INPUT_MODEL}")
    logger.info(f"Output: {OUTPUT_MODEL}")
    logger.info("")
    
    # Create output directory
    output_path = Path(OUTPUT_MODEL)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model in BF16
    logger.info("[1/4] Loading BF16 model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        INPUT_MODEL,
        dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU first for conversion
        trust_remote_code=True,
    )
    
    logger.info("[2/4] Converting to FP8_E4M3FN...")
    # Convert all parameters to FP8
    # Store as FP8 but the model will cast to compute dtype during forward pass
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16 and param.numel() > 0:
                # Convert to FP8_E4M3FN (stored precision)
                # Model will auto-cast to BF16/FP16 during compute
                param.data = param.data.to(torch.float8_e4m3fn)
    
    # Update config to use FP8 storage with BF16 compute
    if hasattr(model, 'config'):
        # Set torch_dtype to indicate storage format
        model.config.torch_dtype = torch.float8_e4m3fn
    
    logger.info("[3/4] Saving FP8 model...")
    model.save_pretrained(
        OUTPUT_MODEL,
        safe_serialization=True,  # Use safetensors
        max_shard_size="5GB"
    )
    
    # Copy processor and tokenizer configs
    logger.info("[4/4] Copying processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(INPUT_MODEL, trust_remote_code=True)
    processor.save_pretrained(OUTPUT_MODEL)
    
    tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_MODEL)
    
    logger.info("")
    logger.info("✅ Conversion complete!")
    logger.info(f"📁 Output saved to: {OUTPUT_MODEL}")
    logger.info("")
    logger.info("Verify with:")
    logger.info(f"  ls -lh '{OUTPUT_MODEL}/*.safetensors'")
    logger.info("")
    logger.info("Expected size: ~4.1 GB (vs 8.3 GB original)")

if __name__ == "__main__":
    try:
        convert_to_fp8()
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise
