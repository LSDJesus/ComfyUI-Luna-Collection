# luna_daemon/qwen3_encoder.py

## Purpose
Unified Qwen3-VL encoder service providing both Z-IMAGE CLIP encoding and vision-language capabilities using architecturally compatible text embeddings.

## Exports
- `Qwen3VLEncoder`: Main encoder class for text and vision operations
- `Qwen3VLConfig`: Configuration dataclass for encoder settings
- `get_encoder()`: Factory function for global encoder instance

## Key Imports
- `os`, `torch`, `logging`, `typing`, `dataclasses`, `pathlib`

## ComfyUI Node Configuration
N/A - Utility class

## Input Schema
N/A

## Key Methods
- `Qwen3VLEncoder.load_model(model_path)` - Load Qwen3-VL model (HF or GGUF)
- `Qwen3VLEncoder.encode_text(text, output_type, normalize)` - Extract text embeddings for Z-IMAGE
- `Qwen3VLEncoder.encode_text_for_zimage(text, negative_text)` - Encode text specifically for Z-IMAGE conditioning
- `Qwen3VLEncoder.describe_image(image, prompt, max_tokens)` - Generate image descriptions using VLM
- `Qwen3VLEncoder.extract_style(image)` - Extract style descriptors from images
- `Qwen3VLEncoder.caption_for_training(image, style)` - Generate training captions

## Dependencies
- `transformers` (for HuggingFace models)
- `llama_cpp` (for GGUF models)
- `PIL` (for image processing)

## Integration Points
- Used by Luna Daemon for Z-IMAGE CLIP encoding and VLM operations
- Provides embeddings compatible with Z-IMAGE's Qwen3-4B CLIP encoder
- Integrates with daemon's client interface for remote access

## Notes
Leverages architectural compatibility between Qwen3-VL-4B and Qwen3-4B text encoders (vocab_size=151936, hidden_size=2560), supports both HuggingFace and GGUF formats with fallback capabilities.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\luna_daemon\qwen3_encoder.md