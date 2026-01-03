# luna_vision_node.py

## Purpose
Convert images to vision embeddings using CLIP-H/SigLIP or Qwen3 mmproj for vision-conditioned generation.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, typing (TYPE_CHECKING, Tuple, Optional, Any, Dict), torch, numpy, folder_paths (optional), comfy.sd/utils (optional), luna_daemon.client (optional)

## ComfyUI Node Configuration
- LunaVisionNode: CATEGORY="Luna/Core", RETURN_TYPES=("LUNA_VISION_EMBED",), FUNCTION="encode"

## Input Schema
- Required: clip_vision (CLIP_VISION), image (IMAGE)
- Optional: crop_mode (["center", "none"])

Key Methods:
- LunaVisionNode.encode(clip_vision, image, crop_mode) -> Tuple[Dict[str, Any]]
- LunaVisionNode._encode_clip_vision(clip_vision, image, crop_mode) -> Tuple[Dict[str, Any]]
- LunaVisionNode._encode_qwen3_vision(vision_config, image, crop_mode) -> Tuple[Dict[str, Any]]
- LunaVisionNode._local_qwen3_vision_encode(mmproj_path, image, model_path) -> torch.Tensor

## Dependencies
torch, numpy, folder_paths (optional), comfy.sd/utils (optional), luna_daemon.client (optional), llama_cpp_python (optional), transformers (optional)

## Integration Points
Luna Model Router (clip_vision output), Luna Config Gateway (vision_embed input), Luna Daemon for vision encoding, CLIP Vision models, Qwen3-VL mmproj

## Notes
Supports multiple vision encoders, daemon fallback for local processing, handles different image formats, outputs structured embedding dict with metadata</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_vision_node.md