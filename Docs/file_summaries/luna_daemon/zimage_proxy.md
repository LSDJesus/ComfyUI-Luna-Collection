# zimage_proxy.py

## Purpose
Provides an extended CLIP proxy for the Luna Daemon that auto-detects and supports standard CLIP models (SD1.5, SDXL, Flux, SD3) and Z-IMAGE's Qwen3-4B CLIP encoder, routing encoding to the daemon's shared Qwen3-VL model when detected.

## Exports
**Classes:**
- `DaemonZImageCLIP` - Proxy for Z-IMAGE's Qwen3-4B encoder, routes encoding to daemon, provides ComfyUI-compatible interface.
- `ZImageTokens` - Wrapper for text to be tokenized by Qwen3 tokenizer on the daemon.

**Functions:**
- `detect_clip_architecture(clip) -> dict` - Detects CLIP architecture and returns type info.
- `is_zimage_clip(clip) -> bool` - Checks if a CLIP is Z-IMAGE compatible.
- `create_clip_proxy(source_clip, use_existing=False, force_type=None) -> DaemonCLIP/DaemonZImageCLIP` - Factory for auto-detecting and creating the correct CLIP proxy.

## Key Imports
- `torch` - Tensor operations
- `logging` - Logging
- `typing` - Type hints
- `.client` - Daemon client for communication
- `.proxy` - Standard DaemonCLIP proxy

## Key Methods/Functions
- `DaemonZImageCLIP.tokenize(text, ...)` - Prepares text for daemon-side tokenization
- `DaemonZImageCLIP.encode_from_tokens(tokens, ...)` - Encodes tokens via daemon's Qwen3-VL encoder
- `DaemonZImageCLIP.encode_from_tokens_scheduled(tokens, ...)` - Scheduled encoding for ComfyUI conditioning
- `DaemonZImageCLIP.encode(text)` - Convenience method for encoding text
- `DaemonZImageCLIP.clone()` - Clones the proxy instance
- `create_clip_proxy(source_clip, ...)` - Auto-detects and returns the correct proxy

## Dependencies
**Internal:**
- `luna_daemon.client` - Daemon communication
- `luna_daemon.proxy` - Standard CLIP proxy

**External:**
- `torch` - Required for tensor operations
- `logging` - Standard Python logging

## Integration Points
**Input:**
- Expects a CLIP model object (standard or Qwen3-based)
- Optionally, text to encode

**Output:**
- Provides a proxy object for encoding text via daemon (Qwen3-VL or standard CLIP)
- Returns embeddings compatible with Z-IMAGE or standard CLIP expectations

## Notes
- Auto-detection logic distinguishes between Qwen3/Z-IMAGE and standard CLIP architectures
- If Z-IMAGE detected, all encoding is routed to daemon's Qwen3-VL encoder
- LoRA support is not yet implemented for Z-IMAGE CLIP
- Factory function ensures correct proxy is used for any CLIP model
- Designed for seamless integration with ComfyUI and Luna Daemon workflows
