# luna_daemon/proxy.py

## Purpose
Proxy classes that wrap VAE and CLIP objects to route all operations to the Luna Daemon, enabling maximum VRAM sharing across ComfyUI instances.

## Exports
- `DaemonVAE`: Proxy VAE class that routes encode/decode to daemon
- `DaemonCLIP`: Proxy CLIP class that routes tokenization/encoding to daemon
- `DaemonTokens`: Token wrapper for daemon-side processing
- `detect_clip_type()`: Function to detect CLIP model type
- `detect_vae_type()`: Function to detect VAE model type

## Key Imports
- `torch`, `hashlib`, `typing`

## ComfyUI Node Configuration
N/A - Utility classes

## Input Schema
N/A

## Key Methods
- `DaemonVAE.encode(pixel_samples, auto_tile)` - Encode pixels to latents via daemon
- `DaemonVAE.decode(samples_in, vae_options, auto_tile)` - Decode latents to pixels via daemon
- `DaemonVAE.encode_tiled(pixel_samples, tile_x, tile_y, overlap)` - Tiled encoding for large images
- `DaemonVAE.decode_tiled(samples, tile_x, tile_y, overlap)` - Tiled decoding for large latents
- `DaemonCLIP.tokenize(text, return_word_ids, **kwargs)` - Tokenize text for daemon processing
- `DaemonCLIP.encode_from_tokens(tokens, return_pooled, return_dict)` - Encode tokens via daemon
- `DaemonCLIP.encode_from_tokens_scheduled(tokens, unprojected, add_dict, show_pbar)` - Scheduled encoding for ComfyUI
- `DaemonCLIP.add_patches(patches, strength_patch, strength_model)` - Add LoRA patches (F-150 architecture)
- `DaemonCLIP.add_lora_by_name(lora_name, model_strength, clip_strength)` - Add LoRA by filename

## Dependencies
- `luna_daemon.client` for daemon communication

## Integration Points
- Used by LunaCheckpointTunnel to create proxy objects from real VAE/CLIP
- Compatible with any ComfyUI node expecting VAE or CLIP (including third-party)
- Integrates with LoRA loading nodes via add_patches() interception

## Notes
Component-based architecture allows CLIP component sharing across model families, F-150 LoRA architecture for transient LoRA application with daemon-side caching, automatic tiling for large images/latents.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\luna_daemon\proxy.md