# luna_daemon_loader.py

## Purpose
Provides proxy VAE and CLIP objects that route all operations to the Luna Daemon for VRAM sharing across ComfyUI instances. Enables transparent integration with any ComfyUI node expecting VAE/CLIP.

## Exports
**Classes:**
- `LunaDaemonVAELoader` - Loads VAE via daemon proxy
- `LunaDaemonCLIPLoader` - Loads CLIP via daemon proxy
- `LunaCheckpointTunnel` - Transparent tunnel after checkpoint loaders for auto-routing
- `LunaUNetTunnel` - Provides daemon CLIP/VAE for UNet-only workflows (GGUF)

**Functions:**
- None

**Constants:**
- None

## Key Imports
- `folder_paths` - ComfyUI model path resolution
- `luna_daemon.proxy` - DaemonVAE, DaemonCLIP proxy classes
- `luna_daemon.zimage_proxy` - DaemonZImageCLIP for Qwen3-VL support
- `luna_daemon.client` - Daemon communication client

## ComfyUI Node Configuration
- **LunaDaemonVAELoader**
  - Category: `Luna/Daemon`
  - Display Name: `Luna Daemon VAE Loader`
  - Return Types: `(VAE,)`
  - Return Names: `(vae,)`
  - Function: `load_vae`
- **LunaDaemonCLIPLoader**
  - Category: `Luna/Daemon`
  - Display Name: `Luna Daemon CLIP Loader`
  - Return Types: `(CLIP,)`
  - Return Names: `(clip,)`
  - Function: `load_clip`
- **LunaCheckpointTunnel**
  - Category: `Luna/Daemon`
  - Display Name: `Luna Checkpoint Tunnel`
  - Return Types: `(MODEL, CLIP, VAE, STRING)`
  - Return Names: `(model, clip, vae, status)`
  - Function: `tunnel`
- **LunaUNetTunnel**
  - Category: `Luna/Daemon`
  - Display Name: `Luna UNet Tunnel (GGUF)`
  - Return Types: `(MODEL, CLIP, VAE, STRING)`
  - Return Names: `(model, clip, vae, status)`
  - Function: `tunnel`

## Input Schema
**LunaDaemonVAELoader:**
- `vae_name` (vae_list): VAE model to load via daemon

**LunaDaemonCLIPLoader:**
- `clip_name1` (clip_list): Primary CLIP model
- `clip_name2` (none_option + clip_list, default="None"): Secondary CLIP model

**LunaCheckpointTunnel:**
- `model` (MODEL): MODEL from checkpoint loader - passed through unchanged
- `clip` (CLIP): CLIP from checkpoint loader - auto-detects Z-IMAGE vs standard CLIP
- `vae` (VAE): VAE from checkpoint loader - may be proxied to daemon

**LunaUNetTunnel:**
- `model` (MODEL): MODEL/UNet from GGUF loader or other UNet-only source
- `model_type` (["sdxl", "sd15", "flux", "sd3", "auto"], default="sdxl"): Model architecture for CLIP/VAE selection
- `clip` (CLIP, optional): Optional CLIP override
- `vae` (VAE, optional): Optional VAE override

## Key Methods/Functions
- `LunaCheckpointTunnel.tunnel(model, clip, vae) -> (Any, Any, Any, str)`
  - Routes VAE/CLIP through daemon with intelligent sharing
  - Auto-detects Z-IMAGE (Qwen3-VL) vs standard CLIP architectures
  - Returns proxy objects for sharing, status message
- `LunaUNetTunnel.tunnel(model, model_type, clip=None, vae=None) -> (Any, Any, Any, str)`
  - Provides daemon's shared CLIP/VAE for UNet-only workflows
  - Falls back to user-provided CLIP/VAE if connected
  - Validates daemon has required components loaded
- `LunaDaemonVAELoader.load_vae(vae_name) -> (Any,)`
  - Creates daemon proxy VAE for specified model
  - Requires daemon to be running
- `LunaDaemonCLIPLoader.load_clip(clip_name1, clip_name2="None") -> (Any,)`
  - Creates daemon proxy CLIP from one or two CLIP models
  - Auto-detects SD1.5 vs SDXL based on component count

## Dependencies
**Internal:**
- Requires: `luna_daemon.proxy`, `luna_daemon.zimage_proxy`, `luna_daemon.client`

**External:**
- Required: `folder_paths`
- Optional: None

## Integration Points
**Input:** MODEL/CLIP/VAE from checkpoint loaders or UNet-only sources
**Output:** Proxy objects that transparently route operations to daemon
**Side Effects:** Registers models with daemon, enables VRAM sharing across ComfyUI instances

## Notes
- Component-based architecture: CLIP components (clip_l, clip_g, t5xxl) shared across model families
- Auto-detection for Z-IMAGE: Routes Qwen3-VL CLIP to daemon's Qwen3 encoder
- Transparent passthrough when daemon unavailable or not running
- VAE components are family-specific (sdxl_vae, flux_vae, etc.)
- Designed for integration with third-party nodes expecting standard VAE/CLIP interfaces