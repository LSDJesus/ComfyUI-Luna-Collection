# luna_daemon/__init__.py

## Purpose
Module initialization for Luna VAE/CLIP Daemon, providing shared model server for multi-instance ComfyUI setups.

## Exports
N/A - Documentation-only init file

## Key Imports
N/A

## ComfyUI Node Configuration
N/A - Module init file

## Input Schema
N/A

## Key Methods
N/A

## Dependencies
- All daemon components: config.py, client.py, server.py, proxy.py, zimage_proxy.py, qwen3_encoder.py

## Integration Points
- Provides shared VAE/CLIP models for multi-instance workflows
- Used by Luna nodes for VRAM-efficient model operations

## Notes
Documentation init file describing the Luna Daemon architecture: shared model server with dynamic worker scaling, proxy objects for routing, and unified Qwen3-VL encoder for Z-IMAGE and VLM tasks.</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\luna_daemon\__init__.md