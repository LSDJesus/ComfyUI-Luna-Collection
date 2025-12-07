# luna_config_gateway.py

## Purpose
Central configuration gateway for image generation workflows. Extracts LoRAs from prompts, applies CLIP skip, encodes prompts, creates latents, handles vision conditioning.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, re, comfy.samplers, comfy.sd, comfy.utils, folder_paths, nodes

## ComfyUI Node Configuration
- LunaConfigGateway: CATEGORY="Luna/Parameters", RETURN_TYPES=("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "STRING", "STRING", "STRING", "LORA_STACK", "METADATA"), FUNCTION="process"

## Input Schema
- Required: model (MODEL), clip (CLIP), vae (VAE), width/height/batch_size (INT), seed/steps (INT), cfg/denoise (FLOAT), clip_skip (INT), clip_skip_timing (["before_lora", "after_lora"]), sampler/scheduler (from comfy.samplers)
- Optional: model_name (STRING), positive_prompt/negative_prompt (STRING), lora_stack (LORA_STACK), vision_embed (LUNA_VISION_EMBED), vision_strength (FLOAT)

## Key Methods
- LunaConfigGateway.process(model, clip, vae, width, height, batch_size, seed, steps, cfg, denoise, clip_skip, clip_skip_timing, sampler, scheduler, model_name, positive_prompt, negative_prompt, lora_stack, vision_embed, vision_strength) -> tuple
- LunaConfigGateway.extract_loras_from_prompt(prompt) -> tuple
- LunaConfigGateway.load_loras(model, clip, lora_stack) -> tuple
- LunaConfigGateway._combine_vision_conditioning(text_cond, vision_embed, strength) -> list

## Dependencies
comfy.samplers, comfy.sd, comfy.utils, folder_paths, nodes

## Integration Points
ComfyUI model loading, LoRA stack format, CLIP skip nodes, vision embedding types (CLIP Vision, Qwen3), Luna Daemon CLIP proxy

## Notes
Handles DaemonCLIP vs standard CLIP differently, supports inline <lora:> tags, deduplicates LoRAs, combines vision and text conditioning, outputs complete metadata dict</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_config_gateway.md