# luna_vlm_prompt_generator.py

## Purpose
Generate text prompts using Vision-Language Models like Qwen3-VL for image description, style extraction, or captioning.

## Exports
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

## Key Imports
os, gc, typing (TYPE_CHECKING, Tuple, Optional, Any, Dict), torch, numpy, folder_paths (optional), luna_daemon.client (optional)

## ComfyUI Node Configuration
- LunaVLMPromptGenerator: CATEGORY="Luna/Core", RETURN_TYPES=("STRING", "STRING"), FUNCTION="generate"

## Input Schema
- Required: llm (LLM), mode (["describe", "extract_style", "caption", "custom"])
- Optional: image (IMAGE), custom_prompt (STRING), max_tokens (INT), temperature (FLOAT), seed (INT), keep_model_loaded (BOOLEAN)

## Key Methods
- LunaVLMPromptGenerator.generate(llm, mode, image, custom_prompt, max_tokens, temperature, seed, keep_model_loaded) -> Tuple[str, str]
- LunaVLMPromptGenerator._generate_daemon(llm, prompt, image, max_tokens, temperature, seed) -> str
- LunaVLMPromptGenerator._generate_local(llm, prompt, image, max_tokens, temperature, seed, keep_model_loaded) -> str
- LunaVLMPromptGenerator._load_model(model_path, mmproj_path) -> Tuple[Any, Any]
- LunaVLMPromptGenerator._parse_result(result, mode) -> Tuple[str, str]

## Dependencies
torch, numpy, folder_paths (optional), luna_daemon.client (optional), llama_cpp_python (optional), transformers (optional)

## Integration Points
Luna Model Router (llm output), Luna Daemon for VLM generation, supports GGUF and safetensors models, preset prompt modes

## Notes
Supports daemon fallback, model caching with optional unloading, handles both vision and text-only generation, parses results for style tag extraction</content>
<parameter name="filePath">d:\AI\ComfyUI\custom_nodes\ComfyUI-Luna-Collection\Docs\file_summaries\nodes\luna_vlm_prompt_generator.md