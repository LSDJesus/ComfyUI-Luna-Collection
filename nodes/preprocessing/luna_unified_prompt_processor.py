import os
import json
import torch
from safetensors.torch import save_file, load_file
import folder_paths
from nodes import CLIPTextEncode
import comfy.utils

class LunaUnifiedPromptProcessor:
    """
    Unified node that combines prompt preprocessing and loading functionality
    """
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "original_prompt", "index", "json_path")
    FUNCTION = "process_or_load_prompts"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # Get available negative prompt files from models/luna_prompts
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        negative_files = []
        if os.path.exists(luna_prompts_dir):
            for root, dirs, files in os.walk(luna_prompts_dir):
                for file in files:
                    if file.endswith('.safetensors') and not file.startswith('prompt_'):
                        rel_path = os.path.relpath(os.path.join(root, file), luna_prompts_dir)
                        negative_files.append(rel_path)
            negative_files.sort()

        return {
            "required": {
                "mode": (["preprocess", "load"], {"tooltip": "Choose to preprocess new prompts or load existing preprocessed prompts"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding (required for preprocess mode)"}),
            },
            "optional": {
                # Preprocessing inputs
                "prompt_list_path": ("STRING", {"default": "", "tooltip": "Path to text file containing prompts (preprocess mode only)"}),
                "filename_prefix": ("STRING", {"default": "prompt", "tooltip": "Prefix for output filenames (preprocess mode only)"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Number of prompts to process before saving (preprocess mode only)"}),
                "start_index": ("INT", {"default": 0, "min": 0, "tooltip": "Starting index for processing (preprocess mode only)"}),
                "max_prompts": ("INT", {"default": -1, "min": -1, "tooltip": "Maximum number of prompts to process (preprocess mode only)"}),
                "overwrite_existing": ("BOOLEAN", {"default": False, "tooltip": "Overwrite existing files (preprocess mode only)"}),
                "custom_output_dir": ("STRING", {"default": "", "tooltip": "Custom output directory (preprocess mode only)"}),
                "prepend_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to prepend (preprocess mode only)"}),
                "append_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to append (preprocess mode only)"}),

                # Loading inputs
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to prompt folder (load mode only)"}),
                "prompt_key": ("STRING", {"default": "", "tooltip": "Key/name of prompt to load (load mode only)"}),
                "negative_prompt_file": (negative_files, {"tooltip": "Negative prompt file (load mode only)"}),
            }
        }

    def process_or_load_prompts(self, mode, clip, **kwargs):
        if mode == "preprocess":
            return self._preprocess_mode(clip, **kwargs)
        else:  # mode == "load"
            return self._load_mode(**kwargs)

    def _preprocess_mode(self, clip, prompt_list_path="", filename_prefix="prompt",
                        batch_size=10, start_index=0, max_prompts=-1, overwrite_existing=False,
                        custom_output_dir="", prepend_text="", append_text="", **kwargs):
        """Handle preprocessing functionality"""
        # Validate inputs
        if not prompt_list_path or not os.path.exists(prompt_list_path):
            raise FileNotFoundError(f"Prompt list file not found: {prompt_list_path}")

        # Determine output directory
        if custom_output_dir:
            output_dir = os.path.join(folder_paths.get_output_directory(), custom_output_dir)
        else:
            prompt_list_basename = os.path.splitext(os.path.basename(prompt_list_path))[0]
            output_dir = os.path.join(folder_paths.get_output_directory(), prompt_list_basename)

        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{filename_prefix}_mappings.json")

        # Load existing mappings
        mappings = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
            except:
                mappings = {}

        # Read prompts
        with open(prompt_list_path, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]

        if max_prompts > 0:
            all_prompts = all_prompts[:max_prompts]

        if start_index >= len(all_prompts):
            return (None, None, "", 0, json_path)

        prompts_to_process = all_prompts[start_index:]

        # Process first prompt as example
        if prompts_to_process:
            prompt = prompts_to_process[0]
            combined_prompt = ""
            if prepend_text.strip():
                combined_prompt += prepend_text.strip() + " "
            combined_prompt += prompt.strip()
            if append_text.strip():
                combined_prompt += " " + append_text.strip()
            combined_prompt = combined_prompt.strip()

            # Encode and return
            text_encoder = CLIPTextEncode()
            encoded_result = text_encoder.encode(clip, combined_prompt)
            positive_conditioning = encoded_result[0]

            return (positive_conditioning, None, prompt, start_index, json_path)

        return (None, None, "", 0, json_path)

    def _load_mode(self, folder_path="", prompt_key="", negative_prompt_file="", **kwargs):
        """Handle loading functionality"""
        if not folder_path or not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find JSON file
        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        # Load mappings
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        if prompt_key not in mappings:
            available_keys = list(mappings.keys())[:10]
            raise ValueError(f"Prompt key '{prompt_key}' not found. Available: {available_keys}")

        prompt_filepath = mappings[prompt_key]

        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Preprocessed prompt file not found: {prompt_filepath}")

        # Load tensors
        try:
            tensors = load_file(prompt_filepath)
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")

        if "clip_embeddings" not in tensors:
            raise ValueError("Invalid safetensors file: missing 'clip_embeddings'")

        positive_conditioning = tensors["clip_embeddings"]
        original_prompt = tensors.get("original_prompt", prompt_key)
        index = tensors.get("index", -1)

        # Load negative prompt if specified
        negative_conditioning = None
        if negative_prompt_file:
            luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
            negative_filepath = os.path.join(luna_prompts_dir, negative_prompt_file)

            if os.path.exists(negative_filepath):
                try:
                    neg_tensors = load_file(negative_filepath)
                    if "clip_embeddings" in neg_tensors:
                        negative_conditioning = neg_tensors["clip_embeddings"]
                except Exception as e:
                    print(f"Error loading negative prompt: {e}")

        return (positive_conditioning, negative_conditioning, original_prompt, index, json_path)

NODE_CLASS_MAPPINGS = {
    "LunaUnifiedPromptProcessor": LunaUnifiedPromptProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaUnifiedPromptProcessor": "Luna Unified Prompt Processor",
}