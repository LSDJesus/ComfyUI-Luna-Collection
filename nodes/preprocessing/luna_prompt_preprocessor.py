import os
import json
import torch
from safetensors.torch import save_file
import folder_paths
import comfy.utils
from datetime import datetime
import gzip
import bz2
import lzma

# Import CLIPTextEncode conditionally to avoid circular imports
try:
    # Try importing from main ComfyUI nodes
    import sys
    comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    from nodes import CLIPTextEncode
except ImportError:
    # Fallback for when running in test environment
    CLIPTextEncode = None

class LunaPromptPreprocessor:
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("json_path", "processed_count")
    FUNCTION = "preprocess_prompts"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "prompt_list_path": ("STRING", {"default": "", "tooltip": "Path to text file containing prompts (one per line)"}),
                "filename_prefix": ("STRING", {"default": "prompt", "tooltip": "Prefix for output filenames"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Number of prompts to process before saving progress"}),
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "tooltip": "Starting index for processing (useful for resuming)"}),
                "max_prompts": ("INT", {"default": -1, "min": -1, "tooltip": "Maximum number of prompts to process (-1 for all)"}),
                "overwrite_existing": ("BOOLEAN", {"default": False, "tooltip": "Overwrite existing .safetensors files"}),
                "custom_output_dir": ("STRING", {"default": "", "tooltip": "Custom output directory (leave empty for auto-generated based on prompt list filename)"}),
                "prepend_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to prepend to each prompt (supports embeddings with <embedding:name> syntax)"}),
                "append_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to append to each prompt (supports embeddings with <embedding:name> syntax)"}),
                "quantize_embeddings": ("BOOLEAN", {"default": False, "tooltip": "Quantize embeddings to reduce VRAM usage (converts to float16)"}),
                "compression_level": ("INT", {"default": 0, "min": 0, "max": 9, "tooltip": "Compression level for safetensors (0 = no compression, higher values not yet implemented)"}),
            }
        }

    def preprocess_prompts(self, clip, prompt_list_path, filename_prefix="prompt",
                          batch_size=10, start_index=0, max_prompts=-1, overwrite_existing=False,
                          custom_output_dir="", prepend_text="", append_text="", quantize_embeddings=False,
                          compression_level=0):

        # Validate inputs
        if not os.path.exists(prompt_list_path):
            raise FileNotFoundError(f"Prompt list file not found: {prompt_list_path}")

        # Determine output directory
        if custom_output_dir:
            output_dir = os.path.join(folder_paths.get_output_directory(), custom_output_dir)
        else:
            # Auto-generate directory name from prompt list filename
            prompt_list_basename = os.path.splitext(os.path.basename(prompt_list_path))[0]
            output_dir = os.path.join(folder_paths.get_output_directory(), prompt_list_basename)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # JSON mapping file
        json_path = os.path.join(output_dir, f"{filename_prefix}_mappings.json")

        # Load existing mappings if they exist
        mappings = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
            except:
                mappings = {}

        # Read prompts from file
        with open(prompt_list_path, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]

        # Apply limits
        if max_prompts > 0:
            all_prompts = all_prompts[:max_prompts]

        if start_index >= len(all_prompts):
            print(f"[LunaPromptPreprocessor] Start index {start_index} is beyond available prompts ({len(all_prompts)})")
            return (json_path, 0)

        prompts_to_process = all_prompts[start_index:]

        print(f"[LunaPromptPreprocessor] Processing {len(prompts_to_process)} prompts starting from index {start_index}")
        print(f"[LunaPromptPreprocessor] Output directory: {output_dir}")

        # Initialize text encoder
        if CLIPTextEncode is None:
            raise ImportError("CLIPTextEncode not available. This node requires ComfyUI to be properly installed.")
        text_encoder = CLIPTextEncode()

        processed_count = 0
        batch_count = 0

        for i, prompt in enumerate(prompts_to_process):
            current_index = start_index + i
            filename = "04d"
            filepath = os.path.join(output_dir, filename)

            # Skip if file exists and not overwriting
            if os.path.exists(filepath) and not overwrite_existing:
                if filename not in mappings:
                    mappings[filename] = filepath
                continue

            try:
                # Combine prepend, original prompt, and append text
                combined_prompt = ""
                if prepend_text.strip():
                    combined_prompt += prepend_text.strip() + " "
                combined_prompt += prompt.strip()
                if append_text.strip():
                    combined_prompt += " " + append_text.strip()

                combined_prompt = combined_prompt.strip()

                # Encode the combined prompt
                encoded_result = text_encoder.encode(clip, combined_prompt)
                encoded_tensor = encoded_result[0]  # CLIPTextEncode returns a tuple

                # Quantize if requested
                if quantize_embeddings:
                    # Quantize to half precision for VRAM savings
                    encoded_tensor = encoded_tensor.to(torch.float16)
                    print(f"[LunaPromptPreprocessor] Quantized embedding to float16 for prompt {current_index}")

                # Handle compression level (placeholder for future implementation)
                if compression_level > 0:
                    print(f"[LunaPromptPreprocessor] Compression level {compression_level} requested (not yet implemented) for prompt {current_index}")

                # Save as safetensors
                tensors_dict = {
                    "clip_embeddings": encoded_tensor,
                    "original_prompt": prompt,  # Keep original prompt for reference
                    "combined_prompt": combined_prompt,  # Store the final combined prompt
                    "prepend_text": prepend_text,
                    "append_text": append_text,
                    "index": current_index,
                    "quantized": quantize_embeddings,  # Track quantization status
                    "compression_level": compression_level,  # Track compression level
                    "created": str(datetime.now())  # Add timestamp
                }

                save_file(tensors_dict, filepath)

                # Apply compression if requested
                if compression_level > 0:
                    compressed_path = self._compress_file(filepath, compression_level)
                    # Update mappings with compressed path
                    mappings[filename] = compressed_path
                else:
                    mappings[filename] = filepath

                processed_count += 1

                # Save progress every batch_size prompts
                if processed_count % batch_size == 0:
                    batch_count += 1
                    self._save_mappings(json_path, mappings)
                    print(f"[LunaPromptPreprocessor] Processed {processed_count} prompts (batch {batch_count})")

            except Exception as e:
                print(f"[LunaPromptPreprocessor] Error processing prompt {current_index}: {e}")
                continue

        # Final save
        self._save_mappings(json_path, mappings)
        print(f"[LunaPromptPreprocessor] Completed! Processed {processed_count} prompts total")
        print(f"[LunaPromptPreprocessor] Mappings saved to: {json_path}")

        return (json_path, processed_count)

    def _save_mappings(self, json_path, mappings):
        """Save the mappings dictionary to JSON file"""
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LunaPromptPreprocessor] Error saving mappings: {e}")

    def _compress_file(self, filepath, compression_level):
        """Compress a safetensors file based on compression level"""
        if compression_level == 0:
            return filepath  # No compression

        # Determine compression method and extension
        if 1 <= compression_level <= 3:
            compressor = gzip
            extension = '.gz'
            level = compression_level  # gzip levels 1-3
        elif 4 <= compression_level <= 6:
            compressor = bz2
            extension = '.bz2'
            level = compression_level - 3  # bz2 levels 1-3
        elif 7 <= compression_level <= 9:
            compressor = lzma
            extension = '.xz'
            level = compression_level - 6  # lzma levels 1-3
        else:
            print(f"[LunaPromptPreprocessor] Invalid compression level {compression_level}, using no compression")
            return filepath

        compressed_path = filepath + extension

        try:
            with open(filepath, 'rb') as f_in:
                data = f_in.read()

            compressed_data = None

            # Compress based on compression level range
            if 1 <= compression_level <= 3:
                compressed_data = gzip.compress(data, compresslevel=level)
            elif 4 <= compression_level <= 6:
                compressed_data = bz2.compress(data, compresslevel=level)
            elif 7 <= compression_level <= 9:
                compressed_data = lzma.compress(data, preset=level)

            if compressed_data is None:
                print(f"[LunaPromptPreprocessor] Unsupported compression level {compression_level}")
                return filepath

            # Write compressed data
            with open(compressed_path, 'wb') as f_out:
                f_out.write(compressed_data)

            # Remove original uncompressed file
            os.remove(filepath)

            # Calculate and display compression ratio
            original_size = len(data)
            compressed_size = len(compressed_data)
            ratio = compressed_size / original_size if original_size > 0 else 1.0
            savings = (1 - ratio) * 100
            print(f"[LunaPromptPreprocessor] Compressed {os.path.basename(filepath)} -> {os.path.basename(compressed_path)}")
            print(f"[LunaPromptPreprocessor] Size: {original_size:,} -> {compressed_size:,} bytes ({ratio:.2%}, {savings:.1f}% savings)")
            return compressed_path

        except Exception as e:
            print(f"[LunaPromptPreprocessor] Compression failed: {e}")
            return filepath  # Return original path on failure

NODE_CLASS_MAPPINGS = {
    "LunaPromptPreprocessor": LunaPromptPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaPromptPreprocessor": "Luna Prompt Preprocessor",
}