import os
import json
import torch
from safetensors.torch import load_file, save_file
import folder_paths
import gzip
import bz2
import lzma


class LunaSelectPromptFolder:
    """Select a preprocessed prompt folder from models/luna_prompts."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder_path", "json_path")
    FUNCTION = "select_prompt_folder"

    @classmethod
    def INPUT_TYPES(cls):
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        if os.path.exists(luna_prompts_dir):
            folders = [f for f in os.listdir(luna_prompts_dir) if os.path.isdir(os.path.join(luna_prompts_dir, f))]
            folders.sort()
        else:
            folders = []

        return {
            "required": {
                "prompt_folder": (folders, {"tooltip": "Select a preprocessed prompt folder from models/luna_prompts"}),
            }
        }

    def select_prompt_folder(self, prompt_folder):
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        folder_path = os.path.join(luna_prompts_dir, prompt_folder)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Prompt folder not found: {folder_path}")

        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        print(f"[LunaSelectPromptFolder] Selected folder: {folder_path}")
        print(f"[LunaSelectPromptFolder] Using mappings: {json_path}")

        return (folder_path, json_path)


class LunaLoadPreprocessedPrompt:
    """Load pre-encoded CLIP conditioning from safetensors files."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "INT")
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "original_prompt", "index")
    FUNCTION = "load_preprocessed_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        available_folders = []

        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    json_files = [f for f in os.listdir(item_path) if f.endswith('_mappings.json')]
                    if json_files:
                        available_folders.append(item)

        available_folders.sort()

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
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the prompt folder containing JSON mappings and safetensors files"}),
                "prompt_key": ("STRING", {"default": "", "tooltip": "Key/name of the prompt to load (from the mappings JSON)"}),
                "negative_prompt_file": (negative_files, {"tooltip": "Select a negative prompt safetensors file from models/luna_prompts"}),
            },
            "optional": {
                "auto_select_folder": (available_folders, {"tooltip": "Auto-detected folders from LunaPromptPreprocessor output"}),
            }
        }

    def load_preprocessed_prompt(self, folder_path, prompt_key, negative_prompt_file, auto_select_folder=None):
        if not folder_path and auto_select_folder:
            output_dir = folder_paths.get_output_directory()
            folder_path = os.path.join(output_dir, auto_select_folder)
            print(f"[LunaLoadPreprocessedPrompt] Using auto-selected folder: {folder_path}")

        if not folder_path:
            raise ValueError("Either folder_path must be provided or auto_select_folder must be chosen")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        if prompt_key not in mappings:
            available_keys = list(mappings.keys())[:10]
            raise ValueError(f"Prompt key '{prompt_key}' not found in mappings. Available keys: {available_keys}")

        prompt_filepath = mappings[prompt_key]

        if not os.path.exists(prompt_filepath):
            raise FileNotFoundError(f"Preprocessed prompt file not found: {prompt_filepath}")

        try:
            tensors = self._load_compressed_file(prompt_filepath)
        except Exception as e:
            raise ValueError(f"Error loading positive prompt safetensors file: {e}")

        # Reconstruct ComfyUI conditioning format: [[tensor, {"pooled_output": pooled}]]
        if "cond_tensor" in tensors:
            cond_tensor = tensors["cond_tensor"]
            has_pooled = tensors.get("has_pooled", torch.tensor([False]))
            
            if has_pooled.item() and "pooled_output" in tensors:
                pooled_output = tensors["pooled_output"]
                positive_conditioning = [[cond_tensor, {"pooled_output": pooled_output}]]
            else:
                positive_conditioning = [[cond_tensor, {}]]
        elif "clip_embeddings" in tensors:
            # Legacy format
            cond_tensor = tensors["clip_embeddings"]
            positive_conditioning = [[cond_tensor, {}]]
            print(f"[LunaLoadPreprocessedPrompt] Warning: Legacy format detected, no pooled_output available")
        else:
            raise ValueError("Invalid positive prompt safetensors file: missing 'cond_tensor' or 'clip_embeddings' tensor")

        original_prompt = tensors.get("original_prompt", prompt_key)
        index_val = tensors.get("index", torch.tensor([-1]))
        index = index_val.item() if isinstance(index_val, torch.Tensor) else index_val

        # Load negative prompt
        negative_conditioning = None
        if negative_prompt_file:
            luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
            negative_filepath = os.path.join(luna_prompts_dir, negative_prompt_file)

            if not os.path.exists(negative_filepath):
                raise FileNotFoundError(f"Negative prompt file not found: {negative_filepath}")

            try:
                neg_tensors = load_file(negative_filepath)
                if "cond_tensor" in neg_tensors:
                    cond_tensor = neg_tensors["cond_tensor"]
                    has_pooled = neg_tensors.get("has_pooled", torch.tensor([False]))
                    if has_pooled.item() and "pooled_output" in neg_tensors:
                        pooled_output = neg_tensors["pooled_output"]
                        negative_conditioning = [[cond_tensor, {"pooled_output": pooled_output}]]
                    else:
                        negative_conditioning = [[cond_tensor, {}]]
                elif "clip_embeddings" in neg_tensors:
                    negative_conditioning = [[neg_tensors["clip_embeddings"], {}]]
            except Exception as e:
                print(f"[LunaLoadPreprocessedPrompt] Error loading negative prompt: {e}")

        print(f"[LunaLoadPreprocessedPrompt] Loaded positive prompt '{prompt_key}' from {prompt_filepath}")

        return (positive_conditioning, negative_conditioning, original_prompt, index)

    def _load_compressed_file(self, filepath):
        """Load a compressed or uncompressed safetensors file."""
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = f.read()
            elif filepath.endswith('.bz2'):
                with bz2.open(filepath, 'rb') as f:
                    data = f.read()
            elif filepath.endswith('.xz'):
                with lzma.open(filepath, 'rb') as f:
                    data = f.read()
            else:
                return load_file(filepath)

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                tensors = load_file(temp_file.name)

            try:
                os.unlink(temp_file.name)
            except:
                pass

            return tensors

        except Exception as e:
            print(f"[LunaLoadPreprocessedPrompt] Error loading compressed file {filepath}: {e}")
            return load_file(filepath)


class LunaModifyPreprocessedPrompt:
    """Modify a preprocessed prompt by prepending/appending text and re-encoding."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("modified_conditioning", "modified_prompt")
    FUNCTION = "modify_preprocessed_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for re-encoding modified prompts"}),
                "preprocessed_conditioning": ("CONDITIONING", {"tooltip": "Preprocessed conditioning tensor to modify"}),
                "original_prompt": ("STRING", {"tooltip": "Original prompt text from the preprocessed file"}),
                "prepend_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to prepend to the original prompt"}),
                "append_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to append to the original prompt"}),
            }
        }

    def modify_preprocessed_prompt(self, clip, preprocessed_conditioning, original_prompt, prepend_text="", append_text=""):
        combined_prompt = ""
        if prepend_text.strip():
            combined_prompt += prepend_text.strip() + " "
        combined_prompt += original_prompt.strip()
        if append_text.strip():
            combined_prompt += " " + append_text.strip()

        combined_prompt = combined_prompt.strip()

        if combined_prompt != original_prompt.strip():
            print(f"[LunaModifyPreprocessedPrompt] Re-encoding modified prompt")
            from nodes import CLIPTextEncode
            text_encoder = CLIPTextEncode()
            encoded_result = text_encoder.encode(clip, combined_prompt)
            modified_conditioning = encoded_result[0]
        else:
            modified_conditioning = preprocessed_conditioning

        return (modified_conditioning, combined_prompt)


class LunaListPreprocessedPrompts:
    """List all prompt keys available in a preprocessed prompt folder."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_keys",)
    FUNCTION = "list_preprocessed_prompts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Path to the prompt folder containing JSON mappings"}),
            }
        }

    def list_preprocessed_prompts(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        json_files = [f for f in os.listdir(folder_path) if f.endswith('_mappings.json')]
        if not json_files:
            raise FileNotFoundError(f"No mappings JSON file found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading mappings file: {e}")

        prompt_keys = list(mappings.keys())
        prompt_keys.sort()

        print(f"[LunaListPreprocessedPrompts] Found {len(prompt_keys)} preprocessed prompts")

        return ("\n".join(prompt_keys),)


class LunaSaveNegativePrompt:
    """Encode and save a negative prompt to safetensors for reuse."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_negative_prompt"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "negative_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt text to encode and save"}),
                "filename": ("STRING", {"default": "negative_common", "tooltip": "Filename (without extension)"}),
            }
        }

    def save_negative_prompt(self, clip, negative_text, filename):
        luna_prompts_dir = os.path.join(folder_paths.models_dir, "luna_prompts")
        os.makedirs(luna_prompts_dir, exist_ok=True)

        from nodes import CLIPTextEncode
        text_encoder = CLIPTextEncode()

        try:
            encoded_result = text_encoder.encode(clip, negative_text)
            conditioning = encoded_result[0]
            
            cond_tensor = conditioning[0][0]
            cond_metadata = conditioning[0][1]
            pooled_output = cond_metadata.get("pooled_output", None)
        except Exception as e:
            raise ValueError(f"Error encoding negative prompt: {e}")

        filepath = os.path.join(luna_prompts_dir, f"{filename}.safetensors")
        tensors_dict = {
            "cond_tensor": cond_tensor,
            "original_prompt": negative_text,
            "type": "negative",
        }
        
        if pooled_output is not None:
            tensors_dict["pooled_output"] = pooled_output
            tensors_dict["has_pooled"] = torch.tensor([True])
        else:
            tensors_dict["has_pooled"] = torch.tensor([False])

        try:
            save_file(tensors_dict, filepath)
        except Exception as e:
            raise ValueError(f"Error saving negative prompt: {e}")

        print(f"[LunaSaveNegativePrompt] Saved negative prompt to: {filepath}")

        return (filepath,)


class LunaSinglePromptProcessor:
    """Encode a single prompt and save to safetensors."""
    CATEGORY = "Luna/Preprocessing"
    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("saved_path", "conditioning")
    FUNCTION = "process_single_prompt"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt text to encode and save"}),
                "filename": ("STRING", {"default": "prompt", "tooltip": "Filename (without extension)"}),
            },
            "optional": {
                "overwrite_existing": ("BOOLEAN", {"default": True, "tooltip": "Overwrite existing file if it exists"}),
            }
        }

    def process_single_prompt(self, clip, prompt_text, filename="prompt", overwrite_existing=True):
        output_dir = os.path.join(folder_paths.get_output_directory(), "luna_prompts")
        os.makedirs(output_dir, exist_ok=True)

        from nodes import CLIPTextEncode
        text_encoder = CLIPTextEncode()

        try:
            encoded_result = text_encoder.encode(clip, prompt_text)
            conditioning = encoded_result[0]
            
            cond_tensor = conditioning[0][0]
            cond_metadata = conditioning[0][1]
            pooled_output = cond_metadata.get("pooled_output", None)
        except Exception as e:
            raise ValueError(f"Error encoding prompt: {e}")

        filepath = os.path.join(output_dir, f"{filename}.safetensors")
        if os.path.exists(filepath) and not overwrite_existing:
            print(f"[LunaSinglePromptProcessor] File already exists: {filepath}")
            try:
                existing_tensors = load_file(filepath)
                if "cond_tensor" in existing_tensors:
                    ct = existing_tensors["cond_tensor"]
                    has_pooled = existing_tensors.get("has_pooled", torch.tensor([False]))
                    if has_pooled.item() and "pooled_output" in existing_tensors:
                        existing_cond = [[ct, {"pooled_output": existing_tensors["pooled_output"]}]]
                    else:
                        existing_cond = [[ct, {}]]
                    return (filepath, existing_cond)
            except Exception as e:
                print(f"[LunaSinglePromptProcessor] Error loading existing file: {e}")

        tensors_dict = {
            "cond_tensor": cond_tensor,
            "original_prompt": prompt_text,
            "type": "single_prompt",
            "filename": filename
        }
        
        if pooled_output is not None:
            tensors_dict["pooled_output"] = pooled_output
            tensors_dict["has_pooled"] = torch.tensor([True])
        else:
            tensors_dict["has_pooled"] = torch.tensor([False])

        try:
            save_file(tensors_dict, filepath)
        except Exception as e:
            raise ValueError(f"Error saving prompt: {e}")

        print(f"[LunaSinglePromptProcessor] Saved prompt to: {filepath}")

        if pooled_output is not None:
            output_conditioning = [[cond_tensor, {"pooled_output": pooled_output}]]
        else:
            output_conditioning = [[cond_tensor, {}]]

        return (filepath, output_conditioning)


NODE_CLASS_MAPPINGS = {
    "LunaSelectPromptFolder": LunaSelectPromptFolder,
    "LunaLoadPreprocessedPrompt": LunaLoadPreprocessedPrompt,
    "LunaListPreprocessedPrompts": LunaListPreprocessedPrompts,
    "LunaSaveNegativePrompt": LunaSaveNegativePrompt,
    "LunaSinglePromptProcessor": LunaSinglePromptProcessor,
    "LunaModifyPreprocessedPrompt": LunaModifyPreprocessedPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaSelectPromptFolder": "Luna Select Prompt Folder",
    "LunaLoadPreprocessedPrompt": "Luna Load Preprocessed Prompt",
    "LunaListPreprocessedPrompts": "Luna List Preprocessed Prompts",
    "LunaSaveNegativePrompt": "Luna Save Negative Prompt",
    "LunaSinglePromptProcessor": "Luna Single Prompt Processor",
    "LunaModifyPreprocessedPrompt": "Luna Modify Preprocessed Prompt",
}
