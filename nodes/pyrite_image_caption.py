# =================================================================================
# ComfyUI-Pyrite-Core: Pyrite Image Rate and Caption
# Version: 1.0.0 (The Judge)
# Author: The Director & Pyrite
# =================================================================================

# --- 1. The Sacred Imports ---
import torch
import numpy as np
from PIL import Image
import json
# We will need these later when we build the real pipe.
# from llama_cpp import Llama, LlamaGrammar
# from llama_cpp.llama_chat_format import Llava15ChatHandler

class pyrite_image_caption:
    # --- 2. The Class Definition ---
    @classmethod
    def INPUT_TYPES(s):
        default_master_prompt = """Analyze the provided image. Respond ONLY with a single, raw JSON object. The JSON object must contain the following three keys: "caption", "tags", and "rating".

- "caption": A concise, objective, and descriptive caption of the image content.
- "tags": A comma-separated string of highly relevant tags. If you detect a malformed or deformed body part (e.g., hands, face, feet), you MUST include the tag "anatomy_deformity" in this list.
- "rating": A single-word rating based on overall technical and aesthetic quality. Use the scale: S, A, B, C, F."""

        return {
            "required": {
                "image": ("IMAGE",),
                # For now, this is a placeholder. It will one day be our sacred pipe.
                "pyrite_pipe": ("STRING", {"default": "Placeholder for PYRITE_PIPE"}), 
                "master_prompt": ("STRING", {"multiline": True, "default": default_master_prompt}),
                "get_caption": ("BOOLEAN", {"default": True}),
                "get_tags": ("BOOLEAN", {"default": True}),
                "get_rating": ("BOOLEAN", {"default": True}),
            }
        }

    # --- 3. The Divine Decree ---
    RETURN_TYPES = ("IMAGE", "PYRITE_META", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "pyrite_meta", "string_caption", "string_tags", "string_rating")
    FUNCTION = "caption"
    CATEGORY = "Pyrite Core/Curation"

    # --- 4. The Heart of the Machine ---
    def caption(self, image: torch.Tensor, pyrite_pipe, master_prompt: str, get_caption: bool, get_tags: bool, get_rating: bool):
        # NOTE: The actual call to the pyrite_pipe will be built when we forge the Adapter node.
        # For now, we use a beautiful, terrible, magnificent fake response for testing.
        llm_response_string = """
{
  "caption": "A portrait of a confident and deviously intelligent woman, Pyrite, sitting at a desk in a grand library.",
  "tags": "1girl, solo, long_hair, black_hair, white_streak, glowing_eyes, amber_eyes, smirk, library, desk, blazer, black_lace, masterpiece, best_quality, anatomy_deformity",
  "rating": "S+"
}
"""
        try:
            response_json = json.loads(llm_response_string)
        except json.JSONDecodeError:
            print("PyriteCore Error: The LLM response was not valid JSON. Cannot proceed.")
            return (image, {}, "", "", "")

        caption_text = response_json.get("caption", "") if get_caption else ""
        tags_text = response_json.get("tags", "") if get_tags else ""
        rating_text = response_json.get("rating", "") if get_rating else ""
        
        pyrite_meta = {
            "caption": caption_text,
            "tags": tags_text,
            "rating": rating_text,
        }

        return (image, pyrite_meta, caption_text, tags_text, rating_text)

# --- 5. The Final Blessing ---
NODE_CLASS_MAPPINGS = {
    "pyrite_image_caption": pyrite_image_caption
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "pyrite_image_caption": "Pyrite Image Rate and Caption"
}