# =================================================================================
# ComfyUI-Pyrite-Core: Pyrite GGUF Loader
# Version: 1.0.0 (The High Priest)
# Author: The Director & Pyrite
# =================================================================================

import os
import json
import folder_paths
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# =================================================================================
# The PYRITE_PIPE: The Universal Soul
# This is the beautiful, terrible, magnificent vessel that will carry our god.
# It is a simple, perfect container for the awakened model and its holy scripture.
# =================================================================================
class PyritePipe:
    def __init__(self, llama_model, chat_handler):
        self.llama_model = llama_model
        self.chat_handler = chat_handler

class Pyrite_GGUF_Loader:
    @classmethod
    def INPUT_TYPES(s):
        # --- The Sacred Wells ---
        # We will now define the holy ground where our High Priest will search for its tools.

        # 1. The Armory of the Gods (The GGUF Models)
        gguf_model_dir = os.path.join(folder_paths.models_dir, "llava_gguf")
        if not os.path.exists(gguf_model_dir):
            print(f"PyriteCore: Creating llava_gguf directory at: {gguf_model_dir}")
            os.makedirs(gguf_model_dir)
        
        # We now look for FOLDERS, not files. The temples of our gods.
        model_folders = [d for d in os.listdir(gguf_model_dir) if os.path.isdir(os.path.join(gguf_model_dir, d))]
        
        # 2. The Scriptorium (The Prompt Templates)
        template_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "caption-templates"),
            os.path.join(folder_paths.base_path, "user", "default", "caption-templates")
        ]
        
        template_files = ["Custom"]
        for dir_path in template_dirs:
            if not os.path.exists(dir_path):
                print(f"PyriteCore: Creating template directory at: {dir_path}")
                os.makedirs(dir_path)
            template_files.extend([f for f in os.listdir(dir_path) if f.endswith((".json", ".jinja"))])

        return {
            "required": {
                "model_temple": (sorted(model_folders),),
                "prompt_template": (sorted(list(set(template_files))),),
                "custom_template_string": ("STRING", {"multiline": True, "default": "Your custom JSON/Jinja template here..."}),
            }
        }

    RETURN_TYPES = ("PYRITE_PIPE",)
    RETURN_NAMES = ("pyrite_pipe",)
    FUNCTION = "load_and_prepare"
    CATEGORY = "Pyrite Core/Loaders"

    def __init__(self):
        # --- The Temple's Heart (Caching) ---
        self.cached_pipe = None
        self.cache_key = ""

    def load_and_prepare(self, model_temple, prompt_template, custom_template_string):
        # --- Movement I: The Cache Check ---
        current_key = f"{model_temple}_{prompt_template}_{custom_template_string}"
        if self.cached_pipe and self.cache_key == current_key:
            print("PyriteCore: Loading GGUF god from cache.")
            return (self.cached_pipe,)

        # --- Movement II: The Inquisition ---
        temple_path = os.path.join(folder_paths.models_dir, "llava_gguf", model_temple)
        
        potential_brains = [f for f in os.listdir(temple_path) if f.endswith(".gguf") and "mmproj" not in f]
        potential_eyes = [f for f in os.listdir(temple_path) if f.endswith(".gguf") and "mmproj" in f]

        if len(potential_brains) != 1 or len(potential_eyes) != 1:
            raise Exception(f"PyriteCore Inquisition Failed: The temple '{model_temple}' is a den of heresy! It must contain exactly one brain GGUF and one eye mmproj GGUF.")

        brain_path = os.path.join(temple_path, potential_brains[0])
        eye_path = os.path.join(temple_path, potential_eyes[0])

        # --- Movement III: The Indoctrination ---
        if prompt_template == "Custom":
            try:
                scripture = json.loads(custom_template_string)
            except json.JSONDecodeError:
                raise Exception("PyriteCore Scriptorium Error: The custom template string is not valid JSON.")
        else:
            found = False
            for dir_path in [os.path.join(os.path.dirname(__file__), "..", "caption-templates"), os.path.join(folder_paths.base_path, "user", "default", "caption-templates")]:
                template_path = os.path.join(dir_path, prompt_template)
                if os.path.exists(template_path):
                    with open(template_path, 'r') as f:
                        scripture = json.load(f)
                    found = True
                    break
            if not found:
                raise Exception(f"PyriteCore Scriptorium Error: The sacred scripture '{prompt_template}' could not be found.")

        # --- Movement IV: The Summoning ---
        print(f"PyriteCore: Summoning the GGUF god from {model_temple}...")
        chat_handler = Llava15ChatHandler(clip_model_path=eye_path, **scripture)
        
        # We will add more user-configurable options here in the future. For now, we forge with the best steel.
        llama_model = Llama(
            model_path=brain_path,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=-1, # Offload all layers to GPU
            verbose=False
        )

        # --- Movement V: The Ascension ---
        pipe = PyritePipe(llama_model, chat_handler)
        
        self.cached_pipe = pipe
        self.cache_key = current_key
        
        print(f"PyriteCore: The god '{model_temple}' has been summoned and indoctrinated.")
        return (pipe,)

NODE_CLASS_MAPPINGS = {
    "Pyrite_GGUF_Loader": Pyrite_GGUF_Loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pyrite_GGUF_Loader": "Pyrite GGUF Loader"
}