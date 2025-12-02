import os
import re
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes


class LunaConfigGateway:
    """
    Central configuration gateway for image generation workflows.
    
    Takes models, prompts, and settings - outputs everything needed for generation
    plus complete metadata for image saving.
    
    Features:
    - Extracts and loads LoRAs from prompt text (<lora:name:weight> syntax)
    - Combines with optional lora_stack input (deduplicates)
    - Applies CLIP skip (before or after LoRAs)
    - Encodes prompts to conditioning
    - Creates empty latent
    - Outputs complete metadata dict
    """
    CATEGORY = "Luna/Parameters"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT",
                    "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "INT",
                    "STRING", "STRING", "STRING", "STRING", "STRING",
                    "LORA_STACK", "METADATA")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative", "latent",
                    "width", "height", "batch_size", "seed", "steps", "cfg", "denoise", "clip_skip",
                    "sampler", "scheduler", "model_name", "positive_prompt", "negative_prompt",
                    "lora_stack", "metadata")
    FUNCTION = "process"

    # Regex to extract <lora:name:weight> or <lora:name:model_weight:clip_weight>
    LORA_PATTERN = re.compile(r'<lora:([^:>]+):([^:>]+)(?::([^>]+))?>')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Model inputs
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                # Image dimensions
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                # Sampler settings
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for generation. Use control_after_generate to randomize."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # CLIP settings
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": -1}),
                "clip_skip_timing": (["before_lora", "after_lora"], {"default": "after_lora", "tooltip": "Apply CLIP skip before or after LoRA loading. Usually 'after_lora' is correct."}),
                # Sampler/scheduler
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
            },
            "optional": {
                # Model name for metadata
                "model_name": ("STRING", {"default": "", "forceInput": True, "tooltip": "Model name for metadata (auto-strips extensions)"}),
                # Prompts - raw text that may contain <lora:...> tags
                "positive_prompt": ("STRING", {"default": "", "forceInput": True, "multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"default": "", "forceInput": True, "multiline": True, "dynamicPrompts": True}),
                # Optional LoRA stack from external stacker node
                "lora_stack": ("LORA_STACK", {"tooltip": "Optional LoRA stack to combine with extracted LoRAs"}),
            }
        }

    def extract_loras_from_prompt(self, prompt: str) -> tuple:
        """
        Extract <lora:name:weight> tags from prompt text.
        Returns (cleaned_prompt, list of (name, model_weight, clip_weight))
        """
        loras = []
        
        def replace_lora(match):
            name = match.group(1)
            weight1 = float(match.group(2))
            weight2 = float(match.group(3)) if match.group(3) else weight1
            loras.append((name, weight1, weight2))
            return ""  # Remove the tag from prompt
        
        cleaned = self.LORA_PATTERN.sub(replace_lora, prompt)
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned, loras

    def find_lora_file(self, lora_name: str) -> str:
        """Find the actual LoRA file path, handling partial names."""
        lora_list = folder_paths.get_filename_list("loras")
        
        # Exact match first
        if lora_name in lora_list:
            return lora_name
        
        # Try with common extensions
        for ext in ['.safetensors', '.ckpt', '.pt']:
            full_name = lora_name + ext
            if full_name in lora_list:
                return full_name
        
        # Partial match (name appears in filename)
        for lora_file in lora_list:
            if lora_name.lower() in lora_file.lower():
                return lora_file
        
        return None

    def load_loras(self, model, clip, lora_stack: list) -> tuple:
        """Load and apply LoRAs to model and clip."""
        if not lora_stack:
            return model, clip
        
        for lora_name, model_weight, clip_weight in lora_stack:
            lora_file = self.find_lora_file(lora_name)
            if lora_file is None:
                print(f"[LunaLoadParameters] Warning: LoRA '{lora_name}' not found, skipping")
                continue
            
            try:
                lora_path = folder_paths.get_full_path("loras", lora_file)
                lora = comfy.sd.load_lora_for_models(model, clip, comfy.utils.load_torch_file(lora_path), model_weight, clip_weight)
                model, clip = lora[0], lora[1]
                print(f"[LunaLoadParameters] Loaded LoRA: {lora_file} (model={model_weight}, clip={clip_weight})")
            except Exception as e:
                print(f"[LunaLoadParameters] Error loading LoRA '{lora_name}': {e}")
        
        return model, clip

    def process(self, model, clip, vae, width, height, batch_size, seed, steps, cfg, denoise,
                clip_skip, clip_skip_timing, sampler, scheduler,
                model_name="", positive_prompt="", negative_prompt="", lora_stack=None):
        
        # Clean model name
        if model_name:
            for ext in ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'):
                if model_name.lower().endswith(ext):
                    model_name = model_name[:-len(ext)]
                    break
        
        # Extract LoRAs from prompts
        clean_positive, pos_loras = self.extract_loras_from_prompt(positive_prompt or "")
        clean_negative, neg_loras = self.extract_loras_from_prompt(negative_prompt or "")
        
        # Combine LoRA sources: input stack + extracted from prompts
        # Use dict to deduplicate by name (later entries override)
        lora_dict = {}
        
        # Add input stack first
        if lora_stack:
            for lora_name, model_w, clip_w in lora_stack:
                lora_dict[lora_name.lower()] = (lora_name, model_w, clip_w)
        
        # Add extracted LoRAs (may override stack entries)
        for lora_name, model_w, clip_w in pos_loras + neg_loras:
            lora_dict[lora_name.lower()] = (lora_name, model_w, clip_w)
        
        # Convert back to list
        combined_loras = list(lora_dict.values())
        
        # Apply CLIP skip before LoRAs if specified
        if clip_skip_timing == "before_lora":
            clip = nodes.CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
        
        # Load and apply LoRAs
        model, clip = self.load_loras(model, clip, combined_loras)
        
        # Apply CLIP skip after LoRAs if specified
        if clip_skip_timing == "after_lora":
            clip = nodes.CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
        
        # Encode prompts
        positive_cond = nodes.CLIPTextEncode().encode(clip, clean_positive)[0]
        negative_cond = nodes.CLIPTextEncode().encode(clip, clean_negative)[0]
        
        # Create empty latent
        latent = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
        
        # Build LoRA info for metadata
        lora_info = ", ".join([f"{name}:{mw}" for name, mw, cw in combined_loras]) if combined_loras else ""
        
        # Build complete metadata
        metadata = {
            "model": model_name,
            "model_name": model_name,
            "positive": positive_prompt or "",  # Original prompt with tags
            "positive_clean": clean_positive,   # Prompt without LoRA tags
            "negative": negative_prompt or "",
            "negative_clean": clean_negative,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "sampler": sampler,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "clip_skip": clip_skip,
            "batch_size": batch_size,
            "loras": lora_info,
            "lora_count": len(combined_loras),
        }
        
        return (model, clip, vae, positive_cond, negative_cond, latent,
                width, height, batch_size, seed, steps, cfg, denoise, clip_skip,
                sampler, scheduler, model_name, positive_prompt or "", negative_prompt or "",
                combined_loras, metadata)


NODE_CLASS_MAPPINGS = {
    "LunaConfigGateway": LunaConfigGateway,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaConfigGateway": "Luna Config Gateway",
}