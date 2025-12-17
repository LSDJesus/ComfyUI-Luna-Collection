import re
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes

from utils.lora_weight_cache import LoRAWeightCache


# Global weight cache for LoRA restoration
_lora_weight_cache = LoRAWeightCache()


def is_daemon_clip(clip) -> bool:
    """Check if clip is a DaemonCLIP proxy (routes to Luna Daemon)."""
    # Check class name - DaemonCLIP is from luna_daemon.proxy
    return type(clip).__name__ == "DaemonCLIP"


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
    - Accepts optional vision_embed for vision-conditioned generation
    - Outputs complete metadata dict
    """
    CATEGORY = "Luna"
    CATEGORY = "Luna"
    RETURN_TYPES = ("LUNA_PIPE", "METADATA")
    RETURN_NAMES = ("luna_pipe", "metadata")
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
                # Optional vision embedding for vision-conditioned generation
                "vision_embed": ("LUNA_VISION_EMBED", {
                    "tooltip": "Optional vision embedding from Luna Vision Node. When connected, combines with text conditioning for vision-guided generation."
                }),
                # Vision conditioning strength
                "vision_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Strength of vision conditioning (0 = text only, 1 = balanced, 2 = vision dominant)"
                }),
                # FB Cache (First-Block Cache) settings for 2x speedup on final denoising steps
                "fb_cache_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable First-Block Cache for ~2x speedup on final denoising steps (daemon mode only)"
                }),
                "fb_cache_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Start caching at this denoising percentage (0.0 = start, 1.0 = end)"
                }),
                "fb_cache_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stop caching at this denoising percentage (typical: 1.0 for final steps only)"
                }),
                "fb_cache_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Residual diff tolerance for cache hits (0.0=strict, higher=more caching but less accurate)"
                }),
                "fb_cache_object_to_patch": (("diffusion_model",), {
                    "default": "diffusion_model",
                    "tooltip": "Model object to patch (typically 'diffusion_model')"
                }),
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
        
        return ""

    def load_loras(self, model, clip, lora_stack: list) -> tuple:
        """
        Load and apply LoRAs to model and clip.
        
        Architecture:
        - DaemonCLIP: Uses add_lora_by_name (daemon loads from disk)
        - InferenceModeWrapper: Uses standard ComfyUI LoRA loading
        - Standard local: Uses comfy.sd.load_lora_for_models
        
        This centralizes LoRA loading with intelligent routing based on proxy type.
        
        Weight Caching:
        - Before applying LoRAs, caches affected model weights
        - Allows transient LoRA application with restoration via reset node
        - Reduces disk I/O on repeated runs with same LoRAs
        """
        if not lora_stack:
            return model, clip
        
        # Cache model weights before LoRA application
        global _lora_weight_cache
        layers_cached = _lora_weight_cache.cache_weights_for_loras(
            model, lora_stack, self.find_lora_file
        )
        if layers_cached > 0:
            print(f"[LunaConfigGateway] Cached {layers_cached} layer weights for LoRA restoration")
        
        # Detect proxy types
        use_daemon_clip = is_daemon_clip(clip)
        # InferenceModeWrapper wraps model, but LoRAs apply normally through the wrapper
        is_wrapped_model = type(model).__name__ == "InferenceModeWrapper"
        
        if use_daemon_clip:
            print("[LunaConfigGateway] DaemonCLIP detected - CLIP LoRAs via daemon")
        if is_wrapped_model:
            print("[LunaConfigGateway] InferenceModeWrapper detected - LoRAs apply through wrapper")
        
        for lora_name, model_weight, clip_weight in lora_stack:
            lora_file = self.find_lora_file(lora_name)
            if lora_file is None:
                print(f"[LunaConfigGateway] Warning: LoRA '{lora_name}' not found, skipping")
                continue
            
            try:
                # CLIP LoRA handling
                if use_daemon_clip:
                    clip = clip.add_lora_by_name(lora_file, model_weight, clip_weight)
                
                # UNet LoRA handling - standard loading for all model types
                # (InferenceModeWrapper forwards add_patches transparently)
                if use_daemon_clip:
                    # DaemonCLIP but local model: Load UNet weights only
                    lora_path = folder_paths.get_full_path("loras", lora_file)
                    lora_data = comfy.utils.load_torch_file(lora_path)  # type: ignore
                    # Filter to only UNet/model weights (not CLIP)
                    model_weights = {k: v for k, v in lora_data.items() 
                                    if not any(p in k.lower() for p in 
                                              ['clip_l', 'clip_g', 'te1', 'te2', 'text_encoder', 'lora_te'])}
                    if model_weights:
                        model, _ = comfy.sd.load_lora_for_models(  # type: ignore
                            model, None, model_weights, model_weight, 0.0
                        )
                    print(f"[LunaConfigGateway] LoRA '{lora_file}' - CLIP via daemon, UNet local")
                
                else:
                    # Standard local: Load both CLIP and UNet
                    lora_path = folder_paths.get_full_path("loras", lora_file)
                    lora = comfy.sd.load_lora_for_models(  # type: ignore
                        model, clip, comfy.utils.load_torch_file(lora_path),  # type: ignore 
                        model_weight, clip_weight
                    )
                    model, clip = lora[0], lora[1]
                    print(f"[LunaConfigGateway] Loaded LoRA: {lora_file} (model={model_weight}, clip={clip_weight})")
                    
            except Exception as e:
                print(f"[LunaConfigGateway] Error loading LoRA '{lora_name}': {e}")
        
        return model, clip

    def process(self, model, clip, vae, width, height, batch_size, seed, steps, cfg, denoise,
                clip_skip, clip_skip_timing, sampler, scheduler,
                model_name="", positive_prompt="", negative_prompt="", lora_stack=None,
                vision_embed=None, vision_strength=1.0,
                fb_cache_enabled=False, fb_cache_start=0.0, fb_cache_end=1.0, fb_cache_threshold=0.1,
                fb_cache_object_to_patch="diffusion_model"):
        
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
            clip = nodes.CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]  # type: ignore
        
        # Load and apply LoRAs
        model, clip = self.load_loras(model, clip, combined_loras)
        
        # Configure FB cache if enabled
        if fb_cache_enabled:
            try:
                from luna_daemon.wavespeed_utils import apply_fb_cache_to_model
                
                model = apply_fb_cache_to_model(
                    model,
                    start_percent=fb_cache_start,
                    end_percent=fb_cache_end,
                    residual_diff_threshold=fb_cache_threshold,
                    object_to_patch=fb_cache_object_to_patch
                )
                print(f"[LunaConfigGateway] FB cache enabled: {fb_cache_start:.0%}-{fb_cache_end:.0%} (threshold={fb_cache_threshold}, patch={fb_cache_object_to_patch})")
            except ImportError:
                print("[LunaConfigGateway] Warning: FB cache not available (wavespeed_utils not found)")
            except Exception as e:
                print(f"[LunaConfigGateway] Warning: FB cache error: {e}")
        
        # Apply CLIP skip after LoRAs if specified
        if clip_skip_timing == "after_lora":
            clip = nodes.CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]  # type: ignore
        
        # Encode prompts
        positive_cond = nodes.CLIPTextEncode().encode(clip, clean_positive)[0]  # type: ignore
        negative_cond = nodes.CLIPTextEncode().encode(clip, clean_negative)[0]  # type: ignore
        
        # Combine with vision embedding if provided
        vision_used = False
        if vision_embed is not None:
            positive_cond = self._combine_vision_conditioning(
                positive_cond, vision_embed, vision_strength
            )
            vision_used = True
        
        # Create empty latent
        latent = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]  # type: ignore
        
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
            "vision_conditioning": vision_used,
            "vision_strength": vision_strength if vision_used else 0.0,
            "fb_cache_enabled": fb_cache_enabled,
            "fb_cache_range": f"{fb_cache_start:.0%}-{fb_cache_end:.0%}" if fb_cache_enabled else "disabled",
        }
        
        # Build luna_pipe: tuple containing all generation components
        luna_pipe = (
            model, clip, vae,
            positive_cond, negative_cond,
            latent,
            width, height, seed, steps, cfg, denoise,
            sampler, scheduler
        )
        
        return (luna_pipe, metadata)
    
    def _combine_vision_conditioning(
        self,
        text_cond: list,
        vision_embed: dict,
        strength: float
    ) -> list:
        """
        Combine text conditioning with vision embedding.
        
        The vision embedding is concatenated/combined with the text conditioning
        based on the model architecture:
        - For SDXL/Flux + Vision: Concatenate to pooled output
        - For Z-IMAGE (Qwen3): Combine in the embedding space
        """
        import torch
        
        if not vision_embed or "embedding" not in vision_embed:
            return text_cond
        
        vision_tensor = vision_embed["embedding"]
        if isinstance(vision_tensor, (list, tuple)):
            vision_tensor = torch.tensor(vision_tensor)
        
        # Scale by strength
        vision_tensor = vision_tensor * strength
        
        # text_cond is a list of (cond_tensor, cond_dict) tuples
        new_cond = []
        for cond_tensor, cond_dict in text_cond:
            # Create a copy of the dict
            new_dict = dict(cond_dict)
            
            # Add vision embedding to the conditioning
            # Different strategies based on vision type
            vision_type = vision_embed.get("type", "unknown")
            
            if vision_type in ["clip_vision", "clip_h_or_siglip"]:
                # Standard CLIP Vision - add to pooled output or concat
                if "pooled_output" in new_dict:
                    # Concatenate vision to pooled
                    pooled = new_dict["pooled_output"]
                    # Ensure compatible shapes
                    if vision_tensor.dim() == 2 and pooled.dim() == 2:
                        if vision_tensor.shape[-1] == pooled.shape[-1]:
                            # Same dimension - can average or concat
                            new_dict["pooled_output"] = pooled + vision_tensor * strength
                        else:
                            # Different dimensions - store separately
                            new_dict["vision_embed"] = vision_tensor
                else:
                    new_dict["vision_embed"] = vision_tensor
                    
            elif vision_type in ["qwen3_vision", "qwen3_mmproj"]:
                # Qwen3-VL vision - concatenate to conditioning tokens
                if vision_tensor.dim() == 2:
                    vision_tensor = vision_tensor.unsqueeze(0)
                if vision_tensor.shape[0] != cond_tensor.shape[0]:
                    vision_tensor = vision_tensor.expand(cond_tensor.shape[0], -1, -1)
                
                # Concatenate vision tokens to the conditioning sequence
                # Vision tokens come first (like in LLaVA-style architectures)
                if vision_tensor.shape[-1] == cond_tensor.shape[-1]:
                    combined = torch.cat([vision_tensor, cond_tensor], dim=1)
                    new_cond.append((combined, new_dict))
                    continue
                else:
                    # Incompatible dimensions, store separately
                    new_dict["vision_embed"] = vision_tensor
            
            else:
                # Unknown type - store for model to handle
                new_dict["vision_embed"] = vision_tensor
            
            new_cond.append((cond_tensor, new_dict))
        
        return new_cond


NODE_CLASS_MAPPINGS = {
    "LunaConfigGateway": LunaConfigGateway,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaConfigGateway": "Luna Config Gateway",
}