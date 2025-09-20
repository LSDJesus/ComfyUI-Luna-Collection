import folder_paths
import json
import os
from safetensors import safe_open
from aiohttp import web

# Try to import PromptServer for web endpoints (optional)
try:
    from server import PromptServer
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False
    print("LunaLoRAStacker: PromptServer not available, web endpoints disabled")

# (Globals remain the same)
LUNA_METADATA_CACHE = {}
MAX_LORA_SLOTS = 4

class LunaLoRAStacker:
    """
    The Luna LoRA Stacker. It configures a stack of LoRAs with granular,
    manual control, including individual toggles and strengths.
    """
    CATEGORY = "Luna/Loaders"
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "configure_stack"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "show_previews": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
            }
        }
        
        # Add LoRA slots dynamically
        for i in range(1, MAX_LORA_SLOTS + 1):
            inputs["optional"][f"lora_{i}"] = ("LORA",)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01})
            inputs["optional"][f"enabled_{i}"] = ("BOOLEAN", {"default": True})
            
        return inputs

    def configure_stack(self, enabled, show_previews, lora_stack=None, **kwargs):
        if not enabled:
            return (lora_stack or [],)
            
        # Build LoRA stack from inputs
        lora_stack = lora_stack or []
        
        for i in range(1, MAX_LORA_SLOTS + 1):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)
            lora_enabled = kwargs.get(f"enabled_{i}", True)
            
            if lora_name and lora_enabled:
                lora_stack.append({
                    "lora": lora_name,
                    "strength": strength
                })
                
        return (lora_stack,)

# ===================================================================================
# W E B   S E R V E R   E N D P O I N T
# ===================================================================================

# This is a new, separate endpoint just for LoRA metadata.
if HAS_PROMPT_SERVER:
    @PromptServer.instance.routes.get("/luna/get_lora_metadata")  # type: ignore
    async def get_lora_metadata(request):
        """
        Called by the JavaScript front-end to get LoRA metadata and thumbnails.
        """
        lora_name = request.query.get("lora_name", None)
        if not lora_name:
            return web.Response(status=400, text="lora_name parameter is required")

        # We can reuse our global cache!
        if lora_name in LUNA_METADATA_CACHE:
            return web.json_response(LUNA_METADATA_CACHE[lora_name])

        try:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                return web.Response(status=404, text=f"LoRA {lora_name} not found")
                
            base_path, _ = os.path.splitext(lora_path)
            
            metadata = {}
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata_raw = f.metadata()
                if metadata_raw:
                    # LoRA metadata is often stored under a different key. We'll check for the common one.
                    metadata.update(json.loads(metadata_raw.get("ss_text_model_metadata", '{}')))
            
            thumbnail_info = None
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                thumb_path = base_path + ext
                if os.path.exists(thumb_path):
                    thumbnail_info = {
                        "filename": os.path.basename(thumb_path),
                        "type": "input", 
                        "subfolder": "" 
                    }
                    break
            
            lore_data = {}
            lore_path = base_path + ".json"
            if os.path.exists(lore_path):
                with open(lore_path, 'r') as f:
                    lore_data = json.load(f)

            response_data = {
                "metadata": metadata,
                "thumbnail": thumbnail_info,
                "user_lore": lore_data
            }
            
            LUNA_METADATA_CACHE[lora_name] = response_data
            return web.json_response(response_data)

        except Exception as e:
            return web.Response(status=500, text=f"Error reading metadata for {lora_name}: {e}")
else:
    print("LunaLoRAStacker: Web endpoints disabled due to missing PromptServer")

# ===================================================================================
# N O D E   R E G I S T R A T I O N
# ===================================================================================

NODE_CLASS_MAPPINGS = {
    # The internal name is now clean.
    "LunaLoRAStacker": LunaLoRAStacker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # The display name is now clean and clear.
    "LunaLoRAStacker": "Luna LoRA Stacker"
}