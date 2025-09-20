import os
import json
from safetensors import safe_open
import folder_paths
import comfy.sd # I notice this was missing from my original scripture, a necessary import. My apologies.
from aiohttp import web # Also missing. The mind races ahead of the quill.

# Import Luna validation system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from validation import luna_validator, validate_node_input
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    validate_node_input = None

def conditional_validate(*args, **kwargs):
    """Conditionally apply validation decorator."""
    def decorator(func):
        if VALIDATION_AVAILABLE and validate_node_input:
            return validate_node_input(*args, **kwargs)(func)
        return func
    return decorator

# Try to import PromptServer for web endpoints (optional)
try:
    from server import PromptServer
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False
    print("LunaCheckpointLoader: PromptServer not available, web endpoints disabled")

# (Global cache remains the same)
LUNA_METADATA_CACHE = {}

class LunaCheckpointLoader:
    CATEGORY = "Luna/Loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name")
    FUNCTION = "load_checkpoint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                
                # --- THE SACRED TOGGLE ---
                # We now add the doctrine of elegant design to our first Oracle.
                "show_previews": ("BOOLEAN", {"default": True}),
            },
        }

    # We add 'show_previews' here to accept the value from the UI, even if we don't use it in the backend.
    @conditional_validate('ckpt_name', max_length=255)
    def load_checkpoint(self, ckpt_name, show_previews):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        
        # Handle different return signatures - unpack safely
        if len(out) >= 3:
            model = out[0]
            clip = out[1] 
            vae = out[2]
        else:
            raise ValueError(f"Unexpected return value from load_checkpoint_guess_config: {len(out)} items")
            
        model_name = os.path.splitext(ckpt_name)[0]
        
        return (model, clip, vae, model_name)

# (Web Server Endpoint - only register if PromptServer is available)
if HAS_PROMPT_SERVER:
    @PromptServer.instance.routes.get("/luna/get_checkpoint_metadata")  # type: ignore
    async def get_checkpoint_metadata(request):
        # ... The existing code for this function is perfect and does not need to change ...
        ckpt_name = request.query.get("ckpt_name", None)
        
        if not ckpt_name:
            return web.Response(status=400, text="ckpt_name parameter is required")

        if ckpt_name in LUNA_METADATA_CACHE:
            return web.json_response(LUNA_METADATA_CACHE[ckpt_name])

        try:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if ckpt_path is None:
                return web.Response(status=404, text=f"Checkpoint {ckpt_name} not found")
                
            base_path, _ = os.path.splitext(ckpt_path)
            
            metadata = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                metadata_raw = f.metadata()
                if metadata_raw:
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
            
            LUNA_METADATA_CACHE[ckpt_name] = response_data
            return web.json_response(response_data)
        except Exception as e:
            return web.Response(status=500, text=f"Error reading metadata for {ckpt_name}: {e}")
else:
    print("LunaCheckpointLoader: Web endpoints disabled due to missing PromptServer")

# (Node Registration remains the same)
NODE_CLASS_MAPPINGS = { "LunaCheckpointLoader": LunaCheckpointLoader }
NODE_DISPLAY_NAME_MAPPINGS = { "LunaCheckpointLoader": "Luna Checkpoint Loader" }