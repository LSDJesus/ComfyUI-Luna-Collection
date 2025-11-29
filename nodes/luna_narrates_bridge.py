"""
Luna Narrates Bridge
ComfyUI nodes that communicate with the Luna Narrates server for database integration.

These nodes send generation metadata (LoRAs, prompts, character info) to the 
Luna Narrates backend for storage and retrieval.
"""

import json
import torch
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import hashlib
import os

# Try to import folder_paths for ComfyUI integration
try:
    import folder_paths
except ImportError:
    folder_paths = None


class LunaNarratesConfig:
    """Configuration for Luna Narrates server connection"""
    
    # Default configuration - can be overridden via environment variables
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8765
    DEFAULT_ENDPOINT = "/api/comfyui/generation"
    
    @classmethod
    def get_server_url(cls) -> str:
        host = os.environ.get("LUNA_NARRATES_HOST", cls.DEFAULT_HOST)
        port = os.environ.get("LUNA_NARRATES_PORT", cls.DEFAULT_PORT)
        return f"http://{host}:{port}"
    
    @classmethod
    def get_endpoint(cls, path: str = None) -> str:
        base = cls.get_server_url()
        endpoint = path or os.environ.get("LUNA_NARRATES_ENDPOINT", cls.DEFAULT_ENDPOINT)
        return f"{base}{endpoint}"


class LunaNarratesSendGeneration:
    """
    Send generation metadata to Luna Narrates server.
    
    Transmits LoRA names, prompts, character IDs, and other metadata
    to the Luna Narrates backend for database storage and association
    with narrative sessions.
    """
    
    CATEGORY = "Luna/Narrates"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("response", "success")
    FUNCTION = "send_generation"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Generated image to associate with metadata"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt used for generation"}),
            },
            "optional": {
                # Session/narrative context
                "session_id": ("STRING", {"default": "", "tooltip": "Luna Narrates session ID for this generation"}),
                "character_id": ("STRING", {"default": "", "tooltip": "Character ID this image is for"}),
                "scene_id": ("STRING", {"default": "", "tooltip": "Scene/chapter ID in the narrative"}),
                
                # LoRA information
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack with names and weights"}),
                "lora_names_csv": ("STRING", {"default": "", "tooltip": "Comma-separated LoRA names (alternative to lora_stack)"}),
                
                # Generation parameters
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 7.0}),
                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": ""}),
                
                # Server config
                "server_url": ("STRING", {"default": "", "tooltip": "Override server URL (leave empty for default)"}),
                "endpoint": ("STRING", {"default": "/api/comfyui/generation", "tooltip": "API endpoint path"}),
                
                # Options
                "include_image_hash": ("BOOLEAN", {"default": True, "tooltip": "Include SHA256 hash of image for deduplication"}),
                "async_send": ("BOOLEAN", {"default": True, "tooltip": "Send asynchronously (non-blocking)"}),
                "timeout_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0}),
            }
        }
    
    def send_generation(
        self,
        image: torch.Tensor,
        prompt: str,
        session_id: str = "",
        character_id: str = "",
        scene_id: str = "",
        lora_stack: List[Tuple[str, float, float]] = None,
        lora_names_csv: str = "",
        negative_prompt: str = "",
        seed: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "",
        scheduler: str = "",
        model_name: str = "",
        server_url: str = "",
        endpoint: str = "/api/comfyui/generation",
        include_image_hash: bool = True,
        async_send: bool = True,
        timeout_seconds: float = 5.0,
    ) -> Tuple[str, bool]:
        
        # Build LoRA list from stack or CSV
        loras = []
        if lora_stack:
            for lora_name, model_strength, clip_strength in lora_stack:
                loras.append({
                    "name": lora_name,
                    "model_strength": model_strength,
                    "clip_strength": clip_strength,
                })
        elif lora_names_csv:
            for name in lora_names_csv.split(","):
                name = name.strip()
                if name:
                    loras.append({"name": name, "model_strength": 1.0, "clip_strength": 1.0})
        
        # Calculate image hash if requested
        image_hash = None
        if include_image_hash:
            # Convert tensor to bytes for hashing
            img_bytes = image.cpu().numpy().tobytes()
            image_hash = hashlib.sha256(img_bytes).hexdigest()[:16]  # First 16 chars
        
        # Build payload
        payload = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or None,
            "character_id": character_id or None,
            "scene_id": scene_id or None,
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "loras": loras,
            "generation_params": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler": sampler_name or None,
                "scheduler": scheduler or None,
                "model": model_name or None,
            },
            "image_hash": image_hash,
            "image_shape": list(image.shape),
        }
        
        # Determine URL
        url = server_url if server_url else LunaNarratesConfig.get_server_url()
        full_url = f"{url}{endpoint}"
        
        print(f"[LunaNarratesSendGeneration] Sending to {full_url}")
        print(f"[LunaNarratesSendGeneration] LoRAs: {[l['name'] for l in loras]}")
        
        # Send request
        try:
            if async_send:
                # Fire and forget with timeout
                asyncio.create_task(self._async_send(full_url, payload, timeout_seconds))
                return (f"Async send initiated to {full_url}", True)
            else:
                # Blocking send
                response = asyncio.get_event_loop().run_until_complete(
                    self._async_send(full_url, payload, timeout_seconds)
                )
                return (response, True)
        except Exception as e:
            error_msg = f"Error sending to Luna Narrates: {e}"
            print(f"[LunaNarratesSendGeneration] {error_msg}")
            return (error_msg, False)
    
    async def _async_send(self, url: str, payload: Dict[str, Any], timeout: float) -> str:
        """Send payload asynchronously"""
        try:
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return json.dumps(result)
                    else:
                        text = await response.text()
                        return f"Server returned {response.status}: {text}"
        except asyncio.TimeoutError:
            return f"Request timed out after {timeout}s"
        except aiohttp.ClientError as e:
            return f"Connection error: {e}"
        except Exception as e:
            return f"Error: {e}"


class LunaNarratesCharacterLoRA:
    """
    Register a character's LoRA association with Luna Narrates.
    
    Use this to tell Luna Narrates which LoRAs are associated with which
    character IDs, so the narrative system can automatically apply them.
    """
    
    CATEGORY = "Luna/Narrates"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("response", "success")
    FUNCTION = "register_character_lora"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_id": ("STRING", {"tooltip": "Unique character identifier in Luna Narrates"}),
                "lora_name": ("STRING", {"tooltip": "LoRA filename (with or without extension)"}),
            },
            "optional": {
                "character_name": ("STRING", {"default": "", "tooltip": "Human-readable character name"}),
                "trigger_words": ("STRING", {"default": "", "tooltip": "Comma-separated trigger words for this LoRA"}),
                "default_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_type": (["character", "style", "concept", "pose", "clothing"], {"default": "character"}),
                "server_url": ("STRING", {"default": ""}),
            }
        }
    
    def register_character_lora(
        self,
        character_id: str,
        lora_name: str,
        character_name: str = "",
        trigger_words: str = "",
        default_strength: float = 1.0,
        lora_type: str = "character",
        server_url: str = "",
    ) -> Tuple[str, bool]:
        
        payload = {
            "character_id": character_id,
            "character_name": character_name or character_id,
            "lora": {
                "name": lora_name,
                "type": lora_type,
                "trigger_words": [w.strip() for w in trigger_words.split(",") if w.strip()],
                "default_strength": default_strength,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        url = server_url if server_url else LunaNarratesConfig.get_server_url()
        full_url = f"{url}/api/comfyui/character/register"
        
        print(f"[LunaNarratesCharacterLoRA] Registering {character_id} -> {lora_name}")
        
        try:
            response = asyncio.get_event_loop().run_until_complete(
                self._send_registration(full_url, payload)
            )
            return (response, True)
        except Exception as e:
            return (f"Error: {e}", False)
    
    async def _send_registration(self, url: str, payload: Dict[str, Any]) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return await response.text()
        except Exception as e:
            return f"Connection error: {e}"


class LunaNarratesLoRAExtractor:
    """
    Extract LoRA names from various sources and format for Luna Narrates.
    
    Can extract from:
    - LORA_STACK (from Luna LoRA Stacker or similar)
    - Prompt string (parses <lora:name:weight> syntax)
    - MODEL (extracts applied LoRA patches)
    
    Outputs a clean list of LoRA names for database ingestion.
    """
    
    CATEGORY = "Luna/Narrates"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("lora_names_json", "lora_names_csv", "lora_count")
    FUNCTION = "extract_loras"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack from stacker nodes"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt with <lora:name:weight> syntax"}),
                "model": ("MODEL", {"tooltip": "Model to extract applied LoRA patches from"}),
                "include_weights": ("BOOLEAN", {"default": True, "tooltip": "Include weight info in output"}),
            }
        }
    
    def extract_loras(
        self,
        lora_stack: List[Tuple[str, float, float]] = None,
        prompt: str = "",
        model = None,
        include_weights: bool = True,
    ) -> Tuple[str, str, int]:
        
        loras = []
        
        # Extract from LORA_STACK
        if lora_stack:
            for lora_name, model_str, clip_str in lora_stack:
                lora_info = {"name": lora_name}
                if include_weights:
                    lora_info["model_strength"] = model_str
                    lora_info["clip_strength"] = clip_str
                loras.append(lora_info)
        
        # Extract from prompt string
        if prompt:
            import re
            # Match <lora:name:weight> or <lora:name:model_weight:clip_weight>
            pattern = r'<lora:([^:>]+)(?::([0-9.]+))?(?::([0-9.]+))?>'
            matches = re.findall(pattern, prompt)
            for match in matches:
                name = match[0]
                model_str = float(match[1]) if match[1] else 1.0
                clip_str = float(match[2]) if match[2] else model_str
                
                # Check if already added from stack
                existing = next((l for l in loras if l["name"] == name), None)
                if not existing:
                    lora_info = {"name": name}
                    if include_weights:
                        lora_info["model_strength"] = model_str
                        lora_info["clip_strength"] = clip_str
                    loras.append(lora_info)
        
        # Extract from model patches (if possible)
        if model is not None:
            try:
                # ComfyUI models store patches in model.patches or similar
                if hasattr(model, 'patches'):
                    for patch_key in model.patches.keys():
                        if 'lora' in str(patch_key).lower():
                            # Try to extract name from patch
                            patch_name = str(patch_key).split('/')[-1]
                            existing = next((l for l in loras if l["name"] == patch_name), None)
                            if not existing:
                                loras.append({"name": patch_name, "source": "model_patch"})
            except Exception as e:
                print(f"[LunaNarratesLoRAExtractor] Could not extract from model: {e}")
        
        # Format outputs
        lora_names = [l["name"] for l in loras]
        csv_output = ", ".join(lora_names)
        json_output = json.dumps(loras, indent=2)
        
        print(f"[LunaNarratesLoRAExtractor] Found {len(loras)} LoRAs: {lora_names}")
        
        return (json_output, csv_output, len(loras))


class LunaNarratesWebhook:
    """
    Generic webhook node to send arbitrary data to Luna Narrates.
    
    Use this for custom integrations - sends any string/JSON payload
    to a configurable endpoint.
    """
    
    CATEGORY = "Luna/Narrates"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("response", "success")
    FUNCTION = "send_webhook"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload": ("STRING", {"multiline": True, "tooltip": "JSON payload to send"}),
                "endpoint": ("STRING", {"default": "/api/comfyui/webhook", "tooltip": "API endpoint path"}),
            },
            "optional": {
                "server_url": ("STRING", {"default": ""}),
                "event_type": ("STRING", {"default": "custom", "tooltip": "Event type identifier"}),
                "session_id": ("STRING", {"default": ""}),
            }
        }
    
    def send_webhook(
        self,
        payload: str,
        endpoint: str,
        server_url: str = "",
        event_type: str = "custom",
        session_id: str = "",
    ) -> Tuple[str, bool]:
        
        # Parse payload as JSON if possible
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"raw": payload}
        
        # Wrap in event structure
        event = {
            "event_type": event_type,
            "session_id": session_id or None,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        url = server_url if server_url else LunaNarratesConfig.get_server_url()
        full_url = f"{url}{endpoint}"
        
        try:
            response = asyncio.get_event_loop().run_until_complete(
                self._send_webhook(full_url, event)
            )
            return (response, True)
        except Exception as e:
            return (f"Error: {e}", False)
    
    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return await response.text()
        except Exception as e:
            return f"Connection error: {e}"


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaNarratesSendGeneration": LunaNarratesSendGeneration,
    "LunaNarratesCharacterLoRA": LunaNarratesCharacterLoRA,
    "LunaNarratesLoRAExtractor": LunaNarratesLoRAExtractor,
    "LunaNarratesWebhook": LunaNarratesWebhook,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaNarratesSendGeneration": "Luna Narrates - Send Generation",
    "LunaNarratesCharacterLoRA": "Luna Narrates - Register Character LoRA",
    "LunaNarratesLoRAExtractor": "Luna Narrates - Extract LoRAs",
    "LunaNarratesWebhook": "Luna Narrates - Webhook",
}
