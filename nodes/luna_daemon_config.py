"""
Luna Daemon Config Node
Allows configuring the shared VAE/CLIP daemon from within a workflow.

Instead of hardcoding paths in config.py, this node lets you:
1. Select VAE/CLIP using standard ComfyUI loader dropdowns
2. Send the model selection to the daemon
3. The daemon loads/caches the models as needed

Architecture:
- Workflows use this node to specify which VAE/CLIP the daemon should use
- The daemon maintains a cache of loaded models
- When a different model is requested, the daemon loads it
- Multiple workflows can share the same model (if they select the same one)
"""

import os
from typing import Tuple, Optional, Dict, Any

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import daemon client
try:
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    daemon_client = None
    DAEMON_AVAILABLE = False
    
    class DaemonConnectionError(Exception):
        """Placeholder when daemon not available"""
        pass


class LunaDaemonConfig:
    """
    Configure the Luna VAE/CLIP Daemon with models from the workflow.
    
    Use this node to tell the daemon which VAE and CLIP models to use,
    instead of hardcoding them in the daemon config file.
    
    The daemon will load the requested models and cache them. If the same
    models are already loaded, no reload occurs.
    
    Connect this before any Luna Shared VAE/CLIP nodes to ensure the
    correct models are loaded.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("DAEMON_CONFIG", "STRING")
    RETURN_NAMES = ("config", "status")
    FUNCTION = "configure"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available VAEs
        if folder_paths:
            vae_list = ["(daemon default)"] + folder_paths.get_filename_list("vae")
            clip_list = ["(daemon default)"] + folder_paths.get_filename_list("clip")
        else:
            vae_list = ["(daemon default)"]
            clip_list = ["(daemon default)"]
        
        # For SDXL, we typically need both clip_l and clip_g
        # Let users select them separately or use auto-detection
        
        return {
            "required": {
                "vae": (vae_list, {
                    "default": "(daemon default)",
                    "tooltip": "VAE model for the daemon to use. Select (daemon default) to use config.py setting."
                }),
                "apply_immediately": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Send config to daemon now. If False, config is just passed through."
                }),
            },
            "optional": {
                "clip_l": (clip_list, {
                    "default": "(daemon default)",
                    "tooltip": "CLIP-L model (text encoder). For SDXL, this is the smaller text encoder."
                }),
                "clip_g": (clip_list, {
                    "default": "(daemon default)",
                    "tooltip": "CLIP-G model (text encoder). For SDXL, this is the larger text encoder."
                }),
                "t5xxl": (clip_list, {
                    "default": "(daemon default)",
                    "tooltip": "T5-XXL model for Flux/SD3. Leave as default if not using these models."
                }),
                "embeddings_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Custom embeddings directory path. Leave empty for default."
                }),
                "device": (["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "cuda:1",
                    "tooltip": "GPU device for the daemon to use for shared models."
                }),
            }
        }
    
    def configure(
        self,
        vae: str,
        apply_immediately: bool,
        clip_l: str = "(daemon default)",
        clip_g: str = "(daemon default)",
        t5xxl: str = "(daemon default)",
        embeddings_dir: str = "",
        device: str = "cuda:1"
    ) -> Tuple[Dict, str]:
        """Configure the daemon with selected models."""
        
        if not folder_paths:
            return ({}, "Error: folder_paths not available")
        
        # Build config dict
        config = {
            "device": device,
        }
        
        # Resolve VAE path
        if vae and vae != "(daemon default)":
            vae_path = folder_paths.get_full_path("vae", vae)
            if vae_path:
                config["vae_path"] = vae_path
                config["vae_name"] = vae
        
        # Resolve CLIP paths
        if clip_l and clip_l != "(daemon default)":
            clip_l_path = folder_paths.get_full_path("clip", clip_l)
            if clip_l_path:
                config["clip_l_path"] = clip_l_path
                config["clip_l_name"] = clip_l
        
        if clip_g and clip_g != "(daemon default)":
            clip_g_path = folder_paths.get_full_path("clip", clip_g)
            if clip_g_path:
                config["clip_g_path"] = clip_g_path
                config["clip_g_name"] = clip_g
        
        if t5xxl and t5xxl != "(daemon default)":
            t5_path = folder_paths.get_full_path("clip", t5xxl)
            if t5_path:
                config["t5xxl_path"] = t5_path
                config["t5xxl_name"] = t5xxl
        
        # Embeddings directory
        if embeddings_dir:
            config["embeddings_dir"] = embeddings_dir
        else:
            # Use default embeddings path
            emb_paths = folder_paths.get_folder_paths("embeddings")
            if emb_paths:
                config["embeddings_dir"] = emb_paths[0]
        
        # Apply to daemon if requested
        status_lines = ["Luna Daemon Config"]
        
        if apply_immediately:
            if not DAEMON_AVAILABLE or daemon_client is None:
                status_lines.append("⚠ Daemon client not available")
                return (config, "\n".join(status_lines))
            
            if not daemon_client.is_daemon_running():
                status_lines.append("⚠ Daemon not running")
                status_lines.append("Config saved but not applied")
                return (config, "\n".join(status_lines))
            
            try:
                # Send config to daemon using the configure method
                result = daemon_client.configure(config)
                
                if result.get("status") == "ok":
                    status_lines.append("✓ Config applied to daemon")
                    if result.get("reloaded"):
                        status_lines.append(f"✓ Models reloaded: {', '.join(result.get('reloaded', []))}")
                    else:
                        status_lines.append("✓ Models already loaded")
                else:
                    status_lines.append(f"⚠ {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                status_lines.append(f"⚠ Connection error: {e}")
        else:
            status_lines.append("Config created (not applied)")
        
        # Show config summary
        status_lines.append("")
        status_lines.append("Config:")
        if config.get("vae_name"):
            status_lines.append(f"  VAE: {config['vae_name']}")
        if config.get("clip_l_name"):
            status_lines.append(f"  CLIP-L: {config['clip_l_name']}")
        if config.get("clip_g_name"):
            status_lines.append(f"  CLIP-G: {config['clip_g_name']}")
        if config.get("t5xxl_name"):
            status_lines.append(f"  T5-XXL: {config['t5xxl_name']}")
        status_lines.append(f"  Device: {device}")
        
        return (config, "\n".join(status_lines))


class LunaDaemonConfigFromLoaders:
    """
    Configure the daemon using actual VAE and CLIP objects from loader nodes.
    
    Connect the outputs of standard VAE Loader and CLIP Loader nodes to
    extract their model paths and configure the daemon accordingly.
    
    This provides a more integrated workflow where you use familiar loader
    nodes and this node ensures the daemon has matching models.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("DAEMON_CONFIG", "STRING")
    RETURN_NAMES = ("config", "status")
    FUNCTION = "configure_from_loaders"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apply_to_daemon": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply this config to the running daemon"
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "Connect a VAE from a VAE Loader node"
                }),
                "clip": ("CLIP", {
                    "tooltip": "Connect a CLIP from a CLIP Loader node"
                }),
                "device": (["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "cuda:1"
                }),
            }
        }
    
    def configure_from_loaders(
        self,
        apply_to_daemon: bool,
        vae=None,
        clip=None,
        device: str = "cuda:1"
    ) -> Tuple[Dict, str]:
        """Extract paths from loaded models and configure daemon."""
        
        config = {"device": device}
        status_lines = ["Luna Daemon Config (from loaders)"]
        
        # Try to extract VAE path
        # Note: This requires the VAE object to have path info attached
        # which depends on how ComfyUI's VAE loader works
        if vae is not None:
            # Try to get the source file path
            # This may require modifications to track the path
            if hasattr(vae, 'source_path'):
                config["vae_path"] = vae.source_path
                config["vae_name"] = os.path.basename(vae.source_path)
                status_lines.append(f"VAE: {config['vae_name']}")
            else:
                status_lines.append("⚠ VAE connected but path unknown")
                status_lines.append("  Use LunaDaemonConfig node instead")
        
        # Try to extract CLIP path
        if clip is not None:
            if hasattr(clip, 'source_paths'):
                for i, path in enumerate(clip.source_paths):
                    if 'clip_l' in path.lower():
                        config["clip_l_path"] = path
                        config["clip_l_name"] = os.path.basename(path)
                    elif 'clip_g' in path.lower():
                        config["clip_g_path"] = path
                        config["clip_g_name"] = os.path.basename(path)
                    elif 't5' in path.lower():
                        config["t5xxl_path"] = path
                        config["t5xxl_name"] = os.path.basename(path)
                    else:
                        config[f"clip_{i}_path"] = path
                status_lines.append("CLIP paths extracted")
            else:
                status_lines.append("⚠ CLIP connected but paths unknown")
                status_lines.append("  Use LunaDaemonConfig node instead")
        
        # Apply if requested
        if apply_to_daemon and DAEMON_AVAILABLE and daemon_client is not None:
            if daemon_client.is_daemon_running():
                try:
                    result = daemon_client.configure(config)
                    if result.get("status") == "ok":
                        status_lines.append("✓ Applied to daemon")
                except Exception as e:
                    status_lines.append(f"⚠ Error: {e}")
            else:
                status_lines.append("⚠ Daemon not running")
        
        return (config, "\n".join(status_lines))


class LunaDaemonModelSwitch:
    """
    Quick switch between pre-defined model configurations.
    
    Useful for workflows that need to switch between SD1.5, SDXL, 
    Pony, Illustrious, etc. without editing the daemon config.
    """
    
    CATEGORY = "Luna/Shared"
    RETURN_TYPES = ("DAEMON_CONFIG", "STRING")
    RETURN_NAMES = ("config", "status")
    FUNCTION = "switch_preset"
    OUTPUT_NODE = True
    
    # Pre-defined model presets
    # Users should customize these paths for their setup
    PRESETS = {
        "SDXL": {
            "vae": "sdxl_vae.safetensors",
            "clip_l": "clip_l.safetensors",
            "clip_g": "clip_g.safetensors",
        },
        "SDXL (FP16 VAE)": {
            "vae": "sdxl-vae-fp16-fix.safetensors",
            "clip_l": "clip_l.safetensors",
            "clip_g": "clip_g.safetensors",
        },
        "Pony": {
            "vae": "sdxl_vae.safetensors",  # Pony uses SDXL VAE
            "clip_l": "clip_l.safetensors",
            "clip_g": "clip_g.safetensors",
        },
        "Illustrious": {
            "vae": "sdxl_vae.safetensors",  # Or specific Illustrious VAE
            "clip_l": "clip_l.safetensors",
            "clip_g": "clip_g.safetensors",
        },
        "SD 1.5": {
            "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
            "clip_l": "clip_l.safetensors",
            "clip_g": None,
        },
        "Flux": {
            "vae": "ae.safetensors",  # Flux VAE
            "clip_l": "clip_l.safetensors",
            "t5xxl": "t5xxl_fp16.safetensors",
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "SDXL",
                    "tooltip": "Pre-defined model configuration to load"
                }),
                "device": (["cuda:0", "cuda:1", "cuda:2", "cuda:3"], {
                    "default": "cuda:1"
                }),
            }
        }
    
    def switch_preset(self, preset: str, device: str) -> Tuple[Dict, str]:
        """Switch daemon to a preset configuration."""
        
        if not folder_paths:
            return ({}, "Error: folder_paths not available")
        
        preset_config = self.PRESETS.get(preset, {})
        config = {"device": device}
        status_lines = [f"Luna Daemon Preset: {preset}"]
        
        # Resolve paths for each model in preset
        for key, filename in preset_config.items():
            if filename is None:
                continue
                
            if key == "vae":
                path = folder_paths.get_full_path("vae", filename)
                if path:
                    config["vae_path"] = path
                    config["vae_name"] = filename
                else:
                    status_lines.append(f"⚠ VAE not found: {filename}")
                    
            elif key in ["clip_l", "clip_g", "t5xxl"]:
                path = folder_paths.get_full_path("clip", filename)
                if path:
                    config[f"{key}_path"] = path
                    config[f"{key}_name"] = filename
                else:
                    status_lines.append(f"⚠ {key} not found: {filename}")
        
        # Apply to daemon
        if DAEMON_AVAILABLE and daemon_client is not None and daemon_client.is_daemon_running():
            try:
                result = daemon_client.configure(config)
                if result.get("status") == "ok":
                    status_lines.append("✓ Preset applied")
                    if result.get("reloaded"):
                        status_lines.append(f"✓ Reloaded: {', '.join(result['reloaded'])}")
            except Exception as e:
                status_lines.append(f"⚠ Error: {e}")
        else:
            status_lines.append("⚠ Daemon not running")
        
        return (config, "\n".join(status_lines))


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaDaemonConfig": LunaDaemonConfig,
    "LunaDaemonConfigFromLoaders": LunaDaemonConfigFromLoaders,
    "LunaDaemonModelSwitch": LunaDaemonModelSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaDaemonConfig": "Luna Daemon Config",
    "LunaDaemonConfigFromLoaders": "Luna Daemon Config (From Loaders)",
    "LunaDaemonModelSwitch": "Luna Daemon Model Switch",
}
