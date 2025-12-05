"""
Luna Daemon Model Loader Node

Provides proxy VAE and CLIP objects that route all operations to the Luna Daemon.
These can be wired into ANY node that expects VAE or CLIP, including third-party
nodes like FaceDetailer, UltimateSDUpscale, etc.

Component-based architecture:
- CLIP components (clip_l, clip_g, t5xxl) can be shared across model families
- VAE components are family-specific (sdxl_vae, flux_vae, etc.)

The daemon loads models on-demand from the first workflow request, then
shares components across all ComfyUI instances for maximum VRAM savings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple, List, Any

if TYPE_CHECKING:
    import folder_paths
    from ..luna_daemon.proxy import DaemonVAE, DaemonCLIP
    from ..luna_daemon import client as daemon_client

try:
    import folder_paths
except ImportError:
    folder_paths = None  # type: ignore

# Import proxy classes
try:
    from ..luna_daemon.proxy import DaemonVAE, DaemonCLIP, detect_vae_type, detect_clip_type
    from ..luna_daemon import client as daemon_client
    from ..luna_daemon.client import DaemonConnectionError
    DAEMON_AVAILABLE = True
except ImportError:
    DAEMON_AVAILABLE = False
    DaemonConnectionError = Exception
    DaemonVAE = None  # type: ignore
    DaemonCLIP = None  # type: ignore
    daemon_client = None  # type: ignore
    
    # Dummy functions for when daemon is not available
    def detect_vae_type(vae: Any) -> str:
        return 'unknown'
    def detect_clip_type(clip: Any) -> str:
        return 'unknown'


class LunaDaemonVAELoader:
    """
    Load a VAE that uses the Luna Daemon for all encode/decode operations.
    
    The returned VAE object can be wired into ANY node that expects VAE,
    including third-party nodes like FaceDetailer, UltimateSDUpscale, etc.
    
    On first use, the daemon loads this VAE model. All ComfyUI instances
    share the same VAE via the daemon - massive VRAM savings!
    
    If a different workflow tries to use a different VAE, an error is raised.
    Use the Unload button in the daemon panel to switch VAEs.
    """
    
    CATEGORY = "Luna/Daemon"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of VAE models
        vae_list = folder_paths.get_filename_list("vae")
        
        return {
            "required": {
                "vae_name": (vae_list, {
                    "tooltip": "VAE model to load via daemon. All instances will share this VAE."
                }),
            }
        }
    
    def load_vae(self, vae_name: str) -> Tuple:
        if not DAEMON_AVAILABLE:
            raise RuntimeError(
                "Luna daemon not available. "
                "Make sure the luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel in the sidebar,\n"
                "or run: python -m luna_daemon.server --gpu 1"
            )
        
        # Get full path to VAE
        vae_path = folder_paths.get_full_path("vae", vae_name)
        
        if not vae_path or not os.path.exists(vae_path):
            raise RuntimeError(f"VAE not found: {vae_name}")
        
        # Create proxy VAE - we don't have type detection for file-based loading
        # Default to sdxl, user can specify different VAE loader if needed
        proxy_vae = DaemonVAE(source_vae=None, vae_type='sdxl', use_existing=False)
        # Note: This legacy loader doesn't have access to the actual VAE object,
        # so it can only work if the daemon already has a matching VAE loaded.
        # Use LunaCheckpointTunnel for full functionality.
        
        return (proxy_vae,)


class LunaDaemonCLIPLoader:
    """
    Load CLIP model(s) that use the Luna Daemon for all encoding operations.
    
    The returned CLIP object can be wired into ANY node that expects CLIP,
    including third-party nodes like FaceDetailer, etc.
    
    For SDXL, select both clip_l and clip_g models.
    For SD1.5, select just the clip model.
    
    On first use, the daemon loads these CLIP models. All ComfyUI instances
    share the same CLIP via the daemon!
    """
    
    CATEGORY = "Luna/Daemon"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of CLIP models
        clip_list = folder_paths.get_filename_list("clip")
        none_option = ["None"]
        
        return {
            "required": {
                "clip_name1": (clip_list, {
                    "tooltip": "Primary CLIP model (e.g., clip_l for SDXL)"
                }),
            },
            "optional": {
                "clip_name2": (none_option + clip_list, {
                    "default": "None",
                    "tooltip": "Secondary CLIP model (e.g., clip_g for SDXL). Leave as None for SD1.5."
                }),
            }
        }
    
    def load_clip(self, clip_name1: str, clip_name2: str = "None") -> Tuple:
        if not DAEMON_AVAILABLE:
            raise RuntimeError(
                "Luna daemon not available. "
                "Make sure the luna_daemon package is properly installed."
            )
        
        if not daemon_client.is_daemon_running():
            raise RuntimeError(
                "Luna Daemon is not running!\n"
                "Start it from the Luna Daemon panel in the sidebar,\n"
                "or run: python -m luna_daemon.server --gpu 1"
            )
        
        # Collect CLIP paths
        clip_paths = []
        
        # Primary CLIP
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        if not clip_path1 or not os.path.exists(clip_path1):
            raise RuntimeError(f"CLIP not found: {clip_name1}")
        clip_paths.append(clip_path1)
        
        # Secondary CLIP (optional, for SDXL)
        if clip_name2 and clip_name2 != "None":
            clip_path2 = folder_paths.get_full_path("clip", clip_name2)
            if clip_path2 and os.path.exists(clip_path2):
                clip_paths.append(clip_path2)
        
        # Create proxy CLIP - detect type from number of paths
        clip_type = 'sdxl' if len(clip_paths) == 2 else 'sd15'
        proxy_clip = DaemonCLIP(source_clip=None, clip_type=clip_type, use_existing=False)
        # Note: This legacy loader doesn't have access to the actual CLIP object,
        # so it can only work if the daemon already has matching components loaded.
        # Use LunaCheckpointTunnel for full functionality.
        
        return (proxy_clip,)


class LunaCheckpointTunnel:
    """
    Transparent tunnel that sits after any checkpoint loader.
    
    Intelligently routes VAE and CLIP based on daemon state:
    - Daemon not running: Pass everything through unchanged
    - Daemon running, no models: Load VAE/CLIP into daemon, output proxies
    - Daemon running, matching type: Return existing proxies (share!)
    - Daemon running, different type: Load as additional component
    
    Component-based sharing:
    - CLIP components (clip_l, clip_g, t5xxl) are shared across model families
    - VAE components are family-specific (sdxl_vae, flux_vae, etc.)
    
    Just connect this after your checkpoint loader and it handles everything!
    """
    
    CATEGORY = "Luna/Daemon"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "status")
    FUNCTION = "tunnel"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL from checkpoint loader - passed through unchanged"
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP from checkpoint loader - may be proxied to daemon"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE from checkpoint loader - may be proxied to daemon"
                }),
            }
        }
    
    def tunnel(self, model, clip, vae) -> Tuple:
        """
        Route VAE/CLIP based on daemon state.
        MODEL always passes through unchanged.
        """
        # Detect types using the proxy module's detection functions
        vae_type = detect_vae_type(vae)
        clip_type = detect_clip_type(clip)
        
        # Check if daemon is available
        if not DAEMON_AVAILABLE:
            status = f"[{clip_type}] Daemon module not available - passthrough mode"
            print(f"[LunaCheckpointTunnel] {status}")
            return (model, clip, vae, status)
        
        # Check if daemon is running
        if not daemon_client.is_daemon_running():
            status = f"[{clip_type}] Daemon not running - passthrough mode"
            print(f"[LunaCheckpointTunnel] {status}")
            return (model, clip, vae, status)
        
        # Daemon is running - get current state
        try:
            info = daemon_client.get_daemon_info()
        except Exception as e:
            status = f"[{clip_type}] Daemon error: {e} - passthrough mode"
            print(f"[LunaCheckpointTunnel] {status}")
            return (model, clip, vae, status)
        
        # Check what's currently loaded
        loaded_vaes = info.get('loaded_vaes', {})
        loaded_clips = info.get('loaded_clip_components', {})
        
        # Determine if we can share or need to load
        vae_already_loaded = vae_type in loaded_vaes
        
        # For CLIP, check if all required components are loaded
        clip_components_needed = {
            'sd15': ['clip_l'],
            'sdxl': ['clip_l', 'clip_g'],
            'flux': ['clip_l', 't5xxl'],
            'sd3': ['clip_l', 'clip_g', 't5xxl'],
        }.get(clip_type, ['clip_l'])
        
        clip_components_available = [c for c in clip_components_needed if c in loaded_clips]
        clip_components_missing = [c for c in clip_components_needed if c not in loaded_clips]
        
        # Build status message
        status_parts = []
        
        # Create proxy VAE
        if vae_already_loaded:
            status_parts.append(f"VAE({vae_type}): sharing")
            proxy_vae = DaemonVAE(source_vae=None, vae_type=vae_type, use_existing=True)
        else:
            status_parts.append(f"VAE({vae_type}): registering")
            proxy_vae = DaemonVAE(source_vae=vae, vae_type=vae_type, use_existing=False)
        
        # Create proxy CLIP
        if clip_components_missing:
            if clip_components_available:
                status_parts.append(f"CLIP: sharing [{', '.join(clip_components_available)}], loading [{', '.join(clip_components_missing)}]")
            else:
                status_parts.append(f"CLIP({clip_type}): registering all components")
            proxy_clip = DaemonCLIP(source_clip=clip, clip_type=clip_type, use_existing=False)
        else:
            status_parts.append(f"CLIP: sharing all [{', '.join(clip_components_available)}]")
            proxy_clip = DaemonCLIP(source_clip=None, clip_type=clip_type, use_existing=True)
        
        status = " | ".join(status_parts)
        print(f"[LunaCheckpointTunnel] {status}")
        
        return (model, proxy_clip, proxy_vae, status)


class LunaUNetTunnel:
    """
    Connect a UNet/MODEL (from GGUF loader or similar) to Luna Daemon's CLIP/VAE.
    
    Perfect for GGUF workflow where you only have UNet weights:
    1. Load GGUF UNet with ComfyUI-GGUF or similar
    2. Connect MODEL output to this node
    3. Get daemon's shared CLIP and VAE automatically
    
    No CLIP/VAE inputs needed - daemon provides them!
    
    Workflow:
    ┌─────────────────┐     ┌──────────────────┐
    │ GGUF UNet Loader│────▶│ Luna UNet Tunnel │────▶ MODEL
    └─────────────────┘     │                  │────▶ CLIP (from daemon)
                            │                  │────▶ VAE (from daemon)
                            └──────────────────┘
    """
    
    CATEGORY = "Luna/Daemon"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "status")
    FUNCTION = "tunnel"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Model type hints for daemon to know which CLIP/VAE to provide
        model_types = ["sdxl", "sd15", "flux", "sd3", "auto"]
        
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MODEL/UNet from GGUF loader or other UNet-only source"
                }),
                "model_type": (model_types, {
                    "default": "sdxl",
                    "tooltip": "Model architecture - determines which CLIP/VAE to use from daemon"
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP override - if not connected, uses daemon's CLIP"
                }),
                "vae": ("VAE", {
                    "tooltip": "Optional VAE override - if not connected, uses daemon's VAE"
                }),
            }
        }
    
    def tunnel(self, model, model_type: str, clip=None, vae=None) -> Tuple:
        """
        Route UNet model with daemon CLIP/VAE.
        
        If clip/vae are provided, they're passed through.
        If not provided, daemon's shared CLIP/VAE are used.
        """
        status_parts = []
        
        # Check daemon availability
        if not DAEMON_AVAILABLE:
            if clip is None or vae is None:
                status = "ERROR: Daemon not available and no CLIP/VAE provided!"
                print(f"[LunaUNetTunnel] {status}")
                raise RuntimeError(status)
            return (model, clip, vae, "Passthrough (daemon unavailable)")
        
        if not daemon_client.is_daemon_running():
            if clip is None or vae is None:
                status = "ERROR: Daemon not running and no CLIP/VAE provided!"
                print(f"[LunaUNetTunnel] {status}")
                raise RuntimeError(status)
            return (model, clip, vae, "Passthrough (daemon not running)")
        
        # Get daemon info
        try:
            info = daemon_client.get_daemon_info()
        except Exception as e:
            if clip is None or vae is None:
                raise RuntimeError(f"Daemon error: {e}")
            return (model, clip, vae, f"Passthrough (daemon error: {e})")
        
        loaded_vaes = info.get('loaded_vaes', {})
        loaded_clips = info.get('loaded_clip_components', {})
        
        # Handle VAE
        if vae is not None:
            # User provided VAE - pass through
            output_vae = vae
            status_parts.append("VAE: user-provided")
        else:
            # Use daemon VAE
            vae_type_map = {
                'sdxl': 'sdxl_vae',
                'sd15': 'sd15_vae', 
                'flux': 'flux_vae',
                'sd3': 'sd3_vae',
                'auto': 'sdxl_vae',  # Default to SDXL
            }
            vae_type = vae_type_map.get(model_type, 'sdxl_vae')
            
            if vae_type not in loaded_vaes and 'sdxl_vae' not in loaded_vaes:
                raise RuntimeError(f"Daemon has no VAE loaded for {model_type}. Load VAE in daemon first.")
            
            # Use available VAE (prefer exact match, fall back to sdxl)
            use_vae_type = vae_type if vae_type in loaded_vaes else 'sdxl_vae'
            output_vae = DaemonVAE(source_vae=None, vae_type=use_vae_type, use_existing=True)
            status_parts.append(f"VAE: daemon ({use_vae_type})")
        
        # Handle CLIP
        if clip is not None:
            # User provided CLIP - pass through
            output_clip = clip
            status_parts.append("CLIP: user-provided")
        else:
            # Use daemon CLIP
            clip_components_needed = {
                'sd15': ['clip_l'],
                'sdxl': ['clip_l', 'clip_g'],
                'flux': ['clip_l', 't5xxl'],
                'sd3': ['clip_l', 'clip_g', 't5xxl'],
                'auto': ['clip_l', 'clip_g'],  # Default to SDXL
            }.get(model_type, ['clip_l', 'clip_g'])
            
            missing = [c for c in clip_components_needed if c not in loaded_clips]
            if missing:
                raise RuntimeError(f"Daemon missing CLIP components for {model_type}: {missing}")
            
            output_clip = DaemonCLIP(source_clip=None, clip_type=model_type, use_existing=True)
            status_parts.append(f"CLIP: daemon ({model_type})")
        
        status = " | ".join(status_parts)
        print(f"[LunaUNetTunnel] {status}")
        
        return (model, output_clip, output_vae, status)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LunaDaemonVAELoader": LunaDaemonVAELoader,
    "LunaDaemonCLIPLoader": LunaDaemonCLIPLoader,
    "LunaCheckpointTunnel": LunaCheckpointTunnel,
    "LunaUNetTunnel": LunaUNetTunnel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaDaemonVAELoader": "Luna Daemon VAE Loader",
    "LunaDaemonCLIPLoader": "Luna Daemon CLIP Loader",
    "LunaCheckpointTunnel": "Luna Checkpoint Tunnel",
    "LunaUNetTunnel": "Luna UNet Tunnel (GGUF)",
}
