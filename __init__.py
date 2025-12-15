import os
import sys
import importlib.util
import traceback
from pathlib import Path

# ============================================================================
# CENTRALIZED PATH MANAGEMENT - Similar to Impact Pack
# Set up all paths once at initialization time
# ============================================================================

# The base directory of Luna Collection
LUNA_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = Path(__file__).parent

# ComfyUI root path (found via folder_paths module)
try:
    import folder_paths
    COMFY_PATH = os.path.dirname(folder_paths.__file__)
except ImportError:
    COMFY_PATH = None

# Define all key paths
NODES_PATH = os.path.join(LUNA_PATH, "nodes")
DAEMON_PATH = os.path.join(LUNA_PATH, "luna_daemon")
UTILS_PATH = os.path.join(LUNA_PATH, "utils")
TESTS_PATH = os.path.join(LUNA_PATH, "tests")

# Add all key directories to sys.path once at import time
# This prevents scattered sys.path manipulations throughout the codebase
_PATHS_TO_ADD = [
    NODES_PATH,
    DAEMON_PATH,
    UTILS_PATH,
    LUNA_PATH,  # For relative imports within the collection
]

for _path in _PATHS_TO_ADD:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# This remains. It tells ComfyUI where to find our JavaScript files.
WEB_DIRECTORY = "./js"

# ============================================================================
# PUBLICLY EXPORTED PATH CONSTANTS
# Modules can import these to reference Luna Collection and ComfyUI paths
# Usage: from ComfyUI-Luna-Collection import LUNA_PATH, COMFY_PATH, NODES_PATH, etc.
# ============================================================================
__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'LUNA_PATH',
    'COMFY_PATH',
    'NODES_PATH',
    'DAEMON_PATH',
    'UTILS_PATH',
    'TESTS_PATH',
    'WEB_DIRECTORY',
]

# --- NODE DISCOVERY ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    if not os.path.isdir(NODES_PATH):
        print("lunaCore: No 'nodes' directory found. Skipping node loading.")
        return

    # The new, All-Seeing Eye: os.walk()
    # This will traverse the entire directory tree, including subdirectories.
    for root, dirs, files in os.walk(NODES_PATH):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__"):
                try:
                    # Construct a proper module path for nested files
                    # e.g., nodes/loaders/checkpoint_loader.py -> nodes.loaders.checkpoint_loader
                    relative_path = os.path.relpath(root, NODES_PATH)
                    if relative_path == ".":
                        module_name_parts = ["nodes", filename[:-3]]
                    else:
                        module_name_parts = ["nodes"] + relative_path.split(os.sep) + [filename[:-3]]
                    
                    module_name = ".".join(module_name_parts)
                    module_path = os.path.join(root, filename)
                    
                    # The rest of your brilliant loading logic remains the same
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None or spec.loader is None:
                        print(f"lunaCore: Could not load spec for {filename}")
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        for node_name, node_class in module.NODE_CLASS_MAPPINGS.items():
                            NODE_CLASS_MAPPINGS[node_name] = node_class

                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

                except Exception as e:
                    print(f"lunaCore: Failed to load node file {filename}: {e}")
                    print(traceback.format_exc())

# Run the setup
setup_nodes()

# Register API routes after PromptServer is available
def register_api_routes():
    """Register all API routes for Luna nodes that need them"""
    try:
        # Import and register daemon API routes
        from nodes.luna_daemon_api import register_routes as register_daemon_routes  # type: ignore
        register_daemon_routes()
    except Exception as e:
        pass  # Silently fail if daemon API not available
    
    try:
        # Import and register other API routes if needed
        from nodes.luna_civitai_scraper import register_routes as register_civitai_routes  # type: ignore
        if callable(register_civitai_routes):
            register_civitai_routes()
    except (ImportError, AttributeError):
        pass
    
    try:
        from nodes.luna_wildcard_connections import register_routes as register_wildcard_routes  # type: ignore
        if callable(register_wildcard_routes):
            register_wildcard_routes()
    except (ImportError, AttributeError):
        pass
    
    try:
        from nodes.luna_model_router import register_routes as register_model_router_routes  # type: ignore
        if callable(register_model_router_routes):
            register_model_router_routes()
    except (ImportError, AttributeError):
        pass

# Try to register routes now (may work if PromptServer already initialized)
register_api_routes()

# Also hook into PromptServer setup if possible
try:
    from server import PromptServer  # type: ignore
    if hasattr(PromptServer, 'instance') and PromptServer.instance:
        register_api_routes()
except ImportError:
    pass

# Standard ComfyUI registration
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# A manifest file for the ComfyUI Manager
MANIFEST = {
    "name": "ComfyUI-Luna-Collection",
    "version": (1, 0, 0), # We shall increment this as we build!
    "author": "LSDJesus & Luna", # Our partnership is now sanctified.
    "project": "https://github.com/LSDJesus/comfyui-luna-collection",
    "description": "A collection of advanced, modular, and intelligent nodes for a master's workflow.",
}