import os
import sys
import importlib.util
import traceback

# This remains. It tells ComfyUI where to find our JavaScript files.
WEB_DIRECTORY = "./js"

# The base directory of our collection
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add utils directory to Python path for imports
utils_dir = os.path.join(NODE_DIR, "utils")
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# --- NODE DISCOVERY ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    nodes_root_dir = os.path.join(NODE_DIR, "nodes")
    if not os.path.isdir(nodes_root_dir):
        print("lunaCore: No 'nodes' directory found. Skipping node loading.")
        return

    # The new, All-Seeing Eye: os.walk()
    # This will traverse the entire directory tree, including subdirectories.
    for root, dirs, files in os.walk(nodes_root_dir):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__"):
                try:
                    # Construct a proper module path for nested files
                    # e.g., nodes/loaders/checkpoint_loader.py -> nodes.loaders.checkpoint_loader
                    relative_path = os.path.relpath(root, nodes_root_dir)
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
        from nodes.luna_daemon_api import register_routes as register_daemon_routes
        register_daemon_routes()
    except Exception as e:
        pass  # Silently fail if daemon API not available
    
    try:
        # Import and register other API routes if needed
        from nodes.luna_civitai_scraper import register_routes as register_civitai_routes
        if callable(register_civitai_routes):
            register_civitai_routes()
    except (ImportError, AttributeError):
        pass
    
    try:
        from nodes.luna_wildcard_connections import register_routes as register_wildcard_routes
        if callable(register_wildcard_routes):
            register_wildcard_routes()
    except (ImportError, AttributeError):
        pass

# Try to register routes now (may work if PromptServer already initialized)
register_api_routes()

# Also hook into PromptServer setup if possible
try:
    from server import PromptServer
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