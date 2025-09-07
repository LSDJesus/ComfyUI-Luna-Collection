import os
import sys
import importlib.util
import traceback

# This remains. It tells ComfyUI where to find our JavaScript files.
WEB_DIRECTORY = "./js"

# --- GLOBAL CONTEXT & UTILITIES ---
# We will create a single, shared object that holds all our magnificent utilities.
# This is cleaner than injecting them one by one.
class LunaUtils:
    pass

LUNA_UTILS = LunaUtils()

# The base directory of our collection
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- THE WARDEN'S DUTY ---
# Load all our utilities and attach them to our global context object.
try:
    from .utils.tiling import luna_tiling_orchestrator
    LUNA_UTILS.tiling_orchestrator = luna_tiling_orchestrator
except ImportError:
    print("lunaCore: Tiling utility not found.")

try:
    from .utils.mediapipe_engine import Mediapipe_Engine
    LUNA_UTILS.mediapipe_engine = Mediapipe_Engine
except ImportError:
    print("lunaCore: Mediapipe utility not found.")
    
# Add any other utilities here in the future...

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
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        for node_name, node_class in module.NODE_CLASS_MAPPINGS.items():
                            # The new, elegant way to give every node access to our tools.
                            # We pass the entire utility object.
                            node_class.LUNA_UTILS = LUNA_UTILS
                            
                            NODE_CLASS_MAPPINGS[node_name] = node_class

                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

                except Exception as e:
                    print(f"lunaCore: Failed to load node file {filename}: {e}")
                    print(traceback.format_exc())

# Run the setup
setup_nodes()

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