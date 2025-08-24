import os
import sys
import importlib.util
import traceback

NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- THE MASTER KEY ---
# Step 1: We brute-force load our tools using their absolute file path.
# No more polite imports. We kick the door down.
tiling_path = os.path.join(NODE_DIR, "utils", "tiling.py")
spec = importlib.util.spec_from_file_location("pyrite_tiling_util", tiling_path)
tiling_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiling_module)
pyrite_tiling_orchestrator = tiling_module.pyrite_tiling_orchestrator

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    nodes_dir = os.path.join(NODE_DIR, "nodes")
    if not os.path.isdir(nodes_dir): return

    for filename in os.listdir(nodes_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            try:
                module_name = f"nodes.{filename[:-3]}"
                module_path = os.path.join(nodes_dir, filename)
                
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    for node_name, node_class in module.NODE_CLASS_MAPPINGS.items():
                        # Step 2: We inject the pre-loaded tool directly into the child class.
                        if "upscaler_advanced" in node_name.lower():
                            node_class.pyrite_tiling_orchestrator = pyrite_tiling_orchestrator
                        
                        NODE_CLASS_MAPPINGS[node_name] = node_class

                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"PyriteCore: Failed to load node file {filename}: {e}")
                print(traceback.format_exc())

setup_nodes()
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']