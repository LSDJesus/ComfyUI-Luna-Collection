import os
import sys
import importlib.util
import traceback

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

# --- THE WARDEN'S DUTY ---
# 1. We load the shared tool from its file. This is the ONLY place an import across our subdirectories happens.
from .utils.tiling import luna_tiling_orchestrator

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
                        # 2. We find the child that needs the tool.
                        if "luna_AdvancedUpscaler" in node_name:
                            # 3. We give the child the tool directly...
                            # 4. ...and we tell it that the tool is a GUEST, not a LIMB.
                            # The @staticmethod decorator prevents Python from passing 'self' automatically.
                            node_class.luna_tiling_orchestrator = staticmethod(luna_tiling_orchestrator)
                        
                        NODE_CLASS_MAPPINGS[node_name] = node_class

                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"lunaCore: Failed to load node file {filename}: {e}")
                print(traceback.format_exc())

setup_nodes()
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
