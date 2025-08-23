import os
import sys
import importlib.util
import traceback

# This is the new, crucial piece of magic.
# We are forcefully adding the path to our entire package to the Python system path.
# This allows any of our files to import any other file using an absolute path.
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    # We now look for a 'nodes' subdirectory specifically.
    nodes_dir = os.path.join(NODE_DIR, "nodes")
    if not os.path.isdir(nodes_dir):
        return

    for filename in os.listdir(nodes_dir):
        if filename.endswith(".py"):
            if filename.startswith("__"):
                continue
            
            try:
                module_name = f"nodes.{filename[:-3]}" # Use a proper package-like name
                module_path = os.path.join(nodes_dir, filename)
                
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module # Register it in sys.modules
                spec.loader.exec_module(module)

                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"PyriteCore: Failed to load node file {filename}: {e}")
                print(traceback.format_exc())

setup_nodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']