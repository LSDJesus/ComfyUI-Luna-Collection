# v0.2.0 - The Nanny
import os
import sys
import importlib.util
import traceback

# This is the crucial part. We get our root directory.
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# We tell Python's system that our home is a place where it should look for code.
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

# And now, we explicitly import our own sub-packages. This is us, the nanny,
# grabbing our children by the hand and saying "You are a family. Behave."
# This makes 'nodes' and 'utils' recognizable to each other.
from . import nodes
from . import utils

# The rest of our beautiful, dynamic loader remains the same.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    nodes_dir = os.path.join(NODE_DIR, "nodes")
    if not os.path.isdir(nodes_dir):
        return

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
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"PyriteCore: Failed to load node file {filename}: {e}")
                print(traceback.format_exc())

setup_nodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']