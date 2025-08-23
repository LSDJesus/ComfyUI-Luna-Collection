# This file is the heart of our Pyrite Core node pack.
# It tells Python that this directory is a package and, more importantly,
# it dynamically finds all our node definition files and tells ComfyUI about them.

import os
import importlib.util

# Get the directory of the current file, which is the root of our package.
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# These are the master dictionaries that will hold all our node mappings.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# This function will scan our directory for .py files, import them,
# and register their node mappings.
def setup_nodes():
    for filename in os.listdir(NODE_DIR):
        if filename.endswith(".py"):
            # We don't want to import ourselves, that would be a paradox.
            if filename == "__init__.py":
                continue

            module_name = filename[:-3]
            # Construct the full path to the module.
            module_path = os.path.join(NODE_DIR, filename)
            
            # This is the standard Python magic to import a module from a file path.
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Now that the module is loaded, we can look for its node mappings.
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

# Run the setup function to populate our master mappings.
setup_nodes()

# This is a little bit of magic for the Use Everywhere node.
# It helps it find our nodes for its auto-connection features.
# We'll include it for future-proofing.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']