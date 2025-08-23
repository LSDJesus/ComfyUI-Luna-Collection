import os
import importlib.util
import traceback

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def setup_nodes():
    for dirpath, dirnames, filenames in os.walk(NODE_DIR):
        # We're only interested in the 'nodes' subdirectory for finding nodes.
        if "nodes" in dirpath:
            for filename in filenames:
                if filename.endswith(".py"):
                    if filename == "__init__.py":
                        continue
                    
                    try:
                        module_name = filename[:-3]
                        module_path = os.path.join(dirpath, filename)
                        
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
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