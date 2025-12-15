"""
Luna Daemon - Entry point for running as a module or direct script.
Usage: 
  - python -m luna_daemon (when in custom_nodes/ComfyUI-Luna-Collection)
  - python luna_daemon/__main__.py (direct execution)
"""

import os
import importlib.util

# Try relative import first (package context)
try:
    from .daemon_server import main
except (ImportError, ValueError):
    # Fallback: Direct file load (when run as script, not module)
    daemon_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(daemon_dir, "daemon_server.py")
    
    spec = importlib.util.spec_from_file_location("luna_daemon.daemon_server", server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {server_path}")
    
    server_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_mod)
    main = getattr(server_mod, "main")

if __name__ == "__main__":
    main()
