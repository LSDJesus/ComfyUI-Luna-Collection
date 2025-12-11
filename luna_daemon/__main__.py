"""
Luna Daemon - Entry point for running as a module or direct script.
Usage: 
  - python -m luna_daemon (when in custom_nodes/ComfyUI-Luna-Collection)
  - python luna_daemon/__main__.py (direct execution)
"""

import sys
import os

# Add parent directory to path so we can import server module
daemon_dir = os.path.dirname(os.path.abspath(__file__))
if daemon_dir not in sys.path:
    sys.path.insert(0, daemon_dir)

# Try relative import first (when run as module), fall back to package import or direct file load
try:
    from .server import main
except Exception:
    try:
        # Try importing as a package module (e.g., when parent dir is on PYTHONPATH)
        from luna_daemon.server import main
    except Exception:
        # Final fallback: load server.py directly from the same directory
        import importlib.util
        server_path = os.path.join(daemon_dir, "server.py")
        spec = importlib.util.spec_from_file_location("luna_daemon.server", server_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec or loader for {server_path}")
        server_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_mod)
        main = getattr(server_mod, "main")

if __name__ == "__main__":
    main()
