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

# Try relative import first (when run as module), fall back to direct import
try:
    from .server import main
except ImportError:
    from server import main

if __name__ == "__main__":
    main()
