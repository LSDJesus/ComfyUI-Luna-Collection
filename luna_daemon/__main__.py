"""
Luna Daemon - Entry point for running as a module.
Usage: python -m custom_nodes.ComfyUI-Luna-Collection.luna_daemon
"""

from .server import main

if __name__ == "__main__":
    main()
