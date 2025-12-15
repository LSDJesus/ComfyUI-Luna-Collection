"""
Luna Daemon Server - Entry Point

This file provides backward compatibility for code that imports from server.py.
The actual implementation is now in daemon_server.py.
"""

# Re-export everything from daemon_server for backward compatibility
from .daemon_server import *
from .daemon_server import main, LunaDaemon

if __name__ == "__main__":
    main()
