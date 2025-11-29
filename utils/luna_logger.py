"""
Luna Collection Logging Utility

Centralized logging system for consistent output across all Luna nodes.
"""

import logging
import sys
from typing import Optional
from .constants import LOG_FORMAT, LOG_DATE_FORMAT

class LunaLogger:
    """
    Centralized logger for Luna Collection nodes.

    Provides consistent logging format and level control across all nodes.
    """

    def __init__(self, name: str = "LunaCollection", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter with proper %-style format
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent duplicate logs

    def info(self, message: str, node_name: Optional[str] = None):
        """Log info message"""
        prefix = f"[{node_name}] " if node_name else ""
        self.logger.info(f"{prefix}{message}")

    def warning(self, message: str, node_name: Optional[str] = None):
        """Log warning message"""
        prefix = f"[{node_name}] " if node_name else ""
        self.logger.warning(f"{prefix}{message}")

    def error(self, message: str, node_name: Optional[str] = None):
        """Log error message"""
        prefix = f"[{node_name}] " if node_name else ""
        self.logger.error(f"{prefix}{message}")

    def debug(self, message: str, node_name: Optional[str] = None):
        """Log debug message"""
        prefix = f"[{node_name}] " if node_name else ""
        self.logger.debug(f"{prefix}{message}")

# Global logger instance
luna_logger = LunaLogger()

def get_logger(node_name: str) -> LunaLogger:
    """
    Get a logger instance for a specific node.

    Args:
        node_name: Name of the node for consistent logging

    Returns:
        LunaLogger instance configured for the node
    """
    return LunaLogger(f"Luna{node_name}", logging.INFO)