"""
Configuration Management

Load, validate, and manage configuration settings for Luna Collection.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manage configuration settings for Luna Collection."""

    DEFAULT_CONFIG = {
        'performance': {
            'enable_monitoring': True,
            'log_interval': 1.0,
            'cache_size': 1000
        },
        'validation': {
            'enable_caching': True,
            'strict_mode': False,
            'max_errors': 10
        },
        'memory': {
            'max_usage_mb': 4096,
            'gc_threshold': 0.8,
            'enable_monitoring': True
        }
    }

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self._config = self._load_config()

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration setting."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_setting(self, key: str, value: Any):
        """Set a configuration setting."""
        keys = key.split('.')
        config = self._config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration structure."""
        if config is None:
            config = self._config

        # Basic structure validation
        required_sections = ['performance', 'validation', 'memory']

        for section in required_sections:
            if section not in config:
                return False

            if not isinstance(config[section], dict):
                return False

        return True

    def save_config(self):
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            return True
        except Exception:
            return False

    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self._config = self.DEFAULT_CONFIG.copy()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)

                # Merge with defaults
                config = self.DEFAULT_CONFIG.copy()
                self._deep_update(config, loaded_config)
                return config

            except Exception:
                # If loading fails, use defaults
                pass

        return self.DEFAULT_CONFIG.copy()

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        config_dir = Path.home() / '.luna_collection'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'config.json')