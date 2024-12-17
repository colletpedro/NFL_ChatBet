"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def find_project_root() -> Path:
    """
    Find the project root directory by looking for key indicators.
    
    Returns:
        Path to project root
        
    Raises:
        ConfigurationError: If project root cannot be determined
    """
    current = Path.cwd()
    
    # Look for indicators of project root
    indicators = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt']
    
    while current != current.parent:
        if any((current / indicator).exists() for indicator in indicators):
            return current
        current = current.parent
    
    # If we can't find it, use current directory
    logger.warning("Could not find project root, using current directory")
    return Path.cwd()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, looks for config/config.yml
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration file is not found or invalid
    """
    if config_path is None:
        # Try to find config file
        project_root = find_project_root()
        config_path = project_root / 'config' / 'config.yml'
        
        # Fall back to example config if main config doesn't exist
        if not config_path.exists():
            example_path = project_root / 'config' / 'config.example.yml'
            if example_path.exists():
                logger.warning(f"Using example config from {example_path}")
                config_path = example_path
            else:
                raise ConfigurationError(
                    f"No configuration file found. Please create {config_path} "
                    f"or copy from config.example.yml"
                )
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config or {}
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from nested configuration dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., 'data.sources.nfl_api.base_url')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


class Config:
    """Configuration singleton for easy access throughout the application."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file."""
        self._config = load_config(config_path)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value."""
        if self._config is None:
            self.load()
        return get_config_value(self._config, key_path, default)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        if self._config is None:
            self.load()
        return self._config
