"""Configuration management for the NFL prediction system."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Manage application configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path("config/config.yml")
            if not config_path.exists():
                # Fall back to example config
                config_path = Path("config/config.example.yml")
        
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Loaded configuration from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'sources': {
                    'nfl_api': {
                        'rate_limit': 100
                    }
                },
                'collection': {
                    'seasons': [2020, 2021, 2022, 2023],
                    'update_frequency': 'weekly'
                },
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed'
            },
            'features': {
                'rolling_windows': [3, 5, 10],
                'scaling_method': 'standard'
            },
            'models': {
                'training': {
                    'test_size': 0.2,
                    'validation_size': 0.1,
                    'random_state': 42,
                    'cross_validation_folds': 5
                },
                'model_path': 'models/trained'
            },
            'logging': {
                'level': 'INFO',
                'format': 'text',
                'file': 'logs/nfl_prediction.log'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'data.sources.nfl_api.rate_limit')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        logger.info(f"Set config {key_path} = {value}")
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (uses current path if None)
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {save_path}")
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('models', {})
    
    @property
    def feature_config(self) -> Dict[str, Any]:
        """Get feature configuration section."""
        return self.config.get('features', {})


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(path: Path) -> Config:
    """Load configuration from specific path."""
    global _config
    _config = Config(path)
    return _config
