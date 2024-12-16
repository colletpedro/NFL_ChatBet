"""Base classes and interfaces for data sources."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data source.
        
        Args:
            name: Name of the data source
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._validate_config()
        
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for the data source."""
        pass
    
    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """
        Fetch data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            Data from the source
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the data source is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
