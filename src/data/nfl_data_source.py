"""NFL data source implementation using nfl_data_py."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime

from src.data.data_sources import DataSource

logger = logging.getLogger(__name__)


class NFLDataSource(DataSource):
    """
    Data source for NFL statistics using nfl_data_py.
    
    This wraps the nfl_data_py library which provides access to play-by-play data,
    similar to the R nflfastR package.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NFL data source."""
        super().__init__(name="NFLDataPy", config=config)
        self._nfl_data = None
        
    def _validate_config(self) -> None:
        """Validate NFL data source configuration."""
        # Set default configuration
        self.config.setdefault('seasons', [2023, 2024])
        self.config.setdefault('cache', True)
        
        # Validate season years
        current_year = datetime.now().year
        for season in self.config['seasons']:
            if not isinstance(season, int):
                raise ValueError(f"Season must be integer, got {type(season)}")
            if season < 1999 or season > current_year:
                raise ValueError(f"Invalid season {season}. Must be between 1999 and {current_year}")
    
    def _lazy_import(self):
        """Lazy import nfl_data_py to avoid import errors if not installed."""
        if self._nfl_data is None:
            try:
                import nfl_data_py as nfl
                self._nfl_data = nfl
                logger.info("Successfully imported nfl_data_py")
            except ImportError as e:
                logger.error("Failed to import nfl_data_py. Please install it: pip install nfl_data_py")
                raise ImportError("nfl_data_py is required for NFLDataSource") from e
        return self._nfl_data
    
    def test_connection(self) -> bool:
        """
        Test if nfl_data_py can be imported and basic data can be fetched.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            nfl = self._lazy_import()
            # Try to fetch team information as a test
            teams = nfl.import_team_desc()
            return teams is not None and len(teams) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def fetch(self, data_type: str = 'pbp', **kwargs) -> pd.DataFrame:
        """
        Fetch NFL data based on type.
        
        Args:
            data_type: Type of data to fetch ('pbp', 'schedules', 'rosters', etc.)
            **kwargs: Additional parameters for specific data types
            
        Returns:
            DataFrame with requested data
        """
        nfl = self._lazy_import()
        
        # Map data types to nfl_data_py functions
        data_fetchers = {
            'pbp': self._fetch_play_by_play,
            'schedules': self._fetch_schedules,
            'rosters': self._fetch_rosters,
            'teams': self._fetch_teams,
            'players': self._fetch_players,
            'combine': self._fetch_combine,
            'draft': self._fetch_draft_picks,
            'weekly': self._fetch_weekly_data,
        }
        
        if data_type not in data_fetchers:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(data_fetchers.keys())}")
        
        logger.info(f"Fetching {data_type} data with params: {kwargs}")
        return data_fetchers[data_type](**kwargs)
    
    def _fetch_play_by_play(self, seasons: Optional[List[int]] = None, **kwargs) -> pd.DataFrame:
        """Fetch play-by-play data."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        
        logger.info(f"Fetching play-by-play data for seasons: {seasons}")
        return nfl.import_pbp_data(seasons, cache=self.config['cache'])
    
    def _fetch_schedules(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch game schedules."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        
        logger.info(f"Fetching schedules for seasons: {seasons}")
        return nfl.import_schedules(seasons)
    
    def _fetch_rosters(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch team rosters."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        
        logger.info(f"Fetching rosters for seasons: {seasons}")
        return nfl.import_rosters(seasons)
    
    def _fetch_teams(self) -> pd.DataFrame:
        """Fetch team descriptions."""
        nfl = self._lazy_import()
        return nfl.import_team_desc()
    
    def _fetch_players(self) -> pd.DataFrame:
        """Fetch player information."""
        nfl = self._lazy_import()
        return nfl.import_ids()
    
    def _fetch_combine(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch NFL combine data."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        return nfl.import_combine_data(seasons)
    
    def _fetch_draft_picks(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch NFL draft picks."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        return nfl.import_draft_picks(seasons)
    
    def _fetch_weekly_data(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch weekly player data."""
        nfl = self._lazy_import()
        seasons = seasons or self.config['seasons']
        
        logger.info(f"Fetching weekly data for seasons: {seasons}")
        return nfl.import_weekly_data(seasons)
