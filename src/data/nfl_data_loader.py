"""NFL data loader using nfl_data_py package for accessing nflfastR data."""

import logging
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class NFLDataLoader:
    """Load NFL data from nflfastR datasets via nfl_data_py."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the NFL data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized NFLDataLoader with cache dir: {self.cache_dir}")
    
    def load_pbp_data(
        self, 
        years: List[int],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load play-by-play data for specified years.
        
        Args:
            years: List of years to load data for
            columns: Specific columns to load (None for all)
            
        Returns:
            DataFrame with play-by-play data
        """
        try:
            import nfl_data_py as nfl
            
            logger.info(f"Loading play-by-play data for years: {years}")
            pbp_df = nfl.import_pbp_data(years, columns=columns)
            
            # Cache the data
            cache_file = self.cache_dir / f"pbp_{'_'.join(map(str, years))}.parquet"
            pbp_df.to_parquet(cache_file)
            logger.info(f"Cached {len(pbp_df)} plays to {cache_file}")
            
            return pbp_df
            
        except ImportError:
            logger.error("nfl_data_py not installed. Install with: pip install nfl_data_py")
            raise
        except Exception as e:
            logger.error(f"Error loading play-by-play data: {e}")
            raise
    
    def load_schedules(self, years: List[int]) -> pd.DataFrame:
        """
        Load NFL game schedules.
        
        Args:
            years: List of years to load schedules for
            
        Returns:
            DataFrame with schedule data
        """
        try:
            import nfl_data_py as nfl
            
            logger.info(f"Loading schedules for years: {years}")
            schedules = nfl.import_schedules(years)
            
            # Cache the data
            cache_file = self.cache_dir / f"schedules_{'_'.join(map(str, years))}.parquet"
            schedules.to_parquet(cache_file)
            logger.info(f"Cached {len(schedules)} games to {cache_file}")
            
            return schedules
            
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")
            raise
    
    def load_team_stats(self, years: List[int]) -> pd.DataFrame:
        """
        Load aggregated team statistics.
        
        Args:
            years: List of years to load stats for
            
        Returns:
            DataFrame with team statistics
        """
        try:
            import nfl_data_py as nfl
            
            logger.info(f"Loading weekly team data for years: {years}")
            team_stats = nfl.import_weekly_data(years)
            
            # Cache the data
            cache_file = self.cache_dir / f"team_stats_{'_'.join(map(str, years))}.parquet"
            team_stats.to_parquet(cache_file)
            logger.info(f"Cached team stats to {cache_file}")
            
            return team_stats
            
        except Exception as e:
            logger.error(f"Error loading team stats: {e}")
            raise
    
    def load_rosters(self, years: List[int]) -> pd.DataFrame:
        """
        Load NFL rosters data.
        
        Args:
            years: List of years to load rosters for
            
        Returns:
            DataFrame with roster data
        """
        try:
            import nfl_data_py as nfl
            
            logger.info(f"Loading rosters for years: {years}")
            rosters = nfl.import_rosters(years)
            
            # Cache the data
            cache_file = self.cache_dir / f"rosters_{'_'.join(map(str, years))}.parquet"
            rosters.to_parquet(cache_file)
            logger.info(f"Cached roster data to {cache_file}")
            
            return rosters
            
        except Exception as e:
            logger.error(f"Error loading rosters: {e}")
            raise
    
    def get_cached_data(self, data_type: str, years: List[int]) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available.
        
        Args:
            data_type: Type of data (pbp, schedules, team_stats, rosters)
            years: Years to check cache for
            
        Returns:
            Cached DataFrame or None if not found
        """
        cache_file = self.cache_dir / f"{data_type}_{'_'.join(map(str, years))}.parquet"
        
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        return None
