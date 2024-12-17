"""Data processing and cleaning module for NFL data."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class NFLDataProcessor:
    """Process and clean NFL data for model training."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.team_mappings = self._get_team_mappings()
        logger.info("Initialized NFLDataProcessor")
    
    @staticmethod
    def _get_team_mappings() -> Dict[str, str]:
        """Get standard team abbreviation mappings."""
        return {
            'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
            'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
            'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
            'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC',
            'LA': 'LAR', 'LAR': 'LAR', 'LAC': 'LAC', 'LV': 'LV',
            'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO',
            'NYG': 'NYG', 'NYJ': 'NYJ', 'OAK': 'LV', 'PHI': 'PHI',
            'PIT': 'PIT', 'SD': 'LAC', 'SEA': 'SEA', 'SF': 'SF',
            'STL': 'LAR', 'TB': 'TB', 'TEN': 'TEN', 'WAS': 'WAS'
        }
    
    def clean_pbp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean play-by-play data.
        
        Args:
            df: Raw play-by-play DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning play-by-play data with {len(df)} rows")
        
        df = df.copy()
        
        # Remove rows with missing game_id
        df = df.dropna(subset=['game_id'])
        
        # Standardize team names
        if 'posteam' in df.columns:
            df['posteam'] = df['posteam'].map(self.team_mappings).fillna(df['posteam'])
        if 'defteam' in df.columns:
            df['defteam'] = df['defteam'].map(self.team_mappings).fillna(df['defteam'])
        
        # Convert data types
        numeric_cols = ['yards_gained', 'epa', 'wp', 'wpa']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out special plays if needed
        if 'play_type' in df.columns:
            df = df[df['play_type'].notna()]
        
        logger.info(f"Cleaned data has {len(df)} rows")
        return df
    
    def aggregate_game_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate play-by-play data to game level.
        
        Args:
            pbp_df: Play-by-play DataFrame
            
        Returns:
            Game-level statistics DataFrame
        """
        logger.info("Aggregating play-by-play data to game level")
        
        game_stats = []
        
        for game_id in pbp_df['game_id'].unique():
            game_data = pbp_df[pbp_df['game_id'] == game_id]
            
            # Get teams
            home_team = game_data['home_team'].iloc[0] if 'home_team' in game_data else None
            away_team = game_data['away_team'].iloc[0] if 'away_team' in game_data else None
            
            if not home_team or not away_team:
                continue
            
            # Calculate stats for each team
            for team in [home_team, away_team]:
                team_plays = game_data[game_data['posteam'] == team]
                
                stats = {
                    'game_id': game_id,
                    'team': team,
                    'is_home': team == home_team,
                    'plays': len(team_plays),
                    'total_yards': team_plays['yards_gained'].sum() if 'yards_gained' in team_plays else 0,
                    'passing_yards': team_plays[team_plays['play_type'] == 'pass']['yards_gained'].sum() if 'play_type' in team_plays else 0,
                    'rushing_yards': team_plays[team_plays['play_type'] == 'run']['yards_gained'].sum() if 'play_type' in team_plays else 0,
                    'turnovers': team_plays['interception'].sum() + team_plays['fumble_lost'].sum() if 'interception' in team_plays and 'fumble_lost' in team_plays else 0,
                    'total_epa': team_plays['epa'].sum() if 'epa' in team_plays else 0,
                    'success_rate': (team_plays['epa'] > 0).mean() if 'epa' in team_plays else 0
                }
                
                game_stats.append(stats)
        
        result_df = pd.DataFrame(game_stats)
        logger.info(f"Aggregated to {len(result_df)} team-game records")
        
        return result_df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        group_col: str,
        feature_cols: List[str],
        windows: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create rolling average features.
        
        Args:
            df: DataFrame with game statistics
            group_col: Column to group by (usually 'team')
            feature_cols: Columns to calculate rolling averages for
            windows: Window sizes for rolling averages
            
        Returns:
            DataFrame with rolling features added
        """
        logger.info(f"Creating rolling features for {len(feature_cols)} columns")
        
        df = df.sort_values(['game_id'])
        
        for window in windows:
            for col in feature_cols:
                if col in df.columns:
                    feature_name = f"{col}_rolling_{window}"
                    df[feature_name] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
        
        logger.info(f"Added {len(windows) * len(feature_cols)} rolling features")
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Strategy for handling missing values ('mean', 'median', 'forward_fill', 'drop')
            columns: Specific columns to handle (None for all numeric columns)
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if strategy == 'mean':
            for col in columns:
                if col in df.columns:
                    df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in columns:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'forward_fill':
            df[columns] = df[columns].fillna(method='ffill')
        elif strategy == 'drop':
            df = df.dropna(subset=columns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = df[columns].isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with all data
            target_col: Name of target column
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Preparing training data with {len(feature_cols)} features")
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = set(feature_cols) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Prepared {len(X)} samples for training")
        
        return X, y
