"""Feature engineering for NFL game predictions."""

import logging
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and transform features for NFL predictions."""
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize feature engineer.
        
        Args:
            scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler(scaling_method)
        self.feature_names = []
        logger.info(f"Initialized FeatureEngineer with {scaling_method} scaling")
    
    @staticmethod
    def _get_scaler(method: str):
        """Get scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(method, StandardScaler())
    
    def create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team-level features.
        
        Args:
            df: DataFrame with team statistics
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating team features")
        df = df.copy()
        
        # Offensive efficiency
        if 'total_yards' in df.columns and 'plays' in df.columns:
            df['yards_per_play'] = df['total_yards'] / (df['plays'] + 1)  # Avoid division by zero
        
        # Pass/run ratio
        if 'passing_yards' in df.columns and 'rushing_yards' in df.columns:
            df['pass_run_ratio'] = df['passing_yards'] / (df['rushing_yards'] + 1)
        
        # Turnover differential (requires opponent data)
        if 'turnovers' in df.columns:
            df['turnover_impact'] = -df['turnovers'] * 3  # Approximate point value
        
        # Success rate ranking
        if 'success_rate' in df.columns:
            df['success_rate_rank'] = df.groupby('game_id')['success_rate'].rank(pct=True)
        
        # Home field advantage
        if 'is_home' in df.columns:
            df['home_advantage'] = df['is_home'].astype(int) * 2.5  # Historical average
        
        logger.info(f"Created {len([c for c in df.columns if c not in self.feature_names])} new team features")
        return df
    
    def create_matchup_features(self, home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create matchup-specific features.
        
        Args:
            home_df: Home team statistics
            away_df: Away team statistics
            
        Returns:
            DataFrame with matchup features
        """
        logger.info("Creating matchup features")
        
        matchup_features = pd.DataFrame()
        
        # Ensure same index
        home_df = home_df.reset_index(drop=True)
        away_df = away_df.reset_index(drop=True)
        
        # Differential features
        numeric_cols = home_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in away_df.columns:
                matchup_features[f'{col}_diff'] = home_df[col] - away_df[col]
                matchup_features[f'{col}_ratio'] = home_df[col] / (away_df[col] + 1)
        
        # Combined features
        if 'total_epa' in home_df.columns and 'total_epa' in away_df.columns:
            matchup_features['combined_epa'] = home_df['total_epa'] + away_df['total_epa']
            matchup_features['epa_advantage'] = home_df['total_epa'] - away_df['total_epa']
        
        logger.info(f"Created {len(matchup_features.columns)} matchup features")
        return matchup_features
    
    def create_momentum_features(
        self, 
        df: pd.DataFrame,
        team_col: str = 'team',
        window: int = 3
    ) -> pd.DataFrame:
        """
        Create momentum-based features.
        
        Args:
            df: DataFrame with team statistics
            team_col: Column identifying teams
            window: Window for momentum calculation
            
        Returns:
            DataFrame with momentum features
        """
        logger.info(f"Creating momentum features with window={window}")
        df = df.copy()
        
        # Sort by game for proper ordering
        df = df.sort_values('game_id')
        
        # Win streak
        if 'won' in df.columns:
            df['win_streak'] = df.groupby(team_col)['won'].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        
        # Performance trend
        if 'total_epa' in df.columns:
            df['epa_trend'] = df.groupby(team_col)['total_epa'].transform(
                lambda x: x.diff().rolling(window, min_periods=1).mean()
            )
        
        # Scoring trend
        if 'points_scored' in df.columns:
            df['scoring_trend'] = df.groupby(team_col)['points_scored'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        logger.info(f"Created momentum features")
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced statistical features.
        
        Args:
            df: DataFrame with base statistics
            
        Returns:
            DataFrame with advanced features
        """
        logger.info("Creating advanced features")
        df = df.copy()
        
        # Pythagorean expectation (if we have points for and against)
        if 'points_for' in df.columns and 'points_against' in df.columns:
            df['pythagorean_exp'] = df['points_for']**2.37 / (
                df['points_for']**2.37 + df['points_against']**2.37
            )
        
        # Strength of schedule (requires opponent data)
        if 'opponent_win_pct' in df.columns:
            df['sos_adjusted'] = df['success_rate'] * (1 + df['opponent_win_pct'])
        
        # Efficiency metrics
        if 'total_yards' in df.columns and 'turnovers' in df.columns:
            df['efficiency_score'] = df['total_yards'] / (df['turnovers'] + 1) / 100
        
        logger.info("Created advanced features")
        return df
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Scaled feature array
        """
        logger.info(f"Fitting and transforming {len(X.columns)} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Scaled features to shape {X_scaled.shape}")
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Scaled feature array
        """
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Transform
        return self.scaler.transform(X)
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
