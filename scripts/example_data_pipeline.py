"""Example script demonstrating the NFL data pipeline."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.nfl_data_loader import NFLDataLoader
from data.data_processor import NFLDataProcessor
from utils.config import get_config
from utils.logger import setup_logger
import pandas as pd


def main():
    """Run example data pipeline."""
    # Setup
    logger = setup_logger("example_pipeline")
    config = get_config()
    
    # Initialize components
    loader = NFLDataLoader(cache_dir=Path(config.get('data.raw_data_path', 'data/raw')))
    processor = NFLDataProcessor()
    
    # Get seasons from config
    seasons = config.get('data.collection.seasons', [2023])
    logger.info(f"Processing data for seasons: {seasons}")
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading NFL data...")
        
        # Check cache first
        pbp_data = loader.get_cached_data('pbp', seasons)
        
        if pbp_data is None:
            logger.info("No cached data found. Downloading from nflfastR...")
            # Note: This requires nfl_data_py to be installed
            # For demonstration, we'll create sample data
            pbp_data = create_sample_data()
            
        schedules = loader.get_cached_data('schedules', seasons)
        if schedules is None:
            schedules = create_sample_schedule()
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data...")
        clean_pbp = processor.clean_pbp_data(pbp_data)
        
        # Step 3: Aggregate to game level
        logger.info("Step 3: Aggregating to game level...")
        game_stats = processor.aggregate_game_stats(clean_pbp)
        
        # Step 4: Create features
        logger.info("Step 4: Creating rolling features...")
        feature_cols = ['total_yards', 'turnovers', 'total_epa', 'success_rate']
        windows = config.get('features.rolling_windows', [3, 5, 10])
        
        game_stats_with_features = processor.create_rolling_features(
            game_stats,
            group_col='team',
            feature_cols=feature_cols,
            windows=windows
        )
        
        # Step 5: Handle missing values
        logger.info("Step 5: Handling missing values...")
        final_data = processor.handle_missing_values(
            game_stats_with_features,
            strategy='mean'
        )
        
        # Save processed data
        output_path = Path(config.get('data.processed_data_path', 'data/processed'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"game_stats_{'_'.join(map(str, seasons))}.parquet"
        final_data.to_parquet(output_file)
        logger.info(f"Saved processed data to {output_file}")
        
        # Display summary
        logger.info("\n" + "="*50)
        logger.info("Pipeline Summary:")
        logger.info(f"  - Seasons processed: {seasons}")
        logger.info(f"  - Games processed: {len(game_stats) // 2}")  # Divide by 2 for teams
        logger.info(f"  - Features created: {len(final_data.columns)}")
        logger.info(f"  - Final dataset shape: {final_data.shape}")
        logger.info("="*50)
        
        return final_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def create_sample_data():
    """Create sample play-by-play data for demonstration."""
    import numpy as np
    
    # Create sample data
    teams = ['BUF', 'KC', 'PHI', 'SF', 'DAL', 'MIA']
    plays = []
    
    for i in range(100):
        play = {
            'game_id': f'2023_01_{teams[i % len(teams)]}_{teams[(i + 1) % len(teams)]}',
            'play_id': i,
            'posteam': teams[i % len(teams)],
            'defteam': teams[(i + 1) % len(teams)],
            'home_team': teams[i % len(teams)],
            'away_team': teams[(i + 1) % len(teams)],
            'play_type': np.random.choice(['pass', 'run']),
            'yards_gained': np.random.normal(5, 10),
            'epa': np.random.normal(0, 2),
            'wp': np.random.uniform(0.2, 0.8),
            'interception': np.random.choice([0, 1], p=[0.95, 0.05]),
            'fumble_lost': np.random.choice([0, 1], p=[0.98, 0.02])
        }
        plays.append(play)
    
    return pd.DataFrame(plays)


def create_sample_schedule():
    """Create sample schedule data for demonstration."""
    teams = ['BUF', 'KC', 'PHI', 'SF', 'DAL', 'MIA']
    games = []
    
    for week in range(1, 5):
        for i in range(0, len(teams), 2):
            game = {
                'game_id': f'2023_{week:02d}_{teams[i]}_{teams[i+1]}',
                'week': week,
                'home_team': teams[i],
                'away_team': teams[i+1],
                'game_date': f'2023-09-{week + 7:02d}'
            }
            games.append(game)
    
    return pd.DataFrame(games)


if __name__ == "__main__":
    main()
