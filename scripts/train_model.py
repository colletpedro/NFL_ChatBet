"""Training script for NFL prediction models."""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data.nfl_data_loader import NFLDataLoader
from data.data_processor import NFLDataProcessor
from features.feature_engineer import FeatureEngineer
from models.random_forest_model import RandomForestPredictor
from models.xgboost_model import XGBoostPredictor
from utils.config import get_config
from utils.logger import setup_logger


def prepare_data(seasons: list, config):
    """Load and prepare data for training."""
    logger = setup_logger("data_preparation")
    
    # Initialize components
    loader = NFLDataLoader(cache_dir=Path(config.get('data.raw_data_path', 'data/raw')))
    processor = NFLDataProcessor()
    engineer = FeatureEngineer(scaling_method=config.get('features.scaling_method', 'standard'))
    
    logger.info(f"Loading data for seasons: {seasons}")
    
    # Load data (using sample for demonstration)
    # In production, this would load real data from nflfastR
    pbp_data = create_sample_training_data()
    
    # Process data
    logger.info("Processing data...")
    clean_data = processor.clean_pbp_data(pbp_data)
    game_stats = processor.aggregate_game_stats(clean_data)
    
    # Create features
    logger.info("Engineering features...")
    game_stats = engineer.create_team_features(game_stats)
    
    # Create rolling features
    feature_cols = ['total_yards', 'turnovers', 'total_epa', 'success_rate']
    windows = config.get('features.rolling_windows', [3, 5, 10])
    game_stats = processor.create_rolling_features(
        game_stats,
        group_col='team',
        feature_cols=feature_cols,
        windows=windows
    )
    
    # Handle missing values
    game_stats = processor.handle_missing_values(game_stats, strategy='mean')
    
    # Create target variable (simplified: home team wins)
    game_stats['target'] = (game_stats['total_epa'] > game_stats['total_epa'].median()).astype(int)
    
    return game_stats, engineer


def create_sample_training_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    teams = ['BUF', 'KC', 'PHI', 'SF', 'DAL', 'MIA', 'CIN', 'MIN', 
             'BAL', 'LAC', 'JAX', 'NYG', 'DET', 'SEA', 'GB', 'NO']
    
    data = []
    for game_num in range(100):
        home_idx = np.random.randint(0, len(teams))
        away_idx = np.random.randint(0, len(teams))
        while away_idx == home_idx:
            away_idx = np.random.randint(0, len(teams))
        
        game_id = f"2023_{game_num:03d}_{teams[home_idx]}_{teams[away_idx]}"
        
        # Generate plays for this game
        for play_num in range(np.random.randint(120, 180)):
            play = {
                'game_id': game_id,
                'play_id': play_num,
                'posteam': np.random.choice([teams[home_idx], teams[away_idx]]),
                'defteam': np.random.choice([teams[home_idx], teams[away_idx]]),
                'home_team': teams[home_idx],
                'away_team': teams[away_idx],
                'play_type': np.random.choice(['pass', 'run'], p=[0.6, 0.4]),
                'yards_gained': np.random.normal(5.5, 8),
                'epa': np.random.normal(0, 2),
                'wp': np.random.uniform(0.2, 0.8),
                'interception': np.random.choice([0, 1], p=[0.97, 0.03]),
                'fumble_lost': np.random.choice([0, 1], p=[0.99, 0.01])
            }
            data.append(play)
    
    return pd.DataFrame(data)


def train_model(model_type: str, X_train, y_train, X_test, y_test, config):
    """Train a specific model type."""
    logger = setup_logger(f"train_{model_type}")
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestPredictor()
    elif model_type == 'xgboost':
        model = XGBoostPredictor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training {model_type} model...")
    
    # Train
    model.fit(X_train.values, y_train.values)
    model.is_trained = True
    
    # Evaluate
    y_pred = model.predict(X_test.values)
    metrics = model.evaluate(y_test.values, y_pred)
    
    # Cross-validation
    cv_results = model.cross_validate(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=config.get('models.training.cross_validation_folds', 5)
    )
    
    logger.info(f"Model performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
    logger.info(f"  CV Score: {cv_results['mean']:.3f} (+/- {cv_results['std']:.3f})")
    
    return model, metrics, cv_results


def main(args):
    """Main training pipeline."""
    logger = setup_logger("training_pipeline")
    config = get_config()
    
    # Prepare data
    logger.info("=" * 50)
    logger.info("Starting NFL Prediction Model Training")
    logger.info("=" * 50)
    
    seasons = args.seasons or config.get('data.collection.seasons', [2023])
    game_stats, engineer = prepare_data(seasons, config)
    
    # Prepare features and target
    feature_cols = [col for col in game_stats.columns 
                   if col not in ['game_id', 'team', 'target']]
    
    X = game_stats[feature_cols]
    y = game_stats['target']
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get('models.training.test_size', 0.2),
        random_state=config.get('models.training.random_state', 42),
        stratify=y
    )
    
    # Scale features
    X_train_scaled = engineer.fit_transform(X_train)
    X_test_scaled = engineer.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train models
    results = {}
    
    for model_type in args.models:
        logger.info(f"\nTraining {model_type}...")
        model, metrics, cv_results = train_model(
            model_type, X_train, y_train, X_test, y_test, config
        )
        
        results[model_type] = {
            'metrics': metrics,
            'cv_score': cv_results['mean'],
            'cv_std': cv_results['std']
        }
        
        # Save model
        if args.save_model:
            model_path = Path(config.get('models.model_path', 'models/trained'))
            model_path = model_path / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete - Summary:")
    logger.info("=" * 50)
    
    for model_type, result in results.items():
        logger.info(f"\n{model_type.upper()}:")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.3f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.3f}")
        logger.info(f"  CV Score: {result['cv_score']:.3f} (+/- {result['cv_std']:.3f})")
    
    # Save results
    if args.save_results:
        results_path = Path('results') / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL prediction models")
    parser.add_argument(
        '--models',
        nargs='+',
        default=['random_forest', 'xgboost'],
        choices=['random_forest', 'xgboost'],
        help='Models to train'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        help='Seasons to use for training'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Save trained models'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save training results'
    )
    
    args = parser.parse_args()
    main(args)
