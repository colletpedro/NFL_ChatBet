"""XGBoost model for NFL game predictions."""

import logging
from typing import Optional, Dict, Any
import numpy as np
import xgboost as xgb
from .base_model import BasePredictor

logger = logging.getLogger(__name__)


class XGBoostPredictor(BasePredictor):
    """XGBoost implementation for NFL predictions."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost predictor.
        
        Args:
            params: Model hyperparameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("XGBoost", default_params)
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model."""
        self.model = xgb.XGBClassifier(**self.params)
        logger.info(f"Built XGBoost with params: {self.params}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None, early_stopping_rounds=None):
        """
        Train the XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Validation set for early stopping
            early_stopping_rounds: Rounds for early stopping
        """
        logger.info(f"Training XGBoost on {X.shape} data")
        
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
            
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X, y, **fit_params)
        logger.info("XGBoost training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with XGBoost.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_trained and not hasattr(self.model, 'n_features_in_'):
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if self.model:
            return self.model.get_params()
        return self.params
    
    def optimize_hyperparameters_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials
            cv: Number of CV folds
            
        Returns:
            Best parameters found
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            logger.info(f"Starting Optuna optimization with {n_trials} trials")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                
                model = xgb.XGBClassifier(**params)
                score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
                
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Update model with best parameters
            self.params.update(study.best_params)
            self.build_model()
            
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Best CV score: {study.best_value:.3f}")
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'study': study
            }
            
        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return {'best_params': self.params, 'best_score': None}
