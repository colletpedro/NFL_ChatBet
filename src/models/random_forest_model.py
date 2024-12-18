"""Random Forest model for NFL game predictions."""

import logging
from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BasePredictor

logger = logging.getLogger(__name__)


class RandomForestPredictor(BasePredictor):
    """Random Forest implementation for NFL predictions."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest predictor.
        
        Args:
            params: Model hyperparameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("RandomForest", default_params)
        self.build_model()
    
    def build_model(self):
        """Build Random Forest model."""
        self.model = RandomForestClassifier(**self.params)
        logger.info(f"Built Random Forest with params: {self.params}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info(f"Training Random Forest on {X.shape} data")
        self.model.fit(X, y)
        logger.info("Random Forest training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with Random Forest.
        
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
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Parameter grid for search
            cv: Number of CV folds
            
        Returns:
            Best parameters found
        """
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        logger.info("Starting hyperparameter optimization")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.params = grid_search.best_params_
        self.build_model()
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {self.params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': self.params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
