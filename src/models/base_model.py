"""Base model class for NFL predictions."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Abstract base class for NFL prediction models."""
    
    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base predictor.
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
        """
        self.model_name = model_name
        self.params = params or {}
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}
        logger.info(f"Initialized {model_name} predictor")
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        pass
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train model with train/test split.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Training {self.model_name} with {len(X)} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.fit(X_train.values, y_train.values)
        
        # Evaluate
        y_pred = self.predict(X_test.values)
        self.metrics = self.evaluate(y_test.values, y_pred)
        
        logger.info(f"Training complete. Accuracy: {self.metrics['accuracy']:.3f}")
        self.is_trained = True
        
        return self.metrics
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation")
        
        if not self.model:
            self.build_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'cv_folds': cv
        }
        
        logger.info(f"CV Score: {results['mean']:.3f} (+/- {results['std']:.3f})")
        
        return results
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities if supported.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Return binary predictions as probabilities
            preds = self.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[range(len(preds)), preds.astype(int)] = 1
            return proba
    
    def save_model(self, path: Path):
        """
        Save trained model.
        
        Args:
            path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'params': self.params,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """
        Load trained model.
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning(f"{self.model_name} does not support feature importance")
            return pd.DataFrame()
