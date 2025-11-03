"""
EDDIS (Event-Driven Data-Informed System) - A data-driven model based on event detection
Simplified implementation using pattern matching and regression
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict


class EDDIS:
    """
    EDDIS model - Event-Driven Data-Informed System
    Uses event detection and pattern-based learning
    """
    
    def __init__(self, event_threshold: float = 5.0, poly_degree: int = 2, alpha: float = 1.0):
        """
        Initialize EDDIS model
        
        Args:
            event_threshold: Precipitation threshold to define events (mm)
            poly_degree: Degree of polynomial features
            alpha: Regularization parameter for Ridge regression
        """
        self.event_threshold = event_threshold
        self.poly_degree = poly_degree
        self.alpha = alpha
        
        self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        self.model = Ridge(alpha=alpha)
        self.is_fitted = False
        
    def detect_events(self, precipitation: np.ndarray) -> np.ndarray:
        """
        Detect precipitation events
        
        Args:
            precipitation: Daily precipitation (mm)
            
        Returns:
            Event indicators (1 for event, 0 for non-event)
        """
        events = (precipitation > self.event_threshold).astype(int)
        return events
    
    def create_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create enhanced features including event indicators
        
        Args:
            X: Input features [precipitation, temperature, pet]
            
        Returns:
            Enhanced features
        """
        # Detect events
        events = self.detect_events(X[:, 0]).reshape(-1, 1)
        
        # Add cumulative precipitation
        cum_precip = np.cumsum(X[:, 0]).reshape(-1, 1)
        cum_precip = cum_precip / (np.arange(len(X)) + 1).reshape(-1, 1)  # Average
        
        # Combine features
        features = np.hstack([X, events, cum_precip])
        
        # Apply polynomial transformation
        features_poly = self.poly_features.fit_transform(features)
        
        return features_poly
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit EDDIS model
        
        Args:
            X_train: Training features [precipitation, temperature, pet]
            y_train: Training target (discharge)
        """
        # Create enhanced features
        X_train_enhanced = self.create_features(X_train)
        
        # Fit model
        self.model.fit(X_train_enhanced, y_train)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discharge
        
        Args:
            X: Features [precipitation, temperature, pet]
            
        Returns:
            Predicted discharge
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create enhanced features
        X_enhanced = self.create_features(X)
        
        # Predict
        y_pred = self.model.predict(X_enhanced)
        
        # Ensure non-negative discharge
        y_pred = np.maximum(0, y_pred)
        
        return y_pred
    
    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray,
                  n_iterations: int = 20) -> Dict[str, float]:
        """
        Calibrate EDDIS model by trying different hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            n_iterations: Number of calibration iterations
            
        Returns:
            Best parameters
        """
        from ..utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        for threshold in np.linspace(1, 10, 5):
            for degree in [1, 2, 3]:
                for alpha in [0.1, 1.0, 10.0]:
                    self.event_threshold = threshold
                    self.poly_degree = degree
                    self.alpha = alpha
                    
                    self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                    self.model = Ridge(alpha=alpha)
                    
                    # Fit
                    self.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = self.predict(X_train)
                    
                    # Evaluate
                    nse = nash_sutcliffe_efficiency(y_train, y_pred)
                    
                    if nse > best_nse:
                        best_nse = nse
                        best_params = {
                            'event_threshold': threshold,
                            'poly_degree': degree,
                            'alpha': alpha,
                            'NSE': nse
                        }
        
        # Set best parameters
        if best_params:
            self.event_threshold = best_params['event_threshold']
            self.poly_degree = best_params['poly_degree']
            self.alpha = best_params['alpha']
            
            self.poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            self.model = Ridge(alpha=self.alpha)
            self.fit(X_train, y_train)
        
        return best_params
