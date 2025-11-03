"""
RTREE (Regression Tree) - Decision tree-based regression model for hydrological prediction
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict


class RTREE:
    """
    Regression Tree model for discharge prediction
    Uses decision tree ensemble (Random Forest)
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Initialize RTREE model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def create_features(self, X: np.ndarray, lag_days: int = 7) -> np.ndarray:
        """
        Create lagged features for time series prediction
        
        Args:
            X: Input features [precipitation, temperature, pet]
            lag_days: Number of lag days to include
            
        Returns:
            Enhanced features with lags
        """
        n_samples, n_features = X.shape
        
        # Create lagged features
        features_list = [X]
        
        for lag in range(1, lag_days + 1):
            lagged = np.zeros_like(X)
            lagged[lag:] = X[:-lag]
            features_list.append(lagged)
        
        # Add rolling statistics
        for window in [3, 7]:
            for feature_idx in range(n_features):
                rolling_mean = np.zeros(n_samples)
                for i in range(window - 1, n_samples):
                    rolling_mean[i] = np.mean(X[i - window + 1:i + 1, feature_idx])
                features_list.append(rolling_mean.reshape(-1, 1))
        
        features_enhanced = np.hstack(features_list)
        
        return features_enhanced
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit RTREE model
        
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
                  n_iterations: int = 10) -> Dict[str, float]:
        """
        Calibrate RTREE model by trying different hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            n_iterations: Number of calibration iterations (not used, grid search)
            
        Returns:
            Best parameters
        """
        try:
            from utils.metrics import nash_sutcliffe_efficiency
        except ImportError:
            from ...utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        # Grid search over hyperparameters
        for n_est in [50, 100, 150]:
            for depth in [5, 10, 15]:
                for min_split in [2, 5, 10]:
                    self.n_estimators = n_est
                    self.max_depth = depth
                    self.min_samples_split = min_split
                    
                    self.model = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_split=min_split,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    
                    # Fit
                    self.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = self.predict(X_train)
                    
                    # Evaluate
                    nse = nash_sutcliffe_efficiency(y_train, y_pred)
                    
                    if nse > best_nse:
                        best_nse = nse
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'min_samples_split': min_split,
                            'NSE': nse
                        }
        
        # Set best parameters
        if best_params:
            self.n_estimators = best_params['n_estimators']
            self.max_depth = best_params['max_depth']
            self.min_samples_split = best_params['min_samples_split']
            
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.fit(X_train, y_train)
        
        return best_params
