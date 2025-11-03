"""
ANN (Artificial Neural Network) - Multi-layer perceptron for hydrological prediction
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import Dict


class ANN:
    """
    Artificial Neural Network model for discharge prediction
    Uses Multi-Layer Perceptron (MLP)
    """
    
    def __init__(self, hidden_layers: tuple = (50, 25), 
                 learning_rate: float = 0.001, max_iter: int = 500,
                 random_state: int = 42):
        """
        Initialize ANN model
        
        Args:
            hidden_layers: Tuple defining hidden layer sizes
            learning_rate: Initial learning rate
            max_iter: Maximum number of iterations
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.is_fitted = False
        
        # Store normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def create_features(self, X: np.ndarray, lag_days: int = 5) -> np.ndarray:
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
        
        features_enhanced = np.hstack(features_list)
        
        return features_enhanced
    
    def normalize(self, X: np.ndarray, y: np.ndarray = None, 
                  fit: bool = True) -> tuple:
        """
        Normalize features and target
        
        Args:
            X: Input features
            y: Target values (optional)
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized X and y
        """
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1.0
            
            if y is not None:
                self.y_mean = np.mean(y)
                self.y_std = np.std(y)
                if self.y_std == 0:
                    self.y_std = 1.0
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if y is not None:
            y_norm = (y - self.y_mean) / self.y_std
            return X_norm, y_norm
        else:
            return X_norm
    
    def denormalize(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions
        
        Args:
            y_norm: Normalized predictions
            
        Returns:
            Original scale predictions
        """
        return y_norm * self.y_std + self.y_mean
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit ANN model
        
        Args:
            X_train: Training features [precipitation, temperature, pet]
            y_train: Training target (discharge)
        """
        # Create enhanced features
        X_train_enhanced = self.create_features(X_train)
        
        # Normalize data
        X_train_norm, y_train_norm = self.normalize(X_train_enhanced, y_train, fit=True)
        
        # Fit model
        self.model.fit(X_train_norm, y_train_norm)
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
        
        # Normalize
        X_norm = self.normalize(X_enhanced, fit=False)
        
        # Predict
        y_pred_norm = self.model.predict(X_norm)
        
        # Denormalize
        y_pred = self.denormalize(y_pred_norm)
        
        # Ensure non-negative discharge
        y_pred = np.maximum(0, y_pred)
        
        return y_pred
    
    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray,
                  n_iterations: int = 10) -> Dict[str, float]:
        """
        Calibrate ANN model by trying different hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            n_iterations: Number of calibration iterations (not used, grid search)
            
        Returns:
            Best parameters
        """
        from ..utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        # Grid search over hyperparameters
        for hidden in [(30,), (50,), (50, 25), (100, 50)]:
            for lr in [0.001, 0.01]:
                self.hidden_layers = hidden
                self.learning_rate = lr
                
                self.model = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    learning_rate_init=lr,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20
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
                        'hidden_layers': hidden,
                        'learning_rate': lr,
                        'NSE': nse
                    }
        
        # Set best parameters
        if best_params:
            self.hidden_layers = best_params['hidden_layers']
            self.learning_rate = best_params['learning_rate']
            
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                learning_rate_init=self.learning_rate,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            self.fit(X_train, y_train)
        
        return best_params
