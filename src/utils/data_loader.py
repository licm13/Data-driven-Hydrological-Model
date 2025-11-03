"""
Data loading and preprocessing utilities for hydrological modeling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class HydrologicalDataLoader:
    """Load and preprocess hydrological data for modeling"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_sample_data(self, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic hydrological data for demonstration
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with hydrological features
        """
        np.random.seed(seed)
        
        # Runoff coefficient (typical value for moderately permeable soils)
        # Represents the fraction of precipitation that becomes runoff
        RUNOFF_COEFFICIENT = 0.7
        
        # Generate synthetic meteorological and hydrological data
        dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'precipitation': np.maximum(0, np.random.gamma(2, 2, n_samples)),  # mm/day
            'temperature': 15 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 2, n_samples),  # Celsius
            'pet': np.maximum(0, 3 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 0.5, n_samples)),  # Potential ET mm/day
        })
        
        # Generate synthetic discharge (simplified rainfall-runoff relationship)
        discharge = np.zeros(n_samples)
        storage = 50.0  # Initial storage
        
        for i in range(n_samples):
            # Simple storage-discharge model
            inflow = data.loc[i, 'precipitation'] * RUNOFF_COEFFICIENT
            et = min(storage * 0.05, data.loc[i, 'pet'])
            storage += inflow - et
            discharge[i] = max(0, storage * 0.1)
            storage = max(0, storage - discharge[i])
            
        data['discharge'] = discharge + np.random.normal(0, 0.1, n_samples)  # Add noise
        data['discharge'] = np.maximum(0, data['discharge'])
        
        self.data = data
        return data
    
    def prepare_training_data(self, data: pd.DataFrame, 
                             train_size: int,
                             features: list = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data
        
        Args:
            data: Input DataFrame
            train_size: Number of training samples
            features: List of feature columns (default: precipitation, temperature, pet)
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        if features is None:
            features = ['precipitation', 'temperature', 'pet']
            
        X = data[features].values
        y = data['discharge'].values
        
        # Split data
        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        return X_train, X_val, y_train, y_val
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        seq_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models (LSTM)
        
        Args:
            X: Input features
            y: Target values
            seq_length: Length of input sequences
            
        Returns:
            X_seq, y_seq
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
            
        return np.array(X_seq), np.array(y_seq)


def normalize_data(X: np.ndarray, mean: np.ndarray = None, 
                   std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data using z-score normalization
    
    Args:
        X: Input data
        mean: Mean values (computed if not provided)
        std: Standard deviation (computed if not provided)
        
    Returns:
        X_normalized, mean, std
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        
    X_normalized = (X - mean) / std
    return X_normalized, mean, std
