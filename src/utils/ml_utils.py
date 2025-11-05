"""
Machine Learning utility functions for hydrological models
"""
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def normalize_data(
    features: np.ndarray,
    targets: Optional[np.ndarray] = None,
    feature_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple:
    """
    Normalize features and optionally targets using StandardScaler
    
    Parameters:
    -----------
    features : np.ndarray
        Input features (can be 2D or 3D)
    targets : Optional[np.ndarray]
        Target values to normalize
    feature_scaler : Optional[StandardScaler]
        Pre-fitted scaler for features (used when fit=False)
    target_scaler : Optional[StandardScaler]
        Pre-fitted scaler for targets (used when fit=False)
    fit : bool
        Whether to fit new scalers or use provided ones
        
    Returns:
    --------
    normalized_features : np.ndarray
        Normalized features
    normalized_targets : Optional[np.ndarray]
        Normalized targets (if provided)
    feature_scaler : StandardScaler
        Feature scaler (new or provided)
    target_scaler : Optional[StandardScaler]
        Target scaler (new or provided, if targets were given)
    """
    original_shape = features.shape
    is_3d = len(original_shape) == 3
    
    # Reshape 3D to 2D for scaling
    if is_3d:
        num_samples, seq_len, num_features = original_shape
        features_2d = features.reshape(-1, num_features)
    else:
        features_2d = features
    
    # Normalize features
    if fit:
        if feature_scaler is None:
            feature_scaler = StandardScaler()
        normalized_features_2d = feature_scaler.fit_transform(features_2d)
    else:
        if feature_scaler is None:
            raise ValueError("feature_scaler must be provided when fit=False")
        normalized_features_2d = feature_scaler.transform(features_2d)
    
    # Reshape back if needed
    if is_3d:
        normalized_features = normalized_features_2d.reshape(original_shape)
    else:
        normalized_features = normalized_features_2d
    
    # Normalize targets if provided
    normalized_targets = None
    if targets is not None:
        if fit:
            if target_scaler is None:
                target_scaler = StandardScaler()
            normalized_targets = target_scaler.fit_transform(
                targets.reshape(-1, 1)
            ).flatten()
        else:
            if target_scaler is None:
                raise ValueError("target_scaler must be provided when fit=False")
            normalized_targets = target_scaler.transform(
                targets.reshape(-1, 1)
            ).flatten()
    
    return normalized_features, normalized_targets, feature_scaler, target_scaler


def calculate_array_statistics(data: np.ndarray, include_percentiles: bool = False) -> dict:
    """
    Calculate common statistics for an array
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    include_percentiles : bool
        Whether to include percentile statistics
        
    Returns:
    --------
    stats : dict
        Dictionary containing statistical measures
    """
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'max': np.max(data),
        'min': np.min(data),
    }
    
    if include_percentiles:
        stats.update({
            'q95': np.percentile(data, 95),
            'q5': np.percentile(data, 5),
        })
    
    return stats


def create_lagged_features(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    time_lag: int
) -> np.ndarray:
    """
    Create time-lagged features efficiently using numpy stride tricks
    
    For each time step t, creates features:
    [precip[t-lag:t+1], temp[t-lag:t+1], pet[t-lag:t+1]]
    
    Parameters:
    -----------
    precip : np.ndarray
        Precipitation array
    temp : np.ndarray
        Temperature array
    pet : np.ndarray
        Potential evapotranspiration array
    time_lag : int
        Number of time steps to look back
        
    Returns:
    --------
    features : np.ndarray
        Array of shape (n - lag, (lag+1) * 3)
    """
    num_timesteps = len(precip)
    window_size = time_lag + 1
    num_valid_samples = num_timesteps - time_lag
    
    # Stack all variables
    data = np.column_stack([precip, temp, pet])
    
    # Use numpy's lib.stride_tricks for efficient sliding window
    # This creates views without copying data
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create sliding windows for each variable
    windowed_data = sliding_window_view(data, window_shape=(window_size, 3))
    
    # Reshape to (num_samples, window_size * num_features)
    features = windowed_data.reshape(num_valid_samples, -1)
    
    return features


def create_sequences_for_lstm(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    sequence_length: int,
    discharge: Optional[np.ndarray] = None
) -> Tuple:
    """
    Create sequences for LSTM efficiently using numpy operations
    
    For each time step t, creates sequence:
    X[i] = [[P[t-seq:t], T[t-seq:t], PET[t-seq:t]]]
    y[i] = Q[t]
    
    Parameters:
    -----------
    precip : np.ndarray
        Precipitation array
    temp : np.ndarray
        Temperature array
    pet : np.ndarray
        Potential evapotranspiration array
    sequence_length : int
        Length of input sequences
    discharge : Optional[np.ndarray]
        Discharge array for targets
        
    Returns:
    --------
    sequences : np.ndarray
        Array of shape (n_samples, seq_len, 3)
    targets : Optional[np.ndarray]
        Array of shape (n_samples,) or None
    """
    num_timesteps = len(precip)
    num_samples = num_timesteps - sequence_length
    
    # Stack all input features - shape: (num_timesteps, 3)
    data = np.stack([precip, temp, pet], axis=-1)
    
    # Use sliding_window_view for efficient window creation
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create sliding windows along the first axis (time)
    sequences = sliding_window_view(data, window_shape=sequence_length, axis=0)
    # Result has shape: (num_timesteps - sequence_length + 1, 3, sequence_length)
    # We need: (num_samples, sequence_length, 3)
    # So transpose the last two dimensions
    sequences = sequences[:num_samples].transpose(0, 2, 1)
    
    if discharge is not None:
        # Target: discharge values after each sequence
        targets = discharge[sequence_length:]
        return sequences, targets
    else:
        return sequences, None

