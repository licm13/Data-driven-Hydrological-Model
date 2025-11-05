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
