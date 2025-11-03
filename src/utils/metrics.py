"""
Evaluation metrics for hydrological models
"""

import numpy as np
from typing import Dict


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE)
    
    NSE = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        NSE value (ranges from -inf to 1, with 1 being perfect)
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        return -np.inf
    
    nse = 1 - (numerator / denominator)
    return nse


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def kling_gupta_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Kling-Gupta Efficiency (KGE)
    
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
        r = correlation coefficient
        alpha = std(y_pred) / std(y_true)
        beta = mean(y_pred) / mean(y_true)
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        KGE value (ranges from -inf to 1, with 1 being perfect)
    """
    # Check for constant arrays
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return -np.inf
    
    # Correlation coefficient
    r = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Handle NaN correlation (shouldn't happen after std check, but be safe)
    if np.isnan(r):
        return -np.inf
    
    # Variability ratio
    alpha = np.std(y_pred) / np.std(y_true)
    
    # Bias ratio
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else 0
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge


def percent_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Percent Bias (PBIAS)
    
    PBIAS = 100 * sum(y_true - y_pred) / sum(y_true)
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        PBIAS value (0 is perfect, negative means overestimation, positive means underestimation)
    """
    return 100 * np.sum(y_true - y_pred) / np.sum(y_true) if np.sum(y_true) > 0 else 0


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    
    Args:
        y_true: Observed values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'NSE': nash_sutcliffe_efficiency(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'KGE': kling_gupta_efficiency(y_true, y_pred),
        'PBIAS': percent_bias(y_true, y_pred)
    }
    
    return metrics
