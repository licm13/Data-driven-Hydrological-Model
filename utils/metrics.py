"""
Utility functions for hydrological modeling
"""

import numpy as np


def nse(observed, simulated):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        NSE value (ranges from -∞ to 1, where 1 is perfect)
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]
    
    if len(observed) == 0:
        return np.nan
        
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    
    if denominator == 0:
        return np.nan
        
    return 1 - (numerator / denominator)


def rmse(observed, simulated):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        RMSE value
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]
    
    if len(observed) == 0:
        return np.nan
        
    return np.sqrt(np.mean((observed - simulated) ** 2))


def pbias(observed, simulated):
    """
    Calculate Percent Bias (PBIAS).
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        PBIAS value (optimal value is 0)
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]
    
    if len(observed) == 0:
        return np.nan
        
    numerator = np.sum(simulated - observed)
    denominator = np.sum(observed)
    
    if denominator == 0:
        return np.nan
        
    return 100 * numerator / denominator


def kge(observed, simulated):
    """
    Calculate Kling-Gupta Efficiency (KGE).
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        KGE value (ranges from -∞ to 1, where 1 is perfect)
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]
    
    if len(observed) == 0:
        return np.nan
    
    # Check for zero variance or zero mean
    obs_std = np.std(observed)
    sim_std = np.std(simulated)
    obs_mean = np.mean(observed)
    
    if obs_std == 0 or sim_std == 0:
        return np.nan
    
    if obs_mean == 0:
        return np.nan
        
    # Correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]
    
    # Handle NaN from correlation (e.g., constant arrays)
    if np.isnan(r):
        return np.nan
    
    # Relative variability
    alpha = sim_std / obs_std
    
    # Relative mean
    beta = np.mean(simulated) / obs_mean
    
    # KGE
    kge_value = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge_value


def split_data(data, train_ratio=0.7):
    """
    Split data into training and testing sets.
    
    Args:
        data: Data array or dictionary
        train_ratio: Ratio of training data (default: 0.7)
        
    Returns:
        Training and testing data
    """
    if isinstance(data, dict):
        n = len(data[list(data.keys())[0]])
        split_idx = int(n * train_ratio)
        
        train_data = {}
        test_data = {}
        
        for key, values in data.items():
            train_data[key] = values[:split_idx]
            test_data[key] = values[split_idx:]
            
        return train_data, test_data
    else:
        data = np.array(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]


def calculate_all_metrics(observed, simulated):
    """
    Calculate all performance metrics.
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        Dictionary with all metrics
    """
    return {
        'NSE': nse(observed, simulated),
        'RMSE': rmse(observed, simulated),
        'PBIAS': pbias(observed, simulated),
        'KGE': kge(observed, simulated)
    }
