"""
信息论评估指标
"""
import numpy as np
from typing import Tuple, Optional
from scipy.stats import entropy as scipy_entropy

def bin_data(data: np.ndarray, n_bins: int = 12, method: str = 'kmeans') -> Tuple[np.ndarray, np.ndarray]:
    """
    数据离散化（分箱）
    
    Parameters:
    -----------
    data : array, 连续数据
    n_bins : int, 箱数
    method : str, 分箱方法 ('equal', 'quantile', 'kmeans')
    
    Returns:
    --------
    binned_data : array, 离散化后的数据
    bin_edges : array, 箱边界
    """
    from sklearn.cluster import KMeans
    
    if method == 'equal':
        # 等宽分箱
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
        binned_data = np.digitize(data, bin_edges[1:-1])
        
    elif method == 'quantile':
        # 等频分箱
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, quantiles)
        binned_data = np.digitize(data, bin_edges[1:-1])
        
    elif method == 'kmeans':
        # K-means聚类分箱（最小化SSE）
        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        binned_data = kmeans.fit_predict(data.reshape(-1, 1))
        
        # 计算箱中心和边界
        centers = kmeans.cluster_centers_.flatten()
        sorted_idx = np.argsort(centers)
        bin_edges = np.zeros(n_bins + 1)
        bin_edges[0] = data.min()
        bin_edges[-1] = data.max()
        for i in range(1, n_bins):
            bin_edges[i] = (centers[sorted_idx[i-1]] + centers[sorted_idx[i]]) / 2
        
        # 重新映射binned_data为排序后的索引
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
        binned_data = np.array([mapping[val] for val in binned_data])
    else:
        raise ValueError(f"Unknown binning method: {method}")
    
    return binned_data, bin_edges


def joint_entropy(features: np.ndarray, targets: Optional[np.ndarray] = None, n_bins: int = 12) -> float:
    """
    计算联合熵 H(features, targets) 或边际熵 H(features)
    
    Parameters:
    -----------
    features : array, 变量1（可以是多维）
    targets : array, 变量2（可选）
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_joint : float, 联合熵 [bits]
    """
    # 确保是2D数组
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    # 分箱
    features_binned = np.zeros_like(features, dtype=int)
    for i in range(features.shape[1]):
        features_binned[:, i], _ = bin_data(features[:, i], n_bins, method='kmeans')
    
    if targets is not None:
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        targets_binned = np.zeros_like(targets, dtype=int)
        for i in range(targets.shape[1]):
            targets_binned[:, i], _ = bin_data(targets[:, i], n_bins, method='kmeans')
        
        # 合并
        combined_data = np.hstack([features_binned, targets_binned])
    else:
        combined_data = features_binned
    
    # 计算联合概率分布
    # 使用更高效的方法：将多维索引转为1D，使用numpy的unique函数
    num_samples = combined_data.shape[0]
    num_dims = combined_data.shape[1]
    
    # Convert multi-dimensional indices to unique 1D keys using numpy operations
    # More efficient than string concatenation
    if num_dims == 1:
        unique_vals, counts = np.unique(combined_data, return_counts=True)
    else:
        # Use a hash-like approach with prime number multiplication for multi-dimensional data
        # This is much faster than string concatenation
        combined_keys = combined_data[:, 0].copy()
        for dim in range(1, num_dims):
            combined_keys = combined_keys * (n_bins + 1) + combined_data[:, dim]
        unique_vals, counts = np.unique(combined_keys, return_counts=True)
    
    # 概率
    probs = counts / num_samples
    
    # 熵（以bits为单位，使用log2）
    # Add small epsilon to avoid log(0)
    entropy_value = -np.sum(probs * np.log2(probs + 1e-10))
    
    return entropy_value


def conditional_entropy(observed: np.ndarray, predicted: np.ndarray, n_bins: int = 12) -> float:
    """
    计算条件熵 H(observed | predicted)
    
    衡量给定模型预测后，观测值的剩余不确定性
    
    Parameters:
    -----------
    observed : array, 观测值
    predicted : array, 预测值
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_cond : float, 条件熵 [bits]
    """
    # 分箱
    observed_binned, _ = bin_data(observed, n_bins, method='kmeans')
    predicted_binned, _ = bin_data(predicted, n_bins, method='kmeans')
    
    num_samples = len(observed)
    
    # Use numpy operations for efficiency instead of dictionaries and loops
    # Create joint distribution using 2D histogram
    joint_hist, _, _ = np.histogram2d(
        observed_binned, predicted_binned, 
        bins=[n_bins, n_bins],
        range=[[0, n_bins-1], [0, n_bins-1]]
    )
    
    # Get marginal distribution for predictions
    pred_hist = np.bincount(predicted_binned, minlength=n_bins)
    
    # Calculate conditional entropy using vectorized operations
    # H(Y_obs | Y_pred) = -Σ P(y_obs, y_pred) * log2[P(y_obs, y_pred) / P(y_pred)]
    joint_probs = joint_hist / num_samples
    pred_probs = pred_hist / num_samples
    
    # Avoid division by zero and log of zero
    mask = (joint_probs > 0) & (pred_probs[np.newaxis, :] > 0)
    conditional_probs = np.zeros_like(joint_probs)
    conditional_probs[mask] = joint_probs[mask] / pred_probs[np.newaxis, :][mask]
    
    H_cond = -np.sum(joint_probs[mask] * np.log2(conditional_probs[mask]))
    
    return H_cond


def mutual_information(features: np.ndarray, targets: np.ndarray, n_bins: int = 12) -> float:
    """
    计算互信息 I(features; targets) = H(targets) - H(targets|features)
    
    Parameters:
    -----------
    features : array, 变量1
    targets : array, 变量2
    n_bins : int, 分箱数
    
    Returns:
    --------
    MI : float, 互信息 [bits]
    """
    H_targets = joint_entropy(targets, n_bins=n_bins)
    H_targets_given_features = conditional_entropy(targets, features, n_bins=n_bins)
    
    mutual_info = H_targets - H_targets_given_features
    return mutual_info


def normalized_conditional_entropy(observed: np.ndarray, 
                                   predicted: np.ndarray, 
                                   n_bins: int = 12) -> float:
    """
    归一化条件熵 H_norm = H(observed | predicted) / H(observed)
    
    范围 [0, 1]，0表示完美预测
    
    Parameters:
    -----------
    observed : array, 观测值
    predicted : array, 预测值
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_norm : float, 归一化条件熵
    """
    H_cond = conditional_entropy(observed, predicted, n_bins)
    H_obs = joint_entropy(observed, n_bins=n_bins)
    
    H_norm = H_cond / (H_obs + 1e-10)
    return H_norm


# 主评估函数
def evaluate_model_entropy(observed: np.ndarray, 
                          simulated: np.ndarray, 
                          n_bins: int = 12) -> dict:
    """
    使用信息熵评估模型性能
    
    Parameters:
    -----------
    observed : array, 观测径流
    simulated : array, 模拟径流
    n_bins : int, 分箱数
    
    Returns:
    --------
    metrics : dict, 包含各种熵指标
    """
    # 移除缺测值
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]
    
    # 计算各种熵
    H_obs = joint_entropy(observed, n_bins=n_bins)
    H_sim = joint_entropy(simulated, n_bins=n_bins)
    H_joint = joint_entropy(np.column_stack([observed, simulated]), n_bins=n_bins)
    H_cond = conditional_entropy(observed, simulated, n_bins=n_bins)
    H_norm = H_cond / H_obs if H_obs > 0 else 1.0
    mutual_info = mutual_information(simulated, observed, n_bins=n_bins)
    
    metrics = {
        'H_obs': H_obs,              # 观测熵
        'H_sim': H_sim,              # 模拟熵
        'H_joint': H_joint,          # 联合熵
        'H_conditional': H_cond,     # 条件熵
        'H_normalized': H_norm,      # 归一化条件熵
        'mutual_information': mutual_info,    # 互信息
        'explained_variance': 1 - H_norm,  # 解释方差（类似R²）
    }
    
    return metrics