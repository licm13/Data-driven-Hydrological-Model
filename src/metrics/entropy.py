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


def joint_entropy(X: np.ndarray, Y: Optional[np.ndarray] = None, n_bins: int = 12) -> float:
    """
    计算联合熵 H(X, Y) 或边际熵 H(X)
    
    Parameters:
    -----------
    X : array, 变量1（可以是多维）
    Y : array, 变量2（可选）
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_joint : float, 联合熵 [bits]
    """
    # 确保是2D数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # 分箱
    X_binned = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        X_binned[:, i], _ = bin_data(X[:, i], n_bins, method='kmeans')
    
    if Y is not None:
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_binned = np.zeros_like(Y, dtype=int)
        for i in range(Y.shape[1]):
            Y_binned[:, i], _ = bin_data(Y[:, i], n_bins, method='kmeans')
        
        # 合并
        XY = np.hstack([X_binned, Y_binned])
    else:
        XY = X_binned
    
    # 计算联合概率分布
    # 将多维索引转为1D
    n_samples = XY.shape[0]
    n_dims = XY.shape[1]
    
    # 创建唯一键
    keys = [''.join(map(str, row)) for row in XY]
    unique_keys, counts = np.unique(keys, return_counts=True)
    
    # 概率
    probs = counts / n_samples
    
    # 熵（以bits为单位，使用log2）
    H = -np.sum(probs * np.log2(probs + 1e-10))
    
    return H


def conditional_entropy(Y_obs: np.ndarray, Y_pred: np.ndarray, n_bins: int = 12) -> float:
    """
    计算条件熵 H(Y_obs | Y_pred)
    
    衡量给定模型预测后，观测值的剩余不确定性
    
    Parameters:
    -----------
    Y_obs : array, 观测值
    Y_pred : array, 预测值
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_cond : float, 条件熵 [bits]
    """
    # 分箱
    Y_obs_binned, _ = bin_data(Y_obs, n_bins, method='kmeans')
    Y_pred_binned, _ = bin_data(Y_pred, n_bins, method='kmeans')
    
    n_samples = len(Y_obs)
    
    # 计算 P(Y_obs, Y_pred)
    joint_counts = {}
    for i in range(n_samples):
        key = (Y_obs_binned[i], Y_pred_binned[i])
        joint_counts[key] = joint_counts.get(key, 0) + 1
    
    # 计算 P(Y_pred)
    pred_counts = {}
    for i in range(n_samples):
        key = Y_pred_binned[i]
        pred_counts[key] = pred_counts.get(key, 0) + 1
    
    # H(Y_obs | Y_pred) = -Σ P(y_obs, y_pred) * log2[P(y_obs, y_pred) / P(y_pred)]
    H_cond = 0.0
    for (y_obs, y_pred), joint_count in joint_counts.items():
        p_joint = joint_count / n_samples
        p_pred = pred_counts[y_pred] / n_samples
        p_cond = p_joint / p_pred
        H_cond -= p_joint * np.log2(p_cond + 1e-10)
    
    return H_cond


def mutual_information(X: np.ndarray, Y: np.ndarray, n_bins: int = 12) -> float:
    """
    计算互信息 I(X; Y) = H(Y) - H(Y|X)
    
    Parameters:
    -----------
    X : array, 变量1
    Y : array, 变量2
    n_bins : int, 分箱数
    
    Returns:
    --------
    MI : float, 互信息 [bits]
    """
    H_Y = joint_entropy(Y, n_bins=n_bins)
    H_Y_given_X = conditional_entropy(Y, X, n_bins=n_bins)
    
    MI = H_Y - H_Y_given_X
    return MI


def normalized_conditional_entropy(Y_obs: np.ndarray, 
                                   Y_pred: np.ndarray, 
                                   n_bins: int = 12) -> float:
    """
    归一化条件熵 H_norm = H(Y_obs | Y_pred) / H(Y_obs)
    
    范围 [0, 1]，0表示完美预测
    
    Parameters:
    -----------
    Y_obs : array, 观测值
    Y_pred : array, 预测值
    n_bins : int, 分箱数
    
    Returns:
    --------
    H_norm : float, 归一化条件熵
    """
    H_cond = conditional_entropy(Y_obs, Y_pred, n_bins)
    H_obs = joint_entropy(Y_obs, n_bins=n_bins)
    
    H_norm = H_cond / (H_obs + 1e-10)
    return H_norm


# 主评估函数
def evaluate_model_entropy(obs: np.ndarray, 
                          sim: np.ndarray, 
                          n_bins: int = 12) -> dict:
    """
    使用信息熵评估模型性能
    
    Parameters:
    -----------
    obs : array, 观测径流
    sim : array, 模拟径流
    n_bins : int, 分箱数
    
    Returns:
    --------
    metrics : dict, 包含各种熵指标
    """
    # 移除缺测值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    # 计算各种熵
    H_obs = joint_entropy(obs, n_bins=n_bins)
    H_sim = joint_entropy(sim, n_bins=n_bins)
    H_joint = joint_entropy(np.column_stack([obs, sim]), n_bins=n_bins)
    H_cond = conditional_entropy(obs, sim, n_bins=n_bins)
    H_norm = H_cond / H_obs if H_obs > 0 else 1.0
    MI = mutual_information(sim, obs, n_bins=n_bins)
    
    metrics = {
        'H_obs': H_obs,              # 观测熵
        'H_sim': H_sim,              # 模拟熵
        'H_joint': H_joint,          # 联合熵
        'H_conditional': H_cond,     # 条件熵
        'H_normalized': H_norm,      # 归一化条件熵
        'mutual_information': MI,    # 互信息
        'explained_variance': 1 - H_norm,  # 解释方差（类似R²）
    }
    
    return metrics