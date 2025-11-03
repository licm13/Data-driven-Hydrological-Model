"""
Kling-Gupta效率系数 (KGE)
参考: Gupta et al. (2009)
"""
import numpy as np

def kge(obs: np.ndarray, sim: np.ndarray, s: tuple = (1, 1, 1)) -> float:
    """
    计算Kling-Gupta效率系数
    
    KGE = 1 - sqrt[(r-1)² + (α-1)² + (β-1)²]
    
    其中:
    - r: 相关系数
    - α: 标准差比
    - β: 均值比
    
    Parameters:
    -----------
    obs : array, 观测值
    sim : array, 模拟值
    s : tuple, (r, α, β)的权重
    
    Returns:
    --------
    kge_value : float, KGE值，最优为1
    """
    # 移除缺测值
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return -np.inf
    
    # 相关系数
    r = np.corrcoef(obs, sim)[0, 1]
    
    # 标准差比
    alpha = np.std(sim) / (np.std(obs) + 1e-10)
    
    # 均值比
    beta = np.mean(sim) / (np.mean(obs) + 1e-10)
    
    # KGE
    kge_value = 1 - np.sqrt(
        s[0] * (r - 1)**2 + 
        s[1] * (alpha - 1)**2 + 
        s[2] * (beta - 1)**2
    )
    
    return kge_value


def kge_components(obs: np.ndarray, sim: np.ndarray) -> dict:
    """
    返回KGE的各个组成部分
    
    Parameters:
    -----------
    obs : array, 观测值
    sim : array, 模拟值
    
    Returns:
    --------
    components : dict, {'r', 'alpha', 'beta', 'kge'}
    """
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-10)
    beta = np.mean(sim) / (np.mean(obs) + 1e-10)
    
    kge_val = kge(obs, sim)
    
    return {
        'r': r,
        'alpha': alpha,
        'beta': beta,
        'kge': kge_val
    }


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Nash-Sutcliffe效率系数
    
    Parameters:
    -----------
    obs : array, 观测值
    sim : array, 模拟值
    
    Returns:
    --------
    nse_value : float, NSE值，最优为1
    """
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return -np.inf
    
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    
    nse_value = 1 - numerator / (denominator + 1e-10)
    
    return nse_value