"""
训练数据采样策略
"""
import numpy as np
from typing import List, Tuple

def random_sampling(n_total: int, n_sample: int, n_replicates: int = 30, seed: int = 42) -> List[np.ndarray]:
    """
    完全随机采样
    
    Parameters:
    -----------
    n_total : int, 总数据量
    n_sample : int, 采样数量
    n_replicates : int, 重复次数
    seed : int, 随机种子
    
    Returns:
    --------
    samples : list of array, 采样索引列表
    """
    np.random.seed(seed)
    samples = []
    
    for i in range(n_replicates):
        indices = np.random.choice(n_total, size=n_sample, replace=False)
        indices = np.sort(indices)
        samples.append(indices)
    
    return samples


def consecutive_random_sampling(n_total: int, n_sample: int, n_replicates: int = 30, seed: int = 42) -> List[np.ndarray]:
    """
    随机连续采样（随机起点+连续片段）
    
    Parameters:
    -----------
    n_total : int, 总数据量
    n_sample : int, 采样数量
    n_replicates : int, 重复次数
    seed : int, 随机种子
    
    Returns:
    --------
    samples : list of array, 采样索引列表
    """
    np.random.seed(seed)
    samples = []
    
    for i in range(n_replicates):
        # 随机选择起点
        if n_sample < n_total:
            start = np.random.randint(0, n_total - n_sample + 1)
            indices = np.arange(start, start + n_sample)
        else:
            indices = np.arange(n_total)
        
        samples.append(indices)
    
    return samples


def douglas_peucker_sampling(data: np.ndarray, n_sample: int, epsilon: float = None) -> np.ndarray:
    """
    Douglas-Peucker算法采样（选择拐点）
    
    用于选择时间序列中最有信息量的点
    
    Parameters:
    -----------
    data : array, 时间序列数据
    n_sample : int, 目标采样数量
    epsilon : float, 容差阈值（可选）
    
    Returns:
    --------
    indices : array, 采样索引
    """
    def perpendicular_distance(point, line_start, line_end):
        """计算点到线段的垂直距离"""
        if np.all(line_start == line_end):
            return np.linalg.norm(point - line_start)
        
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
    
    def rdp_recursive(points, indices, epsilon):
        """递归Douglas-Peucker算法"""
        if len(points) < 3:
            return indices
        
        # 找到距离起点和终点连线最远的点
        start = points[0]
        end = points[-1]
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = perpendicular_distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # 如果最大距离大于阈值，递归处理
        if max_dist > epsilon:
            # 分为两段
            left_indices = rdp_recursive(points[:max_idx+1], indices[:max_idx+1], epsilon)
            right_indices = rdp_recursive(points[max_idx:], indices[max_idx:], epsilon)
            
            # 合并（去除重复的中间点）
            return np.concatenate([left_indices[:-1], right_indices])
        else:
            # 只保留起点和终点
            return np.array([indices[0], indices[-1]])
    
    # 如果没有指定epsilon，自适应调整以达到目标样本数
    if epsilon is None:
        # 二分搜索epsilon
        points = np.column_stack([np.arange(len(data)), data])
        indices = np.arange(len(data))
        
        epsilon_low = 0.0
        epsilon_high = np.max(data) - np.min(data)
        
        for _ in range(20):  # 最多20次迭代
            epsilon_mid = (epsilon_low + epsilon_high) / 2
            selected = rdp_recursive(points, indices, epsilon_mid)
            
            if len(selected) > n_sample:
                epsilon_low = epsilon_mid
            elif len(selected) < n_sample:
                epsilon_high = epsilon_mid
            else:
                break
        
        epsilon = epsilon_mid
    
    # 最终采样
    points = np.column_stack([np.arange(len(data)), data])
    indices = np.arange(len(data))
    selected_indices = rdp_recursive(points, indices, epsilon)
    
    # 如果选中的点太多，随机删除一些
    if len(selected_indices) > n_sample:
        keep_indices = np.sort(np.random.choice(len(selected_indices), n_sample, replace=False))
        selected_indices = selected_indices[keep_indices]
    
    # 如果选中的点太少，随机添加一些
    elif len(selected_indices) < n_sample:
        remaining = np.setdiff1d(np.arange(len(data)), selected_indices)
        additional = np.random.choice(remaining, n_sample - len(selected_indices), replace=False)
        selected_indices = np.sort(np.concatenate([selected_indices, additional]))
    
    return selected_indices


# 采样策略工厂
def get_sampling_strategy(strategy: str, data: np.ndarray, n_sample: int, **kwargs) -> np.ndarray:
    """
    获取采样策略
    
    Parameters:
    -----------
    strategy : str, 策略名称 ('random', 'consecutive', 'douglas_peucker')
    data : array, 数据（仅DP需要）
    n_sample : int, 采样数量
    **kwargs : 其他参数
    
    Returns:
    --------
    indices : array or list of array, 采样索引
    """
    n_total = len(data) if data is not None else kwargs.get('n_total')
    
    if strategy == 'random':
        return random_sampling(n_total, n_sample, **kwargs)
    elif strategy == 'consecutive':
        return consecutive_random_sampling(n_total, n_sample, **kwargs)
    elif strategy == 'douglas_peucker':
        return [douglas_peucker_sampling(data, n_sample)]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")