"""
训练数据采样策略
"""
import numpy as np
from typing import List, Tuple

def random_sampling(num_total: int, num_samples: int, num_replicates: int = 30, seed: int = 42) -> List[np.ndarray]:
    """
    完全随机采样
    
    Parameters:
    -----------
    num_total : int, 总数据量
    num_samples : int, 采样数量
    num_replicates : int, 重复次数
    seed : int, 随机种子
    
    Returns:
    --------
    samples : list of array, 采样索引列表
    """
    np.random.seed(seed)
    sample_list = []
    
    for _ in range(num_replicates):
        indices = np.random.choice(num_total, size=num_samples, replace=False)
        indices = np.sort(indices)
        sample_list.append(indices)
    
    return sample_list


def consecutive_random_sampling(num_total: int, num_samples: int, num_replicates: int = 30, seed: int = 42) -> List[np.ndarray]:
    """
    随机连续采样（随机起点+连续片段）
    
    Parameters:
    -----------
    num_total : int, 总数据量
    num_samples : int, 采样数量
    num_replicates : int, 重复次数
    seed : int, 随机种子
    
    Returns:
    --------
    samples : list of array, 采样索引列表
    """
    np.random.seed(seed)
    sample_list = []
    
    for _ in range(num_replicates):
        # 随机选择起点
        if num_samples < num_total:
            start_index = np.random.randint(0, num_total - num_samples + 1)
            indices = np.arange(start_index, start_index + num_samples)
        else:
            indices = np.arange(num_total)
        
        sample_list.append(indices)
    
    return sample_list


def douglas_peucker_sampling(data: np.ndarray, num_samples: int, epsilon: float = None) -> np.ndarray:
    """
    Douglas-Peucker算法采样（选择拐点）
    
    用于选择时间序列中最有信息量的点
    
    Parameters:
    -----------
    data : array, 时间序列数据
    num_samples : int, 目标采样数量
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
    
    def rdp_recursive(points, indices, epsilon_threshold):
        """递归Douglas-Peucker算法"""
        if len(points) < 3:
            return indices
        
        # 找到距离起点和终点连线最远的点
        start_point = points[0]
        end_point = points[-1]
        
        max_distance = 0
        max_index = 0
        
        for idx in range(1, len(points) - 1):
            distance = perpendicular_distance(points[idx], start_point, end_point)
            if distance > max_distance:
                max_distance = distance
                max_index = idx
        
        # 如果最大距离大于阈值，递归处理
        if max_distance > epsilon_threshold:
            # 分为两段
            left_indices = rdp_recursive(points[:max_index+1], indices[:max_index+1], epsilon_threshold)
            right_indices = rdp_recursive(points[max_index:], indices[max_index:], epsilon_threshold)
            
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
            selected_indices = rdp_recursive(points, indices, epsilon_mid)
            
            if len(selected_indices) > num_samples:
                epsilon_low = epsilon_mid
            elif len(selected_indices) < num_samples:
                epsilon_high = epsilon_mid
            else:
                break
        
        epsilon = epsilon_mid
    
    # 最终采样
    points = np.column_stack([np.arange(len(data)), data])
    indices = np.arange(len(data))
    selected_indices = rdp_recursive(points, indices, epsilon)
    
    # 如果选中的点太多，随机删除一些
    if len(selected_indices) > num_samples:
        keep_indices = np.sort(np.random.choice(len(selected_indices), num_samples, replace=False))
        selected_indices = selected_indices[keep_indices]
    
    # 如果选中的点太少，随机添加一些
    elif len(selected_indices) < num_samples:
        remaining = np.setdiff1d(np.arange(len(data)), selected_indices)
        additional = np.random.choice(remaining, num_samples - len(selected_indices), replace=False)
        selected_indices = np.sort(np.concatenate([selected_indices, additional]))
    
    return selected_indices


# 采样策略工厂
def get_sampling_strategy(strategy: str, data: np.ndarray, num_samples: int, **kwargs) -> np.ndarray:
    """
    获取采样策略
    
    Parameters:
    -----------
    strategy : str, 策略名称 ('random', 'consecutive', 'douglas_peucker')
    data : array, 数据（仅DP需要）
    num_samples : int, 采样数量
    **kwargs : 其他参数
    
    Returns:
    --------
    indices : array or list of array, 采样索引
    """
    num_total = len(data) if data is not None else kwargs.get('num_total')
    
    if strategy == 'random':
        return random_sampling(num_total, num_samples, **kwargs)
    elif strategy == 'consecutive':
        return consecutive_random_sampling(num_total, num_samples, **kwargs)
    elif strategy == 'douglas_peucker':
        return [douglas_peucker_sampling(data, num_samples)]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")