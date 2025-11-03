"""
EDDIS (Empirical Discrete Distributions) - 经验离散分布模型
最简单的数据驱动模型，作为下限基准
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from .base_model import BaseHydrologicalModel

class EDDIS(BaseHydrologicalModel):
    """
    经验离散分布模型
    
    本质上是一个概率查找表，基于训练数据的多元联合分布
    给定输入条件，返回目标变量的条件概率分布
    
    特点：
    - 无需训练（直接从数据构建）
    - 无法外推到训练数据范围外
    - 作为最简单的下限基准
    """
    
    def __init__(self, 
                 n_bins: int = 2,
                 use_temporal_aggregation: bool = True):
        """
        Parameters:
        -----------
        n_bins : int, 每个变量的分箱数
        use_temporal_aggregation : bool, 是否使用时间聚合特征
        """
        super().__init__("EDDIS")
        self.n_bins = n_bins
        self.use_temporal_aggregation = use_temporal_aggregation
        self.joint_distribution = None
        self.bin_edges = {}
        
    def initialize(self, params: Dict = None) -> None:
        """EDDIS无需初始化参数"""
        self.parameters = {}
        self.states = {}
    
    def _create_temporal_features(self, 
                                  precip: np.ndarray, 
                                  temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间聚合特征
        
        Returns:
        --------
        precip_features : array, shape (n, 3)
            [当天, 前一天, 前一周平均]
        temp_features : array, shape (n, 2)
            [当天, 前30天平均]
        """
        n = len(precip)
        
        # 降水特征
        precip_features = np.zeros((n, 3))
        precip_features[:, 0] = precip  # 当天
        precip_features[1:, 1] = precip[:-1]  # 前一天
        
        # 前一周平均（day -2 到 -6）
        for i in range(6, n):
            precip_features[i, 2] = np.mean(precip[i-6:i-1])
        
        # 温度特征
        temp_features = np.zeros((n, 2))
        temp_features[:, 0] = temp  # 当天
        
        # 前30天平均
        for i in range(30, n):
            temp_features[i, 1] = np.mean(temp[i-30:i])
        
        return precip_features, temp_features
    
    def _bin_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        使用K-means聚类对数据分箱
        
        Parameters:
        -----------
        data : array, 原始数据
        var_name : str, 变量名（用于存储bin_edges）
        
        Returns:
        --------
        binned : array, 离散化后的数据
        """
        from sklearn.cluster import KMeans
        
        if var_name not in self.bin_edges:
            # 训练时：创建bins
            kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data.reshape(-1, 1))
            
            # 保存中心点（排序后）
            centers = kmeans.cluster_centers_.flatten()
            sorted_idx = np.argsort(centers)
            
            # 创建映射
            mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_idx)}
            self.bin_edges[var_name] = {
                'centers': centers[sorted_idx],
                'mapping': mapping
            }
            
            # 重新映射标签
            binned = np.array([mapping[label] for label in labels])
        else:
            # 预测时：使用已有bins（最近邻分配）
            centers = self.bin_edges[var_name]['centers']
            binned = np.zeros(len(data), dtype=int)
            for i, val in enumerate(data):
                binned[i] = np.argmin(np.abs(centers - val))
        
        return binned
    
    def train(self, 
              precip: np.ndarray, 
              temp: np.ndarray, 
              pet: np.ndarray,
              discharge: np.ndarray) -> None:
        """
        从训练数据构建联合分布
        
        Parameters:
        -----------
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        discharge : array, 径流（目标变量）
        """
        n = len(discharge)
        
        if self.use_temporal_aggregation:
            # 创建时间特征
            precip_features, temp_features = self._create_temporal_features(precip, temp)
            
            # 只使用有效数据（跳过前30天）
            valid_idx = 30
            precip_features = precip_features[valid_idx:]
            temp_features = temp_features[valid_idx:]
            discharge = discharge[valid_idx:]
            
            # 分箱所有特征
            precip_binned = np.zeros((len(discharge), 3), dtype=int)
            for j in range(3):
                precip_binned[:, j] = self._bin_data(
                    precip_features[:, j], f'precip_{j}'
                )
            
            temp_binned = np.zeros((len(discharge), 2), dtype=int)
            for j in range(2):
                temp_binned[:, j] = self._bin_data(
                    temp_features[:, j], f'temp_{j}'
                )
        else:
            # 简单：只用当天值
            precip_binned = self._bin_data(precip, 'precip_0').reshape(-1, 1)
            temp_binned = self._bin_data(temp, 'temp_0').reshape(-1, 1)
        
        # 目标变量分箱（使用更多bins）
        discharge_binned = self._bin_data(discharge, 'discharge')
        
        # 构建联合分布：P(precip, temp, discharge)
        # 使用字典存储：key = (precip_bins, temp_bins), value = discharge分布
        self.joint_distribution = {}
        
        for i in range(len(discharge)):
            # 创建输入键
            if self.use_temporal_aggregation:
                key = tuple(precip_binned[i]) + tuple(temp_binned[i])
            else:
                key = (precip_binned[i, 0], temp_binned[i, 0])
            
            # 添加到分布
            if key not in self.joint_distribution:
                self.joint_distribution[key] = []
            self.joint_distribution[key].append(discharge_binned[i])
        
        # 转换为概率分布
        for key in self.joint_distribution:
            bins = self.joint_distribution[key]
            counts = np.bincount(bins, minlength=self.n_bins)
            probs = counts / np.sum(counts)
            self.joint_distribution[key] = probs
        
        print(f"EDDIS trained with {len(self.joint_distribution)} unique input combinations")
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """
        EDDIS的预测是基于查找表的
        这个方法在单步调用时不适用
        请使用 predict() 方法
        """
        raise NotImplementedError(
            "EDDIS requires full time series for temporal features. Use predict() method."
        )
    
    def predict(self, 
                precip: np.ndarray, 
                temp: np.ndarray, 
                pet: np.ndarray) -> np.ndarray:
        """
        预测径流
        
        Parameters:
        -----------
        precip : array, 降水时间序列
        temp : array, 温度时间序列
        pet : array, 蒸发时间序列
        
        Returns:
        --------
        discharge : array, 预测径流
        """
        if self.joint_distribution is None:
            raise ValueError("Model not trained. Call train() first.")
        
        n = len(precip)
        discharge = np.zeros(n)
        
        if self.use_temporal_aggregation:
            # 创建时间特征
            precip_features, temp_features = self._create_temporal_features(precip, temp)
            
            # 从第30天开始预测
            for i in range(30, n):
                # 分箱
                p_bins = tuple([
                    self._bin_data(np.array([precip_features[i, j]]), f'precip_{j}')[0]
                    for j in range(3)
                ])
                t_bins = tuple([
                    self._bin_data(np.array([temp_features[i, j]]), f'temp_{j}')[0]
                    for j in range(2)
                ])
                
                key = p_bins + t_bins
                
                # 查找分布
                if key in self.joint_distribution:
                    probs = self.joint_distribution[key]
                    # 使用期望值作为预测
                    centers = self.bin_edges['discharge']['centers']
                    discharge[i] = np.sum(probs * centers)
                else:
                    # 如果没有找到，使用均匀分布
                    centers = self.bin_edges['discharge']['centers']
                    discharge[i] = np.mean(centers)
        else:
            # 简单预测
            for i in range(n):
                p_bin = self._bin_data(np.array([precip[i]]), 'precip_0')[0]
                t_bin = self._bin_data(np.array([temp[i]]), 'temp_0')[0]
                
                key = (p_bin, t_bin)
                
                if key in self.joint_distribution:
                    probs = self.joint_distribution[key]
                    centers = self.bin_edges['discharge']['centers']
                    discharge[i] = np.sum(probs * centers)
                else:
                    centers = self.bin_edges['discharge']['centers']
                    discharge[i] = np.mean(centers)
        
        return discharge
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """EDDIS无参数"""
        return {}