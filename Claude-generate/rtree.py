"""
RTREE (Regression Tree) - 回归树模型
使用决策树进行径流预测
"""
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from .base_model import BaseHydrologicalModel

class RTREE(BaseHydrologicalModel):
    """
    回归树模型
    
    使用sklearn的决策树，可以处理非线性关系
    """
    
    def __init__(self, 
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 use_temporal_features: bool = True):
        """
        Parameters:
        -----------
        max_depth : int, 树的最大深度
        min_samples_split : int, 分裂所需的最小样本数
        min_samples_leaf : int, 叶节点最小样本数
        use_temporal_features : bool, 是否使用时间特征
        """
        super().__init__("RTREE")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_temporal_features = use_temporal_features
        self.model = None
        
    def initialize(self, params: Dict = None) -> None:
        """初始化回归树"""
        self.parameters = {
            'max_depth': params.get('max_depth', self.max_depth) if params else self.max_depth,
            'min_samples_split': params.get('min_samples_split', self.min_samples_split) if params else self.min_samples_split,
            'min_samples_leaf': params.get('min_samples_leaf', self.min_samples_leaf) if params else self.min_samples_leaf,
        }
        
        self.model = DecisionTreeRegressor(
            max_depth=int(self.parameters['max_depth']),
            min_samples_split=int(self.parameters['min_samples_split']),
            min_samples_leaf=int(self.parameters['min_samples_leaf']),
            random_state=42
        )
        
        self.states = {}
    
    def _create_features(self, 
                        precip: np.ndarray, 
                        temp: np.ndarray, 
                        pet: np.ndarray) -> np.ndarray:
        """
        创建输入特征
        
        Returns:
        --------
        features : array, shape (n, n_features)
        """
        n = len(precip)
        
        if self.use_temporal_features:
            # 使用时间聚合特征
            features_list = []
            
            # 当天值
            features_list.append(precip)
            features_list.append(temp)
            features_list.append(pet)
            
            # 前一天
            features_list.append(np.concatenate([[0], precip[:-1]]))
            features_list.append(np.concatenate([[temp[0]], temp[:-1]]))
            
            # 前一周平均降水 (day -2 to -6)
            precip_week = np.zeros(n)
            for i in range(6, n):
                precip_week[i] = np.mean(precip[i-6:i-1])
            features_list.append(precip_week)
            
            # 前30天平均温度
            temp_month = np.zeros(n)
            for i in range(30, n):
                temp_month[i] = np.mean(temp[i-30:i])
            features_list.append(temp_month)
            
            features = np.column_stack(features_list)
        else:
            # 简单：只用当天值
            features = np.column_stack([precip, temp, pet])
        
        return features
    
    def train(self, 
              precip: np.ndarray, 
              temp: np.ndarray, 
              pet: np.ndarray,
              discharge: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> None:
        """
        训练回归树
        
        Parameters:
        -----------
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        discharge : array, 径流（目标）
        sample_weight : array, 样本权重（可选）
        """
        if self.model is None:
            self.initialize()
        
        # 创建特征
        X = self._create_features(precip, temp, pet)
        y = discharge
        
        # 使用有效数据（跳过前30天）
        if self.use_temporal_features:
            valid_idx = 30
            X = X[valid_idx:]
            y = y[valid_idx:]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_idx:]
        
        # 训练
        self.model.fit(X, y, sample_weight=sample_weight)
        
        # 报告特征重要性
        importances = self.model.feature_importances_
        print(f"RTREE trained. Feature importances: {importances}")
    
    def predict(self, 
                precip: np.ndarray, 
                temp: np.ndarray, 
                pet: np.ndarray) -> np.ndarray:
        """
        预测径流
        
        Returns:
        --------
        discharge : array, 预测径流
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # 创建特征
        X = self._create_features(precip, temp, pet)
        
        # 预测
        discharge = self.model.predict(X)
        
        # 确保非负
        discharge = np.maximum(discharge, 0)
        
        return discharge
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """单步预测（需要历史数据）"""
        raise NotImplementedError(
            "RTREE requires full time series for temporal features. Use predict() method."
        )
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回超参数范围"""
        return {
            'max_depth': (3, 20),
            'min_samples_split': (2, 50),
            'min_samples_leaf': (1, 20),
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """返回特征重要性"""
        if self.model is None:
            return {}
        
        feature_names = [
            'precip_t0', 'temp_t0', 'pet_t0',
            'precip_t-1', 'temp_t-1',
            'precip_week', 'temp_month'
        ] if self.use_temporal_features else ['precip', 'temp', 'pet']
        
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))