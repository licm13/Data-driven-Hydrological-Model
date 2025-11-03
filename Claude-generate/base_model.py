"""
基础水文模型抽象类
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional

class BaseHydrologicalModel(ABC):
    """所有水文模型的基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.parameter_ranges = {}
        self.states = {}
        
    @abstractmethod
    def initialize(self, params: Dict[str, float]) -> None:
        """初始化模型参数和状态"""
        pass
    
    @abstractmethod
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """运行单个时间步"""
        pass
    
    def simulate(self, 
                 precip: np.ndarray,
                 temp: np.ndarray,
                 pet: np.ndarray,
                 warmup_steps: int = 365) -> np.ndarray:
        """
        运行完整模拟
        
        Parameters:
        -----------
        precip : array, 降水 [mm/day]
        temp : array, 气温 [°C]
        pet : array, 潜在蒸散发 [mm/day]
        warmup_steps : int, 预热期步数
        
        Returns:
        --------
        discharge : array, 模拟径流 [mm/day]
        """
        n_steps = len(precip)
        discharge = np.zeros(n_steps)
        
        # 模拟
        for t in range(n_steps):
            discharge[t] = self.run_timestep(
                precip[t], temp[t], pet[t], t
            )
        
        # 返回预热期后的结果
        return discharge[warmup_steps:]
    
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回参数范围"""
        pass
    
    def reset_states(self) -> None:
        """重置模型状态"""
        for key in self.states:
            self.states[key] = 0.0