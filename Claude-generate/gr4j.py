"""
GR4J (Génie Rural à 4 paramètres Journalier) 日尺度径流模型
参考: Perrin et al. (2003)
"""
import numpy as np
from typing import Dict, Tuple
from .base_model import BaseHydrologicalModel

class GR4J(BaseHydrologicalModel):
    """
    GR4J模型 + CemaNeige雪模块
    
    Parameters:
    -----------
    X1 : float, 生产库最大容量 [mm]
    X2 : float, 地下水交换系数 [mm/day]
    X3 : float, 汇流库容量 [mm]
    X4 : float, 单位线时间基数 [day]
    CTG : float, 雪融化温度阈值 [°C] (CemaNeige)
    Kf : float, 日融雪因子 [mm/°C/day] (CemaNeige)
    """
    
    def __init__(self, with_snow: bool = True):
        super().__init__("GR4J")
        self.with_snow = with_snow
        
    def initialize(self, params: Dict[str, float]) -> None:
        """初始化模型参数"""
        self.parameters = {
            'X1': params.get('X1', 350.0),  # 生产库容量
            'X2': params.get('X2', 0.0),    # 地下水交换
            'X3': params.get('X3', 90.0),   # 汇流库容量
            'X4': params.get('X4', 1.7),    # UH时间基数
        }
        
        if self.with_snow:
            self.parameters.update({
                'CTG': params.get('CTG', 0.0),   # 融雪温度阈值
                'Kf': params.get('Kf', 3.5),     # 融雪系数
            })
        
        # 初始化状态变量
        self.states = {
            'S': self.parameters['X1'] * 0.5,  # 生产库存储
            'R': self.parameters['X3'] * 0.5,  # 汇流库存储
            'UH1': np.zeros(20),  # 单位线1
            'UH2': np.zeros(40),  # 单位线2
            'snow_pack': 0.0,  # 积雪
        }
        
        # 计算单位线序数
        self._compute_unit_hydrographs()
        
    def _compute_unit_hydrographs(self) -> None:
        """计算S型单位线"""
        X4 = self.parameters['X4']
        
        # UH1 (9成径流)
        nUH1 = int(np.ceil(X4))
        self.UH1_ordinates = np.zeros(nUH1)
        for t in range(nUH1):
            if t < X4:
                self.UH1_ordinates[t] = ((t / X4) ** 2.5)
            else:
                self.UH1_ordinates[t] = 1.0
        
        # 归一化
        if len(self.UH1_ordinates) > 1:
            self.UH1_ordinates = np.diff(np.append(0, self.UH1_ordinates))
        
        # UH2 (1成径流)
        nUH2 = int(np.ceil(2 * X4))
        self.UH2_ordinates = np.zeros(nUH2)
        for t in range(nUH2):
            if t < X4:
                self.UH2_ordinates[t] = 0.5 * ((t / X4) ** 2.5)
            elif t < 2 * X4:
                self.UH2_ordinates[t] = 1 - 0.5 * (2 - t / X4) ** 2.5
            else:
                self.UH2_ordinates[t] = 1.0
        
        if len(self.UH2_ordinates) > 1:
            self.UH2_ordinates = np.diff(np.append(0, self.UH2_ordinates))
    
    def _snow_routine(self, precip: float, temp: float) -> Tuple[float, float]:
        """
        CemaNeige雪模块
        
        Returns:
        --------
        liquid_precip : 液态降水 [mm]
        snow_pack : 积雪量 [mm]
        """
        if not self.with_snow:
            return precip, 0.0
        
        CTG = self.parameters['CTG']
        Kf = self.parameters['Kf']
        
        # 判断降雪还是降雨
        if temp < CTG:
            # 降雪
            snow_fall = precip
            rain_fall = 0.0
        else:
            # 降雨
            snow_fall = 0.0
            rain_fall = precip
        
        # 更新积雪
        self.states['snow_pack'] += snow_fall
        
        # 融雪
        if temp > CTG and self.states['snow_pack'] > 0:
            melt = min(Kf * (temp - CTG), self.states['snow_pack'])
            self.states['snow_pack'] -= melt
            rain_fall += melt
        
        return rain_fall, self.states['snow_pack']
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """运行GR4J的一个时间步"""
        
        # 1. 雪模块
        liquid_precip, _ = self._snow_routine(precip, temp)
        
        # 2. 净降水和净蒸发
        if liquid_precip >= pet:
            Pn = liquid_precip - pet
            En = 0.0
            ratio_PE = 0.0
        else:
            Pn = 0.0
            En = pet - liquid_precip
            ratio_PE = En / self.parameters['X1'] if self.parameters['X1'] > 0 else 0.0
        
        # 3. 生产库
        S = self.states['S']
        X1 = self.parameters['X1']
        
        if Pn > 0:
            # 降水情况
            ratio_S = S / X1 if X1 > 0 else 0.0
            Ps = X1 * (1 - ratio_S**2) * np.tanh(Pn / X1) / (1 + ratio_S * np.tanh(Pn / X1))
            S = S + Ps
        else:
            # 蒸发情况
            ratio_S = S / X1 if X1 > 0 else 0.0
            Es = S * (2 - ratio_S) * np.tanh(ratio_PE) / (1 + (1 - ratio_S) * np.tanh(ratio_PE))
            S = S - Es
        
        # 生产库渗流
        Perc = S * (1 - (1 + (4/9 * S / X1)**4)**(-0.25))
        S = S - Perc
        self.states['S'] = max(0, S)
        
        # 4. 总径流 = 渗流 + 超渗降水
        Pr = Perc + (Pn - Ps if Pn > 0 else 0)
        
        # 5. 分配到两条径流路径
        Q9 = 0.9 * Pr  # UH1路径
        Q1 = 0.1 * Pr  # UH2路径
        
        # 6. 单位线卷积
        self.states['UH1'] = np.roll(self.states['UH1'], 1)
        self.states['UH1'][0] = Q9
        Q9_routed = np.sum(self.states['UH1'] * self.UH1_ordinates[:len(self.states['UH1'])])
        
        self.states['UH2'] = np.roll(self.states['UH2'], 1)
        self.states['UH2'][0] = Q1
        Q1_routed = np.sum(self.states['UH2'] * self.UH2_ordinates[:len(self.states['UH2'])])
        
        # 7. 地下水交换
        X2 = self.parameters['X2']
        X3 = self.parameters['X3']
        R = self.states['R']
        
        F = X2 * (R / X3) ** 3.5 if X3 > 0 else 0
        
        # 8. 汇流库
        R = max(0, R + Q9_routed + F)
        Qr = R * (1 - (1 + (R / X3)**4)**(-0.25)) if X3 > 0 else R
        R = R - Qr
        self.states['R'] = R
        
        # 9. 总径流
        Qd = max(0, Q1_routed + F)
        Q = Qr + Qd
        
        return Q
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回参数范围"""
        bounds = {
            'X1': (100, 1200),   # 生产库容量
            'X2': (-5, 3),       # 地下水交换
            'X3': (20, 300),     # 汇流库容量
            'X4': (1.1, 2.9),    # UH时间
        }
        
        if self.with_snow:
            bounds.update({
                'CTG': (-2, 2),    # 融雪温度
                'Kf': (1, 10),     # 融雪系数
            })
        
        return bounds