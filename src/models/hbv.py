"""
HBV (Hydrologiska Byråns Vattenbalansavdelning) 水文模型
参考: Seibert & Vis (2012) - HBV-light
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from .base_model import BaseHydrologicalModel

class HBV(BaseHydrologicalModel):
    """
    HBV-light半分布式水文模型
    
    支持子流域和高程带划分
    """
    
    def __init__(self, 
                 n_elevation_zones: int = 1,
                 elevation_ranges: Optional[List[Tuple[float, float]]] = None,
                 area_fractions: Optional[List[float]] = None):
        """
        Parameters:
        -----------
        n_elevation_zones : int, 高程带数量
        elevation_ranges : list, 各高程带的范围 [(min, max), ...]
        area_fractions : list, 各高程带的面积比例
        """
        super().__init__("HBV")
        self.n_zones = n_elevation_zones
        self.elevation_ranges = elevation_ranges or [(0, 1000)] * n_elevation_zones
        self.area_fractions = area_fractions or [1.0 / n_elevation_zones] * n_elevation_zones
        
    def initialize(self, params: Dict[str, float]) -> None:
        """初始化HBV模型参数"""
        
        # 雪模块参数
        self.parameters = {
            'TT': params.get('TT', 0.0),      # 阈值温度 [°C]
            'CFMAX': params.get('CFMAX', 3.5),  # 度日因子 [mm/°C/day]
            'SFCF': params.get('SFCF', 1.0),    # 降雪修正系数
            'CFR': params.get('CFR', 0.05),     # 再冻结系数
            'CWH': params.get('CWH', 0.1),      # 持水能力
            
            # 土壤模块参数
            'FC': params.get('FC', 250.0),      # 土壤最大含水量 [mm]
            'LP': params.get('LP', 0.7),        # 蒸发限制
            'BETA': params.get('BETA', 2.0),    # 形状系数
            
            # 响应模块参数
            'PERC': params.get('PERC', 2.0),    # 渗漏率 [mm/day]
            'UZL': params.get('UZL', 50.0),     # 快速径流阈值 [mm]
            'K0': params.get('K0', 0.2),        # 快速径流系数 [1/day]
            'K1': params.get('K1', 0.1),        # 中速径流系数 [1/day]
            'K2': params.get('K2', 0.05),       # 慢速径流系数 [1/day]
            
            # 汇流模块参数
            'MAXBAS': params.get('MAXBAS', 3.0), # 汇流时间 [day]
        }
        
        # 温度和降水随高程的梯度
        self.parameters['TCALT'] = params.get('TCALT', -0.6)  # 温度梯度 [°C/100m]
        self.parameters['PCALT'] = params.get('PCALT', 10.0)  # 降水梯度 [%/100m]
        
        # 为每个高程带初始化状态
        self.states = {
            'snow': np.zeros(self.n_zones),      # 雪水当量
            'snow_water': np.zeros(self.n_zones), # 积雪液态水
            'soil': np.ones(self.n_zones) * self.parameters['FC'] * 0.5,  # 土壤水
            'SUZ': 0.0,  # 上层地下水库
            'SLZ': 0.0,  # 下层地下水库
            'routing_store': np.zeros(int(self.parameters['MAXBAS']) + 1),  # 汇流存储
        }
    
    def _snow_routine(self, precip: float, temp: float, zone_idx: int) -> Tuple[float, float]:
        """
        雪累积和融化
        
        Returns:
        --------
        rainfall : 降雨量 [mm]
        snowmelt : 融雪量 [mm]
        """
        TT = self.parameters['TT']
        CFMAX = self.parameters['CFMAX']
        SFCF = self.parameters['SFCF']
        CFR = self.parameters['CFR']
        CWH = self.parameters['CWH']
        
        # 区分降雪和降雨
        if temp < TT:
            snowfall = precip * SFCF
            rainfall = 0.0
        else:
            snowfall = 0.0
            rainfall = precip
        
        # 更新积雪
        self.states['snow'][zone_idx] += snowfall
        
        # 融雪
        snowmelt = 0.0
        if temp > TT:
            potential_melt = CFMAX * (temp - TT)
            snowmelt = min(potential_melt, self.states['snow'][zone_idx])
            self.states['snow'][zone_idx] -= snowmelt
        
        # 再冻结
        if temp < TT and self.states['snow_water'][zone_idx] > 0:
            refreezing = min(CFR * CFMAX * (TT - temp), self.states['snow_water'][zone_idx])
            self.states['snow'][zone_idx] += refreezing
            self.states['snow_water'][zone_idx] -= refreezing
        
        # 液态水滞留
        self.states['snow_water'][zone_idx] += snowmelt
        max_water = CWH * self.states['snow'][zone_idx]
        
        if self.states['snow_water'][zone_idx] > max_water:
            water_release = self.states['snow_water'][zone_idx] - max_water
            self.states['snow_water'][zone_idx] = max_water
            rainfall += water_release
        
        return rainfall, snowmelt
    
    def _soil_routine(self, water_input: float, pet: float, zone_idx: int) -> float:
        """
        土壤水分平衡
        
        Returns:
        --------
        recharge : 地下水补给 [mm]
        """
        FC = self.parameters['FC']
        LP = self.parameters['LP']
        BETA = self.parameters['BETA']
        
        # 实际蒸发
        soil_moisture = self.states['soil'][zone_idx]
        if soil_moisture > LP * FC:
            actual_et = pet
        else:
            actual_et = pet * (soil_moisture / (LP * FC))
        
        # 更新土壤水分
        soil_moisture = max(0, soil_moisture - actual_et)
        
        # 地下水补给（非线性）
        if water_input > 0:
            soil_ratio = soil_moisture / FC if FC > 0 else 0
            recharge = water_input * (soil_ratio ** BETA)
            soil_moisture = min(FC, soil_moisture + water_input - recharge)
        else:
            recharge = 0.0
        
        self.states['soil'][zone_idx] = soil_moisture
        return recharge
    
    def _response_routine(self, recharge: float) -> float:
        """
        响应函数（地下水出流）
        
        Returns:
        --------
        total_runoff : 总径流 [mm]
        """
        PERC = self.parameters['PERC']
        UZL = self.parameters['UZL']
        K0 = self.parameters['K0']
        K1 = self.parameters['K1']
        K2 = self.parameters['K2']
        
        # 上层地下水库
        SUZ = self.states['SUZ'] + recharge
        
        # 快速径流（超过阈值）
        Q0 = 0.0
        if SUZ > UZL:
            Q0 = K0 * (SUZ - UZL)
            SUZ -= Q0
        
        # 中速径流
        Q1 = K1 * SUZ
        SUZ -= Q1
        
        # 渗漏到下层
        perc = min(PERC, SUZ)
        SUZ -= perc
        
        self.states['SUZ'] = max(0, SUZ)
        
        # 下层地下水库
        SLZ = self.states['SLZ'] + perc
        Q2 = K2 * SLZ  # 基流
        SLZ -= Q2
        
        self.states['SLZ'] = max(0, SLZ)
        
        total_runoff = Q0 + Q1 + Q2
        return total_runoff
    
    def _routing_routine(self, runoff: float) -> float:
        """
        三角形汇流
        
        Returns:
        --------
        routed_discharge : 汇流后径流 [mm]
        """
        MAXBAS = self.parameters['MAXBAS']
        
        # 三角形单位线权重
        n = int(MAXBAS) + 1
        weights = np.zeros(n)
        for i in range(n):
            if i <= MAXBAS / 2:
                weights[i] = i / (MAXBAS / 2)
            else:
                weights[i] = (MAXBAS - i) / (MAXBAS / 2)
        
        # 归一化
        weights = weights / np.sum(weights)
        
        # 更新存储
        self.states['routing_store'] = np.roll(self.states['routing_store'], 1)
        self.states['routing_store'][0] = runoff
        
        # 卷积
        routed = np.sum(self.states['routing_store'][:len(weights)] * weights)
        
        return routed
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """运行HBV的一个时间步"""
        
        # 对每个高程带进行计算
        total_recharge = 0.0
        
        for zone_idx in range(self.n_zones):
            # 高程修正
            mean_elevation = np.mean(self.elevation_ranges[zone_idx])
            temp_corrected = temp + self.parameters['TCALT'] * mean_elevation / 100
            precip_corrected = precip * (1 + self.parameters['PCALT'] * mean_elevation / 100 / 100)
            
            # 雪模块
            rainfall, _ = self._snow_routine(precip_corrected, temp_corrected, zone_idx)
            
            # 土壤模块
            recharge = self._soil_routine(rainfall, pet, zone_idx)
            
            # 按面积加权
            total_recharge += recharge * self.area_fractions[zone_idx]
        
        # 响应模块
        runoff = self._response_routine(total_recharge)
        
        # 汇流
        discharge = self._routing_routine(runoff)
        
        return discharge
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回参数范围"""
        return {
            'TT': (-2.5, 2.5),
            'CFMAX': (0.5, 10),
            'SFCF': (0.4, 1.6),
            'CFR': (0, 0.1),
            'CWH': (0, 0.2),
            'FC': (50, 500),
            'LP': (0.3, 1.0),
            'BETA': (1, 6),
            'PERC': (0, 6),
            'UZL': (0, 100),
            'K0': (0.05, 0.5),
            'K1': (0.01, 0.4),
            'K2': (0.001, 0.15),
            'MAXBAS': (1, 7),
            'TCALT': (-1.0, 0.0),
            'PCALT': (0, 20),
        }