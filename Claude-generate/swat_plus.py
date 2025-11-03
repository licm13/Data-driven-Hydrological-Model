"""
SWAT+ (Soil Water Assessment Tool Plus) 简化版
参考: Bieger et al. (2017)

注意：这是一个高度简化的版本，仅包含核心水文过程
完整SWAT+需要大量输入数据和复杂的空间离散化
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from .base_model import BaseHydrologicalModel

class SWATPlus(BaseHydrologicalModel):
    """
    SWAT+ 简化水文模型
    
    包含HRU(Hydrologic Response Unit)划分
    """
    
    def __init__(self, 
                 hru_areas: List[float],
                 hru_slopes: List[float],
                 hru_cn: List[float],
                 soil_properties: List[Dict]):
        """
        Parameters:
        -----------
        hru_areas : list, HRU面积比例
        hru_slopes : list, HRU平均坡度 [%]
        hru_cn : list, SCS曲线数
        soil_properties : list of dict, 每个HRU的土壤属性
        """
        super().__init__("SWAT+")
        self.n_hrus = len(hru_areas)
        self.hru_areas = np.array(hru_areas) / np.sum(hru_areas)  # 归一化
        self.hru_slopes = np.array(hru_slopes)
        self.hru_cn = np.array(hru_cn)
        self.soil_properties = soil_properties
        
    def initialize(self, params: Dict[str, float]) -> None:
        """初始化SWAT+参数"""
        
        # 径流参数
        self.parameters = {
            'CN2': params.get('CN2', 75.0),  # 基础曲线数修正
            'SURLAG': params.get('SURLAG', 4.0),  # 地表滞后系数
            
            # 土壤参数
            'ESCO': params.get('ESCO', 0.95),  # 土壤蒸发补偿系数
            'EPCO': params.get('EPCO', 1.0),   # 植物吸收补偿系数
            
            # 地下水参数
            'GW_DELAY': params.get('GW_DELAY', 31.0),  # 地下水延迟时间 [day]
            'ALPHA_BF': params.get('ALPHA_BF', 0.048),  # 基流alpha因子
            'GWQMN': params.get('GWQMN', 0.0),  # 回渗阈值深度 [mm]
            
            # 通道汇流参数
            'CH_N2': params.get('CH_N2', 0.014),  # Manning系数
            'CH_K2': params.get('CH_K2', 0.0),    # 有效水力传导度
            
            # 雪参数
            'SFTMP': params.get('SFTMP', 1.0),    # 降雪温度 [°C]
            'SMTMP': params.get('SMTMP', 0.5),    # 融雪温度 [°C]
            'SMFMX': params.get('SMFMX', 4.5),    # 融雪因子最大值
            'SMFMN': params.get('SMFMN', 4.5),    # 融雪因子最小值
            'TIMP': params.get('TIMP', 1.0),      # 雪包温度滞后因子
        }
        
        # 每个HRU的状态变量
        self.states = {
            'snow': np.zeros(self.n_hrus),        # 雪水当量 [mm]
            'soil_water': np.zeros(self.n_hrus),  # 土壤水分 [mm]
            'shallow_gw': np.zeros(self.n_hrus),  # 浅层地下水 [mm]
            'deep_gw': np.zeros(self.n_hrus),     # 深层地下水 [mm]
            'revap': np.zeros(self.n_hrus),       # 再蒸发 [mm]
            'percolate_queue': np.zeros((self.n_hrus, 32)),  # 渗漏队列
        }
        
        # 初始化土壤含水量为田间持水量的50%
        for i in range(self.n_hrus):
            awc = self.soil_properties[i].get('awc', 200)  # 有效含水量
            self.states['soil_water'][i] = awc * 0.5
    
    def _snow_module(self, precip: float, temp: float, hru_idx: int) -> Tuple[float, float]:
        """
        雪累积和融化
        
        Returns:
        --------
        rain_liquid : 液态水 [mm]
        snowmelt : 融雪量 [mm]
        """
        SFTMP = self.parameters['SFTMP']
        SMTMP = self.parameters['SMTMP']
        SMFMX = self.parameters['SMFMX']
        
        # 降雪或降雨
        if temp <= SFTMP:
            snow = precip
            rain = 0.0
        else:
            snow = 0.0
            rain = precip
        
        # 更新雪包
        self.states['snow'][hru_idx] += snow
        
        # 融雪
        snowmelt = 0.0
        if temp > SMTMP and self.states['snow'][hru_idx] > 0:
            snowmelt = SMFMX * (temp - SMTMP)
            snowmelt = min(snowmelt, self.states['snow'][hru_idx])
            self.states['snow'][hru_idx] -= snowmelt
        
        return rain + snowmelt, snowmelt
    
    def _surface_runoff_cn(self, precip: float, hru_idx: int) -> float:
        """
        SCS曲线数法计算地表径流
        
        Returns:
        --------
        surface_runoff : [mm]
        """
        CN = self.hru_cn[hru_idx]
        CN_mod = self.parameters['CN2']
        
        # 调整CN值
        CN = CN * CN_mod / 75.0
        CN = max(35, min(98, CN))
        
        # 计算最大潜在滞留量
        S = 25.4 * (1000 / CN - 10)  # mm
        
        # 初始抽象（通常为0.2S）
        Ia = 0.2 * S
        
        # SCS方程
        if precip > Ia:
            Q_surf = ((precip - Ia) ** 2) / (precip - Ia + S)
        else:
            Q_surf = 0.0
        
        return Q_surf
    
    def _soil_water_balance(self, water_input: float, pet: float, hru_idx: int) -> Tuple[float, float, float]:
        """
        土壤水分平衡
        
        Returns:
        --------
        actual_et : 实际蒸散发 [mm]
        lateral_flow : 侧向流 [mm]
        percolation : 渗漏 [mm]
        """
        ESCO = self.parameters['ESCO']
        EPCO = self.parameters['EPCO']
        
        # 获取土壤属性
        soil = self.soil_properties[hru_idx]
        awc = soil.get('awc', 200)  # 有效含水量 [mm]
        fc = soil.get('fc', 0.3)    # 田间持水量
        wp = soil.get('wp', 0.1)    # 凋萎点
        sat = soil.get('sat', 0.4)  # 饱和含水量
        k_sat = soil.get('k_sat', 10)  # 饱和水力传导度 [mm/day]
        
        # 加入降水/融雪
        sw = self.states['soil_water'][hru_idx] + water_input
        
        # 实际蒸散发
        if sw > wp * awc:
            et_demand = pet * EPCO
            et_supply = min(et_demand, (sw - wp * awc))
            actual_et = et_supply * ESCO
        else:
            actual_et = 0.0
        
        sw -= actual_et
        
        # 渗漏（基于饱和度）
        saturation = sw / (sat * awc) if awc > 0 else 0
        percolation = k_sat * saturation * 0.1  # 简化
        percolation = min(percolation, sw - fc * awc) if sw > fc * awc else 0
        percolation = max(0, percolation)
        
        sw -= percolation
        
        # 侧向流（简化：基于坡度和饱和度）
        if sw > fc * awc:
            slope_factor = np.tan(self.hru_slopes[hru_idx] / 100)
            lateral_flow = 0.024 * slope_factor * (sw - fc * awc)
        else:
            lateral_flow = 0.0
        
        sw -= lateral_flow
        
        self.states['soil_water'][hru_idx] = max(0, min(sw, awc))
        
        return actual_et, lateral_flow, percolation
    
    def _groundwater_module(self, percolation: float, hru_idx: int) -> float:
        """
        地下水模块
        
        Returns:
        --------
        baseflow : 基流 [mm]
        """
        GW_DELAY = int(self.parameters['GW_DELAY'])
        ALPHA_BF = self.parameters['ALPHA_BF']
        GWQMN = self.parameters['GWQMN']
        
        # 渗漏延迟队列
        self.states['percolate_queue'][hru_idx] = np.roll(
            self.states['percolate_queue'][hru_idx], 1
        )
        self.states['percolate_queue'][hru_idx][0] = percolation
        
        # 到达浅层地下水的水量
        if GW_DELAY < len(self.states['percolate_queue'][hru_idx]):
            recharge = self.states['percolate_queue'][hru_idx][GW_DELAY]
        else:
            recharge = 0.0
        
        # 更新浅层地下水
        gw = self.states['shallow_gw'][hru_idx] + recharge
        
        # 基流
        if gw > GWQMN:
            baseflow = ALPHA_BF * gw
            gw -= baseflow
        else:
            baseflow = 0.0
        
        # 深层渗漏（简化：10%进入深层）
        deep_perc = 0.1 * gw
        gw -= deep_perc
        self.states['deep_gw'][hru_idx] += deep_perc
        
        self.states['shallow_gw'][hru_idx] = gw
        
        return baseflow
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """运行SWAT+的一个时间步"""
        
        total_discharge = 0.0
        
        # 对每个HRU进行模拟
        for hru_idx in range(self.n_hrus):
            # 1. 雪模块
            liquid_water, _ = self._snow_module(precip, temp, hru_idx)
            
            # 2. 地表径流
            surface_runoff = self._surface_runoff_cn(liquid_water, hru_idx)
            
            # 3. 土壤水分
            infiltration = liquid_water - surface_runoff
            actual_et, lateral_flow, percolation = self._soil_water_balance(
                infiltration, pet, hru_idx
            )
            
            # 4. 地下水和基流
            baseflow = self._groundwater_module(percolation, hru_idx)
            
            # 5. HRU总径流
            hru_discharge = surface_runoff + lateral_flow + baseflow
            
            # 按面积加权
            total_discharge += hru_discharge * self.hru_areas[hru_idx]
        
        # 6. 通道汇流（简化）
        SURLAG = self.parameters['SURLAG']
        lagged_discharge = total_discharge / SURLAG
        
        return lagged_discharge
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回参数范围"""
        return {
            'CN2': (35, 98),
            'SURLAG': (0.05, 24),
            'ESCO': (0.01, 1.0),
            'EPCO': (0.01, 1.0),
            'GW_DELAY': (0, 500),
            'ALPHA_BF': (0.0, 1.0),
            'GWQMN': (0, 5000),
            'CH_N2': (0.01, 0.3),
            'CH_K2': (0, 150),
            'SFTMP': (-5, 5),
            'SMTMP': (-5, 5),
            'SMFMX': (0, 10),
            'SMFMN': (0, 10),
            'TIMP': (0, 1),
        }