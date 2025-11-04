"""
HBV (Hydrologiska Byråns Vattenbalansavdelning) 水文模型
===========================================================

HBV模型是由瑞典水文研究所开发的概念性水文模型，广泛应用于全球各种气候条件下的径流模拟。

模型结构：
---------
1. 雪模块 (Snow Routine)：
   - 采用度日法计算雪的累积和融化
   - 考虑液态水在雪层中的滞留
   - 可模拟再冻结过程

2. 土壤模块 (Soil Routine)：
   - 非线性的土壤水分平衡
   - BETA参数控制产流的非线性程度
   - 考虑实际蒸散发随土壤湿度的变化

3. 响应模块 (Response Routine)：
   - 双层地下水库结构（上层和下层）
   - 三种径流分量：快速流、中速流、基流
   - PERC参数控制层间渗漏

4. 汇流模块 (Routing Routine)：
   - 三角形单位线进行汇流演算
   - MAXBAS参数控制汇流时间

空间离散化：
-----------
- 支持多个高程带划分（半分布式）
- 各高程带独立计算雪和土壤过程
- 地下水响应和汇流在流域尺度进行

参考文献：
---------
Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a
user-friendly catchment-runoff-model software package. Hydrology and Earth
System Sciences, 16(9), 3315-3325.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from .base_model import BaseHydrologicalModel

class HBV(BaseHydrologicalModel):
    """
    HBV-light半分布式水文模型

    特点：
    -----
    - 概念性模型，参数有明确的物理意义
    - 支持多高程带，适合山区流域
    - 计算效率高，适合长时间序列模拟
    - 参数相对较少（10-15个），易于校准

    适用场景：
    ---------
    - 山区和雪融主导的流域
    - 日尺度或小时尺度径流模拟
    - 气候变化影响评估
    - 水资源管理和洪水预报
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
        雪累积和融化子程序（度日法）

        原理：
        -----
        1. 根据阈值温度(TT)区分降雪和降雨
        2. 使用度日因子(CFMAX)计算融雪量：融雪 = CFMAX × (T - TT)
        3. 模拟液态水在雪层中的滞留（持水能力CWH）
        4. 考虑再冻结过程（CFR）

        物理过程：
        ---------
        当温度 < TT 时：降水视为降雪，累积在雪层
        当温度 > TT 时：积雪融化，融化速率与温度成正比
        融化的水一部分被雪层保持（类似海绵），超过持水能力的水释放出来

        Parameters:
        -----------
        precip : float
            降水量 [mm/day]
        temp : float
            温度 [°C]
        zone_idx : int
            高程带索引

        Returns:
        --------
        rainfall : float
            液态水输入到土壤模块的水量 [mm]，包括降雨和释放的融雪水
        snowmelt : float
            融雪量 [mm]

        状态变量：
        ---------
        snow[zone_idx] : 雪水当量 (SWE) [mm]
        snow_water[zone_idx] : 积雪中的液态水量 [mm]
        """
        # 获取雪模块参数
        TT = self.parameters['TT']        # 阈值温度 [°C]
        CFMAX = self.parameters['CFMAX']  # 度日因子 [mm/°C/day]
        SFCF = self.parameters['SFCF']    # 降雪修正系数 [-]
        CFR = self.parameters['CFR']      # 再冻结系数 [-]
        CWH = self.parameters['CWH']      # 持水能力 [-]，雪层能保持的液态水与雪水当量的比例

        # ========================================
        # 步骤1：区分降雪和降雨
        # ========================================
        if temp < TT:
            # 温度低于阈值，全部视为降雪
            # SFCF用于修正降雪观测误差（如风吹雪的影响）
            snowfall = precip * SFCF
            rainfall = 0.0
        else:
            # 温度高于阈值，全部视为降雨
            snowfall = 0.0
            rainfall = precip

        # 更新雪水当量
        self.states['snow'][zone_idx] += snowfall

        # ========================================
        # 步骤2：计算融雪（度日法）
        # ========================================
        snowmelt = 0.0
        if temp > TT:
            # 潜在融雪量 = 度日因子 × 温度超出阈值的部分
            # 例如：CFMAX=3.5 mm/°C/day, T=5°C, TT=0°C
            # 则 potential_melt = 3.5 × 5 = 17.5 mm
            potential_melt = CFMAX * (temp - TT)

            # 实际融雪量不能超过现有积雪
            snowmelt = min(potential_melt, self.states['snow'][zone_idx])
            self.states['snow'][zone_idx] -= snowmelt

        # ========================================
        # 步骤3：再冻结过程
        # ========================================
        # 当温度下降且雪层中有液态水时，部分液态水会再次冻结
        if temp < TT and self.states['snow_water'][zone_idx] > 0:
            # 潜在再冻结量
            potential_refreeze = CFR * CFMAX * (TT - temp)

            # 实际再冻结量不能超过现有液态水
            refreezing = min(potential_refreeze, self.states['snow_water'][zone_idx])

            # 液态水转换为固态雪
            self.states['snow'][zone_idx] += refreezing
            self.states['snow_water'][zone_idx] -= refreezing

        # ========================================
        # 步骤4：液态水滞留（雪层持水能力）
        # ========================================
        # 融化的水首先进入雪层的孔隙
        self.states['snow_water'][zone_idx] += snowmelt

        # 雪层最大持水能力 = CWH × 雪水当量
        # 例如：CWH=0.1, SWE=100mm，则最多能保持10mm液态水
        max_water = CWH * self.states['snow'][zone_idx]

        # 超过持水能力的水释放出来，成为降雨输入到土壤
        if self.states['snow_water'][zone_idx] > max_water:
            water_release = self.states['snow_water'][zone_idx] - max_water
            self.states['snow_water'][zone_idx] = max_water
            rainfall += water_release

        return rainfall, snowmelt
    
    def _soil_routine(self, water_input: float, pet: float, zone_idx: int) -> float:
        """
        土壤水分平衡子程序（非线性产流）

        核心思想：
        ---------
        1. 土壤湿度越高，产流能力越强（非线性关系）
        2. 实际蒸散发随土壤湿度线性减少
        3. BETA参数控制产流的非线性程度

        物理机制：
        ---------
        - 当土壤很湿时：大部分降水转化为径流（高产流率）
        - 当土壤很干时：降水优先补充土壤水分（低产流率）
        - 这种非线性关系由幂函数 (SM/FC)^BETA 描述

        计算步骤：
        ---------
        1. 计算实际蒸散发（受土壤湿度限制）
        2. 扣除蒸散发后更新土壤湿度
        3. 根据当前土壤湿度计算产流（地下水补给）
        4. 剩余水分补充土壤蓄水

        Parameters:
        -----------
        water_input : float
            输入到土壤的水量 [mm]，包括降雨和融雪水
        pet : float
            潜在蒸散发 [mm/day]
        zone_idx : int
            高程带索引

        Returns:
        --------
        recharge : float
            地下水补给量 [mm]，即产流量

        状态变量：
        ---------
        soil[zone_idx] : 土壤水分储量 [mm]

        参数说明：
        ---------
        FC : 土壤最大含水量 [mm]，反映土壤的蓄水能力
        LP : 蒸发限制参数 [-]，当SM < LP×FC时蒸发受限
        BETA : 形状系数 [-]，控制产流非线性程度
             BETA=1：线性产流
             BETA>1：凸型曲线，湿态下产流率快速增加
             BETA越大，非线性越强
        """
        # 获取土壤模块参数
        FC = self.parameters['FC']      # 土壤最大含水量 [mm]
        LP = self.parameters['LP']      # 蒸发限制参数 [-]
        BETA = self.parameters['BETA']  # 形状系数 [-]

        # ========================================
        # 步骤1：计算实际蒸散发
        # ========================================
        soil_moisture = self.states['soil'][zone_idx]

        # 当土壤湿度高于阈值(LP×FC)时，蒸发不受限制
        # 当土壤湿度低于阈值时，蒸发随湿度线性减少
        if soil_moisture > LP * FC:
            # 土壤足够湿润，实际蒸散发 = 潜在蒸散发
            actual_et = pet
        else:
            # 土壤较干，实际蒸散发按比例减少
            # 例如：LP=0.7, FC=250mm, SM=100mm
            # actual_ET = PET × (100 / (0.7×250)) = PET × 0.57
            actual_et = pet * (soil_moisture / (LP * FC))

        # 扣除蒸散发
        soil_moisture = max(0, soil_moisture - actual_et)

        # ========================================
        # 步骤2：计算产流（非线性）
        # ========================================
        if water_input > 0:
            # 当前土壤相对湿度
            soil_ratio = soil_moisture / FC if FC > 0 else 0

            # 产流率 = (SM/FC)^BETA
            # 例如：SM=200mm, FC=250mm, BETA=2.0
            # 产流率 = (200/250)^2 = 0.64
            # 即64%的输入水转化为产流，36%补充土壤

            # 产流量（地下水补给）
            recharge = water_input * (soil_ratio ** BETA)

            # 剩余水分补充土壤
            soil_moisture = min(FC, soil_moisture + water_input - recharge)
        else:
            recharge = 0.0

        # 更新状态
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
        
        # 确保MAXBAS至少为1
        MAXBAS = max(MAXBAS, 1.0)
        
        # 三角形单位线权重
        n = int(MAXBAS) + 1
        weights = np.zeros(n)
        for i in range(n):
            if i <= MAXBAS / 2:
                weights[i] = i / (MAXBAS / 2) if MAXBAS > 0 else 1.0
            else:
                weights[i] = (MAXBAS - i) / (MAXBAS / 2) if MAXBAS > 0 else 0.0
        
        # 归一化（避免除零）
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # 如果权重和为0，设置简单的单位权重
            weights = np.ones(n) / n
        
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