"""
数据加载和预处理工具
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml

class CatchmentData:
    """流域数据类"""
    
    def __init__(self, 
                 name: str,
                 precip: np.ndarray,
                 temp: np.ndarray,
                 pet: np.ndarray,
                 discharge: np.ndarray,
                 dates: pd.DatetimeIndex,
                 area: float = None,
                 elevation_range: Tuple[float, float] = None,
                 metadata: Dict = None):
        """
        Parameters:
        -----------
        name : str, 流域名称
        precip : array, 降水 [mm/day]
        temp : array, 温度 [°C]
        pet : array, 潜在蒸散发 [mm/day]
        discharge : array, 径流 [mm/day]
        dates : DatetimeIndex, 时间索引
        area : float, 流域面积 [km²]
        elevation_range : tuple, 高程范围 [m]
        metadata : dict, 其他元数据
        """
        self.name = name
        self.precip = precip
        self.temp = temp
        self.pet = pet
        self.discharge = discharge
        self.dates = dates
        self.area = area
        self.elevation_range = elevation_range
        self.metadata = metadata or {}
        
        # 验证数据长度一致
        assert len(precip) == len(temp) == len(pet) == len(discharge) == len(dates), \
            "All data arrays must have the same length"
    
    def __len__(self):
        return len(self.dates)
    
    def __repr__(self):
        return (f"CatchmentData(name='{self.name}', "
                f"n_days={len(self)}, "
                f"period={self.dates[0].date()} to {self.dates[-1].date()})")
    
    def split(self, 
              train_period: Tuple[str, str],
              test_period: Tuple[str, str],
              warmup_days: int = 365) -> Tuple['CatchmentData', 'CatchmentData']:
        """
        划分训练/测试数据
        
        Parameters:
        -----------
        train_period : tuple, 训练期 (start, end)
        test_period : tuple, 测试期 (start, end)
        warmup_days : int, 预热期天数
        
        Returns:
        --------
        train_data : CatchmentData
        test_data : CatchmentData
        """
        # 训练期（包含预热）
        train_start = pd.Timestamp(train_period[0]) - pd.Timedelta(days=warmup_days)
        train_end = pd.Timestamp(train_period[1])
        train_mask = (self.dates >= train_start) & (self.dates <= train_end)
        
        train_data = CatchmentData(
            name=self.name + '_train',
            precip=self.precip[train_mask],
            temp=self.temp[train_mask],
            pet=self.pet[train_mask],
            discharge=self.discharge[train_mask],
            dates=self.dates[train_mask],
            area=self.area,
            elevation_range=self.elevation_range,
            metadata={**self.metadata, 'warmup_days': warmup_days}
        )
        
        # 测试期（包含预热）
        test_start = pd.Timestamp(test_period[0]) - pd.Timedelta(days=warmup_days)
        test_end = pd.Timestamp(test_period[1])
        test_mask = (self.dates >= test_start) & (self.dates <= test_end)
        
        test_data = CatchmentData(
            name=self.name + '_test',
            precip=self.precip[test_mask],
            temp=self.temp[test_mask],
            pet=self.pet[test_mask],
            discharge=self.discharge[test_mask],
            dates=self.dates[test_mask],
            area=self.area,
            elevation_range=self.elevation_range,
            metadata={**self.metadata, 'warmup_days': warmup_days}
        )
        
        return train_data, test_data
    
    def get_statistics(self) -> Dict:
        """计算数据统计特征"""
        return {
            'precip': {
                'mean': np.mean(self.precip),
                'std': np.std(self.precip),
                'max': np.max(self.precip),
                'min': np.min(self.precip),
            },
            'temp': {
                'mean': np.mean(self.temp),
                'std': np.std(self.temp),
                'max': np.max(self.temp),
                'min': np.min(self.temp),
            },
            'pet': {
                'mean': np.mean(self.pet),
                'std': np.std(self.pet),
                'max': np.max(self.pet),
                'min': np.min(self.pet),
            },
            'discharge': {
                'mean': np.mean(self.discharge),
                'std': np.std(self.discharge),
                'max': np.max(self.discharge),
                'min': np.min(self.discharge),
                'q95': np.percentile(self.discharge, 95),
                'q5': np.percentile(self.discharge, 5),
            }
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame({
            'date': self.dates,
            'precip': self.precip,
            'temp': self.temp,
            'pet': self.pet,
            'discharge': self.discharge,
        }).set_index('date')


def load_catchment_from_csv(data_dir: str, 
                            catchment_name: str,
                            config: Dict = None) -> CatchmentData:
    """
    从CSV文件加载流域数据
    
    预期文件结构：
    data_dir/
        catchment_name/
            meteorology.csv  (columns: date, precip, temp, pet)
            discharge.csv    (columns: date, discharge)
            config.yaml      (可选)
    
    Parameters:
    -----------
    data_dir : str, 数据目录
    catchment_name : str, 流域名称
    config : dict, 配置（如果不提供，从config.yaml读取）
    
    Returns:
    --------
    data : CatchmentData
    """
    data_path = Path(data_dir) / catchment_name
    
    # 加载配置
    if config is None:
        config_file = data_path / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # 加载气象数据
    meteo_file = data_path / 'meteorology.csv'
    meteo_df = pd.read_csv(meteo_file, parse_dates=['date'])
    
    # 加载径流数据
    discharge_file = data_path / 'discharge.csv'
    discharge_df = pd.read_csv(discharge_file, parse_dates=['date'])
    
    # 合并
    df = pd.merge(meteo_df, discharge_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 创建CatchmentData对象
    data = CatchmentData(
        name=catchment_name,
        precip=df['precip'].values,
        temp=df['temp'].values,
        pet=df['pet'].values,
        discharge=df['discharge'].values,
        dates=pd.DatetimeIndex(df['date']),
        area=config.get('area'),
        elevation_range=config.get('elevation_range'),
        metadata=config
    )
    
    return data


def generate_synthetic_data(n_days: int = 3650,
                           seed: int = 42) -> CatchmentData:
    """
    生成合成数据用于测试
    
    Parameters:
    -----------
    n_days : int, 天数
    seed : int, 随机种子
    
    Returns:
    --------
    data : CatchmentData
    """
    np.random.seed(seed)
    
    # 生成日期
    dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
    
    # 生成气象数据（带季节性）
    t = np.arange(n_days)
    
    # 降水：泊松过程 + 季节性
    seasonal_precip = 5 + 3 * np.sin(2 * np.pi * t / 365)
    precip = np.random.gamma(2, seasonal_precip / 2)
    
    # 温度：正弦波 + 噪声
    temp = 10 + 10 * np.sin(2 * np.pi * (t - 80) / 365) + np.random.randn(n_days) * 2
    
    # 蒸发：与温度相关
    pet = np.maximum(0, 2 + 0.3 * temp + np.random.randn(n_days) * 0.5)
    
    # 径流：简单水量平衡 + 延迟
    storage = 100.0
    discharge = np.zeros(n_days)
    
    for i in range(n_days):
        # 入流
        inflow = max(0, precip[i] - pet[i])
        storage += inflow
        
        # 出流（线性水库）
        outflow = 0.1 * storage
        storage -= outflow
        discharge[i] = outflow
        
        # 限制存储
        storage = max(0, min(storage, 300))
    
    data = CatchmentData(
        name='synthetic',
        precip=precip,
        temp=temp,
        pet=pet,
        discharge=discharge,
        dates=dates,
        area=1000.0,
        elevation_range=(100, 500),
        metadata={'type': 'synthetic'}
    )
    
    return data


def load_multiple_catchments(data_dir: str, 
                             catchment_names: List[str]) -> Dict[str, CatchmentData]:
    """
    加载多个流域数据
    
    Parameters:
    -----------
    data_dir : str, 数据目录
    catchment_names : list, 流域名称列表
    
    Returns:
    --------
    catchments : dict, {name: CatchmentData}
    """
    catchments = {}
    
    for name in catchment_names:
        print(f"Loading {name}...")
        try:
            data = load_catchment_from_csv(data_dir, name)
            catchments[name] = data
            print(f"  Loaded {len(data)} days of data")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    
    return catchments