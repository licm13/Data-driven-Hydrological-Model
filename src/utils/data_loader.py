"""
数据加载和预处理工具
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re
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
                 area: Optional[float] = None,
                 elevation_range: Optional[Tuple[float, float]] = None,
                 metadata: Optional[Dict] = None):
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
                            config: Optional[Dict] = None) -> CatchmentData:
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
    
    # 加载气象与径流数据（支持CSV或通用ASCII回退）
    meteo_file = data_path / 'meteorology.csv'
    discharge_file = data_path / 'discharge.csv'

    if meteo_file.exists() and discharge_file.exists():
        meteo_df = pd.read_csv(meteo_file, parse_dates=['date'])
        discharge_df = pd.read_csv(discharge_file, parse_dates=['date'])
    else:
        # 回退：尝试从通用文件结构（ASCII/空白分隔/TSV）自动解析
        meteo_df, discharge_df = _load_catchment_from_folder_generic(
            data_path, config
        )
    
    # 合并
    df = pd.merge(meteo_df, discharge_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 创建CatchmentData对象
    data = CatchmentData(
        name=catchment_name,
        precip=np.asarray(df['precip'].to_numpy(dtype=float)),
        temp=np.asarray(df['temp'].to_numpy(dtype=float)),
        pet=np.asarray(df['pet'].to_numpy(dtype=float)),
        discharge=np.asarray(df['discharge'].to_numpy(dtype=float)),
        dates=pd.DatetimeIndex(df['date']),
        area=(config or {}).get('area'),
        elevation_range=(config or {}).get('elevation_range'),
        metadata=config
    )
    
    return data


# ===============
# 通用加载工具
# ===============

DATE_CANDIDATES = ['date', 'Date', 'DATE', 'time', 'Time']


def _read_table_with_date(
    file_path: Path,
    specified_sep: Optional[str] = None,
    decimal: Optional[str] = None,
    columns_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read a table and return a dataframe with a 'date' column.

    - Tries multiple separators (comma, semicolon, tab, whitespace)
    - Detects date column by common names or by (year, month, day)
    - Applies optional columns_map to rename columns
    """
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    seps = [specified_sep] if specified_sep else [',', ';', '\t', None]
    errors = []
    df: Optional[pd.DataFrame] = None
    for sep in seps:
        try:
            if sep is None:
                # Let pandas infer, try delim_whitespace
                df = pd.read_csv(
                    file_path, engine='python', delim_whitespace=True, comment='#'
                )
            else:
                if decimal is not None:
                    df = pd.read_csv(
                        file_path, engine='python', sep=sep, decimal=decimal, comment='#'
                    )
                else:
                    df = pd.read_csv(
                        file_path, engine='python', sep=sep, comment='#'
                    )
            break
        except Exception as e:
            errors.append((sep, str(e)))
            df = None
    if df is None:
        raise ValueError(f"Failed to read {file_path} with tried seps: {errors}")

    # Normalize columns
    if columns_map:
        df = df.rename(columns=columns_map)

    # Try to construct 'date'
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = next((cols_lower[c.lower()] for c in DATE_CANDIDATES if c.lower() in cols_lower), None)

    if date_col is not None:
        df['date'] = pd.to_datetime(df[date_col])
    else:
        # Try year, month, day pattern
        # Look for columns representing year/month/day
        def find_col(names: List[str]) -> Optional[str]:
            for n in names:
                for col in df.columns:
                    if col.lower() == n:
                        return col
            return None
        year_col = find_col(['year', 'yy', 'yyyy'])
        month_col = find_col(['month', 'mm'])
        day_col = find_col(['day', 'dd'])
        if all([year_col, month_col, day_col]):
            tmp = df[[year_col, month_col, day_col]].astype(int).copy()
            tmp.columns = ['year', 'month', 'day']
            df['date'] = pd.to_datetime(tmp)
        else:
            # Give up on auto date; try index as day count (very rare); raise helpful error
            raise ValueError(
                f"Could not infer date column in {file_path}. Provide config.yaml with columns mapping."
            )

    # Ensure 'date' is first-class and sorted
    if 'date' not in df.columns:
        raise ValueError(f"No date parsed from {file_path}")
    return df


def _load_catchment_from_folder_generic(
    catchment_dir: Path,
    config: Optional[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt to load meteorology (precip,temp,pet) and discharge from a folder.

    Priority:
      1) Use explicit paths and column maps from config.yaml if present
      2) Try to detect combined meteorology file containing precip/temp/pet
      3) Try to detect separate files for precip, temp, pet and merge on date
      4) Detect discharge file by common keywords
    """
    cfg = config or {}

    # 1) Explicit config mapping
    if 'meteorology' in cfg and 'discharge' in cfg:
        met_cfg = cfg['meteorology']
        q_cfg = cfg['discharge']
        met_file = (catchment_dir / met_cfg.get('file')).resolve()
        q_file = (catchment_dir / q_cfg.get('file')).resolve()
        met_df = _read_table_with_date(
            met_file,
            specified_sep=met_cfg.get('sep'),
            decimal=met_cfg.get('decimal'),
            columns_map=met_cfg.get('columns'),
        )
        q_df = _read_table_with_date(
            q_file,
            specified_sep=q_cfg.get('sep'),
            decimal=q_cfg.get('decimal'),
            columns_map=q_cfg.get('columns'),
        )
        # Standardize names if provided
        met_df = _standardize_meteorology_columns(met_df)
        q_df = _standardize_discharge_columns(q_df)
        return met_df, q_df

    # 2/3) Heuristic detection
    files = list(catchment_dir.glob('**/*'))
    cand = [f for f in files if f.suffix.lower() in ('.csv', '.txt', '.dat', '.asc', '.tsv')]

    def has_keywords(path: Path, keywords: List[str]) -> bool:
        name = path.name.lower()
        return any(k in name for k in keywords)

    # Discharge candidates
    q_keywords = ['discharge', 'runoff', 'flow', 'qobs', 'q_', 'q-']
    q_files = [f for f in cand if has_keywords(f, q_keywords)]
    q_df = None
    for f in q_files:
        try:
            df = _read_table_with_date(f)
            df = _standardize_discharge_columns(df)
            q_df = df
            break
        except Exception:
            continue
    if q_df is None:
        raise FileNotFoundError(
            f"Could not find discharge file in {catchment_dir}. Provide config.yaml with 'discharge' mapping."
        )

    # Meteorology: combined file first
    met_keywords = ['meteo', 'meteor', 'met', 'climate', 'weather']
    met_files = [f for f in cand if has_keywords(f, met_keywords)]
    met_df = None
    for f in met_files:
        try:
            df = _read_table_with_date(f)
            df = _standardize_meteorology_columns(df)
            if {'precip', 'temp', 'pet'}.issubset(df.columns):
                met_df = df
                break
        except Exception:
            continue

    # If not combined, try separate series and merge
    if met_df is None:
        precip_files = [f for f in cand if has_keywords(f, ['precip', 'ppt', 'pr'])]
        temp_files = [f for f in cand if has_keywords(f, ['temp', 'tmean', 't'])]
        pet_files = [f for f in cand if has_keywords(f, ['pet', 'evap', 'et0', 'eto', 'evapo'])]

        parts = {}
        for label, flist in [('precip', precip_files), ('temp', temp_files), ('pet', pet_files)]:
            for f in flist:
                try:
                    df = _read_table_with_date(f)
                    # Pick first numeric column (besides date)
                    value_cols = [c for c in df.columns if c != 'date']
                    if not value_cols:
                        continue
                    parts[label] = df[['date', value_cols[0]]].rename(columns={value_cols[0]: label})
                    break
                except Exception:
                    continue
        if set(parts.keys()) == {'precip', 'temp', 'pet'}:
            met_df = parts['precip']
            met_df = met_df.merge(parts['temp'], on='date', how='inner')
            met_df = met_df.merge(parts['pet'], on='date', how='inner')
        else:
            raise FileNotFoundError(
                f"Could not detect meteorology files in {catchment_dir}. Provide config.yaml with 'meteorology' mapping."
            )

    # Final standardization and sorting
    met_df = _standardize_meteorology_columns(met_df)
    q_df = _standardize_discharge_columns(q_df)
    return met_df.sort_values('date'), q_df.sort_values('date')


def _standardize_meteorology_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    # Precipitation
    for k in ['precip', 'ppt', 'pr', 'p']:
        if k in cols:
            rename_map[cols[k]] = 'precip'
            break
    # Temperature
    for k in ['temp', 'tmean', 't']:
        if k in cols:
            rename_map[cols[k]] = 'temp'
            break
    # PET
    for k in ['pet', 'et0', 'eto', 'evap', 'evapo']:
        if k in cols:
            rename_map[cols[k]] = 'pet'
            break
    # Date
    for k in DATE_CANDIDATES:
        if k.lower() in cols:
            rename_map[cols[k.lower()]] = 'date'
            break
    df2 = df.rename(columns=rename_map)
    needed = {'date', 'precip', 'temp', 'pet'}
    if not needed.issubset(df2.columns):
        missing = needed - set(df2.columns)
        raise ValueError(f"Meteorology file missing columns: {missing}")
    return df2[['date', 'precip', 'temp', 'pet']]


def _standardize_discharge_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    # Discharge
    for k in ['discharge', 'qobs', 'runoff', 'flow', 'q']:
        if k in cols:
            rename_map[cols[k]] = 'discharge'
            break
    # Date
    for k in DATE_CANDIDATES:
        if k.lower() in cols:
            rename_map[cols[k.lower()]] = 'date'
            break
    df2 = df.rename(columns=rename_map)
    needed = {'date', 'discharge'}
    if not needed.issubset(df2.columns):
        missing = needed - set(df2.columns)
        raise ValueError(f"Discharge file missing columns: {missing}")
    return df2[['date', 'discharge']]


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