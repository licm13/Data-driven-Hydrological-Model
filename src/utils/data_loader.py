"""
数据加载和预处理工具
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re
import yaml
from .ml_utils import calculate_array_statistics

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
            'precip': calculate_array_statistics(self.precip),
            'temp': calculate_array_statistics(self.temp),
            'pet': calculate_array_statistics(self.pet),
            'discharge': calculate_array_statistics(self.discharge, include_percentiles=True),
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
    
    支持的文件结构：
    1) data_dir/catchment_name/[meteorology.csv, discharge.csv]
    2) data_dir/catchment_name/[单独的 .csv/.asc 文件]
    3) data_dir/[catchment_specific_files.csv/.asc]
    4) IMPRO数据集特殊格式
    
    Parameters:
    -----------
    data_dir : str, 数据目录
    catchment_name : str, 流域名称（支持大小写变体）
    config : dict, 配置（如果不提供，从config.yaml读取）
    
    Returns:
    --------
    data : CatchmentData
    """
    # 检查是否为IMPRO数据集
    data_path = Path(data_dir)
    if 'IMPRO' in str(data_path).upper() or any((data_path / variant).exists() for variant in [catchment_name.lower(), catchment_name.upper()]):
        try:
            print(f"Detected IMPRO dataset format for {catchment_name}")
            from .impro_loader import load_impro_catchment, convert_impro_to_catchment_data
            impro_data = load_impro_catchment(data_dir, catchment_name)
            return convert_impro_to_catchment_data(impro_data, catchment_name)
        except Exception as e:
            print(f"IMPRO loader failed: {e}")
            print("Falling back to generic loader...")
    
    # 原有的通用加载逻辑
    # 尝试匹配流域目录（支持大小写变体）
    catchment_dir = None
    catchment_name_variants = [
        catchment_name,
        catchment_name.lower(),
        catchment_name.upper(),
        catchment_name.capitalize(),
        catchment_name.title()
    ]
    
    for variant in catchment_name_variants:
        candidate_dir = data_path / variant
        if candidate_dir.exists() and candidate_dir.is_dir():
            catchment_dir = candidate_dir
            print(f"Found catchment directory: {catchment_dir}")
            break
    
    # 如果没有找到流域子目录，使用数据根目录
    if catchment_dir is None:
        catchment_dir = data_path
        print(f"Using data root directory: {catchment_dir}")
    
    # 加载配置
    if config is None:
        config_file = catchment_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    
    # 尝试标准CSV结构
    meteo_file = catchment_dir / 'meteorology.csv'
    discharge_file = catchment_dir / 'discharge.csv'

    if meteo_file.exists() and discharge_file.exists():
        meteo_df = pd.read_csv(meteo_file, parse_dates=['date'])
        discharge_df = pd.read_csv(discharge_file, parse_dates=['date'])
    else:
        # 回退：尝试从通用文件结构（ASCII/空白分隔/TSV）自动解析
        meteo_df, discharge_df = _load_catchment_from_folder_generic(
            catchment_dir, config, catchment_name
        )
    
    # 合并
    df = pd.merge(meteo_df, discharge_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 验证数据质量
    if len(df) == 0:
        raise ValueError(f"No overlapping dates found between meteorology and discharge data for {catchment_name}")
    
    # 检查必需列
    required_cols = ['precip', 'temp', 'pet', 'discharge']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data for {catchment_name}: {missing_cols}")
    
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
    
    print(f"Successfully loaded {len(data)} days of data for {catchment_name}")
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
    - Handles ASCII raster format and other specialized formats
    """
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    # Special handling for .asc files which might be ASCII raster format
    if file_path.suffix.lower() == '.asc':
        try:
            return _read_asc_time_series(file_path)
        except Exception as e:
            print(f"Failed to read as ASCII time series, trying generic: {e}")
            # Fall through to generic reading

    separators = [specified_sep] if specified_sep else [',', ';', '\t', None]
    errors = []
    dataframe: Optional[pd.DataFrame] = None
    
    for sep in separators:
        try:
            read_kwargs = {
                'engine': 'python',
                'comment': '#',
                'skipinitialspace': True,
            }
            
            if sep is None:
                # Let pandas infer, try delim_whitespace
                read_kwargs['delim_whitespace'] = True
            else:
                read_kwargs['sep'] = sep
                
            if decimal is not None:
                read_kwargs['decimal'] = decimal
                
            dataframe = pd.read_csv(file_path, **read_kwargs)
            
            # Check if we got reasonable data
            if len(dataframe) > 0 and len(dataframe.columns) > 1:
                break
            else:
                dataframe = None
                
        except Exception as e:
            errors.append((sep, str(e)))
            dataframe = None
            
    if dataframe is None:
        raise ValueError(f"Failed to read {file_path} with tried seps: {errors}")

    # Clean up column names (remove whitespace, handle encoding issues)
    dataframe.columns = [str(col).strip() for col in dataframe.columns]

    # Normalize columns
    if columns_map:
        dataframe = dataframe.rename(columns=columns_map)

    # Try to construct 'date'
    cols_lower = {col.lower(): col for col in dataframe.columns}
    date_col = next((cols_lower[c.lower()] for c in DATE_CANDIDATES if c.lower() in cols_lower), None)

    if date_col is not None:
        try:
            dataframe['date'] = pd.to_datetime(dataframe[date_col])
        except Exception as e:
            print(f"Failed to parse date column {date_col}: {e}")
            # Try alternative date parsing
            dataframe['date'] = pd.to_datetime(dataframe[date_col], errors='coerce')
            if dataframe['date'].isna().all():
                raise ValueError(f"Could not parse any dates from column {date_col}")
    else:
        # Try year, month, day pattern
        def find_col(names: List[str]) -> Optional[str]:
            for name in names:
                for col in dataframe.columns:
                    if col.lower() == name:
                        return col
            return None
            
        year_col = find_col(['year', 'yy', 'yyyy'])
        month_col = find_col(['month', 'mm'])
        day_col = find_col(['day', 'dd'])
        
        if all([year_col, month_col, day_col]):
            try:
                temp_df = dataframe[[year_col, month_col, day_col]].astype(int).copy()
                temp_df.columns = ['year', 'month', 'day']
                dataframe['date'] = pd.to_datetime(temp_df)
            except Exception as e:
                raise ValueError(f"Could not construct date from year/month/day columns: {e}")
        else:
            # Try to infer from index or create artificial dates
            if len(dataframe) > 0:
                print(f"Warning: No date column found in {file_path}, creating artificial dates")
                dataframe['date'] = pd.date_range('2000-01-01', periods=len(dataframe), freq='D')
            else:
                raise ValueError(
                    f"Could not infer date column in {file_path}. Provide config.yaml with columns mapping."
                )

    # Ensure 'date' is first-class and sorted
    if 'date' not in dataframe.columns:
        raise ValueError(f"No date parsed from {file_path}")
    
    # Remove rows with invalid dates
    dataframe = dataframe.dropna(subset=['date'])
    
    return dataframe


def _read_asc_time_series(file_path: Path) -> pd.DataFrame:
    """
    Read ASCII time series format commonly used in hydrology.
    
    Expected formats:
    1) Simple format: date value
    2) Multi-column format: year month day value
    3) Header with metadata followed by data
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Remove empty lines and comments
    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            data_lines.append(line)
    
    if not data_lines:
        raise ValueError(f"No data lines found in {file_path}")
    
    # Try to detect format by examining first few lines
    first_line_parts = data_lines[0].split()
    
    if len(first_line_parts) >= 2:
        # Check if first column looks like a date
        try:
            pd.to_datetime(first_line_parts[0])
            # Simple date value format
            data = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        date = pd.to_datetime(parts[0])
                        value = float(parts[1])
                        data.append({'date': date, 'value': value})
                    except:
                        continue
            return pd.DataFrame(data)
        except:
            pass
    
    # Try year month day value format
    if len(first_line_parts) >= 4:
        try:
            year = int(first_line_parts[0])
            month = int(first_line_parts[1])
            day = int(first_line_parts[2])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                data = []
                for line in data_lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            year = int(parts[0])
                            month = int(parts[1])
                            day = int(parts[2])
                            value = float(parts[3])
                            date = pd.Timestamp(year, month, day)
                            data.append({'date': date, 'value': value})
                        except:
                            continue
                return pd.DataFrame(data)
        except:
            pass
    
    # Try generic whitespace-separated format
    try:
        # Convert to a string that pandas can read
        text_data = '\n'.join(data_lines)
        from io import StringIO
        df = pd.read_csv(StringIO(text_data), sep=r'\s+', header=None, engine='python')
        
        # Auto-assign column names
        if len(df.columns) == 2:
            df.columns = ['date', 'value']
        elif len(df.columns) >= 4:
            df.columns = ['year', 'month', 'day', 'value'] + [f'col_{i}' for i in range(4, len(df.columns))]
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
        
        return df
    except Exception as e:
        raise ValueError(f"Could not parse ASCII file {file_path}: {e}")


def _load_catchment_from_folder_generic(
    catchment_dir: Path,
    config: Optional[Dict],
    catchment_name: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt to load meteorology (precip,temp,pet) and discharge from a folder.

    Priority:
      1) Use explicit paths and column maps from config.yaml if present
      2) Try to detect files with catchment-specific naming (e.g., discharge_iller.csv)
      3) Try to detect combined meteorology file containing precip/temp/pet
      4) Try to detect separate files for precip, temp, pet and merge on date
      5) Detect discharge file by common keywords
    """
    cfg = config or {}

    # 1) Explicit config mapping
    if 'meteorology' in cfg and 'discharge' in cfg:
        met_cfg = cfg['meteorology']
        discharge_cfg = cfg['discharge']
        met_file = (catchment_dir / met_cfg.get('file')).resolve()
        discharge_file = (catchment_dir / discharge_cfg.get('file')).resolve()
        met_dataframe = _read_table_with_date(
            met_file,
            specified_sep=met_cfg.get('sep'),
            decimal=met_cfg.get('decimal'),
            columns_map=met_cfg.get('columns'),
        )
        discharge_dataframe = _read_table_with_date(
            discharge_file,
            specified_sep=discharge_cfg.get('sep'),
            decimal=discharge_cfg.get('decimal'),
            columns_map=discharge_cfg.get('columns'),
        )
        # Standardize names if provided
        met_dataframe = _standardize_meteorology_columns(met_dataframe)
        discharge_dataframe = _standardize_discharge_columns(discharge_dataframe)
        return met_dataframe, discharge_dataframe

    # 2) Catchment-specific file detection
    files = list(catchment_dir.glob('**/*'))
    candidate_files = [file for file in files if file.suffix.lower() in ('.csv', '.txt', '.dat', '.asc', '.tsv')]
    
    # Create catchment name variants for file matching
    catchment_variants = []
    if catchment_name:
        catchment_variants = [
            catchment_name.lower(),
            catchment_name.upper(),
            catchment_name.capitalize(),
            catchment_name.title()
        ]

    def has_keywords(path: Path, keywords: List[str]) -> bool:
        name = path.name.lower()
        return any(keyword in name for keyword in keywords)
    
    def has_catchment_specific(path: Path, variable_keywords: List[str]) -> bool:
        """Check if file matches catchment-specific pattern like 'discharge_iller.csv'"""
        name = path.name.lower()
        for var_keyword in variable_keywords:
            for catchment_var in catchment_variants:
                pattern = f"{var_keyword}_{catchment_var}"
                if pattern in name:
                    return True
        return False

    # Discharge detection (prioritize catchment-specific files)
    discharge_keywords = ['discharge', 'runoff', 'flow', 'qobs', 'q_', 'q-']
    discharge_files = []
    
    # First, try catchment-specific files
    if catchment_variants:
        discharge_files.extend([file for file in candidate_files if has_catchment_specific(file, discharge_keywords)])
    
    # Then, try general keyword matching
    if not discharge_files:
        discharge_files = [file for file in candidate_files if has_keywords(file, discharge_keywords)]
    
    discharge_dataframe = None
    for file_path in discharge_files:
        try:
            print(f"Trying to load discharge from: {file_path}")
            dataframe = _read_table_with_date(file_path)
            dataframe = _standardize_discharge_columns(dataframe)
            discharge_dataframe = dataframe
            print(f"Successfully loaded discharge data with {len(dataframe)} records")
            break
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue
    
    if discharge_dataframe is None:
        available_files = [file.name for file in candidate_files]
        raise FileNotFoundError(
            f"Could not find discharge file in {catchment_dir}. "
            f"Available files: {available_files}. "
            f"Provide config.yaml with 'discharge' mapping."
        )

    # Meteorology detection (prioritize catchment-specific files)
    met_keywords = ['meteo', 'meteor', 'met', 'climate', 'weather']
    met_files = []
    
    # First, try catchment-specific combined meteorology files
    if catchment_variants:
        met_files.extend([file for file in candidate_files if has_catchment_specific(file, met_keywords)])
    
    # Then, try general keyword matching
    if not met_files:
        met_files = [file for file in candidate_files if has_keywords(file, met_keywords)]
    
    met_dataframe = None
    for file_path in met_files:
        try:
            print(f"Trying to load meteorology from: {file_path}")
            dataframe = _read_table_with_date(file_path)
            dataframe = _standardize_meteorology_columns(dataframe)
            if {'precip', 'temp', 'pet'}.issubset(dataframe.columns):
                met_dataframe = dataframe
                print(f"Successfully loaded combined meteorology data with {len(dataframe)} records")
                break
        except Exception as e:
            print(f"Failed to load combined meteorology from {file_path}: {e}")
            continue

    # If not combined, try separate series and merge
    if met_dataframe is None:
        print("Trying to load separate meteorology files...")
        # Prioritize catchment-specific files
        precip_files = []
        temp_files = []
        pet_files = []
        
        if catchment_variants:
            precip_files.extend([file for file in candidate_files if has_catchment_specific(file, ['precip', 'precipitation', 'ppt', 'pr'])])
            temp_files.extend([file for file in candidate_files if has_catchment_specific(file, ['temp', 'temperature', 'tmean', 't'])])
            pet_files.extend([file for file in candidate_files if has_catchment_specific(file, ['pet', 'evap', 'et0', 'eto', 'evapo'])])
        
        # Fallback to general keyword matching
        if not precip_files:
            precip_files = [file for file in candidate_files if has_keywords(file, ['precip', 'ppt', 'pr'])]
        if not temp_files:
            temp_files = [file for file in candidate_files if has_keywords(file, ['temp', 'tmean', 't'])]
        if not pet_files:
            pet_files = [file for file in candidate_files if has_keywords(file, ['pet', 'evap', 'et0', 'eto', 'evapo'])]

        parts = {}
        for label, file_list in [('precip', precip_files), ('temp', temp_files), ('pet', pet_files)]:
            for file_path in file_list:
                try:
                    print(f"Trying to load {label} from: {file_path}")
                    dataframe = _read_table_with_date(file_path)
                    # Pick first numeric column (besides date)
                    value_cols = [col for col in dataframe.columns if col != 'date' and pd.api.types.is_numeric_dtype(dataframe[col])]
                    if not value_cols:
                        print(f"No numeric columns found in {file_path}")
                        continue
                    parts[label] = dataframe[['date', value_cols[0]]].rename(columns={value_cols[0]: label})
                    print(f"Successfully loaded {label} data with {len(parts[label])} records")
                    break
                except Exception as e:
                    print(f"Failed to load {label} from {file_path}: {e}")
                    continue
        
        if set(parts.keys()) == {'precip', 'temp', 'pet'}:
            met_dataframe = parts['precip']
            met_dataframe = met_dataframe.merge(parts['temp'], on='date', how='inner')
            met_dataframe = met_dataframe.merge(parts['pet'], on='date', how='inner')
            print(f"Successfully merged separate meteorology files, {len(met_dataframe)} records")
        else:
            missing = {'precip', 'temp', 'pet'} - set(parts.keys())
            available_files = [file.name for file in candidate_files]
            raise FileNotFoundError(
                f"Could not detect meteorology files in {catchment_dir}. "
                f"Missing: {missing}. Available files: {available_files}. "
                f"Provide config.yaml with 'meteorology' mapping."
            )

    # Final standardization and sorting
    met_dataframe = _standardize_meteorology_columns(met_dataframe)
    discharge_dataframe = _standardize_discharge_columns(discharge_dataframe)
    return met_dataframe.sort_values('date'), discharge_dataframe.sort_values('date')


def _standardize_meteorology_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = {col.lower(): col for col in dataframe.columns}
    # Precipitation
    for key in ['precip', 'ppt', 'pr', 'p']:
        if key in cols:
            rename_map[cols[key]] = 'precip'
            break
    # Temperature
    for key in ['temp', 'tmean', 't']:
        if key in cols:
            rename_map[cols[key]] = 'temp'
            break
    # PET
    for key in ['pet', 'et0', 'eto', 'evap', 'evapo']:
        if key in cols:
            rename_map[cols[key]] = 'pet'
            break
    # Date
    for key in DATE_CANDIDATES:
        if key.lower() in cols:
            rename_map[cols[key.lower()]] = 'date'
            break
    standardized_df = dataframe.rename(columns=rename_map)
    needed = {'date', 'precip', 'temp', 'pet'}
    if not needed.issubset(standardized_df.columns):
        missing = needed - set(standardized_df.columns)
        raise ValueError(f"Meteorology file missing columns: {missing}")
    return standardized_df[['date', 'precip', 'temp', 'pet']]


def _standardize_discharge_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = {col.lower(): col for col in dataframe.columns}
    # Discharge
    for key in ['discharge', 'qobs', 'runoff', 'flow', 'q']:
        if key in cols:
            rename_map[cols[key]] = 'discharge'
            break
    # Date
    for key in DATE_CANDIDATES:
        if key.lower() in cols:
            rename_map[cols[key.lower()]] = 'date'
            break
    standardized_df = dataframe.rename(columns=rename_map)
    needed = {'date', 'discharge'}
    if not needed.issubset(standardized_df.columns):
        missing = needed - set(standardized_df.columns)
        raise ValueError(f"Discharge file missing columns: {missing}")
    return standardized_df[['date', 'discharge']]


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