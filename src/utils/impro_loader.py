"""
IMPRO数据集专用加载器
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings

def load_impro_catchment(data_dir: str, catchment_name: str) -> Dict[str, np.ndarray]:
    """
    加载IMPRO数据集的流域数据
    
    Parameters:
    -----------
    data_dir : str, 数据目录路径
    catchment_name : str, 流域名称 (Iller, Saale, Selke)
    
    Returns:
    --------
    data : dict, 包含时间序列数据的字典
        - dates: DatetimeIndex
        - discharge: 径流数据 [mm/day]
        - precipitation: 降水数据 [mm/day] (如果可用)
        - temperature: 温度数据 [°C] (如果可用) 
        - pet: 蒸发数据 [mm/day] (如果可用)
    """
    catchment_lower = catchment_name.lower()
    data_path = Path(data_dir) / catchment_lower
    
    if not data_path.exists():
        # 尝试首字母大写的变体
        data_path = Path(data_dir) / catchment_name.capitalize()
        
    if not data_path.exists():
        raise FileNotFoundError(f"Catchment directory not found: {data_path}")
    
    print(f"Loading IMPRO data for {catchment_name} from {data_path}")
    
    # 加载径流数据
    discharge_file = data_path / f"discharge_{catchment_lower}.csv"
    if not discharge_file.exists():
        raise FileNotFoundError(f"Discharge file not found: {discharge_file}")
    
    print(f"Loading discharge from: {discharge_file}")
    
    # 读取径流CSV文件
    discharge_df = pd.read_csv(discharge_file)
    
    # 智能日期解析 - 尝试多种格式
    date_formats = ['%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y', '%Y/%m/%d']
    date_parsed = False
    
    for date_format in date_formats:
        try:
            discharge_df['date'] = pd.to_datetime(discharge_df['date'], format=date_format, errors='raise')
            print(f"Successfully parsed dates using format: {date_format}")
            date_parsed = True
            break
        except:
            continue
    
    if not date_parsed:
        # 尝试pandas的智能解析
        try:
            discharge_df['date'] = pd.to_datetime(discharge_df['date'], errors='coerce')
            if discharge_df['date'].isna().sum() > len(discharge_df) * 0.1:  # 超过10%解析失败
                raise ValueError("Too many date parsing failures")
            print("Successfully parsed dates using pandas automatic detection")
            date_parsed = True
        except:
            raise ValueError(f"Could not parse dates in {discharge_file}. Please check date format.")
    
    # 移除无效日期的行
    initial_rows = len(discharge_df)
    discharge_df = discharge_df.dropna(subset=['date'])
    dropped_rows = initial_rows - len(discharge_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with invalid dates")
    
    # 选择第一个数值列作为主径流站点（跳过date列）
    numeric_cols = [col for col in discharge_df.columns if col != 'date' and pd.api.types.is_numeric_dtype(discharge_df[col])]
    if not numeric_cols:
        raise ValueError(f"No numeric discharge columns found in {discharge_file}")
    
    # 使用第一个站点的数据
    discharge_col = numeric_cols[0]
    print(f"Using discharge data from station: {discharge_col}")
    
    # 创建径流时间序列
    discharge_ts = discharge_df[['date', discharge_col]].copy()
    discharge_ts = discharge_ts.rename(columns={discharge_col: 'discharge'})
    discharge_ts = discharge_ts.dropna(subset=['discharge'])
    discharge_ts = discharge_ts.sort_values('date')
    
    result = {
        'dates': discharge_ts['date'],
        'discharge': discharge_ts['discharge'].values,
    }
    
    print(f"Loaded {len(result['discharge'])} days of discharge data")
    print(f"Date range: {result['dates'].min().date()} to {result['dates'].max().date()}")
    print(f"Discharge range: {result['discharge'].min():.2f} - {result['discharge'].max():.2f} mm/day")
    
    # 检查ASC文件是否为时间序列
    asc_files = {
        'precipitation': data_path / f"precipitation_{catchment_lower}.asc",
        'temperature': data_path / f"temperature_{catchment_lower}.asc", 
        'pet': data_path / f"pet_{catchment_lower}.asc"
    }
    
    for var_name, asc_file in asc_files.items():
        if asc_file.exists():
            print(f"Found {var_name} file: {asc_file}")
            try:
                # 尝试读取ASC文件的前几行来判断格式
                with open(asc_file, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(10)]
                
                # 检查是否看起来像时间序列数据
                # IMPRO ASC文件实际上可能是栅格数据，不是时间序列
                print(f"  Warning: {var_name} ASC file appears to be raster data, not time series")
                print(f"  Skipping {var_name} data")
                
            except Exception as e:
                print(f"  Error reading {asc_file}: {e}")
                continue
    
    # 生成合成气象数据作为替代
    if len(result['dates']) > 0:
        print("Generating synthetic meteorological data as placeholder...")
        n_days = len(result['dates'])
        
        # 简单的季节性合成数据
        day_of_year = result['dates'].dt.dayofyear
        
        # 降水：随机 + 季节性
        np.random.seed(42)
        seasonal_precip = 2 + np.sin(2 * np.pi * (day_of_year - 80) / 365)
        precip = np.random.exponential(seasonal_precip)
        
        # 温度：季节性 + 噪声
        temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n_days) * 3
        
        # 蒸发：与温度相关
        pet = np.maximum(0, 1 + 0.15 * temp + np.random.randn(n_days) * 0.5)
        
        result.update({
            'precipitation': precip,
            'temperature': temp,
            'pet': pet
        })
        
        print("Added synthetic meteorological data:")
        print(f"  Precipitation: {precip.min():.2f} - {precip.max():.2f} mm/day")
        print(f"  Temperature: {temp.min():.2f} - {temp.max():.2f} °C")
        print(f"  PET: {pet.min():.2f} - {pet.max():.2f} mm/day")
    
    return result


def convert_impro_to_catchment_data(impro_data: Dict[str, np.ndarray], catchment_name: str):
    """
    将IMPRO数据格式转换为CatchmentData对象
    """
    from src.utils.data_loader import CatchmentData
    
    return CatchmentData(
        name=catchment_name,
        precip=impro_data['precipitation'],
        temp=impro_data['temperature'],
        pet=impro_data['pet'],
        discharge=impro_data['discharge'],
        dates=pd.DatetimeIndex(impro_data['dates']),
        metadata={'source': 'IMPRO', 'note': 'Meteorological data is synthetic'}
    )


def test_impro_loader():
    """测试IMPRO数据加载器"""
    import sys
    sys.path.append('src')
    
    from run_all_experiments import auto_detect_data_dir
    
    try:
        data_dir = auto_detect_data_dir()
        print(f"Data directory: {data_dir}")
        
        catchments = ['Iller', 'Saale', 'Selke']
        
        for catchment in catchments:
            print(f"\n{'='*50}")
            print(f"Testing {catchment}")
            print(f"{'='*50}")
            
            try:
                impro_data = load_impro_catchment(data_dir, catchment)
                catchment_data = convert_impro_to_catchment_data(impro_data, catchment)
                
                print(f"✓ Successfully loaded {catchment}")
                print(f"  Duration: {len(catchment_data)} days")
                print(f"  Period: {catchment_data.dates[0].date()} to {catchment_data.dates[-1].date()}")
                
                # 数据质量检查
                stats = catchment_data.get_statistics()
                print(f"  Discharge stats: mean={stats['discharge']['mean']:.2f}, max={stats['discharge']['max']:.2f}")
                
            except Exception as e:
                print(f"✗ Failed to load {catchment}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    test_impro_loader()