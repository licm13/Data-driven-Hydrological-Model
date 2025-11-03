import sys
sys.path.append('src')
from utils.impro_loader import load_impro_catchment

# 测试加载Iller流域数据
try:
    data = load_impro_catchment('../Dataset/IMPRO_catchment_data_infotheo', 'iller')
    print(f"Loaded {len(data['discharge'])} days of data")
    print(f"Date range: {data['dates'].min()} to {data['dates'].max()}")
    print(f"Discharge range: {data['discharge'].min():.2f} - {data['discharge'].max():.2f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()