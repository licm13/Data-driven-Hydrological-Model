"""
测试更新后的数据加载器
"""
import sys
import os
sys.path.append('src')

from utils.data_loader import load_catchment_from_csv

def test_data_loading():
    """测试数据加载功能"""
    print("="*60)
    print("测试数据加载功能")
    print("="*60)
    
    # 使用自动检测的数据目录
    from run_all_experiments import auto_detect_data_dir
    
    try:
        data_dir = auto_detect_data_dir()
        print(f"数据目录: {data_dir}")
        
        # 测试加载流域数据
        catchments = ['Iller', 'Saale', 'Selke']
        
        for catchment in catchments:
            print(f"\n{'='*40}")
            print(f"测试加载 {catchment} 流域数据")
            print(f"{'='*40}")
            
            try:
                data = load_catchment_from_csv(data_dir, catchment)
                print(f"✓ 成功加载 {catchment} 数据")
                print(f"  - 数据天数: {len(data)}")
                print(f"  - 时间范围: {data.dates[0].date()} 到 {data.dates[-1].date()}")
                print(f"  - 降水范围: {data.precip.min():.2f} - {data.precip.max():.2f} mm/day")
                print(f"  - 温度范围: {data.temp.min():.2f} - {data.temp.max():.2f} °C")
                print(f"  - 蒸发范围: {data.pet.min():.2f} - {data.pet.max():.2f} mm/day")
                print(f"  - 径流范围: {data.discharge.min():.2f} - {data.discharge.max():.2f} mm/day")
                
                # 检查数据质量
                nan_counts = {
                    'precip': sum(pd.isna(data.precip)),
                    'temp': sum(pd.isna(data.temp)),
                    'pet': sum(pd.isna(data.pet)),
                    'discharge': sum(pd.isna(data.discharge))
                }
                
                if any(nan_counts.values()):
                    print(f"  - 缺失值: {nan_counts}")
                else:
                    print(f"  - 无缺失值")
                    
            except Exception as e:
                print(f"✗ 加载 {catchment} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("数据加载测试完成")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import pandas as pd
    test_data_loading()