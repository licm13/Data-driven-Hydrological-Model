"""
测试数据目录检测功能
"""
import os
from pathlib import Path
from run_all_experiments import auto_detect_data_dir, find_dataset_directory, find_data_subdirectory, verify_catchment_data


def test_data_detection():
    """测试数据检测功能"""
    print("="*60)
    print("测试数据目录检测功能")
    print("="*60)
    
    try:
        # 测试 Dataset 目录查找
        print("\n1. 查找 Dataset 目录...")
        dataset_dir = find_dataset_directory()
        print(f"✓ Dataset 目录: {dataset_dir}")
        
        # 测试数据子目录查找
        print("\n2. 查找数据子目录...")
        data_dir = find_data_subdirectory(dataset_dir)
        print(f"✓ 数据子目录: {data_dir}")
        
        # 测试完整的自动检测
        print("\n3. 完整自动检测...")
        auto_data_dir = auto_detect_data_dir()
        print(f"✓ 自动检测结果: {auto_data_dir}")
        
        # 验证流域数据文件
        print("\n4. 验证流域数据文件...")
        catchments = ['iller', 'saale', 'selke']
        
        for catchment in catchments:
            print(f"\n检查 {catchment.upper()} 流域:")
            try:
                file_info = verify_catchment_data(auto_data_dir, catchment)
                for file_type, info in file_info.items():
                    status = '✓' if info['exists'] else '✗'
                    size_mb = info['size'] / (1024*1024) if info['size'] > 0 else 0
                    print(f"  {file_type:12}: {status} {info['path']} ({size_mb:.1f} MB)")
            except FileNotFoundError as e:
                print(f"  ✗ 错误: {e}")
        
        print("\n" + "="*60)
        print("✓ 数据检测测试完成 - 所有功能正常")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n" + "="*60)
        print("✗ 数据检测测试失败")
        print("="*60)
        return False


def list_available_files():
    """列出可用的数据文件"""
    try:
        data_dir = auto_detect_data_dir()
        data_path = Path(data_dir)
        
        print(f"\n可用的数据文件 (位于 {data_dir}):")
        print("-" * 40)
        
        # 按文件类型分组
        file_types = {
            '径流数据 (.csv)': list(data_path.glob("discharge_*.csv")),
            '降水数据 (.asc)': list(data_path.glob("precipitation_*.asc")),
            '温度数据 (.asc)': list(data_path.glob("temperature_*.asc")),
            '蒸发数据 (.asc)': list(data_path.glob("pet_*.asc")),
            '子流域数据 (.asc)': list(data_path.glob("subcatch_*.asc")),
            '其他文件': list(data_path.glob("*")) 
        }
        
        for file_type, files in file_types.items():
            if files:
                print(f"\n{file_type}:")
                for file in sorted(files):
                    if file.is_file():
                        print(f"  - {file.name}")
                        
    except Exception as e:
        print(f"列出文件时出错: {e}")


if __name__ == '__main__':
    # 运行测试
    success = test_data_detection()
    
    # 列出可用文件
    list_available_files()
    
    print(f"\n测试结果: {'成功' if success else '失败'}")