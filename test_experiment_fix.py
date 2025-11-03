"""
快速测试实验修复
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hbv import HBV
from src.utils.data_loader import generate_synthetic_data
from src.calibration.spotpy_wrapper import calibrate_model

def test_simple_hbv():
    """测试基本HBV模型运行"""
    print("测试基本HBV模型...")
    
    # 生成小量合成数据
    data = generate_synthetic_data(n_days=100)
    
    # 创建HBV模型
    model = HBV(n_elevation_zones=1)
    
    # 测试校准
    print("运行小规模校准...")
    try:
        results = calibrate_model(
            model,
            precip=data.precip[:50],
            temp=data.temp[:50],
            pet=data.pet[:50],
            discharge_obs=data.discharge[:50],
            algorithm='lhs',
            n_iterations=10,  # 很少的迭代
            warmup_period=10,
            objective_function='kge'
        )
        print(f"✓ 校准成功，最佳KGE: {results['best_objective']:.4f}")
        
        # 测试模拟
        print("测试模拟...")
        model.initialize(results['best_params'])
        sim = model.simulate(
            data.precip[50:70],
            data.temp[50:70], 
            data.pet[50:70],
            warmup_steps=5
        )
        print(f"✓ 模拟成功，长度: {len(sim)}")
        print(f"  期望长度: {20-5} (输入20天 - 预热5天)")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

if __name__ == '__main__':
    success = test_simple_hbv()
    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 测试失败！")