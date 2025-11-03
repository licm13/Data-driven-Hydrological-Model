"""
实验2：采样策略对比
测试不同采样策略对HBV模型性能的影响
"""
import numpy as np
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

# Ensure project root is on sys.path so `src.*` imports work when running this file directly
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HBV
from src.utils.data_loader import load_catchment_from_csv, generate_synthetic_data
from src.sampling.strategies import (random_sampling, consecutive_random_sampling,
                                    douglas_peucker_sampling)
from src.metrics.entropy import evaluate_model_entropy
from src.calibration.spotpy_wrapper import calibrate_model
from src.utils.visualization import plot_sampling_strategy_comparison


def run_experiment_2(catchment_name: str,
                    data_dir: str,
                    output_dir: str,
                    sample_sizes: list = None,
                    n_replicates: int = 30,
                    use_synthetic: bool = False):
    """
    运行实验2：采样策略对比
    
    Parameters:
    -----------
    catchment_name : str, 流域名称
    data_dir : str, 数据目录
    output_dir : str, 输出目录
    sample_sizes : list, 样本量列表
    n_replicates : int, 重复次数
    use_synthetic : bool, 是否使用合成数据
    """
    # 创建输出目录
    output_path = Path(output_dir) / 'experiment_2' / catchment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认样本量
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2000, 3000, 3654]
    
    print(f"=" * 60)
    print(f"Experiment 2: Sampling Strategies - {catchment_name}")
    print(f"=" * 60)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Replicates: {n_replicates}")
    print()
    
    # 加载数据
    if use_synthetic:
        print("Generating synthetic data...")
        data = generate_synthetic_data(n_days=5000)
    else:
        print(f"Loading data from {data_dir}...")
        data = load_catchment_from_csv(data_dir, catchment_name)
    
    # 划分训练/测试
    train_data, test_data = data.split(
        train_period=('2001-01-01', '2010-12-31'),
        test_period=('2012-01-01', '2015-12-31'),
        warmup_days=365
    )
    
    print(f"Training period: {len(train_data)} days")
    print(f"Testing period: {len(test_data)} days")
    print()
    
    # 存储结果
    results = {
        'random': {},
        'consecutive': {},
        'douglas_peucker': {}
    }
    
    # 测试每种采样策略
    strategies = {
        'random': lambda n, size: random_sampling(n, size, n_replicates, seed=42),
        'consecutive': lambda n, size: consecutive_random_sampling(n, size, n_replicates, seed=42),
        'douglas_peucker': lambda n, size: [douglas_peucker_sampling(
            train_data.discharge[365:365+n], size
        )]
    }
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy_name}")
        print(f"{'='*60}\n")
        
        for sample_size in sample_sizes:
            print(f"\nSample size: {sample_size}")
            
            # 生成采样索引
            n_train = len(train_data.discharge) - 365
            if sample_size > n_train:
                sample_size = n_train
            
            sampling_indices = strategy_func(n_train, sample_size)
            
            # 存储结果
            H_conditionals = []
            kges = []
            
            # 对每个重复
            n_reps = len(sampling_indices)
            for rep_idx, indices in enumerate(tqdm(sampling_indices, desc=f"  Replicates")):
                # 调整索引（加上预热期偏移）
                train_indices = indices + 365
                max_idx = np.max(train_indices)
                
                try:
                    # 创建HBV模型
                    model = HBV(n_elevation_zones=1)
                    
                    # 确定需要的数据长度（到最大索引）
                    data_length = max_idx + 1
                    
                    # 校准（使用到最大索引的所有数据）
                    calib_results = calibrate_model(
                        model,
                        precip=train_data.precip[:data_length],
                        temp=train_data.temp[:data_length],
                        pet=train_data.pet[:data_length],
                        discharge_obs=train_data.discharge[:data_length],
                        algorithm='lhs',
                        n_iterations=min(500, sample_size * 2),
                        warmup_period=365,
                        objective_function='kge'
                    )
                    
                    # 测试期模拟
                    discharge_sim = model.simulate(
                        test_data.precip,
                        test_data.temp,
                        test_data.pet,
                        warmup_steps=365
                    )
                    
                    # 评估（simulate方法已经移除了预热期）
                    discharge_obs = test_data.discharge[365:]
                    # discharge_sim 已经是移除预热期后的结果
                    
                    # 熵
                    entropy_metrics = evaluate_model_entropy(discharge_obs, discharge_sim)
                    H_conditionals.append(entropy_metrics['H_conditional'])
                    
                    # KGE
                    from src.metrics.kge import kge
                    kge_value = kge(discharge_obs, discharge_sim)
                    kges.append(kge_value)
                
                except Exception as e:
                    print(f"    Error in replicate {rep_idx}: {e}")
                    continue
            
            # 保存结果
            results[strategy_name][sample_size] = {
                'H_conditional': H_conditionals,
                'kge': kges
            }
            
            # 打印统计
            if len(H_conditionals) > 0:
                print(f"  H_cond: {np.median(H_conditionals):.3f} "
                      f"({np.percentile(H_conditionals, 25):.3f}-"
                      f"{np.percentile(H_conditionals, 75):.3f})")
                print(f"  KGE: {np.median(kges):.3f} "
                      f"({np.percentile(kges, 25):.3f}-"
                      f"{np.percentile(kges, 75):.3f})")
    
    # 保存结果
    results_file = output_path / 'sampling_strategies_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {results_file}")
    
    # 绘制对比图
    # 提取H_conditional用于绘图
    plot_data = {}
    for strategy in results:
        plot_data[strategy] = {
            size: results[strategy][size]['H_conditional']
            for size in results[strategy]
        }
    
    plot_sampling_strategy_comparison(
        plot_data,
        catchment_name,
        save_path=output_path / 'sampling_strategies_comparison.png'
    )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Experiment 2: Sampling Strategies')
    parser.add_argument('--catchment', type=str, default='Iller',
                       help='Catchment name')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--n_replicates', type=int, default=30,
                       help='Number of replicates')
    
    args = parser.parse_args()
    
    run_experiment_2(
        catchment_name=args.catchment,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_replicates=args.n_replicates,
        use_synthetic=args.synthetic
    )