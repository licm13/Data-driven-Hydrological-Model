"""
实验1：学习曲线
比较不同模型在不同训练数据量下的学习能力
"""
import numpy as np
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

from src.models import get_model, PROCESS_BASED_MODELS, DATA_DRIVEN_MODELS
from src.utils.data_loader import load_catchment_from_csv, generate_synthetic_data
from src.sampling.strategies import consecutive_random_sampling
from src.metrics.entropy import evaluate_model_entropy
from src.metrics.kge import kge
from src.calibration.spotpy_wrapper import calibrate_model
from src.utils.visualization import plot_learning_curves


def run_experiment_1(catchment_name: str,
                    data_dir: str,
                    output_dir: str,
                    sample_sizes: list = None,
                    n_replicates: int = 30,
                    models_to_test: list = None,
                    use_synthetic: bool = False):
    """
    运行实验1：学习曲线
    
    Parameters:
    -----------
    catchment_name : str, 流域名称
    data_dir : str, 数据目录
    output_dir : str, 输出目录
    sample_sizes : list, 训练样本量列表
    n_replicates : int, 每个样本量的重复次数
    models_to_test : list, 要测试的模型列表
    use_synthetic : bool, 是否使用合成数据
    """
    # 创建输出目录
    output_path = Path(output_dir) / 'experiment_1' / catchment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认样本量
    if sample_sizes is None:
        sample_sizes = [2, 10, 50, 100, 250, 500, 1000, 2000, 3000, 3654]
    
    # 默认测试所有模型
    if models_to_test is None:
        models_to_test = PROCESS_BASED_MODELS + DATA_DRIVEN_MODELS
    
    print(f"=" * 60)
    print(f"Experiment 1: Learning Curves - {catchment_name}")
    print(f"=" * 60)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Models: {models_to_test}")
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
    results = {}
    
    # 对每个模型
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}\n")
        
        model_results = {}
        
        # 对每个样本量
        for sample_size in sample_sizes:
            print(f"\nSample size: {sample_size}")
            
            # 生成采样索引
            n_train = len(train_data.discharge) - 365  # 去除预热期
            if sample_size > n_train:
                print(f"  Warning: Sample size {sample_size} > available data {n_train}")
                sample_size = n_train
            
            sampling_indices = consecutive_random_sampling(
                n_train, sample_size, n_replicates, seed=42
            )
            
            # 存储该样本量的结果
            size_results = {
                'H_conditional': [],
                'H_normalized': [],
                'kge': [],
            }
            
            # 对每个重复
            for rep_idx, indices in enumerate(tqdm(sampling_indices, desc=f"  Replicates")):
                # 调整索引（加上预热期）
                train_indices = indices + 365
                
                # 提取训练数据
                train_precip = train_data.precip[train_indices]
                train_temp = train_data.temp[train_indices]
                train_pet = train_data.pet[train_indices]
                train_discharge = train_data.discharge[train_indices]
                
                # 根据模型类型进行训练/校准
                try:
                    if model_name in PROCESS_BASED_MODELS:
                        # 过程驱动模型：使用SPOTPY校准
                        model = get_model(model_name)
                        
                        calib_results = calibrate_model(
                            model,
                            precip=train_data.precip[:len(train_indices)+365],
                            temp=train_data.temp[:len(train_indices)+365],
                            pet=train_data.pet[:len(train_indices)+365],
                            discharge_obs=train_data.discharge[:len(train_indices)+365],
                            algorithm='lhs',
                            n_iterations=min(500, sample_size * 2),
                            warmup_period=365,
                            objective_function='kge'
                        )
                        
                        # 在测试期模拟
                        discharge_sim = model.simulate(
                            test_data.precip,
                            test_data.temp,
                            test_data.pet,
                            warmup_steps=365
                        )
                    
                    else:
                        # 数据驱动模型
                        if model_name == 'EDDIS':
                            model = get_model('EDDIS', n_bins=2, use_temporal_aggregation=True)
                            model.train(train_precip, train_temp, train_pet, train_discharge)
                            discharge_sim = model.predict(test_data.precip, test_data.temp, test_data.pet)
                        
                        elif model_name == 'RTREE':
                            model = get_model('RTREE', use_temporal_features=True)
                            model.train(train_precip, train_temp, train_pet, train_discharge)
                            discharge_sim = model.predict(test_data.precip, test_data.temp, test_data.pet)
                        
                        elif model_name == 'ANN':
                            model = get_model('ANN', time_lag=7, n_epochs=30)
                            model.train(train_precip, train_temp, train_pet, train_discharge)
                            discharge_sim = model.predict(test_data.precip, test_data.temp, test_data.pet)
                        
                        elif model_name == 'LSTM':
                            model = get_model('LSTM', sequence_length=365, n_epochs=20, n_init=3)
                            model.train(train_precip, train_temp, train_pet, train_discharge)
                            discharge_sim = model.predict(test_data.precip, test_data.temp, test_data.pet)
                    
                    # 评估
                    discharge_obs = test_data.discharge[365:]
                    discharge_sim = discharge_sim[365:]
                    
                    # 信息熵
                    entropy_metrics = evaluate_model_entropy(discharge_obs, discharge_sim, n_bins=12)
                    size_results['H_conditional'].append(entropy_metrics['H_conditional'])
                    size_results['H_normalized'].append(entropy_metrics['H_normalized'])
                    
                    # KGE
                    kge_value = kge(discharge_obs, discharge_sim)
                    size_results['kge'].append(kge_value)
                
                except Exception as e:
                    print(f"    Error in replicate {rep_idx}: {e}")
                    continue
            
            # 保存该样本量的结果
            model_results[sample_size] = size_results
            
            # 打印统计
            if len(size_results['H_conditional']) > 0:
                print(f"  H_cond: {np.median(size_results['H_conditional']):.3f} "
                      f"({np.percentile(size_results['H_conditional'], 25):.3f}-"
                      f"{np.percentile(size_results['H_conditional'], 75):.3f})")
                print(f"  KGE: {np.median(size_results['kge']):.3f} "
                      f"({np.percentile(size_results['kge'], 25):.3f}-"
                      f"{np.percentile(size_results['kge'], 75):.3f})")
        
        # 保存模型结果
        results[model_name] = model_results
    
    # 保存完整结果
    results_file = output_path / 'learning_curves_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {results_file}")
    
    # 绘制学习曲线
    plot_learning_curves(
        results,
        catchment_name,
        metric='H_conditional',
        save_path=output_path / 'learning_curves.png'
    )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Experiment 1: Learning Curves')
    parser.add_argument('--catchment', type=str, default='Iller',
                       help='Catchment name')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Models to test (default: all)')
    parser.add_argument('--n_replicates', type=int, default=30,
                       help='Number of replicates')
    
    args = parser.parse_args()
    
    run_experiment_1(
        catchment_name=args.catchment,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_to_test=args.models,
        n_replicates=args.n_replicates,
        use_synthetic=args.synthetic
    )