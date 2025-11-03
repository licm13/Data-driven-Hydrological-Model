"""
实验3：信息内容分析
分析数据的信息含量与模型学习能力的关系
"""
import numpy as np
import argparse
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import load_catchment_from_csv, load_multiple_catchments
from src.metrics.entropy import joint_entropy, conditional_entropy
from src.utils.visualization import plot_information_content


def analyze_catchment_information(precip: np.ndarray,
                                  temp: np.ndarray,
                                  pet: np.ndarray,
                                  discharge: np.ndarray,
                                  n_bins: int = 8) -> dict:
    """
    分析流域数据的信息含量
    
    Parameters:
    -----------
    precip, temp, pet, discharge : array, 气象和径流数据
    n_bins : int, 分箱数
    
    Returns:
    --------
    info_metrics : dict, 信息指标
    """
    n = len(discharge)
    
    # 1. 无记忆的联合熵和条件熵
    # 只使用当天的值
    X_no_memory = np.column_stack([precip, temp, pet])
    
    H_joint_no_memory = joint_entropy(
        np.column_stack([X_no_memory, discharge.reshape(-1, 1)]),
        n_bins=n_bins
    )
    
    H_cond_no_memory = conditional_entropy(
        discharge, 
        X_no_memory[:, 0],  # 简化：只用降水作为预测变量
        n_bins=n_bins
    )
    
    # 2. 考虑前一天的记忆
    # 创建lag-1特征
    precip_lag1 = np.concatenate([[0], precip[:-1]])
    temp_lag1 = np.concatenate([[temp[0]], temp[:-1]])
    pet_lag1 = np.concatenate([[pet[0]], pet[:-1]])
    
    X_day_memory = np.column_stack([
        precip, temp, pet,
        precip_lag1, temp_lag1, pet_lag1
    ])
    
    H_joint_day = joint_entropy(
        np.column_stack([X_day_memory, discharge.reshape(-1, 1)]),
        n_bins=n_bins
    )
    
    # 使用所有特征计算条件熵（简化）
    H_cond_day = conditional_entropy(
        discharge,
        precip,  # 主要预测变量
        n_bins=n_bins
    )
    
    # 3. 考虑前一周的记忆
    # 创建周平均特征
    precip_week = np.zeros(n)
    temp_week = np.zeros(n)
    pet_week = np.zeros(n)
    
    for i in range(7, n):
        precip_week[i] = np.mean(precip[i-7:i])
        temp_week[i] = np.mean(temp[i-7:i])
        pet_week[i] = np.mean(pet[i-7:i])
    
    X_week_memory = np.column_stack([
        precip, temp, pet,
        precip_lag1, temp_lag1, pet_lag1,
        precip_week, temp_week, pet_week
    ])
    
    # 只使用有效数据（跳过前7天）
    valid_idx = 7
    X_week_memory_valid = X_week_memory[valid_idx:]
    discharge_valid = discharge[valid_idx:]
    
    H_joint_week = joint_entropy(
        np.column_stack([X_week_memory_valid, discharge_valid.reshape(-1, 1)]),
        n_bins=n_bins
    )
    
    H_cond_week = conditional_entropy(
        discharge_valid,
        X_week_memory_valid[:, 0],  # 当天降水
        n_bins=n_bins
    )
    
    # 4. 仅径流的边际熵
    H_discharge = joint_entropy(discharge, n_bins=n_bins)
    
    return {
        'H_discharge': H_discharge,
        'H_joint_no_memory': H_joint_no_memory,
        'H_conditional_no_memory': H_cond_no_memory,
        'H_joint_day_memory': H_joint_day,
        'H_conditional_day_memory': H_cond_day,
        'H_joint_week_memory': H_joint_week,
        'H_conditional_week_memory': H_cond_week,
        'learnability_no_memory': H_discharge - H_cond_no_memory,
        'learnability_day_memory': H_discharge - H_cond_day,
        'learnability_week_memory': H_discharge - H_cond_week,
    }


def run_experiment_3(catchment_names: list,
                    data_dir: str,
                    output_dir: str,
                    learning_results_path: str = None):
    """
    运行实验3：信息内容分析
    
    Parameters:
    -----------
    catchment_names : list, 流域名称列表
    data_dir : str, 数据目录
    output_dir : str, 输出目录
    learning_results_path : str, 实验1结果文件路径（用于关联学习能力）
    """
    # 创建输出目录
    output_path = Path(output_dir) / 'experiment_3'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Experiment 3: Information Content Analysis")
    print(f"=" * 60)
    print(f"Catchments: {catchment_names}")
    print()
    
    # 加载多个流域数据
    catchments = load_multiple_catchments(data_dir, catchment_names)
    
    # 存储结果
    results = {}
    
    # 分析每个流域
    for name, data in catchments.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {name}")
        print(f"{'='*60}\n")
        
        # 使用训练期数据
        train_data, _ = data.split(
            train_period=('2001-01-01', '2010-12-31'),
            test_period=('2012-01-01', '2015-12-31'),
            warmup_days=0
        )
        
        # 分析信息内容
        info_metrics = analyze_catchment_information(
            train_data.precip,
            train_data.temp,
            train_data.pet,
            train_data.discharge,
            n_bins=8
        )
        
        results[name] = info_metrics
        
        # 打印结果
        print(f"H(Q): {info_metrics['H_discharge']:.3f} bits")
        print(f"\nNo memory:")
        print(f"  H(P,T,PET,Q): {info_metrics['H_joint_no_memory']:.3f} bits")
        print(f"  H(Q|P,T,PET): {info_metrics['H_conditional_no_memory']:.3f} bits")
        print(f"  Learnability: {info_metrics['learnability_no_memory']:.3f} bits")
        
        print(f"\nDay memory (t, t-1):")
        print(f"  H(P,T,PET,Q, t-1): {info_metrics['H_joint_day_memory']:.3f} bits")
        print(f"  H(Q|...): {info_metrics['H_conditional_day_memory']:.3f} bits")
        print(f"  Learnability: {info_metrics['learnability_day_memory']:.3f} bits")
        
        print(f"\nWeek memory (t, t-1, week):")
        print(f"  H(P,T,PET,Q, week): {info_metrics['H_joint_week_memory']:.3f} bits")
        print(f"  H(Q|...): {info_metrics['H_conditional_week_memory']:.3f} bits")
        print(f"  Learnability: {info_metrics['learnability_week_memory']:.3f} bits")
    
    # 保存结果
    results_file = output_path / 'information_content_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {results_file}")
    
    # 如果提供了学习结果，进行关联分析
    if learning_results_path and Path(learning_results_path).exists():
        print("\nAnalyzing relationship with model learning...")
        
        with open(learning_results_path, 'rb') as f:
            learning_results = pickle.load(f)
        
        # 计算每个流域的平均学习能力
        avg_learning = {}
        for catchment in catchment_names:
            if catchment in learning_results:
                # 计算所有模型的平均相对学习
                model_learning = []
                for model_name, model_results in learning_results[catchment].items():
                    # 提取最小和最大样本量的性能
                    sizes = sorted(model_results.keys())
                    if len(sizes) >= 2:
                        H_start = np.median(model_results[sizes[0]]['H_conditional'])
                        H_end = np.median(model_results[sizes[-1]]['H_conditional'])
                        learning = H_start - H_end
                        model_learning.append(learning)
                
                avg_learning[catchment] = np.mean(model_learning) if model_learning else 0
        
        # 提取信息指标
        joint_entropy_vals = {name: results[name]['H_joint_week_memory'] 
                             for name in catchment_names}
        conditional_entropy_vals = {name: results[name]['H_conditional_week_memory']
                                   for name in catchment_names}
        
        # 绘制关系图
        plot_information_content(
            joint_entropy_vals,
            conditional_entropy_vals,
            avg_learning,
            save_path=output_path / 'information_vs_learning.png'
        )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Experiment 3: Information Content')
    parser.add_argument('--catchments', nargs='+', 
                       default=['Iller', 'Saale', 'Selke'],
                       help='Catchment names')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--learning_results', type=str, default=None,
                       help='Path to Experiment 1 results (optional)')
    
    args = parser.parse_args()
    
    run_experiment_3(
        catchment_names=args.catchments,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        learning_results_path=args.learning_results
    )