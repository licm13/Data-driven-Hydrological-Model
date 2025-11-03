"""
运行所有实验的主脚本
"""
import argparse
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_1_learning_curves import run_experiment_1
from experiments.experiment_2_sampling_strategies import run_experiment_2
from experiments.experiment_3_information_content import run_experiment_3
from experiments.experiment_4_spatial_distribution import run_experiment_4


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments for hydrological model learning curves study'
    )
    
    parser.add_argument('--experiments', nargs='+', 
                       default=['1', '2', '3', '4'],
                       help='Experiments to run (1, 2, 3, 4)')
    parser.add_argument('--catchments', nargs='+',
                       default=['Iller', 'Saale', 'Selke'],
                       help='Catchment names')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--n_replicates', type=int, default=30,
                       help='Number of replicates per sample size')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced settings')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick_test:
        print("\n" + "="*60)
        print("QUICK TEST MODE")
        print("="*60)
        args.n_replicates = 3
        args.catchments = [args.catchments[0]]  # 只测试第一个流域
        sample_sizes_quick = [50, 500, 1000]
    
    print("\n" + "="*60)
    print("RUNNING ALL EXPERIMENTS")
    print("="*60)
    print(f"Experiments: {args.experiments}")
    print(f"Catchments: {args.catchments}")
    print(f"Replicates: {args.n_replicates}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # 实验1：学习曲线
    if '1' in args.experiments:
        print("\n" + "+"*60)
        print("EXPERIMENT 1: LEARNING CURVES")
        print("+"*60 + "\n")
        
        for catchment in args.catchments:
            try:
                sample_sizes = sample_sizes_quick if args.quick_test else None
                run_experiment_1(
                    catchment_name=catchment,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    sample_sizes=sample_sizes,
                    n_replicates=args.n_replicates,
                    use_synthetic=args.synthetic
                )
            except Exception as e:
                print(f"Error in Experiment 1 for {catchment}: {e}")
                continue
    
    # 实验2：采样策略
    if '2' in args.experiments:
        print("\n" + "+"*60)
        print("EXPERIMENT 2: SAMPLING STRATEGIES")
        print("+"*60 + "\n")
        
        for catchment in args.catchments:
            try:
                sample_sizes = sample_sizes_quick if args.quick_test else None
                run_experiment_2(
                    catchment_name=catchment,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    sample_sizes=sample_sizes,
                    n_replicates=args.n_replicates,
                    use_synthetic=args.synthetic
                )
            except Exception as e:
                print(f"Error in Experiment 2 for {catchment}: {e}")
                continue
    
    # 实验3：信息内容
    if '3' in args.experiments:
        print("\n" + "+"*60)
        print("EXPERIMENT 3: INFORMATION CONTENT")
        print("+"*60 + "\n")
        
        try:
            # 实验3处理所有流域
            run_experiment_3(
                catchment_names=args.catchments,
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Error in Experiment 3: {e}")
    
    # 实验4：空间分布
    if '4' in args.experiments:
        print("\n" + "+"*60)
        print("EXPERIMENT 4: SPATIAL DISTRIBUTION")
        print("+"*60 + "\n")
        
        for catchment in args.catchments:
            try:
                sample_sizes = sample_sizes_quick if args.quick_test else None
                run_experiment_4(
                    catchment_name=catchment,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    sample_sizes=sample_sizes,
                    n_replicates=args.n_replicates,
                    use_synthetic=args.synthetic
                )
            except Exception as e:
                print(f"Error in Experiment 4 for {catchment}: {e}")
                continue
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60 + "\n")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()