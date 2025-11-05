"""
运行所有实验的主脚本
"""
import argparse
import os
from pathlib import Path

from experiments.experiment_1_learning_curves import run_experiment_1
from experiments.experiment_2_sampling_strategies import run_experiment_2
from experiments.experiment_3_information_content import run_experiment_3
from experiments.experiment_4_spatial_distribution import run_experiment_4


def find_dataset_directory():
    """
    自动查找 Dataset 目录的位置
    
    Returns:
    --------
    str: Dataset 目录的路径
    
    Raises:
    -------
    FileNotFoundError: 如果找不到 Dataset 目录
    """
    # 当前项目目录
    current_dir = Path(__file__).parent.absolute()
    
    # 检查可能的位置
    possible_locations = [
        current_dir / "Dataset",                    # 项目根目录下
        current_dir.parent / "Dataset",             # 上一层目录
        current_dir.parent.parent / "Dataset",     # 上两层目录
    ]
    
    for location in possible_locations:
        if location.exists() and location.is_dir():
            print(f"找到 Dataset 目录: {location}")
            return str(location)
    
    # 如果没找到，抛出错误并提供详细信息
    searched_paths = "\n".join([f"  - {path}" for path in possible_locations])
    raise FileNotFoundError(
        f"未找到 Dataset 目录！\n"
        f"已搜索以下路径:\n{searched_paths}\n"
        f"请确保 Dataset 目录存在于项目目录或其父目录中。"
    )


def find_data_subdirectory(dataset_dir):
    """
    在 Dataset 目录中查找数据子目录
    
    Parameters:
    -----------
    dataset_dir : str, Dataset 目录路径
    
    Returns:
    --------
    str: 数据子目录路径
    
    Raises:
    -------
    FileNotFoundError: 如果找不到合适的数据子目录
    """
    dataset_path = Path(dataset_dir)
    
    # 查找可能的数据子目录
    possible_subdirs = [
        "IMPRO_catchment_data_infotheo",
        "catchment_data",
        "data",
        "raw_data",
        "input_data"
    ]
    
    for subdir in possible_subdirs:
        data_path = dataset_path / subdir
        if data_path.exists() and data_path.is_dir():
            # 检查是否包含预期的数据文件（如 .csv 或 .asc 文件）
            # 可能在子目录中，也可能在根目录中
            data_files = list(data_path.glob("**/*.csv")) + list(data_path.glob("**/*.asc"))
            if data_files:
                print(f"找到数据子目录: {data_path}")
                
                # 检查是否有流域子目录结构
                catchment_dirs = []
                for catchment in ['iller', 'saale', 'selke']:
                    catchment_dir = data_path / catchment
                    if catchment_dir.exists() and catchment_dir.is_dir():
                        catchment_files = list(catchment_dir.glob("*.csv")) + list(catchment_dir.glob("*.asc"))
                        if catchment_files:
                            catchment_dirs.append(catchment)
                
                if catchment_dirs:
                    print(f"发现流域子目录结构: {catchment_dirs}")
                    print(f"数据组织方式: {data_path}/[catchment_name]/[data_files]")
                else:
                    print(f"数据组织方式: {data_path}/[data_files]")
                
                return str(data_path)
    
    # 如果没找到合适的子目录，列出所有可用的子目录
    available_subdirs = [directory.name for directory in dataset_path.iterdir() if directory.is_dir()]
    raise FileNotFoundError(
        f"在 Dataset 目录中未找到包含数据文件的子目录！\n"
        f"Dataset 路径: {dataset_dir}\n"
        f"可用的子目录: {available_subdirs}\n"
        f"请检查数据文件是否存在，或手动指定 --data_dir 参数。"
    )


def auto_detect_data_dir():
    """
    自动检测数据目录
    
    Returns:
    --------
    str: 数据目录路径
    """
    try:
        dataset_dir = find_dataset_directory()
        data_dir = find_data_subdirectory(dataset_dir)
        return data_dir
    except FileNotFoundError as e:
        print(f"错误: {e}")
        raise


def verify_catchment_data(data_dir, catchment_name):
    """
    验证特定流域的数据文件是否存在
    
    Parameters:
    -----------
    data_dir : str, 数据目录路径
    catchment_name : str, 流域名称（如 'iller', 'saale', 'selke'）
    
    Returns:
    --------
    dict: 包含文件路径和存在性的字典
    
    Raises:
    -------
    FileNotFoundError: 如果找不到必要的数据文件
    """
    data_path = Path(data_dir)
    catchment_lower = catchment_name.lower()
    
    # 检查两种可能的文件组织方式
    # 方式1: data_dir/catchment_name/files
    # 方式2: data_dir/files_with_catchment_suffix
    
    file_info = {}
    
    # 首先尝试流域子目录方式
    catchment_dir = data_path / catchment_lower
    if catchment_dir.exists() and catchment_dir.is_dir():
        print(f"使用流域子目录: {catchment_dir}")
        base_path = catchment_dir
        file_patterns = {
            'discharge': f"discharge_{catchment_lower}.csv",
            'precipitation': f"precipitation_{catchment_lower}.asc",
            'temperature': f"temperature_{catchment_lower}.asc",
            'pet': f"pet_{catchment_lower}.asc",
        }
    else:
        # 尝试根目录方式
        print(f"使用根目录模式: {data_path}")
        base_path = data_path
        file_patterns = {
            'discharge': f"discharge_{catchment_lower}.csv",
            'precipitation': f"precipitation_{catchment_lower}.asc",
            'temperature': f"temperature_{catchment_lower}.asc",
            'pet': f"pet_{catchment_lower}.asc",
        }
    
    # 检查文件存在性
    missing_files = []
    for file_type, filename in file_patterns.items():
        file_path = base_path / filename
        file_info[file_type] = {
            'path': str(file_path),
            'exists': file_path.exists(),
            'size': file_path.stat().st_size if file_path.exists() else 0
        }
        
        if not file_path.exists():
            missing_files.append(f"{file_type}: {file_path}")
    
    # 如果有缺失文件，抛出错误
    if missing_files:
        available_files = []
        for file_type, info in file_info.items():
            if info['exists']:
                available_files.append(f"{file_type}: {info['path']}")
        
        error_msg = f"流域 {catchment_name.upper()} 的数据文件不完整！\n"
        error_msg += f"缺失文件:\n"
        for missing in missing_files:
            error_msg += f"  - {missing}\n"
        
        if available_files:
            error_msg += f"可用文件:\n"
            for available in available_files:
                error_msg += f"  + {available}\n"
        
        # 列出实际存在的文件供参考
        actual_files = list(base_path.glob("*")) if base_path.exists() else []
        if actual_files:
            error_msg += f"\n实际存在的文件:\n"
            for file in sorted(actual_files):
                if file.is_file():
                    error_msg += f"  * {file.name}\n"
        
        raise FileNotFoundError(error_msg)
    
    print(f"✓ 流域 {catchment_name.upper()} 的所有数据文件已找到")
    return file_info


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
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--n_replicates', type=int, default=30,
                       help='Number of replicates per sample size')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced settings')
    
    args = parser.parse_args()
    
    # 自动检测数据目录（如果未指定）
    if args.data_dir is None:
        print("正在自动检测数据目录...")
        try:
            args.data_dir = auto_detect_data_dir()
            print(f"✓ 成功找到数据目录: {args.data_dir}")
        except FileNotFoundError:
            print("✗ 数据目录检测失败，请手动指定 --data_dir 参数")
            return 1
    else:
        # 验证用户指定的数据目录
        if not Path(args.data_dir).exists():
            print(f"错误: 指定的数据目录不存在: {args.data_dir}")
            return 1
        print(f"使用指定的数据目录: {args.data_dir}")
    
    # 验证流域数据
    print("\n正在验证流域数据...")
    valid_catchments = []
    for catchment in args.catchments:
        try:
            verify_catchment_data(args.data_dir, catchment)
            valid_catchments.append(catchment)
        except FileNotFoundError as e:
            print(f"✗ 流域 {catchment} 数据验证失败:")
            print(f"  {e}")
            if not args.quick_test:  # 在快速测试模式下跳过缺失的流域
                return 1
    
    if not valid_catchments:
        print("✗ 没有找到任何有效的流域数据")
        return 1
    
    # 更新流域列表为有效的流域
    if len(valid_catchments) != len(args.catchments):
        print(f"注意: 将使用有效的流域 {valid_catchments}")
        args.catchments = valid_catchments
    
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
    print(f"Data directory: {args.data_dir}")
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
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)