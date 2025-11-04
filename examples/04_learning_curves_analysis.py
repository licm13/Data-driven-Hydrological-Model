#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例4：学习曲线分析
==================

本示例展示如何分析模型的学习曲线，即模型性能如何随训练数据量变化。
这是论文 Staudinger et al. (2025) 的核心分析之一。

包括：
  - 不同训练数据量下的模型性能
  - 过程驱动模型 vs 数据驱动模型的学习效率
  - 学习曲线可视化
  - 样本高效性分析

作者：Data-driven Hydrological Model Team
日期：2025-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import time

# 导入项目模块
from src.models import get_model
from src.utils.data_loader import generate_synthetic_data
from src.metrics.kge import kge
from src.metrics.entropy import evaluate_model_entropy
from src.sampling.strategies import random_sampling


def train_and_evaluate_model(
    model_name: str,
    data,
    train_size: int,
    warmup_days: int = 365,
    seed: int = 42
):
    """
    训练和评估单个模型

    Parameters:
    -----------
    model_name : str
        模型名称
    data : CatchmentData
        流域数据
    train_size : int
        训练样本数量
    warmup_days : int
        预热期天数
    seed : int
        随机种子

    Returns:
    --------
    metrics : dict
        性能指标字典
    """
    np.random.seed(seed)

    # 确保有足够的数据
    total_days = len(data)
    if warmup_days + train_size >= total_days:
        raise ValueError(f"Not enough data: {total_days} days, need {warmup_days + train_size}")

    # 采样训练数据
    train_indices = random_sampling(
        n_total=total_days - warmup_days,
        n_sample=train_size,
        seed=seed
    )
    train_indices = train_indices + warmup_days  # 调整为全局索引

    # 准备测试数据（使用所有预热期后的数据）
    test_start = warmup_days
    test_end = total_days

    # 根据模型类型选择训练方式
    if model_name in ['GR4J', 'HBV', 'SWAT+']:
        # 过程驱动模型：使用校准方法
        model = get_model(model_name,
                          with_snow=(model_name == 'GR4J'),
                          n_elevation_zones=(3 if model_name == 'HBV' else None))

        # 使用默认参数（实际应用中应进行校准）
        if model_name == 'GR4J':
            params = {'X1': 350.0, 'X2': 0.0, 'X3': 90.0, 'X4': 1.7,
                      'CTG': 0.0, 'Kf': 3.5}
        elif model_name == 'HBV':
            params = {'TT': 0.0, 'CFMAX': 3.5, 'FC': 250.0, 'BETA': 2.0,
                      'K0': 0.2, 'K1': 0.1, 'K2': 0.05, 'MAXBAS': 3.0,
                      'SFCF': 1.0, 'CFR': 0.05, 'CWH': 0.1,
                      'LP': 0.7, 'PERC': 2.0, 'UZL': 50.0,
                      'TCALT': -0.6, 'PCALT': 10.0}

        model.initialize(params)

        # 模拟（过程模型不需要训练，但参数应该用训练数据校准）
        sim_all = model.simulate(data.precip, data.temp, data.pet)
        sim_test = sim_all[test_start:test_end]

    else:
        # 数据驱动模型：需要训练
        model = get_model(model_name)

        # 准备训练特征
        X_train = np.column_stack([
            data.precip[train_indices],
            data.temp[train_indices],
            data.pet[train_indices]
        ])
        y_train = data.discharge[train_indices]

        # 准备测试特征
        X_test = np.column_stack([
            data.precip[test_start:test_end],
            data.temp[test_start:test_end],
            data.pet[test_start:test_end]
        ])

        # 初始化和训练
        if model_name in ['LSTM', 'ANN']:
            model.initialize({
                'n_epochs': 30,  # 用于快速演示
                'learning_rate': 0.001,
                'batch_size': 32,
            })
        else:
            model.initialize({})

        model.fit(X_train, y_train)

        # 预测
        sim_test = model.predict(X_test)

    # 获取观测数据
    obs_test = data.discharge[test_start:test_end]

    # 评估
    kge_value = kge(obs_test, sim_test)
    entropy_metrics = evaluate_model_entropy(obs_test, sim_test, n_bins=20)

    return {
        'model': model_name,
        'train_size': train_size,
        'kge': kge_value,
        'h_conditional': entropy_metrics['H_conditional'],
        'h_conditional_norm': entropy_metrics['H_conditional_normalized'],
    }


def run_learning_curve_experiment(
    models: List[str],
    data,
    train_sizes: List[int],
    n_replicates: int = 5,
    warmup_days: int = 365
):
    """
    运行学习曲线实验

    Parameters:
    -----------
    models : list
        要测试的模型列表
    data : CatchmentData
        流域数据
    train_sizes : list
        训练样本大小列表
    n_replicates : int
        每个样本大小的重复次数
    warmup_days : int
        预热期天数

    Returns:
    --------
    results : DataFrame
        实验结果
    """
    print(f"\n{'='*60}")
    print("学习曲线实验")
    print(f"{'='*60}")
    print(f"  模型数量: {len(models)}")
    print(f"  训练样本大小: {train_sizes}")
    print(f"  每个样本大小的重复次数: {n_replicates}")
    print(f"  总实验次数: {len(models) * len(train_sizes) * n_replicates}")

    all_results = []
    total_experiments = len(models) * len(train_sizes) * n_replicates
    experiment_count = 0

    for model_name in models:
        print(f"\n📊 模型: {model_name}")

        for train_size in train_sizes:
            print(f"\n   训练样本数 = {train_size}:")

            kge_values = []
            h_cond_values = []

            for rep in range(n_replicates):
                experiment_count += 1

                try:
                    # 运行单次实验
                    result = train_and_evaluate_model(
                        model_name=model_name,
                        data=data,
                        train_size=train_size,
                        warmup_days=warmup_days,
                        seed=42 + rep
                    )

                    kge_values.append(result['kge'])
                    h_cond_values.append(result['h_conditional'])
                    all_results.append(result)

                    # 进度显示
                    if (rep + 1) % max(1, n_replicates // 5) == 0:
                        print(f"      重复 {rep+1}/{n_replicates} - "
                              f"KGE={result['kge']:.3f}, "
                              f"H_cond={result['h_conditional']:.3f}")

                except Exception as e:
                    print(f"      ⚠ 重复 {rep+1} 失败: {e}")

            # 统计结果
            if kge_values:
                print(f"      平均KGE: {np.mean(kge_values):.3f} ± {np.std(kge_values):.3f}")
                print(f"      平均条件熵: {np.mean(h_cond_values):.3f} ± {np.std(h_cond_values):.3f}")

        print(f"   进度: {experiment_count}/{total_experiments} "
              f"({experiment_count/total_experiments*100:.1f}%)")

    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    print(f"\n✓ 实验完成！共收集 {len(results_df)} 个数据点")

    return results_df


def plot_learning_curves(results_df: pd.DataFrame, output_path: Path):
    """
    绘制学习曲线

    Parameters:
    -----------
    results_df : DataFrame
        实验结果
    output_path : Path
        输出文件路径
    """
    # 计算统计量
    stats = results_df.groupby(['model', 'train_size']).agg({
        'kge': ['mean', 'std'],
        'h_conditional': ['mean', 'std'],
        'h_conditional_norm': ['mean', 'std']
    }).reset_index()

    # 展平列名
    stats.columns = ['model', 'train_size', 'kge_mean', 'kge_std',
                     'h_cond_mean', 'h_cond_std',
                     'h_cond_norm_mean', 'h_cond_norm_std']

    # 定义颜色和标记
    model_colors = {
        'GR4J': '#1f77b4',
        'HBV': '#ff7f0e',
        'LSTM': '#2ca02c',
        'RTREE': '#d62728',
        'ANN': '#9467bd',
        'EDDIS': '#8c564b',
    }

    model_markers = {
        'GR4J': 'o',
        'HBV': 's',
        'LSTM': '^',
        'RTREE': 'v',
        'ANN': 'D',
        'EDDIS': 'p',
    }

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = stats['model'].unique()

    # 子图1：KGE学习曲线
    ax = axes[0, 0]
    for model in models:
        data = stats[stats['model'] == model]
        ax.plot(data['train_size'], data['kge_mean'],
                marker=model_markers.get(model, 'o'),
                color=model_colors.get(model, 'gray'),
                linewidth=2, markersize=8, label=model)
        ax.fill_between(data['train_size'],
                        data['kge_mean'] - data['kge_std'],
                        data['kge_mean'] + data['kge_std'],
                        alpha=0.2, color=model_colors.get(model, 'gray'))

    ax.set_xlabel('训练样本数', fontsize=11)
    ax.set_ylabel('KGE', fontsize=11)
    ax.set_title('KGE学习曲线', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 子图2：条件熵学习曲线
    ax = axes[0, 1]
    for model in models:
        data = stats[stats['model'] == model]
        ax.plot(data['train_size'], data['h_cond_mean'],
                marker=model_markers.get(model, 'o'),
                color=model_colors.get(model, 'gray'),
                linewidth=2, markersize=8, label=model)
        ax.fill_between(data['train_size'],
                        data['h_cond_mean'] - data['h_cond_std'],
                        data['h_cond_mean'] + data['h_cond_std'],
                        alpha=0.2, color=model_colors.get(model, 'gray'))

    ax.set_xlabel('训练样本数', fontsize=11)
    ax.set_ylabel('条件熵 [bits]', fontsize=11)
    ax.set_title('条件熵学习曲线（越低越好）', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 子图3：归一化条件熵
    ax = axes[1, 0]
    for model in models:
        data = stats[stats['model'] == model]
        ax.plot(data['train_size'], data['h_cond_norm_mean'],
                marker=model_markers.get(model, 'o'),
                color=model_colors.get(model, 'gray'),
                linewidth=2, markersize=8, label=model)
        ax.fill_between(data['train_size'],
                        data['h_cond_norm_mean'] - data['h_cond_norm_std'],
                        data['h_cond_norm_mean'] + data['h_cond_norm_std'],
                        alpha=0.2, color=model_colors.get(model, 'gray'))

    ax.set_xlabel('训练样本数', fontsize=11)
    ax.set_ylabel('归一化条件熵', fontsize=11)
    ax.set_title('归一化条件熵学习曲线（越低越好）', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 子图4：样本效率对比
    ax = axes[1, 1]

    # 计算达到KGE=0.7所需的样本数
    target_kge = 0.7
    sample_efficiency = []

    for model in models:
        data = stats[stats['model'] == model].sort_values('train_size')
        # 找到首次超过目标KGE的样本数
        above_target = data[data['kge_mean'] >= target_kge]

        if len(above_target) > 0:
            min_samples = above_target['train_size'].min()
            sample_efficiency.append({'model': model, 'min_samples': min_samples})
        else:
            # 如果从未达到，使用最大样本数
            sample_efficiency.append({'model': model, 'min_samples': np.nan})

    eff_df = pd.DataFrame(sample_efficiency)
    eff_df = eff_df.sort_values('min_samples')

    colors_list = [model_colors.get(m, 'gray') for m in eff_df['model']]
    ax.barh(range(len(eff_df)), eff_df['min_samples'], color=colors_list, alpha=0.7)
    ax.set_yticks(range(len(eff_df)))
    ax.set_yticklabels(eff_df['model'])
    ax.set_xlabel('达到KGE≥0.7所需样本数', fontsize=11)
    ax.set_title(f'样本效率对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n   学习曲线图已保存至: {output_path}")


def main():
    """主函数：学习曲线分析"""

    print("\n" + "="*70)
    print("示例4：学习曲线分析")
    print("="*70)

    output_dir = Path('results/examples')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 步骤1：生成数据
    # ========================================
    print("\n[步骤1] 生成数据...")

    data = generate_synthetic_data(
        n_days=4000,  # 约11年
        mean_precip=3.0,
        mean_temp=10.0,
        mean_pet=2.0,
        seed=42
    )

    print(f"   数据长度: {len(data)} 天")

    # ========================================
    # 步骤2：定义实验参数
    # ========================================
    print("\n[步骤2] 定义实验参数...")

    # 选择要测试的模型
    models_to_test = ['GR4J', 'HBV', 'LSTM', 'RTREE']

    # 定义训练样本大小（对数刻度）
    train_sizes = [10, 50, 100, 250, 500, 1000, 2000]

    # 重复次数（用于估计不确定性）
    n_replicates = 3  # 实际应用中建议10-30次

    print(f"   模型: {models_to_test}")
    print(f"   训练样本大小: {train_sizes}")
    print(f"   重复次数: {n_replicates}")

    # ========================================
    # 步骤3：运行实验
    # ========================================
    print("\n[步骤3] 运行学习曲线实验...")

    start_time = time.time()

    results_df = run_learning_curve_experiment(
        models=models_to_test,
        data=data,
        train_sizes=train_sizes,
        n_replicates=n_replicates,
        warmup_days=365
    )

    elapsed_time = time.time() - start_time
    print(f"\n   总运行时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")

    # ========================================
    # 步骤4：保存结果
    # ========================================
    print("\n[步骤4] 保存结果...")

    csv_file = output_dir / '04_learning_curves_data.csv'
    results_df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"   数据已保存至: {csv_file}")

    # ========================================
    # 步骤5：可视化
    # ========================================
    print("\n[步骤5] 生成学习曲线图...")

    plot_file = output_dir / '04_learning_curves.png'
    plot_learning_curves(results_df, plot_file)

    # ========================================
    # 步骤6：统计分析
    # ========================================
    print("\n[步骤6] 统计分析...")

    # 计算每个模型的平均性能
    avg_performance = results_df.groupby('model').agg({
        'kge': ['mean', 'std'],
        'h_conditional': ['mean', 'std']
    }).round(4)

    print("\n平均性能（所有训练样本大小）:")
    print(avg_performance)

    # 大样本性能（>=1000样本）
    large_sample = results_df[results_df['train_size'] >= 1000]
    if len(large_sample) > 0:
        print("\n大样本性能（≥1000样本）:")
        print(large_sample.groupby('model')['kge'].agg(['mean', 'std']).round(4))

    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*70)
    print("示例完成！")
    print("="*70)
    print("\n关键发现:")
    print("  ✓ 生成了多个模型的学习曲线")
    print("  ✓ 分析了模型性能随训练数据量的变化")
    print("  ✓ 对比了过程驱动模型和数据驱动模型的学习效率")
    print("  ✓ 使用KGE和条件熵两种评估指标")
    print("\n学习曲线的意义:")
    print("  - 陡峭的曲线：模型能快速从少量数据中学习")
    print("  - 平缓的曲线：模型需要大量数据才能达到好性能")
    print("  - 曲线趋于平稳：增加数据量带来的收益递减")
    print("\n实际应用:")
    print("  - 帮助选择合适的模型（考虑可用数据量）")
    print("  - 指导数据收集策略（性价比分析）")
    print("  - 评估模型的样本效率")
    print("="*70)


if __name__ == '__main__':
    main()
