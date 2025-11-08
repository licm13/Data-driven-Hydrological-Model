#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例2：多模型性能对比
====================

本示例展示如何同时运行和对比多个水文模型的性能。
包括：过程驱动模型和数据驱动模型的对比、性能评估、结果可视化。

对比的模型：
  - GR4J: 4参数日尺度集总式模型
  - HBV: 半分布式模型
  - LSTM: 长短期记忆神经网络
  - RTREE: 回归树模型

作者：Data-driven Hydrological Model Team
日期：2025-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Tuple

# 导入模型和工具
from src.models import get_model
from src.utils.data_loader import generate_synthetic_data, CatchmentData
from src.metrics.kge import kge, kge_components
from src.metrics.entropy import evaluate_model_entropy

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_process_based_model(
    model_name: str,
    data: CatchmentData,
    warmup_days: int = 365,
    **model_kwargs
) -> Tuple[np.ndarray, float]:
    """
    运行过程驱动模型

    Parameters:
    -----------
    model_name : str
        模型名称 ('GR4J' 或 'HBV')
    data : CatchmentData
        流域数据
    warmup_days : int
        预热期天数
    **model_kwargs : dict
        模型特定参数

    Returns:
    --------
    simulated : np.ndarray
        模拟的径流（已去除预热期）
    run_time : float
        运行时间（秒）
    """
    print(f"\n   运行 {model_name} 模型...")

    # 创建模型
    model = get_model(model_name, **model_kwargs)

    # 设置模型参数（使用典型值）
    if model_name == 'GR4J':
        params = {
            'X1': 350.0,   # 生产库容量 [mm]
            'X2': 0.0,     # 地下水交换 [mm/day]
            'X3': 90.0,    # 汇流库容量 [mm]
            'X4': 1.7,     # 单位线时间基数 [day]
            'CTG': 0.0,    # 融雪温度阈值 [°C]
            'Kf': 3.5,     # 融雪系数 [mm/°C/day]
        }
    elif model_name == 'HBV':
        params = {
            'TT': 0.0,      'CFMAX': 3.5,   'SFCF': 1.0,
            'CFR': 0.05,    'CWH': 0.1,     'FC': 250.0,
            'LP': 0.7,      'BETA': 2.0,    'PERC': 2.0,
            'UZL': 50.0,    'K0': 0.2,      'K1': 0.1,
            'K2': 0.05,     'MAXBAS': 3.0,  'TCALT': -0.6,
            'PCALT': 10.0,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 初始化并运行
    model.initialize(params)

    start_time = time.time()
    simulated = model.simulate(data.precip, data.temp, data.pet)
    run_time = time.time() - start_time

    # 去除预热期
    simulated = simulated[warmup_days:]

    print(f"      运行时间: {run_time:.3f} 秒")

    return simulated, run_time


def run_data_driven_model(
    model_name: str,
    data: CatchmentData,
    train_size: int = 2000,
    warmup_days: int = 365,
    **model_kwargs
) -> Tuple[np.ndarray, float]:
    """
    运行数据驱动模型

    Parameters:
    -----------
    model_name : str
        模型名称 ('LSTM', 'ANN', 'RTREE', 'EDDIS')
    data : CatchmentData
        流域数据
    train_size : int
        训练样本数
    warmup_days : int
        预热期天数
    **model_kwargs : dict
        模型特定参数

    Returns:
    --------
    simulated : np.ndarray
        模拟的径流（已去除预热期）
    run_time : float
        运行时间（秒）
    """
    print(f"\n   运行 {model_name} 模型...")

    # 创建模型
    model = get_model(model_name, **model_kwargs)

    # 准备训练数据
    total_days = len(data)
    train_end = warmup_days + train_size

    # 构造输入特征矩阵 [P, T, PET]
    X_train = np.column_stack([
        data.precip[warmup_days:train_end],
        data.temp[warmup_days:train_end],
        data.pet[warmup_days:train_end]
    ])
    y_train = data.discharge[warmup_days:train_end]

    X_test = np.column_stack([
        data.precip,
        data.temp,
        data.pet
    ])

    # 训练并模拟
    start_time = time.time()

    if model_name in ['LSTM', 'ANN']:
        # 神经网络模型需要更少的训练轮次用于演示
        model.initialize({
            'n_epochs': 50 if model_name == 'LSTM' else 100,
            'learning_rate': 0.001,
            'batch_size': 32,
        })
    else:
        model.initialize({})

    # 训练模型
    model.fit(X_train, y_train)

    # 在全部数据上预测
    simulated = model.predict(X_test)

    run_time = time.time() - start_time

    # 去除预热期
    simulated = simulated[warmup_days:]

    print(f"      训练样本数: {train_size}")
    print(f"      运行时间: {run_time:.3f} 秒 (包含训练)")

    return simulated, run_time


def evaluate_model(
    obs: np.ndarray,
    sim: np.ndarray,
    model_name: str
) -> Dict:
    """
    评估模型性能

    Returns:
    --------
    metrics : dict
        包含各种评估指标的字典
    """
    # KGE指标
    kge_value = kge(obs, sim)
    r, alpha, beta = kge_components(obs, sim)

    # 信息熵指标
    entropy_metrics = evaluate_model_entropy(obs, sim, n_bins=20)

    # NSE指标
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

    # RMSE
    rmse = np.sqrt(np.mean((obs - sim)**2))

    # 相对偏差
    bias = (np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100

    return {
        'model': model_name,
        'kge': kge_value,
        'r': r,
        'alpha': alpha,
        'beta': beta,
        'nse': nse,
        'rmse': rmse,
        'bias': bias,
        'h_conditional': entropy_metrics['H_conditional'],
        'h_conditional_norm': entropy_metrics['H_conditional_normalized'],
        'mutual_info': entropy_metrics['I_mutual'],
    }


def plot_comparison(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    simulations: Dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    output_path: Path
):
    """
    绘制多模型对比图

    Parameters:
    -----------
    dates : DatetimeIndex
        时间索引
    obs : np.ndarray
        观测径流
    simulations : dict
        各模型的模拟结果 {model_name: simulated_discharge}
    metrics_df : DataFrame
        性能指标表
    output_path : Path
        输出文件路径
    """
    # 定义模型颜色
    colors = {
        'GR4J': '#1f77b4',
        'HBV': '#ff7f0e',
        'LSTM': '#2ca02c',
        'RTREE': '#d62728',
    }

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # ========================================
    # 子图1：时间序列对比（上半年）
    # ========================================
    ax1 = fig.add_subplot(gs[0, :])
    n_show = 180  # 显示前半年

    ax1.plot(dates[:n_show], obs[:n_show],
             label='观测', color='black', linewidth=2, alpha=0.8)

    for model_name, sim in simulations.items():
        ax1.plot(dates[:n_show], sim[:n_show],
                 label=model_name, color=colors.get(model_name, 'gray'),
                 linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('径流 [mm/day]', fontsize=11)
    ax1.set_title('径流时间序列对比（前180天）', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', ncol=5)
    ax1.grid(True, alpha=0.3)

    # ========================================
    # 子图2-5：散点图（每个模型）
    # ========================================
    for idx, (model_name, sim) in enumerate(simulations.items()):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])

        # 散点图
        ax.scatter(obs, sim, alpha=0.2, s=5,
                   color=colors.get(model_name, 'gray'))

        # 1:1线
        max_val = max(np.max(obs), np.max(sim))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1线')

        # 添加KGE值
        kge_val = metrics_df[metrics_df['model'] == model_name]['kge'].values[0]
        nse_val = metrics_df[metrics_df['model'] == model_name]['nse'].values[0]

        ax.text(0.05, 0.95, f'KGE={kge_val:.3f}\nNSE={nse_val:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('观测径流 [mm/day]', fontsize=10)
        ax.set_ylabel('模拟径流 [mm/day]', fontsize=10)
        ax.set_title(f'{model_name} 模型', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # ========================================
    # 子图6：性能对比雷达图
    # ========================================
    ax6 = fig.add_subplot(gs[3, 0], projection='polar')

    # 准备雷达图数据（归一化到0-1）
    metrics_names = ['KGE', 'NSE', 'r', '1-bias', '1-h_norm']
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for model_name in simulations.keys():
        row = metrics_df[metrics_df['model'] == model_name].iloc[0]

        values = [
            max(0, row['kge']),                      # KGE
            max(0, row['nse']),                      # NSE
            row['r'],                                 # 相关系数
            1 - abs(row['bias']) / 100,              # 1 - 相对偏差
            1 - row['h_conditional_norm'],           # 1 - 归一化条件熵
        ]
        values += values[:1]  # 闭合

        ax6.plot(angles, values, 'o-', linewidth=2,
                 label=model_name, color=colors.get(model_name, 'gray'))
        ax6.fill(angles, values, alpha=0.15,
                 color=colors.get(model_name, 'gray'))

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics_names)
    ax6.set_ylim(0, 1)
    ax6.set_title('综合性能对比', fontsize=11, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.grid(True)

    # ========================================
    # 子图7：指标柱状图
    # ========================================
    ax7 = fig.add_subplot(gs[3, 1])

    x = np.arange(len(simulations))
    width = 0.25

    kge_values = [metrics_df[metrics_df['model'] == m]['kge'].values[0]
                  for m in simulations.keys()]
    nse_values = [metrics_df[metrics_df['model'] == m]['nse'].values[0]
                  for m in simulations.keys()]
    r_values = [metrics_df[metrics_df['model'] == m]['r'].values[0]
                for m in simulations.keys()]

    ax7.bar(x - width, kge_values, width, label='KGE', alpha=0.8)
    ax7.bar(x, nse_values, width, label='NSE', alpha=0.8)
    ax7.bar(x + width, r_values, width, label='r', alpha=0.8)

    ax7.set_ylabel('指标值', fontsize=11)
    ax7.set_title('关键指标对比', fontsize=11, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(simulations.keys())
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='black', linewidth=0.5)

    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n   对比图已保存至: {output_path}")


def main():
    """主函数：多模型对比"""

    print("="*70)
    print("示例2：多模型性能对比")
    print("="*70)

    # ========================================
    # 步骤1：生成数据
    # ========================================
    print("\n[步骤1] 生成合成数据...")

    n_days = 4000  # 约11年
    data = generate_synthetic_data(
        n_days=n_days,
        mean_precip=3.0,
        mean_temp=10.0,
        mean_pet=2.0,
        seed=42
    )

    warmup_days = 365
    print(f"   总天数: {n_days}")
    print(f"   预热期: {warmup_days} 天")
    print(f"   有效数据: {n_days - warmup_days} 天")

    # ========================================
    # 步骤2：运行多个模型
    # ========================================
    print("\n[步骤2] 运行多个模型...")

    simulations = {}
    run_times = {}

    # 过程驱动模型
    print("\n   === 过程驱动模型 ===")

    sim, rt = run_process_based_model('GR4J', data, warmup_days, with_snow=True)
    simulations['GR4J'] = sim
    run_times['GR4J'] = rt

    sim, rt = run_process_based_model('HBV', data, warmup_days, n_elevation_zones=3)
    simulations['HBV'] = sim
    run_times['HBV'] = rt

    # 数据驱动模型
    print("\n   === 数据驱动模型 ===")

    sim, rt = run_data_driven_model('LSTM', data, train_size=2000, warmup_days=warmup_days,
                                     n_layers=1, hidden_size=32, window_size=7)
    simulations['LSTM'] = sim
    run_times['LSTM'] = rt

    sim, rt = run_data_driven_model('RTREE', data, train_size=2000, warmup_days=warmup_days,
                                     lag_days=7)
    simulations['RTREE'] = sim
    run_times['RTREE'] = rt

    # ========================================
    # 步骤3：评估所有模型
    # ========================================
    print("\n[步骤3] 评估所有模型...")

    obs = data.discharge[warmup_days:]
    dates = data.dates[warmup_days:]

    all_metrics = []
    for model_name, sim in simulations.items():
        metrics = evaluate_model(obs, sim, model_name)
        metrics['run_time'] = run_times[model_name]
        all_metrics.append(metrics)

    # 转换为DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # ========================================
    # 步骤4：打印结果表格
    # ========================================
    print("\n[步骤4] 性能对比结果:")
    print("\n" + "="*70)

    # 精简表格
    display_df = metrics_df[[
        'model', 'kge', 'nse', 'r', 'rmse',
        'bias', 'h_conditional', 'run_time'
    ]].copy()

    display_df.columns = [
        '模型', 'KGE', 'NSE', '相关系数(r)',
        'RMSE', '偏差(%)', '条件熵(bits)', '运行时间(s)'
    ]

    print(display_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("="*70)

    # 排名分析
    print("\n模型排名:")
    print(f"  最佳KGE: {metrics_df.loc[metrics_df['kge'].idxmax(), 'model']}")
    print(f"  最佳NSE: {metrics_df.loc[metrics_df['nse'].idxmax(), 'model']}")
    print(f"  最小条件熵: {metrics_df.loc[metrics_df['h_conditional'].idxmin(), 'model']}")
    print(f"  最快运行: {metrics_df.loc[metrics_df['run_time'].idxmin(), 'model']}")

    # ========================================
    # 步骤5：可视化对比
    # ========================================
    print("\n[步骤5] 生成对比可视化...")

    output_dir = Path('results/examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '02_model_comparison.png'

    plot_comparison(dates, obs, simulations, metrics_df, output_file)

    # 保存详细结果
    csv_file = output_dir / '02_model_metrics.csv'
    metrics_df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"   详细指标已保存至: {csv_file}")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*70)
    print("示例完成！")
    print("="*70)
    print("\n关键发现:")
    print("  ✓ 比较了4个不同类型的水文模型")
    print("  ✓ 过程驱动模型和数据驱动模型各有优势")
    print("  ✓ 使用多个指标（KGE、NSE、条件熵）综合评估")
    print("  ✓ 生成了详细的对比可视化和数据表")
    print("\n下一步:")
    print("  - 查看示例3学习如何使用真实流域数据")
    print("  - 查看示例3学习如何校准模型参数")
    print("  - 尝试调整模型参数观察性能变化")
    print("="*70)


if __name__ == '__main__':
    main()
