#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例1：基础模型使用
===================

本示例展示如何使用单个水文模型进行模拟和评估。
包括：模型初始化、参数设置、模拟运行和结果评估。

作者：Data-driven Hydrological Model Team
日期：2025-01
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入模型和工具
from src.models import get_model
from src.utils.data_loader import generate_synthetic_data
from src.metrics.kge import kge, kge_components
from src.metrics.entropy import evaluate_model_entropy

def main():
    """主函数：演示基础模型使用流程"""

    print("="*60)
    print("示例1：基础水文模型使用")
    print("="*60)

    # ========================================
    # 步骤1：生成或加载数据
    # ========================================
    print("\n[步骤1] 生成合成数据...")

    # 生成10年的合成数据用于演示
    # 包含：降水(precip)、温度(temp)、潜在蒸散发(pet)、径流(discharge)
    n_days = 3650  # 10年
    data = generate_synthetic_data(
        n_days=n_days,
        mean_precip=3.0,      # 平均降水量 [mm/day]
        mean_temp=10.0,       # 平均温度 [°C]
        mean_pet=2.0,         # 平均蒸散发 [mm/day]
        seed=42               # 随机种子，保证结果可复现
    )

    print(f"   数据长度: {len(data)} 天")
    print(f"   时间范围: {data.dates[0].date()} 至 {data.dates[-1].date()}")
    print(f"   降水统计: 均值={np.mean(data.precip):.2f} mm/day, "
          f"最大值={np.max(data.precip):.2f} mm/day")
    print(f"   温度统计: 均值={np.mean(data.temp):.2f} °C, "
          f"范围=[{np.min(data.temp):.1f}, {np.max(data.temp):.1f}] °C")

    # ========================================
    # 步骤2：创建和初始化模型
    # ========================================
    print("\n[步骤2] 创建HBV模型...")

    # HBV是一个半分布式水文模型，支持多个高程带
    # n_elevation_zones=3 表示将流域分为3个高程带
    model = get_model('HBV', n_elevation_zones=3)

    # 设置模型参数
    # 这些参数通常通过校准获得，这里使用典型值
    params = {
        # === 雪模块参数 ===
        'TT': 0.0,          # 阈值温度 [°C]，高于此温度降雪融化
        'CFMAX': 3.5,       # 度日因子 [mm/°C/day]，控制融雪速率
        'SFCF': 1.0,        # 降雪修正系数 [-]
        'CFR': 0.05,        # 再冻结系数 [-]
        'CWH': 0.1,         # 持水能力 [-]，积雪可保持的液态水比例

        # === 土壤模块参数 ===
        'FC': 250.0,        # 土壤最大含水量 [mm]，反映土壤蓄水能力
        'LP': 0.7,          # 蒸发限制 [-]，土壤含水量低于此比例时蒸发受限
        'BETA': 2.0,        # 形状系数 [-]，控制产流的非线性程度

        # === 地下水响应模块参数 ===
        'PERC': 2.0,        # 渗漏率 [mm/day]，从上层到下层地下水库的渗漏
        'UZL': 50.0,        # 快速径流阈值 [mm]，上层水库超过此值产生快速径流
        'K0': 0.2,          # 快速径流系数 [1/day]，快速流退水系数
        'K1': 0.1,          # 中速径流系数 [1/day]，中速流退水系数
        'K2': 0.05,         # 慢速径流系数 [1/day]，慢速流退水系数

        # === 汇流模块参数 ===
        'MAXBAS': 3.0,      # 汇流时间 [day]，控制径流汇集到出口的时间

        # === 高程梯度参数 ===
        'TCALT': -0.6,      # 温度梯度 [°C/100m]，温度随海拔的变化率
        'PCALT': 10.0,      # 降水梯度 [%/100m]，降水随海拔的增加率
    }

    model.initialize(params)
    print(f"   模型类型: {model.name}")
    print(f"   高程带数量: {model.n_zones}")
    print(f"   参数数量: {len(params)}")

    # ========================================
    # 步骤3：运行模型模拟
    # ========================================
    print("\n[步骤3] 运行模型模拟...")

    # 定义预热期（让模型状态变量达到稳定状态）
    warmup_days = 365  # 预热1年

    # 运行模拟
    # simulate方法接收气象输入，返回模拟的径流
    simulated_discharge = model.simulate(
        precip=data.precip,
        temp=data.temp,
        pet=data.pet
    )

    # 去除预热期数据
    obs_discharge = data.discharge[warmup_days:]
    sim_discharge = simulated_discharge[warmup_days:]
    dates = data.dates[warmup_days:]

    print(f"   模拟完成！有效数据: {len(sim_discharge)} 天")
    print(f"   观测径流: 均值={np.mean(obs_discharge):.2f} mm/day, "
          f"最大值={np.max(obs_discharge):.2f} mm/day")
    print(f"   模拟径流: 均值={np.mean(sim_discharge):.2f} mm/day, "
          f"最大值={np.max(sim_discharge):.2f} mm/day")

    # ========================================
    # 步骤4：评估模型性能
    # ========================================
    print("\n[步骤4] 评估模型性能...")

    # 计算KGE（Kling-Gupta Efficiency）
    # KGE综合考虑相关性、变异性和偏差
    kge_value = kge(obs_discharge, sim_discharge)
    r, alpha, beta = kge_components(obs_discharge, sim_discharge)

    print(f"\n   === KGE评估指标 ===")
    print(f"   KGE总分: {kge_value:.3f}  (越接近1越好)")
    print(f"   相关系数 (r): {r:.3f}  (观测与模拟的线性关系)")
    print(f"   变异比 (α): {alpha:.3f}  (变异性比值，理想值=1)")
    print(f"   偏差比 (β): {beta:.3f}  (均值比值，理想值=1)")

    # 计算信息熵评估指标
    # 信息熵从信息论角度评估模型的预测不确定性
    entropy_metrics = evaluate_model_entropy(
        obs=obs_discharge,
        sim=sim_discharge,
        n_bins=20  # 离散化的箱数
    )

    print(f"\n   === 信息熵评估指标 ===")
    print(f"   联合熵 H(X,Y): {entropy_metrics['H_joint']:.3f} bits")
    print(f"   条件熵 H(Y|X): {entropy_metrics['H_conditional']:.3f} bits "
          f"(模拟给定时的观测不确定性)")
    print(f"   互信息 I(X;Y): {entropy_metrics['I_mutual']:.3f} bits "
          f"(模拟与观测的共享信息)")
    print(f"   归一化条件熵: {entropy_metrics['H_conditional_normalized']:.3f} "
          f"(越小越好)")

    # ========================================
    # 步骤5：可视化结果
    # ========================================
    print("\n[步骤5] 可视化结果...")

    # 创建图形窗口
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1：时间序列对比（显示前2年）
    n_show = 730  # 显示前2年
    axes[0].plot(dates[:n_show], obs_discharge[:n_show],
                 label='观测径流', color='blue', linewidth=1.5, alpha=0.7)
    axes[0].plot(dates[:n_show], sim_discharge[:n_show],
                 label='模拟径流', color='red', linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel('径流 [mm/day]', fontsize=11)
    axes[0].set_title('径流时间序列对比（前2年）', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 子图2：散点图（观测 vs 模拟）
    axes[1].scatter(obs_discharge, sim_discharge,
                    alpha=0.3, s=10, color='darkblue')
    # 添加1:1线
    max_val = max(np.max(obs_discharge), np.max(sim_discharge))
    axes[1].plot([0, max_val], [0, max_val],
                 'r--', linewidth=2, label='1:1线')
    axes[1].set_xlabel('观测径流 [mm/day]', fontsize=11)
    axes[1].set_ylabel('模拟径流 [mm/day]', fontsize=11)
    axes[1].set_title(f'散点图对比 (KGE={kge_value:.3f})',
                      fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    # 子图3：残差分析
    residuals = sim_discharge - obs_discharge
    axes[2].plot(dates, residuals, color='purple', linewidth=0.5, alpha=0.6)
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].fill_between(dates, residuals, 0, alpha=0.3, color='purple')
    axes[2].set_xlabel('日期', fontsize=11)
    axes[2].set_ylabel('残差 [mm/day]', fontsize=11)
    axes[2].set_title('模拟残差 (模拟 - 观测)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图形
    output_dir = Path('results/examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '01_basic_model_usage.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   图形已保存至: {output_file}")

    # 显示图形（如果在交互环境中）
    # plt.show()

    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("\n总结:")
    print("  ✓ 成功生成合成数据")
    print("  ✓ 成功初始化HBV模型")
    print("  ✓ 完成模拟运行")
    print(f"  ✓ KGE评估: {kge_value:.3f}")
    print(f"  ✓ 条件熵评估: {entropy_metrics['H_conditional']:.3f} bits")
    print("  ✓ 生成可视化结果")
    print("\n下一步:")
    print("  - 查看示例2学习如何对比多个模型")
    print("  - 查看示例3学习如何使用真实流域数据")
    print("  - 查看示例4学习如何校准模型参数")
    print("="*60)

if __name__ == '__main__':
    main()
