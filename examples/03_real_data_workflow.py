#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例3：真实流域数据处理完整工作流
===================================

本示例展示使用真实流域数据的完整工作流程，包括：
  - 数据加载和验证
  - 数据质量检查
  - 训练/测试集划分
  - 模型校准
  - 性能评估
  - 结果分析和可视化

作者：Data-driven Hydrological Model Team
日期：2025-01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 导入项目模块
from src.utils.data_loader import load_catchment_from_csv, generate_synthetic_data
from src.utils.impro_loader import load_impro_catchment
from src.models import get_model
from src.calibration.spotpy_wrapper import calibrate_model
from src.metrics.kge import kge
from src.metrics.entropy import evaluate_model_entropy


def check_data_quality(data, catchment_name: str):
    """
    检查数据质量

    Parameters:
    -----------
    data : CatchmentData
        流域数据对象
    catchment_name : str
        流域名称
    """
    print(f"\n{'='*60}")
    print(f"数据质量检查: {catchment_name}")
    print(f"{'='*60}")

    # 基本信息
    print(f"\n📊 基本信息:")
    print(f"  流域名称: {data.name}")
    print(f"  数据长度: {len(data)} 天 ({len(data)/365:.1f} 年)")
    print(f"  时间范围: {data.dates[0].date()} 至 {data.dates[-1].date()}")
    if data.area:
        print(f"  流域面积: {data.area:.1f} km²")
    if data.elevation_range:
        print(f"  高程范围: {data.elevation_range[0]:.0f} - {data.elevation_range[1]:.0f} m")

    # 缺失值检查
    print(f"\n🔍 缺失值检查:")
    missing_precip = np.isnan(data.precip).sum()
    missing_temp = np.isnan(data.temp).sum()
    missing_pet = np.isnan(data.pet).sum()
    missing_discharge = np.isnan(data.discharge).sum()

    print(f"  降水缺失: {missing_precip} ({missing_precip/len(data)*100:.2f}%)")
    print(f"  温度缺失: {missing_temp} ({missing_temp/len(data)*100:.2f}%)")
    print(f"  蒸散发缺失: {missing_pet} ({missing_pet/len(data)*100:.2f}%)")
    print(f"  径流缺失: {missing_discharge} ({missing_discharge/len(data)*100:.2f}%)")

    if missing_precip + missing_temp + missing_pet + missing_discharge == 0:
        print("  ✓ 数据完整，无缺失值")
    else:
        print("  ⚠ 警告: 存在缺失值，建议进行插值或删除")

    # 数据统计
    print(f"\n📈 数据统计:")
    print(f"  降水 [mm/day]:")
    print(f"    均值={np.nanmean(data.precip):.2f}, 中位数={np.nanmedian(data.precip):.2f}")
    print(f"    最小值={np.nanmin(data.precip):.2f}, 最大值={np.nanmax(data.precip):.2f}")
    print(f"    年总量={np.nansum(data.precip)/len(data)*365:.1f} mm/year")

    print(f"  温度 [°C]:")
    print(f"    均值={np.nanmean(data.temp):.2f}, 中位数={np.nanmedian(data.temp):.2f}")
    print(f"    最小值={np.nanmin(data.temp):.2f}, 最大值={np.nanmax(data.temp):.2f}")

    print(f"  潜在蒸散发 [mm/day]:")
    print(f"    均值={np.nanmean(data.pet):.2f}, 中位数={np.nanmedian(data.pet):.2f}")
    print(f"    年总量={np.nansum(data.pet)/len(data)*365:.1f} mm/year")

    print(f"  径流 [mm/day]:")
    print(f"    均值={np.nanmean(data.discharge):.2f}, 中位数={np.nanmedian(data.discharge):.2f}")
    print(f"    最小值={np.nanmin(data.discharge):.2f}, 最大值={np.nanmax(data.discharge):.2f}")
    print(f"    年总量={np.nansum(data.discharge)/len(data)*365:.1f} mm/year")

    # 水量平衡检查
    print(f"\n💧 水量平衡检查:")
    annual_precip = np.nansum(data.precip) / len(data) * 365
    annual_pet = np.nansum(data.pet) / len(data) * 365
    annual_discharge = np.nansum(data.discharge) / len(data) * 365
    runoff_coefficient = annual_discharge / annual_precip

    print(f"  年均降水: {annual_precip:.1f} mm")
    print(f"  年均蒸散发: {annual_pet:.1f} mm")
    print(f"  年均径流: {annual_discharge:.1f} mm")
    print(f"  径流系数: {runoff_coefficient:.3f}")

    if 0.1 <= runoff_coefficient <= 0.9:
        print(f"  ✓ 径流系数合理")
    else:
        print(f"  ⚠ 警告: 径流系数异常，请检查数据质量")

    # 极端值检查
    print(f"\n⚡ 极端值检查:")
    p95_precip = np.nanpercentile(data.precip, 95)
    p95_discharge = np.nanpercentile(data.discharge, 95)

    print(f"  降水95分位数: {p95_precip:.2f} mm/day")
    print(f"  径流95分位数: {p95_discharge:.2f} mm/day")

    extreme_precip_days = np.sum(data.precip > p95_precip)
    extreme_discharge_days = np.sum(data.discharge > p95_discharge)

    print(f"  极端降水天数: {extreme_precip_days} ({extreme_precip_days/len(data)*100:.1f}%)")
    print(f"  极端径流天数: {extreme_discharge_days} ({extreme_discharge_days/len(data)*100:.1f}%)")

    print(f"\n{'='*60}\n")


def visualize_data(data, output_path: Path):
    """
    可视化数据

    Parameters:
    -----------
    data : CatchmentData
        流域数据
    output_path : Path
        输出文件路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 子图1：降水
    axes[0].bar(data.dates, data.precip, width=1.0, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('降水 [mm/day]', fontsize=11)
    axes[0].set_title('降水时间序列', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 子图2：温度
    axes[1].plot(data.dates, data.temp, color='orangered', linewidth=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('温度 [°C]', fontsize=11)
    axes[1].set_title('温度时间序列', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 子图3：潜在蒸散发
    axes[2].plot(data.dates, data.pet, color='green', linewidth=0.8)
    axes[2].set_ylabel('潜在蒸散发 [mm/day]', fontsize=11)
    axes[2].set_title('潜在蒸散发时间序列', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # 子图4：径流
    axes[3].plot(data.dates, data.discharge, color='blue', linewidth=1.0)
    axes[3].set_ylabel('径流 [mm/day]', fontsize=11)
    axes[3].set_xlabel('日期', fontsize=11)
    axes[3].set_title('径流时间序列', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   数据可视化已保存至: {output_path}")


def run_calibration_and_validation(data, model_name: str, output_dir: Path):
    """
    运行模型校准和验证

    Parameters:
    -----------
    data : CatchmentData
        流域数据
    model_name : str
        模型名称
    output_dir : Path
        输出目录
    """
    print(f"\n{'='*60}")
    print(f"模型校准和验证: {model_name}")
    print(f"{'='*60}")

    # 划分训练/测试期
    # 假设前70%用于校准，后30%用于验证
    n_total = len(data)
    n_calib = int(n_total * 0.7)
    warmup_days = 365

    print(f"\n📅 数据划分:")
    print(f"  总数据: {n_total} 天")
    print(f"  校准期: {warmup_days} - {n_calib} 天 ({(n_calib-warmup_days)/365:.1f} 年)")
    print(f"  验证期: {n_calib} - {n_total} 天 ({(n_total-n_calib)/365:.1f} 年)")

    # 准备校准数据
    calib_data_dict = {
        'precip': data.precip[:n_calib],
        'temp': data.temp[:n_calib],
        'pet': data.pet[:n_calib],
        'discharge': data.discharge[:n_calib],
        'dates': data.dates[:n_calib],
    }

    # 校准模型
    print(f"\n🔧 开始模型校准...")
    print(f"   优化算法: SCE-UA")
    print(f"   目标函数: KGE")
    print(f"   最大迭代次数: 1000")

    if model_name == 'GR4J':
        model = get_model('GR4J', with_snow=True)
        model_kwargs = {'with_snow': True}
    elif model_name == 'HBV':
        model = get_model('HBV', n_elevation_zones=1)
        model_kwargs = {'n_elevation_zones': 1}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 运行校准（使用较少的迭代次数用于演示）
    best_params, best_kge = calibrate_model(
        model=model,
        catchment_data=calib_data_dict,
        objective='kge',
        warmup_days=warmup_days,
        n_iterations=500,  # 实际应用中建议 5000-10000
        algorithm='sceua'
    )

    print(f"\n   ✓ 校准完成!")
    print(f"   最优KGE: {best_kge:.4f}")
    print(f"\n   最优参数:")
    for param, value in best_params.items():
        print(f"     {param}: {value:.4f}")

    # 使用最优参数运行模型
    model.initialize(best_params)

    # 校准期模拟
    sim_calib = model.simulate(data.precip[:n_calib],
                                data.temp[:n_calib],
                                data.pet[:n_calib])
    obs_calib = data.discharge[warmup_days:n_calib]
    sim_calib = sim_calib[warmup_days:]
    dates_calib = data.dates[warmup_days:n_calib]

    # 验证期模拟
    sim_valid = model.simulate(data.precip,
                                data.temp,
                                data.pet)
    obs_valid = data.discharge[n_calib:]
    sim_valid = sim_valid[n_calib:]
    dates_valid = data.dates[n_calib:]

    # 评估
    print(f"\n📊 性能评估:")

    # 校准期
    kge_calib = kge(obs_calib, sim_calib)
    entropy_calib = evaluate_model_entropy(obs_calib, sim_calib, n_bins=20)

    print(f"\n  === 校准期 ===")
    print(f"  KGE: {kge_calib:.4f}")
    print(f"  条件熵: {entropy_calib['H_conditional']:.4f} bits")
    print(f"  归一化条件熵: {entropy_calib['H_conditional_normalized']:.4f}")

    # 验证期
    kge_valid = kge(obs_valid, sim_valid)
    entropy_valid = evaluate_model_entropy(obs_valid, sim_valid, n_bins=20)

    print(f"\n  === 验证期 ===")
    print(f"  KGE: {kge_valid:.4f}")
    print(f"  条件熵: {entropy_valid['H_conditional']:.4f} bits")
    print(f"  归一化条件熵: {entropy_valid['H_conditional_normalized']:.4f}")

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 校准期时间序列
    n_show_calib = min(365, len(dates_calib))
    axes[0, 0].plot(dates_calib[:n_show_calib], obs_calib[:n_show_calib],
                    label='观测', color='blue', linewidth=1.5)
    axes[0, 0].plot(dates_calib[:n_show_calib], sim_calib[:n_show_calib],
                    label='模拟', color='red', linewidth=1.0)
    axes[0, 0].set_ylabel('径流 [mm/day]')
    axes[0, 0].set_title(f'校准期 (KGE={kge_calib:.3f})', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 校准期散点图
    axes[0, 1].scatter(obs_calib, sim_calib, alpha=0.3, s=5)
    max_val = max(np.max(obs_calib), np.max(sim_calib))
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('观测径流 [mm/day]')
    axes[0, 1].set_ylabel('模拟径流 [mm/day]')
    axes[0, 1].set_title('校准期散点图', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal', adjustable='box')

    # 验证期时间序列
    n_show_valid = min(365, len(dates_valid))
    axes[1, 0].plot(dates_valid[:n_show_valid], obs_valid[:n_show_valid],
                    label='观测', color='blue', linewidth=1.5)
    axes[1, 0].plot(dates_valid[:n_show_valid], sim_valid[:n_show_valid],
                    label='模拟', color='red', linewidth=1.0)
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('径流 [mm/day]')
    axes[1, 0].set_title(f'验证期 (KGE={kge_valid:.3f})', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 验证期散点图
    axes[1, 1].scatter(obs_valid, sim_valid, alpha=0.3, s=5)
    max_val = max(np.max(obs_valid), np.max(sim_valid))
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('观测径流 [mm/day]')
    axes[1, 1].set_ylabel('模拟径流 [mm/day]')
    axes[1, 1].set_title('验证期散点图', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output_file = output_dir / f'03_calibration_validation_{model_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   校准验证结果已保存至: {output_file}")

    # 保存参数和指标
    results = {
        'model': model_name,
        'best_params': best_params,
        'kge_calibration': kge_calib,
        'kge_validation': kge_valid,
        'h_conditional_calibration': entropy_calib['H_conditional'],
        'h_conditional_validation': entropy_valid['H_conditional'],
    }

    return results


def main():
    """主函数：真实数据完整工作流"""

    print("\n" + "="*70)
    print("示例3：真实流域数据处理完整工作流")
    print("="*70)

    output_dir = Path('results/examples')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 步骤1：加载数据
    # ========================================
    print("\n[步骤1] 加载流域数据...")

    # 尝试加载真实数据，如果失败则使用合成数据
    data = None
    catchment_name = "示例流域"

    # 尝试从IMPRO数据集加载
    try:
        # 这里需要根据实际数据路径调整
        # data = load_impro_catchment('Iller', data_dir='path/to/IMPRO_data')
        # catchment_name = "Iller"
        # print(f"   ✓ 成功加载真实流域数据: {catchment_name}")
        raise FileNotFoundError("演示：跳过真实数据加载")

    except Exception as e:
        print(f"   ⚠ 未找到真实数据，使用合成数据进行演示")
        print(f"      原因: {e}")

        # 生成合成数据
        data = generate_synthetic_data(
            n_days=3650,  # 10年
            mean_precip=3.0,
            mean_temp=10.0,
            mean_pet=2.0,
            seed=42
        )
        catchment_name = "合成流域"
        print(f"   ✓ 成功生成合成数据")

    # ========================================
    # 步骤2：数据质量检查
    # ========================================
    print("\n[步骤2] 数据质量检查...")
    check_data_quality(data, catchment_name)

    # ========================================
    # 步骤3：数据可视化
    # ========================================
    print("\n[步骤3] 数据可视化...")
    viz_file = output_dir / '03_data_visualization.png'
    visualize_data(data, viz_file)

    # ========================================
    # 步骤4：模型校准和验证
    # ========================================
    print("\n[步骤4] 模型校准和验证...")

    # 使用GR4J模型进行演示
    results = run_calibration_and_validation(data, 'GR4J', output_dir)

    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*70)
    print("示例完成！")
    print("="*70)
    print("\n工作流程总结:")
    print("  ✓ 数据加载和验证")
    print("  ✓ 数据质量检查（缺失值、极端值、水量平衡）")
    print("  ✓ 数据可视化")
    print("  ✓ 模型校准（自动参数优化）")
    print("  ✓ 模型验证（独立数据集）")
    print("\n关键结果:")
    print(f"  模型: {results['model']}")
    print(f"  校准期KGE: {results['kge_calibration']:.4f}")
    print(f"  验证期KGE: {results['kge_validation']:.4f}")
    print("\n实际应用建议:")
    print("  1. 使用真实流域数据替换合成数据")
    print("  2. 增加校准迭代次数（建议5000-10000次）")
    print("  3. 尝试不同的优化算法和目标函数")
    print("  4. 进行敏感性分析和不确定性评估")
    print("  5. 对比多个模型的性能")
    print("="*70)


if __name__ == '__main__':
    main()
