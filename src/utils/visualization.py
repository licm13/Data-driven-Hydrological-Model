"""
可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# 模型颜色方案（与论文一致）
MODEL_COLORS = {
    'EDDIS': '#D4A574',
    'RTREE': '#C4915E',
    'ANN': '#E8857E',
    'LSTM': '#D85C5A',
    'GR4J': '#7EC4CF',
    'HBV': '#5BA4CF',
    'SWAT+': '#4A7BA7',
}

MODEL_ORDER = ['EDDIS', 'RTREE', 'ANN', 'LSTM', 'GR4J', 'HBV', 'SWAT+']


def plot_learning_curves(results: Dict[str, Dict],
                        catchment_name: str,
                        metric: str = 'H_conditional',
                        save_path: Optional[str] = None):
    """
    绘制学习曲线
    
    Parameters:
    -----------
    results : dict, {model_name: {sample_size: {metric: value, ...}}}
    catchment_name : str, 流域名称
    metric : str, 要绘制的指标
    save_path : str, 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取所有样本量
    sample_sizes = None
    for model_name, model_results in results.items():
        if sample_sizes is None:
            sample_sizes = sorted(model_results.keys())
    
    # 绘制每个模型
    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        
        model_results = results[model_name]
        
        # 提取数据
        values = []
        lower = []
        upper = []
        
        for size in sample_sizes:
            if size not in model_results:
                continue
            
            res = model_results[size]
            
            if isinstance(res[metric], (list, np.ndarray)):
                # 多次重复的结果
                values.append(np.median(res[metric]))
                lower.append(np.percentile(res[metric], 25))
                upper.append(np.percentile(res[metric], 75))
            else:
                values.append(res[metric])
                lower.append(res[metric])
                upper.append(res[metric])
        
        # 绘制
        color = MODEL_COLORS.get(model_name, 'gray')
        ax.plot(sample_sizes[:len(values)], values, 
                marker='o', label=model_name, color=color, linewidth=2)
        ax.fill_between(sample_sizes[:len(values)], lower, upper, 
                        alpha=0.2, color=color)
    
    # 添加最大熵基准线（对于条件熵）
    if metric == 'H_conditional':
        # 计算观测的边际熵作为上限
        ax.axhline(y=3.0, color='black', linestyle='--', 
                  label='Max Entropy', alpha=0.5)
    
    ax.set_xlabel('Training Sample Size (days)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title() + ' (bits)', fontsize=12)
    ax.set_title(f'Learning Curves - {catchment_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_relative_learning(results: Dict[str, float],
                          catchment_names: List[str],
                          save_path: Optional[str] = None):
    """
    绘制相对学习能力对比
    
    Parameters:
    -----------
    results : dict, {model_name: relative_learning_value}
    catchment_names : list, 流域名称列表
    save_path : str, 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    models = []
    values = []
    colors = []
    
    for model_name in MODEL_ORDER:
        if model_name in results:
            models.append(model_name)
            values.append(results[model_name])
            colors.append(MODEL_COLORS.get(model_name, 'gray'))
    
    # 水平条形图
    y_pos = np.arange(len(models))
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Relative Learning [-]', fontsize=12)
    ax.set_title('Relative Learning by Model', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hydrograph(obs: np.ndarray,
                   sim: Dict[str, np.ndarray],
                   dates: pd.DatetimeIndex,
                   title: str = 'Hydrograph Comparison',
                   save_path: Optional[str] = None):
    """
    绘制水文过程线对比
    
    Parameters:
    -----------
    obs : array, 观测径流
    sim : dict, {model_name: simulated_discharge}
    dates : DatetimeIndex, 日期
    title : str, 标题
    save_path : str, 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 观测值
    ax.plot(dates, obs, color='black', linewidth=1.5, 
            label='Observed', alpha=0.7)
    
    # 模拟值
    for model_name, sim_values in sim.items():
        color = MODEL_COLORS.get(model_name, 'gray')
        ax.plot(dates, sim_values, color=color, linewidth=1, 
                label=model_name, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Discharge (mm/day)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sampling_strategy_comparison(results: Dict[str, Dict],
                                     catchment_name: str,
                                     save_path: Optional[str] = None):
    """
    绘制采样策略对比（实验2）
    
    Parameters:
    -----------
    results : dict, {strategy: {sample_size: metric_value}}
    catchment_name : str, 流域名称
    save_path : str, 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['random', 'consecutive', 'douglas_peucker']
    strategy_labels = ['Fully Random', 'Random Consecutive', 'Douglas-Peucker']
    colors = ['#E8857E', '#7EC4CF', '#A3A3A3']
    
    sample_sizes = None
    
    for strategy, label, color in zip(strategies, strategy_labels, colors):
        if strategy not in results:
            continue
        
        data = results[strategy]
        
        if sample_sizes is None:
            sample_sizes = sorted(data.keys())
        
        values = []
        lower = []
        upper = []
        
        for size in sample_sizes:
            if isinstance(data[size], (list, np.ndarray)):
                values.append(np.median(data[size]))
                lower.append(np.percentile(data[size], 25))
                upper.append(np.percentile(data[size], 75))
            else:
                values.append(data[size])
                lower.append(data[size])
                upper.append(data[size])
        
        ax.plot(sample_sizes, values, marker='o', label=label, 
                color=color, linewidth=2)
        ax.fill_between(sample_sizes, lower, upper, alpha=0.2, color=color)
    
    ax.set_xlabel('Sample Size (days)', fontsize=12)
    ax.set_ylabel('Conditional Entropy (bits)', fontsize=12)
    ax.set_title(f'Sampling Strategy Comparison - {catchment_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_information_content(joint_entropy: Dict[str, float],
                            conditional_entropy: Dict[str, float],
                            learning: Dict[str, float],
                            save_path: Optional[str] = None):
    """
    绘制信息内容分析（实验3）
    
    Parameters:
    -----------
    joint_entropy : dict, {catchment: H_joint}
    conditional_entropy : dict, {catchment: H_conditional}
    learning : dict, {catchment: learning_value}
    save_path : str, 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    catchments = list(joint_entropy.keys())
    
    # 子图1：联合熵 vs 条件熵
    x = [joint_entropy[c] for c in catchments]
    y = [conditional_entropy[c] for c in catchments]
    
    ax1.scatter(x, y, s=100, alpha=0.7)
    
    for i, catchment in enumerate(catchments):
        ax1.annotate(catchment, (x[i], y[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    ax1.set_xlabel('Joint Entropy H(P, T, PET, Q) [bits]', fontsize=11)
    ax1.set_ylabel('Conditional Entropy H(Q | P, T, PET) [bits]', fontsize=11)
    ax1.set_title('Data Variability vs. Learnability', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：条件熵 vs 学习能力
    y2 = [learning[c] for c in catchments]
    
    ax2.scatter(y, y2, s=100, alpha=0.7, color='coral')
    
    for i, catchment in enumerate(catchments):
        ax2.annotate(catchment, (y[i], y2[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    ax2.set_xlabel('Conditional Entropy H(Q | P, T, PET) [bits]', fontsize=11)
    ax2.set_ylabel('Average Model Learning [-]', fontsize=11)
    ax2.set_title('Learnability vs. Model Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_scatter_obs_sim(obs: np.ndarray,
                        sim: Dict[str, np.ndarray],
                        save_path: Optional[str] = None):
    """
    绘制观测 vs 模拟散点图
    
    Parameters:
    -----------
    obs : array, 观测值
    sim : dict, {model_name: simulated_values}
    save_path : str, 保存路径
    """
    n_models = len(sim)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (model_name, sim_values) in enumerate(sim.items()):
        ax = axes[i]
        
        # 移除NaN
        mask = ~(np.isnan(obs) | np.isnan(sim_values))
        obs_clean = obs[mask]
        sim_clean = sim_values[mask]
        
        # 散点图
        ax.scatter(obs_clean, sim_clean, alpha=0.3, s=10, 
                  color=MODEL_COLORS.get(model_name, 'gray'))
        
        # 1:1线
        max_val = max(np.max(obs_clean), np.max(sim_clean))
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        
        # 计算指标
        from ..metrics.kge import kge
        kge_val = kge(obs_clean, sim_clean)
        
        ax.set_xlabel('Observed (mm/day)', fontsize=10)
        ax.set_ylabel('Simulated (mm/day)', fontsize=10)
        ax.set_title(f'{model_name} (KGE={kge_val:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # 隐藏多余的子图
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()