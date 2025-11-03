"""
Visualization utilities for learning curves and model comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_learning_curves(results_df: pd.DataFrame, metric: str = 'NSE',
                         save_path: Optional[str] = None):
    """
    Plot learning curves for all models
    
    Args:
        results_df: DataFrame with learning curve results
        metric: Metric to plot
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for different model types
    process_colors = {'GR4J': '#1f77b4', 'HBV': '#ff7f0e', 'SWAT+': '#2ca02c'}
    data_colors = {'EDDIS': '#d62728', 'RTREE': '#9467bd', 'ANN': '#8c564b', 'LSTM': '#e377c2'}
    colors = {**process_colors, **data_colors}
    
    # Define line styles
    process_style = '-'
    data_style = '--'
    
    # Plot each model
    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name].sort_values('train_size')
        
        # Determine style
        if model_name in process_colors:
            linestyle = process_style
            label_prefix = 'Process: '
            linewidth = 2.5
        else:
            linestyle = data_style
            label_prefix = 'Data-driven: '
            linewidth = 2.5
        
        ax.plot(model_data['train_size'], model_data[metric],
               marker='o', linestyle=linestyle, linewidth=linewidth,
               color=colors.get(model_name, 'gray'),
               label=label_prefix + model_name, markersize=6)
    
    ax.set_xlabel('Training Data Size (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Process-driven vs Data-driven Models', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")
    
    plt.show()


def plot_performance_comparison(results_df: pd.DataFrame, train_sizes: List[int],
                               metric: str = 'NSE', save_path: Optional[str] = None):
    """
    Plot performance comparison at specific training sizes
    
    Args:
        results_df: DataFrame with learning curve results
        train_sizes: List of training sizes to compare
        metric: Metric to plot
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, len(train_sizes), figsize=(5*len(train_sizes), 5))
    
    if len(train_sizes) == 1:
        axes = [axes]
    
    for idx, train_size in enumerate(train_sizes):
        ax = axes[idx]
        
        # Filter data for this training size
        data = results_df[results_df['train_size'] == train_size].copy()
        
        if len(data) == 0:
            continue
        
        # Separate process-driven and data-driven models
        process_models = ['GR4J', 'HBV', 'SWAT+']
        data['model_type'] = data['model'].apply(
            lambda x: 'Process-driven' if x in process_models else 'Data-driven'
        )
        
        # Create bar plot
        models = data['model'].values
        values = data[metric].values
        colors = ['#2ca02c' if m in process_models else '#d62728' for m in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'Training Size: {train_size} days', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    
    plt.show()


def plot_model_type_comparison(results_df: pd.DataFrame, metric: str = 'NSE',
                               save_path: Optional[str] = None):
    """
    Plot comparison between process-driven and data-driven models
    
    Args:
        results_df: DataFrame with learning curve results
        metric: Metric to plot
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Classify models
    process_models = ['GR4J', 'HBV', 'SWAT+']
    results_df['model_type'] = results_df['model'].apply(
        lambda x: 'Process-driven' if x in process_models else 'Data-driven'
    )
    
    # Calculate mean and std for each model type at each training size
    grouped = results_df.groupby(['train_size', 'model_type'])[metric].agg(['mean', 'std'])
    
    for model_type in ['Process-driven', 'Data-driven']:
        data = grouped.xs(model_type, level='model_type')
        
        train_sizes = data.index.values
        means = data['mean'].values
        stds = data['std'].values
        
        color = '#2ca02c' if model_type == 'Process-driven' else '#d62728'
        linestyle = '-' if model_type == 'Process-driven' else '--'
        
        ax.plot(train_sizes, means, marker='o', linestyle=linestyle,
               linewidth=3, label=model_type, color=color, markersize=8)
        ax.fill_between(train_sizes, means - stds, means + stds,
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Training Data Size (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_title('Process-driven vs Data-driven Models: Average Performance',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model type comparison saved to {save_path}")
    
    plt.show()


def plot_efficiency_analysis(efficiency_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot learning efficiency analysis
    
    Args:
        efficiency_df: DataFrame with learning efficiency metrics
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Learning rate
    ax1 = axes[0]
    process_models = ['GR4J', 'HBV', 'SWAT+']
    colors = ['#2ca02c' if m in process_models else '#d62728' 
             for m in efficiency_df['model']]
    
    bars1 = ax1.barh(efficiency_df['model'], efficiency_df['learning_rate'],
                     color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Learning Rate (NSE improvement per log(samples))', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=11, fontweight='bold')
    ax1.set_title('Learning Efficiency: Rate of Improvement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Total improvement
    ax2 = axes[1]
    bars2 = ax2.barh(efficiency_df['model'], efficiency_df['total_improvement'],
                     color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total NSE Improvement', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=11, fontweight='bold')
    ax2.set_title('Total Performance Improvement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Efficiency analysis saved to {save_path}")
    
    plt.show()


def create_summary_table(results_df: pd.DataFrame, metric: str = 'NSE') -> pd.DataFrame:
    """
    Create a summary table of model performance
    
    Args:
        results_df: DataFrame with learning curve results
        metric: Metric to summarize
        
    Returns:
        Summary DataFrame
    """
    summary = results_df.groupby('model').agg({
        metric: ['mean', 'std', 'min', 'max'],
        'train_size': ['min', 'max', 'count']
    }).round(4)
    
    return summary
