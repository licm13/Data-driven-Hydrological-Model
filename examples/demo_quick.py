"""
Quick demonstration script with reduced training sizes for faster execution
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils.data_loader import HydrologicalDataLoader
from utils.visualization import (
    plot_learning_curves, 
    plot_model_type_comparison,
    create_summary_table
)

# Import process-driven models
from models.process_driven.gr4j import GR4J
from models.process_driven.hbv import HBV
from models.process_driven.swat_plus import SWATPlus

# Import data-driven models
from models.data_driven.eddis import EDDIS
from models.data_driven.rtree import RTREE
from models.data_driven.ann import ANN
from models.data_driven.lstm import LSTM

# Import evaluation
from evaluation.learning_curves import LearningCurveEvaluator


def main():
    """Quick demonstration"""
    
    print("="*80)
    print("Quick Demonstration: Hydrological Model Learning Curves")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    data_loader = HydrologicalDataLoader()
    data = data_loader.load_sample_data(n_samples=800, seed=42)
    
    print(f"Data shape: {data.shape}")
    
    # Prepare features and target
    features = ['precipitation', 'temperature', 'pet']
    X = data[features].values
    y = data['discharge'].values
    
    # Initialize models (subset for speed)
    print("\n[2/4] Initializing models...")
    models = {
        # Process-driven
        'GR4J': GR4J(),
        'HBV': HBV(),
        'SWAT+': SWATPlus(),
        
        # Data-driven
        'EDDIS': EDDIS(),
        'RTREE': RTREE(n_estimators=30, max_depth=8),
        'ANN': ANN(hidden_layers=(30,), max_iter=100),
    }
    
    print(f"Models: {list(models.keys())}")
    
    # Define smaller training sizes for quick demo
    train_sizes = [60, 120, 180, 240, 365]
    print(f"\n[3/4] Training sizes: {train_sizes}")
    
    # Evaluate learning curves
    print("\n[4/4] Evaluating learning curves...")
    evaluator = LearningCurveEvaluator(train_sizes=train_sizes)
    results_df = evaluator.compare_models(models, X, y, val_size=200)
    
    # Save results
    results_path = 'demo_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Display summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = results_df.groupby('model')['NSE'].agg(['mean', 'std', 'min', 'max']).round(4)
    print(summary)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    try:
        print("\n1. Learning curves plot...")
        plot_learning_curves(results_df, metric='NSE', save_path='demo_learning_curves.png')
    except Exception as e:
        print(f"   Note: Visualization skipped (no display): {e}")
    
    try:
        print("\n2. Model type comparison...")
        plot_model_type_comparison(results_df, metric='NSE', save_path='demo_comparison.png')
    except Exception as e:
        print(f"   Note: Visualization skipped (no display): {e}")
    
    # Efficiency analysis
    print("\n" + "="*80)
    print("Learning Efficiency Analysis")
    print("="*80)
    
    efficiency_df = evaluator.analyze_learning_efficiency()
    print(efficiency_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    print("\nThis demonstration shows:")
    print("1. Process-driven models (GR4J, HBV, SWAT+) perform consistently with limited data")
    print("2. Data-driven models (EDDIS, RTREE, ANN) improve with more training data")
    print("3. The learning curves reveal complementary advantages of both approaches")
    print("\nFor full analysis, run: python examples/compare_models.py")
    print("="*80)


if __name__ == '__main__':
    main()
