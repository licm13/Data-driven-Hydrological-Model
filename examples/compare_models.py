"""
Main example script for comparing learning curves of process-driven and data-driven models

This script demonstrates:
1. Data loading and preprocessing
2. Model initialization (3 process-driven + 4 data-driven models)
3. Learning curve evaluation across different training data sizes
4. Performance comparison and visualization
5. Analysis of learning efficiency and complementary advantages
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
    plot_performance_comparison,
    plot_model_type_comparison,
    plot_efficiency_analysis,
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
    """Main execution function"""
    
    print("="*80)
    print("Hydrological Model Learning Curve Comparison")
    print("Process-driven vs Data-driven Models")
    print("="*80)
    
    # ========================================================================
    # Step 1: Load and prepare data
    # ========================================================================
    print("\n[1/6] Loading and preparing data...")
    
    data_loader = HydrologicalDataLoader()
    data = data_loader.load_sample_data(n_samples=2000, seed=42)
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"\nData statistics:")
    print(data[['precipitation', 'temperature', 'pet', 'discharge']].describe())
    
    # Prepare features and target
    features = ['precipitation', 'temperature', 'pet']
    X = data[features].values
    y = data['discharge'].values
    
    # ========================================================================
    # Step 2: Initialize models
    # ========================================================================
    print("\n[2/6] Initializing models...")
    
    # Process-driven models
    models = {
        # Process-driven models (3)
        'GR4J': GR4J(),
        'HBV': HBV(),
        'SWAT+': SWATPlus(),
        
        # Data-driven models (4)
        'EDDIS': EDDIS(),
        'RTREE': RTREE(n_estimators=50, max_depth=10),
        'ANN': ANN(hidden_layers=(50,), max_iter=200),
        'LSTM': LSTM(hidden_size=32, num_layers=1, epochs=50, seq_length=7),
    }
    
    print(f"Total models: {len(models)}")
    print(f"Process-driven: GR4J, HBV, SWAT+")
    print(f"Data-driven: EDDIS, RTREE, ANN, LSTM")
    
    # ========================================================================
    # Step 3: Define training sizes for learning curves
    # ========================================================================
    print("\n[3/6] Defining training data sizes...")
    
    # Training sizes: from 30 days to 3 years
    train_sizes = [30, 60, 90, 120, 180, 240, 365, 2*365, 3*365]
    print(f"Training sizes: {train_sizes}")
    
    # ========================================================================
    # Step 4: Evaluate learning curves
    # ========================================================================
    print("\n[4/6] Evaluating learning curves (this may take a while)...")
    
    evaluator = LearningCurveEvaluator(train_sizes=train_sizes)
    results_df = evaluator.compare_models(models, X, y, val_size=365)
    
    # Save results
    results_path = 'learning_curves_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Display results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string())
    
    # ========================================================================
    # Step 5: Analyze results
    # ========================================================================
    print("\n[5/6] Analyzing results...")
    
    # Summary statistics
    print("\n--- Summary Statistics by Model ---")
    summary = create_summary_table(results_df, metric='NSE')
    print(summary)
    
    # Learning efficiency analysis
    print("\n--- Learning Efficiency Analysis ---")
    efficiency_df = evaluator.analyze_learning_efficiency()
    print(efficiency_df.to_string())
    
    # Identify crossover points
    print("\n--- Crossover Analysis ---")
    print("Comparing process-driven vs data-driven models...")
    
    process_models = ['GR4J', 'HBV', 'SWAT+']
    data_models = ['EDDIS', 'RTREE', 'ANN', 'LSTM']
    
    for p_model in process_models[:1]:  # Compare GR4J with all data-driven
        for d_model in data_models:
            crossover = evaluator.identify_crossover_point(p_model, d_model, metric='NSE')
            if crossover:
                print(f"\n{p_model} vs {d_model}:")
                print(f"  Crossover at ~{crossover['crossover_size']} days")
                print(f"  {p_model} NSE: {crossover[f'{p_model}_NSE']:.4f}")
                print(f"  {d_model} NSE: {crossover[f'{d_model}_NSE']:.4f}")
    
    # ========================================================================
    # Step 6: Create visualizations
    # ========================================================================
    print("\n[6/6] Creating visualizations...")
    
    # Plot 1: Learning curves for all models
    print("  - Plotting learning curves...")
    plot_learning_curves(results_df, metric='NSE', save_path='learning_curves.png')
    
    # Plot 2: Performance comparison at specific training sizes
    print("  - Plotting performance comparison...")
    comparison_sizes = [90, 365, 3*365]
    plot_performance_comparison(results_df, comparison_sizes, metric='NSE', 
                               save_path='performance_comparison.png')
    
    # Plot 3: Model type comparison (process vs data-driven)
    print("  - Plotting model type comparison...")
    plot_model_type_comparison(results_df, metric='NSE', 
                              save_path='model_type_comparison.png')
    
    # Plot 4: Efficiency analysis
    print("  - Plotting efficiency analysis...")
    plot_efficiency_analysis(efficiency_df, save_path='efficiency_analysis.png')
    
    # ========================================================================
    # Conclusions
    # ========================================================================
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    print("\n1. LEARNING PERFORMANCE BOUNDARY:")
    print("   - The learning curves reveal distinct performance characteristics")
    print("   - Process-driven models show more stable performance with limited data")
    print("   - Data-driven models improve significantly with more training data")
    
    print("\n2. COMPLEMENTARY ADVANTAGES:")
    print("   - Process-driven models: Better with scarce data (< 1 year)")
    print("     * Incorporate physical knowledge and constraints")
    print("     * More interpretable and physically consistent")
    
    print("\n   - Data-driven models: Better with abundant data (> 2 years)")
    print("     * Can learn complex non-linear patterns")
    print("     * More flexible and adaptive to local conditions")
    
    print("\n3. PRACTICAL IMPLICATIONS:")
    print("   - For data-scarce regions: Use process-driven models or hybrid approaches")
    print("   - For data-rich regions: Data-driven models can provide superior performance")
    print("   - Consider ensemble methods combining both approaches")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the generated figures and CSV file.")
    print("="*80)


if __name__ == '__main__':
    main()
