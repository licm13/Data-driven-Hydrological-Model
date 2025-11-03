"""
Compare all three hydrological models (GR4J, HBV, SWAT+)

This script demonstrates how to compare the three process-based models
on the same synthetic dataset.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GR4J, HBV, SWATPlus
from utils import calculate_all_metrics


def main():
    print("=" * 80)
    print("Comparison of GR4J, HBV, and SWAT+ Models")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 365 * 2  # 2 years of data
    
    # Precipitation (mm/day)
    P = np.random.exponential(5, n_days)
    P[P < 2] = 0
    
    # Temperature (°C) - for HBV
    t = np.arange(n_days)
    T = 10 + 15 * np.sin(2 * np.pi * (t - 80) / 365)
    
    # Potential evapotranspiration (mm/day)
    PET = np.maximum(0, 2 + 3 * np.sin(2 * np.pi * (t - 80) / 365))
    
    print(f"\nInput Data:")
    print(f"  Duration: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Mean precipitation: {np.mean(P):.2f} mm/day")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Mean temperature: {np.mean(T):.2f} °C")
    print(f"  Mean PET: {np.mean(PET):.2f} mm/day")
    
    # ========== GR4J Model ==========
    print("\n" + "=" * 80)
    print("1. GR4J Model")
    print("=" * 80)
    
    gr4j_params = {
        'X1': 350.0,
        'X2': 0.0,
        'X3': 90.0,
        'X4': 1.7
    }
    
    print("Parameters:", gr4j_params)
    
    gr4j = GR4J(**gr4j_params)
    gr4j_results = gr4j.run(P, PET)
    Q_gr4j = gr4j_results['Q']
    
    print(f"Results:")
    print(f"  Mean discharge: {np.mean(Q_gr4j):.3f} mm/day")
    print(f"  Max discharge: {np.max(Q_gr4j):.3f} mm/day")
    print(f"  Total discharge: {np.sum(Q_gr4j):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(Q_gr4j)/np.sum(P):.3f}")
    
    # ========== HBV Model ==========
    print("\n" + "=" * 80)
    print("2. HBV Model")
    print("=" * 80)
    
    hbv_params = {
        'TT': 0.0, 'CFMAX': 3.5, 'CFR': 0.05, 'CWH': 0.1,
        'FC': 200.0, 'LP': 0.7, 'BETA': 2.0,
        'PERC': 2.0, 'UZL': 20.0, 'K0': 0.5, 'K1': 0.1, 'K2': 0.05,
        'MAXBAS': 3.0
    }
    
    print("Parameters: FC={}, BETA={}, K1={}, K2={}".format(
        hbv_params['FC'], hbv_params['BETA'], hbv_params['K1'], hbv_params['K2']))
    
    hbv = HBV(hbv_params)
    hbv_results = hbv.run(P, T, PET)
    Q_hbv = hbv_results['Q']
    
    print(f"Results:")
    print(f"  Mean discharge: {np.mean(Q_hbv):.3f} mm/day")
    print(f"  Max discharge: {np.max(Q_hbv):.3f} mm/day")
    print(f"  Total discharge: {np.sum(Q_hbv):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(Q_hbv)/np.sum(P):.3f}")
    print(f"  Days with snow: {np.sum(hbv_results['SP'] > 0)}")
    
    # ========== SWAT+ Model ==========
    print("\n" + "=" * 80)
    print("3. SWAT+ Model")
    print("=" * 80)
    
    swat_params = {
        'CN2': 75.0, 'ESCO': 0.95, 'EPCO': 1.0,
        'SURLAG': 4.0, 'ALPHA_BF': 0.048, 'GW_DELAY': 31.0,
        'GW_REVAP': 0.02, 'REVAPMN': 750.0, 'RCHRG_DP': 0.05,
        'GWQMN': 1000.0, 'SOL_AWC': 0.15, 'SOL_Z': 1000.0
    }
    
    print("Parameters: CN2={}, ALPHA_BF={}, SOL_AWC={}".format(
        swat_params['CN2'], swat_params['ALPHA_BF'], swat_params['SOL_AWC']))
    
    swat = SWATPlus(swat_params)
    swat_results = swat.run(P, PET)
    Q_swat = swat_results['Q_total']
    
    print(f"Results:")
    print(f"  Mean total discharge: {np.mean(Q_swat):.3f} mm/day")
    print(f"  Mean surface runoff: {np.mean(swat_results['Q_surf']):.3f} mm/day")
    print(f"  Mean baseflow: {np.mean(swat_results['Q_gw']):.3f} mm/day")
    print(f"  Total discharge: {np.sum(Q_swat):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(Q_swat)/np.sum(P):.3f}")
    
    # ========== Comparison ==========
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format(
        "Metric", "GR4J", "HBV", "SWAT+"))
    print("-" * 80)
    
    print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f}".format(
        "Mean Q (mm/day)", np.mean(Q_gr4j), np.mean(Q_hbv), np.mean(Q_swat)))
    
    print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f}".format(
        "Max Q (mm/day)", np.max(Q_gr4j), np.max(Q_hbv), np.max(Q_swat)))
    
    print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
        "Total Q (mm)", np.sum(Q_gr4j), np.sum(Q_hbv), np.sum(Q_swat)))
    
    print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f}".format(
        "Runoff Coeff.", 
        np.sum(Q_gr4j)/np.sum(P), 
        np.sum(Q_hbv)/np.sum(P), 
        np.sum(Q_swat)/np.sum(P)))
    
    # Correlation between models
    print("\n" + "=" * 80)
    print("Inter-model Correlations")
    print("=" * 80)
    
    corr_gr4j_hbv = np.corrcoef(Q_gr4j, Q_hbv)[0, 1]
    corr_gr4j_swat = np.corrcoef(Q_gr4j, Q_swat)[0, 1]
    corr_hbv_swat = np.corrcoef(Q_hbv, Q_swat)[0, 1]
    
    print(f"GR4J vs HBV:    {corr_gr4j_hbv:.3f}")
    print(f"GR4J vs SWAT+:  {corr_gr4j_swat:.3f}")
    print(f"HBV vs SWAT+:   {corr_hbv_swat:.3f}")
    
    print("\n" + "=" * 80)
    print("Comparison completed successfully!")
    print("=" * 80)
    
    print("\nKey Observations:")
    print("- All three models are process-based hydrological models")
    print("- GR4J is the simplest with only 4 parameters")
    print("- HBV includes explicit snow modeling for cold climates")
    print("- SWAT+ separates surface runoff and baseflow components")
    print("- Each model has different strengths for different applications")


if __name__ == "__main__":
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
