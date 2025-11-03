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
    main()
