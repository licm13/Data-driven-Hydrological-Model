"""
Example usage of SWAT+ model

This script demonstrates how to use the SWAT+ watershed model
with synthetic data.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import SWATPlus
from utils import calculate_all_metrics


def main():
    print("=" * 60)
    print("SWAT+ Model Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 365
    
    # Precipitation (mm/day)
    P = np.random.exponential(5, n_days)
    P[P < 2] = 0
    
    # Potential evapotranspiration (mm/day) - seasonal pattern
    t = np.arange(n_days)
    PET = 3 + 2 * np.sin(2 * np.pi * t / 365)
    
    print(f"\nInput Data:")
    print(f"  Number of days: {n_days}")
    print(f"  Mean precipitation: {np.mean(P):.2f} mm/day")
    print(f"  Mean PET: {np.mean(PET):.2f} mm/day")
    
    # Initialize SWAT+ model with typical parameters
    params = {
        'CN2': 75.0,           # SCS Curve Number
        'ESCO': 0.95,          # Soil evaporation compensation factor
        'EPCO': 1.0,           # Plant uptake compensation factor
        'SURLAG': 4.0,         # Surface runoff lag (days)
        'ALPHA_BF': 0.048,     # Baseflow recession constant (1/days)
        'GW_DELAY': 31.0,      # Groundwater delay (days)
        'GW_REVAP': 0.02,      # Groundwater revap coefficient
        'REVAPMN': 750.0,      # Revap threshold (mm)
        'RCHRG_DP': 0.05,      # Deep aquifer percolation fraction
        'GWQMN': 1000.0,       # Baseflow threshold (mm)
        'SOL_AWC': 0.15,       # Available water capacity (mm/mm)
        'SOL_Z': 1000.0        # Soil depth (mm)
    }
    
    print(f"\nModel Parameters:")
    print(f"  CN2 (Curve Number): {params['CN2']}")
    print(f"  ALPHA_BF (Baseflow recession): {params['ALPHA_BF']} 1/days")
    print(f"  GW_DELAY (GW delay): {params['GW_DELAY']} days")
    print(f"  SOL_AWC (Available water capacity): {params['SOL_AWC']} mm/mm")
    print(f"  SOL_Z (Soil depth): {params['SOL_Z']} mm")
    
    model = SWATPlus(params)
    
    # Run the model
    print("\nRunning SWAT+ model...")
    results = model.run(P, PET)
    
    # Extract results
    Q_total = results['Q_total']
    Q_surf = results['Q_surf']
    Q_gw = results['Q_gw']
    SW = results['SW']
    GW = results['GW']
    
    print("\nModel Results:")
    print(f"  Mean total discharge: {np.mean(Q_total):.2f} mm/day")
    print(f"  Mean surface runoff: {np.mean(Q_surf):.2f} mm/day")
    print(f"  Mean baseflow: {np.mean(Q_gw):.2f} mm/day")
    print(f"  Mean soil water: {np.mean(SW):.2f} mm")
    print(f"  Mean groundwater storage: {np.mean(GW):.2f} mm")
    
    # Flow components
    total_surf = np.sum(Q_surf)
    total_gw = np.sum(Q_gw)
    total_Q = np.sum(Q_total)
    
    print("\nFlow Components:")
    print(f"  Surface runoff: {total_surf:.2f} mm ({100*total_surf/total_Q:.1f}%)")
    print(f"  Baseflow: {total_gw:.2f} mm ({100*total_gw/total_Q:.1f}%)")
    print(f"  Total discharge: {total_Q:.2f} mm")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
