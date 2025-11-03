"""
Example usage of HBV model

This script demonstrates how to use the HBV conceptual hydrological model
with synthetic data including temperature for snow modeling.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import HBV
from utils import calculate_all_metrics


def main():
    print("=" * 60)
    print("HBV Model Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 365
    
    # Precipitation (mm/day)
    P = np.random.exponential(5, n_days)
    P[P < 2] = 0
    
    # Temperature (°C) - seasonal pattern
    t = np.arange(n_days)
    T = 10 + 15 * np.sin(2 * np.pi * (t - 80) / 365)  # Peaks in summer
    
    # Potential evapotranspiration (mm/day)
    PET = np.maximum(0, 2 + 3 * np.sin(2 * np.pi * (t - 80) / 365))
    
    print(f"\nInput Data:")
    print(f"  Number of days: {n_days}")
    print(f"  Mean precipitation: {np.mean(P):.2f} mm/day")
    print(f"  Mean temperature: {np.mean(T):.2f} °C")
    print(f"  Mean PET: {np.mean(PET):.2f} mm/day")
    
    # Initialize HBV model with typical parameters
    params = {
        # Snow routine
        'TT': 0.0,          # Threshold temperature (°C)
        'CFMAX': 3.5,       # Degree-day factor (mm/°C/day)
        'CFR': 0.05,        # Refreezing coefficient
        'CWH': 0.1,         # Water holding capacity of snow
        
        # Soil routine
        'FC': 200.0,        # Field capacity (mm)
        'LP': 0.7,          # Limit for potential evapotranspiration
        'BETA': 2.0,        # Shape coefficient
        
        # Response routine
        'PERC': 2.0,        # Percolation rate (mm/day)
        'UZL': 20.0,        # Threshold for quick runoff (mm)
        'K0': 0.5,          # Recession coefficient for quick flow
        'K1': 0.1,          # Recession coefficient for upper zone
        'K2': 0.05,         # Recession coefficient for lower zone
        
        # Routing
        'MAXBAS': 3.0       # Routing parameter (days)
    }
    
    print(f"\nModel Parameters:")
    print(f"  TT (Threshold temperature): {params['TT']} °C")
    print(f"  CFMAX (Degree-day factor): {params['CFMAX']} mm/°C/day")
    print(f"  FC (Field capacity): {params['FC']} mm")
    print(f"  BETA (Shape coefficient): {params['BETA']}")
    
    model = HBV(params)
    
    # Run the model
    print("\nRunning HBV model...")
    results = model.run(P, T, PET)
    
    # Extract results
    Q = results['Q']
    SP = results['SP']
    SM = results['SM']
    SUZ = results['SUZ']
    SLZ = results['SLZ']
    
    print("\nModel Results:")
    print(f"  Mean discharge: {np.mean(Q):.2f} mm/day")
    print(f"  Max discharge: {np.max(Q):.2f} mm/day")
    print(f"  Mean snow pack: {np.mean(SP):.2f} mm")
    print(f"  Mean soil moisture: {np.mean(SM):.2f} mm")
    print(f"  Mean upper zone storage: {np.mean(SUZ):.2f} mm")
    print(f"  Mean lower zone storage: {np.mean(SLZ):.2f} mm")
    
    # Find snow accumulation period
    snow_days = np.sum(SP > 0)
    print(f"\nSnow Statistics:")
    print(f"  Days with snow cover: {snow_days}")
    print(f"  Maximum snow pack: {np.max(SP):.2f} mm")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
