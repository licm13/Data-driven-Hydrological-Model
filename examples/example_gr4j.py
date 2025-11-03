"""
Example usage of GR4J model

This script demonstrates how to use the GR4J rainfall-runoff model
with synthetic data.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GR4J
from utils import calculate_all_metrics


def main():
    print("=" * 60)
    print("GR4J Model Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 365
    
    # Precipitation (mm/day) - random with some wet periods
    P = np.random.exponential(5, n_days)
    P[P < 2] = 0  # Dry days
    
    # Potential evapotranspiration (mm/day) - seasonal pattern
    t = np.arange(n_days)
    PET = 3 + 2 * np.sin(2 * np.pi * t / 365)
    
    print(f"\nInput Data:")
    print(f"  Number of days: {n_days}")
    print(f"  Mean precipitation: {np.mean(P):.2f} mm/day")
    print(f"  Mean PET: {np.mean(PET):.2f} mm/day")
    
    # Initialize GR4J model with typical parameters
    print(f"\nModel Parameters:")
    X1 = 350.0  # Production store capacity (mm)
    X2 = 0.0    # Groundwater exchange coefficient (mm/day)
    X3 = 90.0   # Routing store capacity (mm)
    X4 = 1.7    # Unit hydrograph time base (days)
    
    print(f"  X1 (Production store): {X1} mm")
    print(f"  X2 (Groundwater exchange): {X2} mm/day")
    print(f"  X3 (Routing store): {X3} mm")
    print(f"  X4 (Unit hydrograph time base): {X4} days")
    
    model = GR4J(X1=X1, X2=X2, X3=X3, X4=X4)
    
    # Run the model
    print("\nRunning GR4J model...")
    results = model.run(P, PET)
    
    # Extract results
    Q = results['Q']
    S = results['S']
    R = results['R']
    
    print("\nModel Results:")
    print(f"  Mean discharge: {np.mean(Q):.2f} mm/day")
    print(f"  Max discharge: {np.max(Q):.2f} mm/day")
    print(f"  Mean production store level: {np.mean(S):.2f} mm")
    print(f"  Mean routing store level: {np.mean(R):.2f} mm")
    
    # Water balance check
    total_P = np.sum(P)
    total_ET = np.sum(PET)
    total_Q = np.sum(Q)
    storage_change = S[-1] + R[-1]
    
    print("\nWater Balance:")
    print(f"  Total precipitation: {total_P:.2f} mm")
    print(f"  Total discharge: {total_Q:.2f} mm")
    print(f"  Final storage: {storage_change:.2f} mm")
    print(f"  Balance (P - Q - ΔS): {total_P - total_Q - storage_change:.2f} mm")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
