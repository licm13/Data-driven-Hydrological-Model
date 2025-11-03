"""
Simple tests to verify model implementations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from models import GR4J, HBV, SWATPlus
from utils import nse, rmse, pbias, kge


def test_gr4j():
    """Test GR4J model basic functionality"""
    print("Testing GR4J model...")
    
    # Simple test data
    P = np.array([10, 0, 5, 0, 8, 0, 0, 3, 0, 0])
    E = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    
    model = GR4J(X1=300, X2=0, X3=80, X4=1.5)
    results = model.run(P, E)
    
    assert 'Q' in results
    assert 'S' in results
    assert 'R' in results
    assert len(results['Q']) == len(P)
    assert np.all(results['Q'] >= 0)
    
    print("  ✓ GR4J model working correctly")


def test_hbv():
    """Test HBV model basic functionality"""
    print("Testing HBV model...")
    
    # Simple test data
    P = np.array([10, 0, 5, 0, 8, 0, 0, 3, 0, 0])
    T = np.array([5, 3, 4, 2, 6, 8, 10, 12, 11, 9])
    E = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    
    params = {
        'TT': 0, 'CFMAX': 3.5, 'CFR': 0.05, 'CWH': 0.1,
        'FC': 200, 'LP': 0.7, 'BETA': 2.0,
        'PERC': 2.0, 'UZL': 20, 'K0': 0.5, 'K1': 0.1, 'K2': 0.05,
        'MAXBAS': 3.0
    }
    
    model = HBV(params)
    results = model.run(P, T, E)
    
    assert 'Q' in results
    assert 'SP' in results
    assert 'SM' in results
    assert len(results['Q']) == len(P)
    assert np.all(results['Q'] >= 0)
    
    print("  ✓ HBV model working correctly")


def test_swatplus():
    """Test SWAT+ model basic functionality"""
    print("Testing SWAT+ model...")
    
    # Simple test data
    P = np.array([10, 0, 5, 0, 8, 0, 0, 3, 0, 0])
    E = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    
    params = {
        'CN2': 75, 'ALPHA_BF': 0.048, 'GW_DELAY': 31,
        'SOL_AWC': 0.15, 'SOL_Z': 1000
    }
    
    model = SWATPlus(params)
    results = model.run(P, E)
    
    assert 'Q_total' in results
    assert 'Q_surf' in results
    assert 'Q_gw' in results
    assert len(results['Q_total']) == len(P)
    assert np.all(results['Q_total'] >= 0)
    
    print("  ✓ SWAT+ model working correctly")


def test_metrics():
    """Test utility metrics"""
    print("Testing utility metrics...")
    
    observed = np.array([1, 2, 3, 4, 5])
    simulated = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    nse_val = nse(observed, simulated)
    rmse_val = rmse(observed, simulated)
    pbias_val = pbias(observed, simulated)
    kge_val = kge(observed, simulated)
    
    assert -1 <= nse_val <= 1
    assert rmse_val >= 0
    assert -100 <= pbias_val <= 100
    assert kge_val <= 1
    
    print("  ✓ Utility metrics working correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Model Tests")
    print("=" * 60)
    print()
    
    test_gr4j()
    test_hbv()
    test_swatplus()
    test_metrics()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
