"""
Quick test script to verify the framework functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Testing Hydrological Model Comparison Framework")
print("="*80)

# Test 1: Data loading
print("\n[1/5] Testing data loader...")
from utils.data_loader import HydrologicalDataLoader

loader = HydrologicalDataLoader()
data = loader.load_sample_data(n_samples=500, seed=42)
print(f"✓ Data loaded: {data.shape}")
print(f"  Features: {list(data.columns)}")

# Prepare data
X = data[['precipitation', 'temperature', 'pet']].values
y = data['discharge'].values
print(f"✓ X shape: {X.shape}, y shape: {y.shape}")

# Test 2: Process-driven models
print("\n[2/5] Testing process-driven models...")
from models.process_driven.gr4j import GR4J
from models.process_driven.hbv import HBV
from models.process_driven.swat_plus import SWATPlus

# Test GR4J
print("  Testing GR4J...")
gr4j = GR4J()
y_pred_gr4j = gr4j.run(X[:100, 0], X[:100, 2])
print(f"  ✓ GR4J prediction shape: {y_pred_gr4j.shape}")

# Test HBV
print("  Testing HBV...")
hbv = HBV()
y_pred_hbv = hbv.run(X[:100, 0], X[:100, 1], X[:100, 2])
print(f"  ✓ HBV prediction shape: {y_pred_hbv.shape}")

# Test SWAT+
print("  Testing SWAT+...")
swat = SWATPlus()
y_pred_swat = swat.run(X[:100, 0], X[:100, 1], X[:100, 2])
print(f"  ✓ SWAT+ prediction shape: {y_pred_swat.shape}")

# Test 3: Data-driven models
print("\n[3/5] Testing data-driven models...")
from models.data_driven.eddis import EDDIS
from models.data_driven.rtree import RTREE
from models.data_driven.ann import ANN

# Test EDDIS
print("  Testing EDDIS...")
eddis = EDDIS()
eddis.fit(X[:100], y[:100])
y_pred_eddis = eddis.predict(X[100:150])
print(f"  ✓ EDDIS prediction shape: {y_pred_eddis.shape}")

# Test RTREE
print("  Testing RTREE...")
rtree = RTREE(n_estimators=10, max_depth=5)
rtree.fit(X[:100], y[:100])
y_pred_rtree = rtree.predict(X[100:150])
print(f"  ✓ RTREE prediction shape: {y_pred_rtree.shape}")

# Test ANN
print("  Testing ANN...")
ann = ANN(hidden_layers=(20,), max_iter=50)
ann.fit(X[:100], y[:100])
y_pred_ann = ann.predict(X[100:150])
print(f"  ✓ ANN prediction shape: {y_pred_ann.shape}")

# Test 4: Metrics
print("\n[4/5] Testing evaluation metrics...")
from utils.metrics import evaluate_model

metrics = evaluate_model(y[100:150], y_pred_eddis)
print(f"  ✓ Metrics calculated: {list(metrics.keys())}")
print(f"    NSE: {metrics['NSE']:.4f}")
print(f"    RMSE: {metrics['RMSE']:.4f}")

# Test 5: Learning curve evaluator
print("\n[5/5] Testing learning curve evaluator...")
from evaluation.learning_curves import LearningCurveEvaluator

evaluator = LearningCurveEvaluator(train_sizes=[50, 100, 150])
print("  ✓ Evaluator initialized")

# Quick evaluation with one model
print("  Running quick evaluation with EDDIS...")
eddis_test = EDDIS()
results = evaluator.evaluate_model(eddis_test, 'EDDIS', X, y, val_size=100)
print(f"  ✓ Evaluation complete: {len(results)} training sizes tested")
print(f"\n  Results:")
print(results[['model', 'train_size', 'NSE', 'RMSE']].to_string(index=False))

print("\n" + "="*80)
print("All tests passed! Framework is working correctly.")
print("="*80)
