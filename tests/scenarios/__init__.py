"""
tests/scenarios/__init__.py

Advanced Application Test Cases for Hydrological Models
========================================================

This module contains complex scenario tests to validate system robustness:

1. Climate Stress Test - Testing model behavior under climate change scenarios
2. PUB (Prediction in Ungauged Basins) - Cross-basin transfer learning
3. Extreme Event Robustness - Testing model constraints under extreme conditions
"""

from .test_climate_stress import ClimateStressTest
from .test_pub_prediction import PUBPredictionTest
from .test_extreme_events import ExtremeEventRobustnessTest

__all__ = [
    'ClimateStressTest',
    'PUBPredictionTest', 
    'ExtremeEventRobustnessTest'
]
