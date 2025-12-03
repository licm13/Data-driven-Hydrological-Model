"""
Climate Stress Test (气候变化压力测试)
======================================

This module tests hydrological model behavior under climate change scenarios.

Scenarios:
----------
1. Temperature increase (+2°C warming)
2. Precipitation variability increase (+20%)
3. Combined scenario

Key Questions:
--------------
- Can physical models (HBV) capture earlier snowmelt due to warming?
- Do data-driven models (LSTM) fail when inputs exceed training distribution (OOD)?
"""

import numpy as np
import unittest
from typing import Dict, Tuple, Optional
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ClimateModifier:
    """
    Data loader decorator that modifies climate inputs dynamically.
    
    This class wraps around the data loading process and applies
    climate change scenarios to the input data.
    
    Parameters:
    -----------
    temp_increase : float
        Temperature increase in °C (default: 0, no change)
    precip_variability_factor : float
        Factor to increase precipitation variability (default: 1.0, no change)
        Values > 1.0 increase variability
    """
    
    def __init__(self, 
                 temp_increase: float = 0.0, 
                 precip_variability_factor: float = 1.0,
                 seed: int = 42):
        self.temp_increase = temp_increase
        self.precip_variability_factor = precip_variability_factor
        self.seed = seed
        np.random.seed(seed)
        
    def modify_temperature(self, temp: np.ndarray) -> np.ndarray:
        """
        Apply temperature increase uniformly.
        
        This simulates a warming climate scenario where all
        temperatures are shifted by a constant amount.
        """
        return temp + self.temp_increase
    
    def modify_precipitation(self, precip: np.ndarray) -> np.ndarray:
        """
        Increase precipitation variability while preserving mean.
        
        This simulates a scenario where extreme events become more frequent
        while the total annual precipitation remains similar.
        
        Method: Scale deviations from the mean by the variability factor
        """
        if self.precip_variability_factor == 1.0:
            return precip
        
        mean_precip = np.mean(precip)
        deviations = precip - mean_precip
        modified_deviations = deviations * self.precip_variability_factor
        modified_precip = mean_precip + modified_deviations
        
        # Ensure non-negative precipitation
        return np.maximum(0, modified_precip)
    
    def apply(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply climate modifications to input data dictionary.
        
        Parameters:
        -----------
        data : dict
            Dictionary with 'precip', 'temp', and optionally 'pet'
            
        Returns:
        --------
        modified_data : dict
            Modified data dictionary
        """
        modified = data.copy()
        
        if 'temp' in data:
            modified['temp'] = self.modify_temperature(data['temp'])
            
        if 'precip' in data:
            modified['precip'] = self.modify_precipitation(data['precip'])
            
        # Recalculate PET if temperature changed and PET exists
        if 'pet' in data and self.temp_increase != 0:
            # Simple adjustment: PET increases with temperature
            # Using a simplified relationship: ~4% increase per °C
            pet_adjustment = 1 + 0.04 * self.temp_increase
            modified['pet'] = data['pet'] * pet_adjustment
            
        return modified


def generate_synthetic_climate_data(n_years: int = 10) -> Dict[str, np.ndarray]:
    """
    Generate synthetic climate data for testing.
    
    Generates realistic-looking hydrological data including:
    - Seasonal temperature patterns
    - Random precipitation events
    - PET correlated with temperature
    """
    n_days = n_years * 365
    day_of_year = np.tile(np.arange(365), n_years)
    
    # Temperature: seasonal cycle + random variation
    temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 91) / 365)
    temp += np.random.normal(0, 3, n_days)
    
    # Precipitation: random events with seasonal variation
    precip = np.zeros(n_days)
    rain_prob = 0.3 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
    for i in range(n_days):
        if np.random.random() < rain_prob[i]:
            precip[i] = np.random.exponential(scale=10)
    
    # PET: temperature-dependent
    pet = np.maximum(0, 0.5 + 0.2 * temp + np.random.normal(0, 0.5, n_days))
    
    return {
        'temp': temp,
        'precip': precip,
        'pet': pet
    }


class ClimateStressTest(unittest.TestCase):
    """
    Test suite for climate stress testing of hydrological models.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and models."""
        np.random.seed(42)
        cls.base_data = generate_synthetic_climate_data(n_years=5)
        
        # Try to import models
        try:
            from src.models.hbv import HBV
            cls.hbv_available = True
        except ImportError:
            cls.hbv_available = False
            
    def test_climate_modifier_temperature(self):
        """Test that temperature modification works correctly."""
        modifier = ClimateModifier(temp_increase=2.0)
        modified = modifier.apply(self.base_data)
        
        # Check temperature increased by exactly 2°C
        temp_diff = np.mean(modified['temp'] - self.base_data['temp'])
        self.assertAlmostEqual(temp_diff, 2.0, places=5)
        
    def test_climate_modifier_precipitation_variability(self):
        """Test that precipitation variability modification works correctly."""
        modifier = ClimateModifier(precip_variability_factor=1.2)
        modified = modifier.apply(self.base_data)
        
        # Check that mean is preserved (approximately)
        mean_diff = abs(np.mean(modified['precip']) - np.mean(self.base_data['precip']))
        self.assertLess(mean_diff, 0.5, "Mean precipitation should be preserved")
        
        # Check that standard deviation increased
        std_ratio = np.std(modified['precip']) / np.std(self.base_data['precip'])
        self.assertGreater(std_ratio, 1.1, "Precipitation variability should increase")
        
    def test_climate_modifier_non_negative_precip(self):
        """Test that precipitation remains non-negative after modification."""
        modifier = ClimateModifier(precip_variability_factor=2.0)  # High variability
        modified = modifier.apply(self.base_data)
        
        self.assertTrue(np.all(modified['precip'] >= 0), 
                       "Precipitation should never be negative")
        
    def test_warming_scenario_snow_timing(self):
        """
        Test whether HBV can capture earlier snowmelt under warming.
        
        Hypothesis: With +2°C warming, snow should melt earlier in the season.
        """
        if not self.hbv_available:
            self.skipTest("HBV model not available")
            
        from src.models.hbv import HBV
        
        # Create HBV model
        hbv = HBV(n_elevation_zones=1)
        params = {
            'TT': 0.0,
            'CFMAX': 4.0,
            'FC': 200.0,
            'BETA': 2.0,
            'K0': 0.2,
            'K1': 0.1,
            'K2': 0.05,
            'MAXBAS': 2.5,
        }
        
        # Run with baseline data
        hbv.initialize(params)
        discharge_baseline = []
        for t in range(len(self.base_data['precip'])):
            q = hbv.run_timestep(
                self.base_data['precip'][t],
                self.base_data['temp'][t],
                self.base_data['pet'][t],
                t
            )
            discharge_baseline.append(q)
        
        # Run with +2°C warming
        modifier = ClimateModifier(temp_increase=2.0)
        warmed_data = modifier.apply(self.base_data)
        
        hbv.initialize(params)  # Reset
        discharge_warmed = []
        for t in range(len(warmed_data['precip'])):
            q = hbv.run_timestep(
                warmed_data['precip'][t],
                warmed_data['temp'][t],
                warmed_data['pet'][t],
                t
            )
            discharge_warmed.append(q)
        
        # Analysis: Check if spring runoff timing shifts earlier
        # (This is a simplified check - in practice, more sophisticated analysis needed)
        discharge_baseline = np.array(discharge_baseline)
        discharge_warmed = np.array(discharge_warmed)
        
        # Both should produce valid discharge
        self.assertTrue(np.all(np.isfinite(discharge_baseline)))
        self.assertTrue(np.all(np.isfinite(discharge_warmed)))
        
    def test_out_of_distribution_detection(self):
        """
        Test detection of out-of-distribution (OOD) inputs.
        
        Under extreme climate scenarios, inputs may exceed the training
        distribution, which could cause data-driven models to fail.
        """
        # Create extreme scenario
        modifier = ClimateModifier(temp_increase=5.0, precip_variability_factor=1.5)
        extreme_data = modifier.apply(self.base_data)
        
        # Check that extreme values exist
        temp_max_baseline = np.max(self.base_data['temp'])
        temp_max_extreme = np.max(extreme_data['temp'])
        
        self.assertGreater(temp_max_extreme, temp_max_baseline + 4,
                          "Extreme scenario should have significantly higher temperatures")
        
        # In a real implementation, this would test LSTM model behavior
        # and check for uncertainty quantification or OOD detection


class SnowmeltTimingAnalysis:
    """
    Utility class for analyzing snowmelt timing changes.
    """
    
    @staticmethod
    def find_peak_discharge_day(discharge: np.ndarray, 
                                window_size: int = 365) -> int:
        """
        Find the day of year with peak discharge (spring runoff).
        
        Parameters:
        -----------
        discharge : np.ndarray
            Daily discharge values
        window_size : int
            Window for finding peak (default: 1 year)
            
        Returns:
        --------
        peak_day : int
            Day of year with maximum discharge
        """
        # Reshape to years and find average annual pattern
        n_years = len(discharge) // window_size
        annual_pattern = np.zeros(window_size)
        
        for year in range(n_years):
            start = year * window_size
            end = start + window_size
            annual_pattern += discharge[start:end]
        
        annual_pattern /= n_years
        
        # Find peak in spring period (day 60-180, roughly March-June)
        spring_start, spring_end = 60, 180
        spring_pattern = annual_pattern[spring_start:spring_end]
        peak_in_spring = np.argmax(spring_pattern)
        
        return spring_start + peak_in_spring


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
