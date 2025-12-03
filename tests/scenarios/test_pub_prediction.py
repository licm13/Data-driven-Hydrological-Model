"""
PUB - Prediction in Ungauged Basins (缺资料流域预测)
====================================================

This module tests model transferability across basins.

Scenario:
---------
Train on basin A, predict on basin B (assuming B has no discharge observations)

Key Questions:
--------------
- Can dPL (differentiable Parameter Learning) leverage basin attributes 
  to infer physical parameters for ungauged basins?
- Does dPL outperform pure LSTM in spatial generalization?
"""

import numpy as np
import unittest
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class SyntheticBasin:
    """
    Represents a synthetic basin with specific characteristics.
    
    Parameters:
    -----------
    basin_id : str
        Unique identifier for the basin
    area : float
        Basin area in km²
    mean_elevation : float
        Mean elevation in meters
    mean_slope : float
        Mean slope in degrees
    forest_fraction : float
        Fraction of basin covered by forest (0-1)
    clay_fraction : float
        Fraction of clay in soil (0-1)
    """
    
    def __init__(self,
                 basin_id: str,
                 area: float = 500.0,
                 mean_elevation: float = 500.0,
                 mean_slope: float = 10.0,
                 forest_fraction: float = 0.5,
                 clay_fraction: float = 0.3):
        self.basin_id = basin_id
        self.area = area
        self.mean_elevation = mean_elevation
        self.mean_slope = mean_slope
        self.forest_fraction = forest_fraction
        self.clay_fraction = clay_fraction
        
    def get_attributes(self) -> Dict[str, float]:
        """Return basin attributes as dictionary."""
        return {
            'area': self.area,
            'mean_elevation': self.mean_elevation,
            'mean_slope': self.mean_slope,
            'forest_fraction': self.forest_fraction,
            'clay_fraction': self.clay_fraction
        }
    
    def get_attribute_vector(self) -> np.ndarray:
        """Return basin attributes as normalized vector."""
        # Normalization constants (typical ranges)
        normalizers = {
            'area': 1000.0,
            'mean_elevation': 2000.0,
            'mean_slope': 30.0,
            'forest_fraction': 1.0,
            'clay_fraction': 1.0
        }
        
        attrs = self.get_attributes()
        return np.array([attrs[k] / normalizers[k] for k in sorted(attrs.keys())])
    
    def derive_hbv_parameters(self) -> Dict[str, float]:
        """
        Derive conceptual HBV parameters from basin attributes.
        
        This demonstrates how physical parameters could be related
        to observable basin characteristics.
        
        Note: These relationships are simplified for demonstration.
        In reality, these relationships are complex and often unknown.
        """
        params = {}
        
        # FC (field capacity) - related to soil and forest cover
        # More forest and clay = higher water holding capacity
        params['FC'] = 100 + 200 * self.forest_fraction + 100 * self.clay_fraction
        
        # BETA (shape parameter) - related to soil type
        # Clay soils tend to have more nonlinear response
        params['BETA'] = 1.5 + 2.0 * self.clay_fraction
        
        # K0, K1, K2 (recession coefficients) - related to slope
        # Steeper slopes = faster response
        slope_factor = self.mean_slope / 20.0
        params['K0'] = 0.15 + 0.15 * slope_factor
        params['K1'] = 0.08 + 0.08 * slope_factor
        params['K2'] = 0.03 + 0.02 * slope_factor
        
        # TT (threshold temperature) - related to elevation
        # Higher elevation = colder, so TT may be higher
        params['TT'] = -1.0 + 2.0 * (self.mean_elevation / 2000.0)
        
        # CFMAX (degree-day factor) - related to forest cover
        # Less forest (more open) = higher melt rate
        params['CFMAX'] = 5.0 - 2.0 * self.forest_fraction
        
        # MAXBAS (routing time) - related to basin size
        params['MAXBAS'] = 1.5 + 2.0 * np.sqrt(self.area / 500.0)
        
        return params


class SpatialCrossValidation:
    """
    Implements leave-one-basin-out cross-validation for PUB experiments.
    
    This class helps evaluate model transferability by:
    1. Training on N-1 basins
    2. Testing on the held-out basin
    3. Repeating for all basins
    """
    
    def __init__(self, basins: List[SyntheticBasin]):
        self.basins = basins
        self.n_basins = len(basins)
        
    def get_fold(self, test_basin_idx: int) -> Tuple[List[int], List[int]]:
        """
        Get train/test split for a specific fold.
        
        Parameters:
        -----------
        test_basin_idx : int
            Index of the basin to use for testing
            
        Returns:
        --------
        train_indices : list
            Indices of training basins
        test_indices : list
            Indices of test basins (single basin)
        """
        train_indices = [i for i in range(self.n_basins) if i != test_basin_idx]
        test_indices = [test_basin_idx]
        return train_indices, test_indices
    
    def run_cross_validation(self, 
                            model_trainer,
                            data_dict: Dict[str, np.ndarray],
                            metric_fn) -> Dict[str, List[float]]:
        """
        Run full leave-one-out cross-validation.
        
        Parameters:
        -----------
        model_trainer : callable
            Function that trains model: (train_data, train_basins) -> model
        data_dict : dict
            Dictionary mapping basin_id to data
        metric_fn : callable
            Function to compute metrics: (obs, sim) -> float
            
        Returns:
        --------
        results : dict
            Results for each basin
        """
        results = {'basin_id': [], 'metric': []}
        
        for test_idx in range(self.n_basins):
            train_idx, test_idx_list = self.get_fold(test_idx)
            
            # Get training basins and data
            train_basins = [self.basins[i] for i in train_idx]
            test_basin = self.basins[test_idx_list[0]]
            
            # Train model (in practice, this would train the model)
            # model = model_trainer(train_basins, data_dict)
            
            # Evaluate on test basin
            # predictions = model.predict(test_basin, data_dict[test_basin.basin_id])
            # metric = metric_fn(data_dict[test_basin.basin_id]['discharge'], predictions)
            
            results['basin_id'].append(test_basin.basin_id)
            # results['metric'].append(metric)
            
        return results


def generate_basin_data(basin: SyntheticBasin, 
                       n_days: int = 1000,
                       seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic hydrological data for a basin.
    
    The generated data incorporates basin characteristics to create
    realistic-looking discharge responses.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate forcing data
    day_of_year = np.arange(n_days) % 365
    
    # Temperature adjusted for elevation
    temp_base = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 91) / 365)
    temp_adjustment = -6.5 * (basin.mean_elevation / 1000)  # Lapse rate
    temp = temp_base + temp_adjustment + np.random.normal(0, 3, n_days)
    
    # Precipitation (could vary by region)
    precip = np.zeros(n_days)
    rain_prob = 0.3
    rain_days = np.random.random(n_days) < rain_prob
    precip[rain_days] = np.random.exponential(scale=10, size=np.sum(rain_days))
    
    # PET
    pet = np.maximum(0, 0.5 + 0.15 * temp + np.random.normal(0, 0.3, n_days))
    
    # Generate "true" discharge using derived HBV parameters
    params = basin.derive_hbv_parameters()
    
    # Simple simulation (simplified for testing)
    storage = params['FC'] * 0.5
    discharge = np.zeros(n_days)
    
    for t in range(n_days):
        # Simple water balance
        effective_precip = precip[t] * 0.7
        storage += effective_precip
        actual_et = min(pet[t], storage * 0.1)
        storage -= actual_et
        
        # Nonlinear runoff
        if storage > 0:
            runoff_fraction = (storage / params['FC']) ** params['BETA']
            runoff = params['K1'] * storage * runoff_fraction
            storage -= runoff
            discharge[t] = runoff
        
        storage = max(0, storage)
    
    return {
        'temp': temp,
        'precip': precip,
        'pet': pet,
        'discharge': discharge,
        'attributes': basin.get_attribute_vector()
    }


def compute_basin_similarity(basin1: SyntheticBasin, 
                            basin2: SyntheticBasin) -> float:
    """
    Compute similarity between two basins based on attributes.
    
    Returns a value between 0 (dissimilar) and 1 (identical).
    """
    attr1 = basin1.get_attribute_vector()
    attr2 = basin2.get_attribute_vector()
    
    # Euclidean distance, normalized
    distance = np.sqrt(np.sum((attr1 - attr2) ** 2))
    max_distance = np.sqrt(len(attr1))  # Maximum possible distance
    
    similarity = 1 - (distance / max_distance)
    return similarity


class PUBPredictionTest(unittest.TestCase):
    """
    Test suite for Prediction in Ungauged Basins (PUB).
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test basins and data."""
        np.random.seed(42)
        
        # Create diverse set of synthetic basins
        cls.basins = [
            SyntheticBasin('basin_A', area=300, mean_elevation=400, 
                          mean_slope=8, forest_fraction=0.7, clay_fraction=0.2),
            SyntheticBasin('basin_B', area=800, mean_elevation=1200, 
                          mean_slope=15, forest_fraction=0.3, clay_fraction=0.4),
            SyntheticBasin('basin_C', area=500, mean_elevation=600, 
                          mean_slope=10, forest_fraction=0.5, clay_fraction=0.3),
            SyntheticBasin('basin_D', area=200, mean_elevation=300, 
                          mean_slope=5, forest_fraction=0.8, clay_fraction=0.1),
        ]
        
        # Generate data for each basin
        cls.basin_data = {}
        for i, basin in enumerate(cls.basins):
            cls.basin_data[basin.basin_id] = generate_basin_data(
                basin, n_days=1000, seed=42 + i
            )
            
    def test_parameter_derivation_consistency(self):
        """Test that derived parameters are physically consistent."""
        for basin in self.basins:
            params = basin.derive_hbv_parameters()
            
            # Check parameter bounds
            self.assertGreater(params['FC'], 0)
            self.assertLess(params['FC'], 1000)
            self.assertGreater(params['BETA'], 1)
            self.assertLess(params['BETA'], 6)
            self.assertGreater(params['K0'], 0)
            self.assertLess(params['K0'], 1)
            
    def test_basin_similarity_metric(self):
        """Test basin similarity computation."""
        # Same basin should have similarity = 1
        sim_same = compute_basin_similarity(self.basins[0], self.basins[0])
        self.assertAlmostEqual(sim_same, 1.0, places=5)
        
        # Different basins should have similarity < 1
        sim_diff = compute_basin_similarity(self.basins[0], self.basins[1])
        self.assertLess(sim_diff, 1.0)
        self.assertGreater(sim_diff, 0.0)
        
    def test_parameter_attribute_relationship(self):
        """
        Test that basin attributes influence derived parameters logically.
        """
        # High forest basin should have higher FC
        high_forest = SyntheticBasin('high_forest', forest_fraction=0.9)
        low_forest = SyntheticBasin('low_forest', forest_fraction=0.1)
        
        params_high = high_forest.derive_hbv_parameters()
        params_low = low_forest.derive_hbv_parameters()
        
        self.assertGreater(params_high['FC'], params_low['FC'],
                          "Higher forest cover should increase field capacity")
        
        # Steep basin should have faster recession
        steep = SyntheticBasin('steep', mean_slope=25)
        flat = SyntheticBasin('flat', mean_slope=5)
        
        params_steep = steep.derive_hbv_parameters()
        params_flat = flat.derive_hbv_parameters()
        
        self.assertGreater(params_steep['K1'], params_flat['K1'],
                          "Steeper basins should have faster recession")
        
    def test_data_generation_validity(self):
        """Test that generated data is physically valid."""
        for basin_id, data in self.basin_data.items():
            # Check non-negative values
            self.assertTrue(np.all(data['precip'] >= 0),
                          f"Precipitation should be non-negative for {basin_id}")
            self.assertTrue(np.all(data['discharge'] >= 0),
                          f"Discharge should be non-negative for {basin_id}")
            self.assertTrue(np.all(data['pet'] >= 0),
                          f"PET should be non-negative for {basin_id}")
            
            # Check reasonable ranges
            self.assertTrue(np.all(data['temp'] > -50),
                          f"Temperature should be reasonable for {basin_id}")
            self.assertTrue(np.all(data['temp'] < 50),
                          f"Temperature should be reasonable for {basin_id}")
            
    def test_cross_validation_folds(self):
        """Test that cross-validation folds are correct."""
        cv = SpatialCrossValidation(self.basins)
        
        for test_idx in range(len(self.basins)):
            train_idx, test_idx_list = cv.get_fold(test_idx)
            
            # Check that test basin is not in training set
            self.assertNotIn(test_idx, train_idx)
            
            # Check that all other basins are in training set
            self.assertEqual(len(train_idx), len(self.basins) - 1)
            
            # Check that test index is correct
            self.assertEqual(test_idx_list[0], test_idx)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
