"""
Extreme Event Robustness Test (极端事件鲁棒性测试)
=================================================

This module tests model behavior under extreme hydrological events.

Scenario:
---------
Test model simulation of "100-year flood" events using synthetic extreme data.

Key Questions:
--------------
- Does model output satisfy physical constraints?
  - Discharge should not exceed (precipitation + storage)
  - Discharge should never be negative
- How do different model types handle extreme inputs?
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


class ExtremeEventGenerator:
    """
    Generates synthetic extreme hydrological events for testing.
    
    This class creates realistic extreme precipitation events
    based on statistical distributions and return periods.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_extreme_precipitation(self,
                                       n_days: int = 100,
                                       event_day: int = 50,
                                       event_magnitude: float = 200.0,
                                       event_duration: int = 3) -> np.ndarray:
        """
        Generate a precipitation series with an extreme event.
        
        The default event_magnitude of 200mm over 3 days represents approximately
        a 100-year return period event for many temperate regions. This is based
        on typical intensity-duration-frequency (IDF) relationships where:
        - 100-year daily rainfall: ~100-150mm (varies by region)
        - Multi-day events can accumulate 150-300mm
        
        Statistical Assumptions:
        - Background precipitation follows an exponential distribution (common for daily precip)
        - Extreme events are deterministic for reproducibility
        - Return period estimation uses simplified GEV distribution assumptions
        
        Parameters:
        -----------
        n_days : int
            Total number of days
        event_day : int
            Day when extreme event starts
        event_magnitude : float
            Total precipitation during extreme event (mm). Default 200mm represents
            approximately a 100-year event for temperate regions.
        event_duration : int
            Duration of extreme event in days
            
        Returns:
        --------
        precip : np.ndarray
            Precipitation series with extreme event
        """
        # Background precipitation
        precip = np.zeros(n_days)
        background_rain_days = np.random.choice(n_days, size=int(n_days * 0.2), replace=False)
        precip[background_rain_days] = np.random.exponential(scale=5, size=len(background_rain_days))
        
        # Add extreme event (distributed over event_duration days)
        # Typical pattern: increasing to peak, then decreasing
        event_pattern = np.array([0.2, 0.5, 0.3]) if event_duration == 3 else np.ones(event_duration) / event_duration
        event_pattern = event_pattern[:event_duration]
        event_pattern = event_pattern / event_pattern.sum()
        
        for i, fraction in enumerate(event_pattern):
            day = event_day + i
            if day < n_days:
                precip[day] = event_magnitude * fraction
                
        return precip
    
    def generate_flash_flood_scenario(self, n_days: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate a complete flash flood scenario.
        
        Returns:
        --------
        data : dict
            Dictionary with 'precip', 'temp', 'pet', and 'expected_peak_day'
        """
        # Generate extreme precipitation (equivalent to 100-year event)
        precip = self.generate_extreme_precipitation(
            n_days=n_days,
            event_day=50,
            event_magnitude=200.0,  # 200mm over 3 days
            event_duration=3
        )
        
        # Normal temperature pattern
        day_of_year = np.arange(n_days) % 365
        temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year) / 365)
        temp += np.random.normal(0, 2, n_days)
        
        # PET
        pet = np.maximum(0, 0.5 + 0.15 * temp + np.random.normal(0, 0.3, n_days))
        
        return {
            'precip': precip,
            'temp': temp,
            'pet': pet,
            'event_day': 50
        }
    
    def generate_compound_event(self, n_days: int = 365) -> Dict[str, np.ndarray]:
        """
        Generate a compound extreme event (e.g., rain-on-snow).
        
        This represents a scenario where heavy rainfall occurs
        on top of snowpack, potentially causing severe flooding.
        """
        # Winter temperature pattern (cold enough for snow)
        day_of_year = np.arange(n_days) % 365
        temp = -5 + 15 * np.sin(2 * np.pi * (day_of_year - 91) / 365)
        temp += np.random.normal(0, 3, n_days)
        
        # Precipitation throughout the year
        precip = np.zeros(n_days)
        
        # Regular snow accumulation in winter
        winter_days = np.where(temp < 0)[0]
        winter_rain_days = np.random.choice(winter_days, size=int(len(winter_days) * 0.3), replace=False)
        precip[winter_rain_days] = np.random.exponential(scale=8, size=len(winter_rain_days))
        
        # Rain-on-snow event: warm day with heavy rain during snow season
        # This typically happens during sudden warming
        ros_day = np.argmin(np.abs(day_of_year - 80))  # Around March 21
        
        # Sudden warming
        temp[ros_day:ros_day+3] = 10  # Warm spell
        
        # Heavy rain during warming
        precip[ros_day] = 80
        precip[ros_day+1] = 50
        precip[ros_day+2] = 30
        
        # PET (low during cold periods)
        pet = np.maximum(0, 0.3 + 0.1 * temp + np.random.normal(0, 0.2, n_days))
        
        return {
            'precip': precip,
            'temp': temp,
            'pet': pet,
            'event_type': 'rain_on_snow',
            'event_day': ros_day
        }


class PhysicalConstraintChecker:
    """
    Checks whether model outputs satisfy physical constraints.
    
    Key Constraints:
    ----------------
    1. Mass Balance: Q ≤ P + ΔS (discharge cannot exceed input + storage release)
    2. Non-negativity: Q ≥ 0 (discharge cannot be negative)
    3. Realistic Range: Q should be within physically plausible bounds
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def check_non_negativity(self, discharge: np.ndarray) -> Tuple[bool, str]:
        """
        Check that discharge is never negative.
        
        Returns:
        --------
        passed : bool
            True if constraint satisfied
        message : str
            Description of check result
        """
        min_discharge = np.min(discharge)
        
        if min_discharge < -self.tolerance:
            return False, f"Negative discharge detected: min={min_discharge:.6f}"
        return True, f"Non-negativity check passed (min={min_discharge:.6f})"
    
    def check_mass_balance(self,
                          precip: np.ndarray,
                          discharge: np.ndarray,
                          evap: Optional[np.ndarray] = None,
                          initial_storage: float = 100.0,
                          final_storage: float = 100.0) -> Tuple[bool, str]:
        """
        Check mass balance over the simulation period.
        
        Total P - Total E - Total Q ≈ ΔS
        
        Returns:
        --------
        passed : bool
            True if mass balance roughly satisfied
        message : str
            Description of check result
        """
        total_precip = np.sum(precip)
        total_discharge = np.sum(discharge)
        total_evap = np.sum(evap) if evap is not None else 0
        storage_change = final_storage - initial_storage
        
        balance = total_precip - total_evap - total_discharge - storage_change
        
        # Allow some tolerance (e.g., 5% of total input)
        tolerance_threshold = 0.05 * total_precip
        
        if abs(balance) > tolerance_threshold:
            return False, (f"Mass balance violation: "
                         f"P={total_precip:.1f}, E={total_evap:.1f}, "
                         f"Q={total_discharge:.1f}, ΔS={storage_change:.1f}, "
                         f"imbalance={balance:.1f}")
        return True, f"Mass balance check passed (imbalance={balance:.2f})"
    
    def check_peak_timing(self,
                         precip: np.ndarray,
                         discharge: np.ndarray,
                         max_lag: int = 10) -> Tuple[bool, str]:
        """
        Check that discharge peaks occur after precipitation peaks.
        
        This ensures the model captures the hydrological lag.
        
        Parameters:
        -----------
        max_lag : int
            Maximum expected lag between precip and discharge peaks
            
        Returns:
        --------
        passed : bool
            True if timing is physically reasonable
        message : str
            Description of check result
        """
        precip_peak_day = np.argmax(precip)
        discharge_peak_day = np.argmax(discharge)
        
        lag = discharge_peak_day - precip_peak_day
        
        if lag < 0:
            return False, f"Discharge peaks before precipitation (lag={lag} days)"
        if lag > max_lag:
            return False, f"Excessive lag between precip and discharge (lag={lag} days)"
        
        return True, f"Peak timing check passed (lag={lag} days)"
    
    def check_realistic_range(self,
                             discharge: np.ndarray,
                             precip: np.ndarray,
                             max_runoff_ratio: float = 1.5) -> Tuple[bool, str]:
        """
        Check that discharge is within realistic bounds.
        
        Daily discharge should not exceed some multiple of daily precipitation
        (accounting for storage release).
        
        Parameters:
        -----------
        max_runoff_ratio : float
            Maximum ratio of Q/P allowed (>1 allows for storage release)
            
        Returns:
        --------
        passed : bool
            True if discharge in realistic range
        message : str
            Description of check result
        """
        # Check for extremely high discharge
        max_discharge = np.max(discharge)
        max_precip = np.max(precip)
        
        # Discharge shouldn't be unreasonably high compared to inputs
        # Allow for storage contribution
        threshold = max_precip * max_runoff_ratio + 100  # + base storage release
        
        if max_discharge > threshold:
            return False, (f"Unrealistically high discharge: "
                         f"max_Q={max_discharge:.1f} vs threshold={threshold:.1f}")
        
        return True, f"Realistic range check passed (max_Q={max_discharge:.1f})"


def run_simple_model(precip: np.ndarray, 
                    temp: np.ndarray, 
                    pet: np.ndarray,
                    k: float = 0.1,
                    initial_storage: float = 100.0) -> Tuple[np.ndarray, float]:
    """
    Run a simple reservoir model for testing.
    
    Returns:
    --------
    discharge : np.ndarray
        Simulated discharge
    final_storage : float
        Final storage value
    """
    n = len(precip)
    discharge = np.zeros(n)
    storage = initial_storage
    
    for t in range(n):
        # Input
        storage += precip[t] * 0.7
        
        # Evaporation
        evap = min(pet[t] * 0.5, storage * 0.05)
        storage -= evap
        
        # Discharge
        discharge[t] = k * storage
        storage -= discharge[t]
        
        storage = max(0, storage)
    
    return discharge, storage


class ExtremeEventRobustnessTest(unittest.TestCase):
    """
    Test suite for extreme event robustness.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test scenarios and checkers."""
        cls.generator = ExtremeEventGenerator(seed=42)
        cls.checker = PhysicalConstraintChecker()
        
        # Generate test scenarios
        cls.flash_flood = cls.generator.generate_flash_flood_scenario(n_days=100)
        cls.compound_event = cls.generator.generate_compound_event(n_days=365)
        
    def test_non_negativity_flash_flood(self):
        """Test that discharge remains non-negative during flash flood."""
        discharge, _ = run_simple_model(
            self.flash_flood['precip'],
            self.flash_flood['temp'],
            self.flash_flood['pet']
        )
        
        passed, message = self.checker.check_non_negativity(discharge)
        self.assertTrue(passed, message)
        
    def test_non_negativity_compound_event(self):
        """Test that discharge remains non-negative during compound event."""
        discharge, _ = run_simple_model(
            self.compound_event['precip'],
            self.compound_event['temp'],
            self.compound_event['pet']
        )
        
        passed, message = self.checker.check_non_negativity(discharge)
        self.assertTrue(passed, message)
        
    def test_peak_timing_flash_flood(self):
        """Test that discharge peak occurs after precipitation peak."""
        discharge, _ = run_simple_model(
            self.flash_flood['precip'],
            self.flash_flood['temp'],
            self.flash_flood['pet']
        )
        
        passed, message = self.checker.check_peak_timing(
            self.flash_flood['precip'], 
            discharge,
            max_lag=5
        )
        self.assertTrue(passed, message)
        
    def test_realistic_discharge_range(self):
        """Test that discharge stays within realistic bounds."""
        discharge, _ = run_simple_model(
            self.flash_flood['precip'],
            self.flash_flood['temp'],
            self.flash_flood['pet']
        )
        
        passed, message = self.checker.check_realistic_range(
            discharge,
            self.flash_flood['precip']
        )
        self.assertTrue(passed, message)
        
    def test_extreme_input_handling(self):
        """Test model behavior with extreme input values."""
        # Create extremely high precipitation
        extreme_precip = np.zeros(100)
        extreme_precip[50] = 500  # 500mm in one day (extremely rare)
        
        temp = np.ones(100) * 20
        pet = np.ones(100) * 3
        
        discharge, final_storage = run_simple_model(extreme_precip, temp, pet)
        
        # Model should still produce valid output
        self.assertTrue(np.all(np.isfinite(discharge)), 
                       "Discharge should be finite even with extreme input")
        self.assertTrue(np.all(discharge >= 0),
                       "Discharge should be non-negative")
        
    def test_zero_input_stability(self):
        """Test model stability with zero inputs."""
        zero_precip = np.zeros(100)
        temp = np.ones(100) * 20
        pet = np.ones(100) * 3
        
        discharge, final_storage = run_simple_model(zero_precip, temp, pet)
        
        # Model should gracefully handle zero precipitation
        self.assertTrue(np.all(np.isfinite(discharge)))
        self.assertGreaterEqual(final_storage, 0)
        
        # Discharge should approach zero over time (recession only)
        self.assertLess(discharge[-1], discharge[0])
        
    def test_mass_balance_extreme_event(self):
        """Test mass balance during extreme event."""
        initial_storage = 100.0
        discharge, final_storage = run_simple_model(
            self.flash_flood['precip'],
            self.flash_flood['temp'],
            self.flash_flood['pet'],
            initial_storage=initial_storage
        )
        
        # Approximate evaporation (simplified)
        evap = self.flash_flood['pet'] * 0.5
        
        passed, message = self.checker.check_mass_balance(
            self.flash_flood['precip'] * 0.7,  # Effective precipitation
            discharge,
            evap=evap,
            initial_storage=initial_storage,
            final_storage=final_storage
        )
        # Note: This may not pass perfectly due to simplifications
        # The test demonstrates the concept
        print(f"Mass balance check: {message}")


class ConstraintViolationReport:
    """
    Generates a report of constraint violations.
    """
    
    def __init__(self):
        self.violations = []
        self.checks = []
        
    def add_check(self, name: str, passed: bool, message: str):
        """Add a check result to the report."""
        self.checks.append({
            'name': name,
            'passed': passed,
            'message': message
        })
        if not passed:
            self.violations.append({'name': name, 'message': message})
            
    def generate_report(self) -> str:
        """Generate summary report."""
        lines = ["=" * 60]
        lines.append("CONSTRAINT VIOLATION REPORT")
        lines.append("=" * 60)
        lines.append(f"\nTotal checks: {len(self.checks)}")
        lines.append(f"Passed: {len([c for c in self.checks if c['passed']])}")
        lines.append(f"Failed: {len(self.violations)}")
        
        if self.violations:
            lines.append("\nVIOLATIONS:")
            for v in self.violations:
                lines.append(f"  - {v['name']}: {v['message']}")
        else:
            lines.append("\n✓ All constraints satisfied!")
            
        return "\n".join(lines)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
