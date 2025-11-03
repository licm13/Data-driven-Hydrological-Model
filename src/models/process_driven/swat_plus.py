"""
SWAT+ (Soil and Water Assessment Tool Plus) - A watershed-scale hydrological model
Simplified implementation focusing on key hydrological processes
"""

import numpy as np
from typing import Dict


class SWATPlus:
    """
    Simplified SWAT+ hydrological model
    
    Parameters:
        CN: Curve Number for runoff estimation
        ESCO: Soil evaporation compensation factor
        EPCO: Plant uptake compensation factor
        GWQMN: Threshold depth of water in shallow aquifer for return flow (mm)
        GW_REVAP: Groundwater "revap" coefficient
        ALPHA_BF: Baseflow alpha factor (days)
        REVAPMN: Threshold depth of water in shallow aquifer for percolation (mm)
    """
    
    def __init__(self, CN: float = 75.0, ESCO: float = 0.95, 
                 EPCO: float = 1.0, GWQMN: float = 1000.0,
                 GW_REVAP: float = 0.02, ALPHA_BF: float = 0.048,
                 REVAPMN: float = 750.0):
        """
        Initialize SWAT+ model
        
        Args:
            CN: Curve Number
            ESCO: Soil evaporation compensation factor
            EPCO: Plant uptake compensation factor
            GWQMN: Threshold for return flow (mm)
            GW_REVAP: Revap coefficient
            ALPHA_BF: Baseflow recession constant
            REVAPMN: Threshold for percolation (mm)
        """
        self.CN = CN
        self.ESCO = ESCO
        self.EPCO = EPCO
        self.GWQMN = GWQMN
        self.GW_REVAP = GW_REVAP
        self.ALPHA_BF = ALPHA_BF
        self.REVAPMN = REVAPMN
        
        # Calculate maximum retention
        self.S = 25.4 * (1000.0 / CN - 10.0)
        
        # State variables
        self.SW = None  # Soil water content (mm)
        self.GW = None  # Groundwater storage (mm)
        
    def initialize_states(self):
        """Initialize model states"""
        self.SW = self.S * 0.5
        self.GW = self.GWQMN
        
    def calculate_runoff(self, P: float) -> float:
        """
        Calculate surface runoff using SCS Curve Number method
        
        Args:
            P: Precipitation (mm)
            
        Returns:
            Surface runoff (mm)
        """
        if P > 0.2 * self.S:
            Q_surf = ((P - 0.2 * self.S) ** 2) / (P + 0.8 * self.S)
        else:
            Q_surf = 0
        
        return Q_surf
    
    def run(self, precipitation: np.ndarray, temperature: np.ndarray,
            evapotranspiration: np.ndarray) -> np.ndarray:
        """
        Run SWAT+ model
        
        Args:
            precipitation: Daily precipitation (mm)
            temperature: Daily temperature (°C)
            evapotranspiration: Daily potential evapotranspiration (mm)
            
        Returns:
            Simulated discharge (mm/day)
        """
        n = len(precipitation)
        discharge = np.zeros(n)
        
        # Initialize states
        self.initialize_states()
        
        # Maximum soil water
        SW_max = self.S * 1.5
        
        for t in range(n):
            P = precipitation[t]
            T = temperature[t]
            EP = evapotranspiration[t]
            
            # Surface runoff
            Q_surf = self.calculate_runoff(P)
            
            # Infiltration
            infiltration = P - Q_surf
            self.SW += infiltration
            
            # Evapotranspiration from soil
            ET = min(EP * self.ESCO, self.SW)
            self.SW -= ET
            
            # Percolation to groundwater
            if self.SW > self.S:
                perc = (self.SW - self.S) * 0.2
                self.SW -= perc
                self.GW += perc
            
            # Ensure soil water is within bounds
            self.SW = np.clip(self.SW, 0, SW_max)
            
            # Baseflow from groundwater
            if self.GW > self.GWQMN:
                Q_base = (self.GW - self.GWQMN) * self.ALPHA_BF
                self.GW -= Q_base
            else:
                Q_base = 0
            
            # Revap (return of water from aquifer to unsaturated zone)
            if self.GW > self.REVAPMN:
                revap = self.GW_REVAP * (self.GW - self.REVAPMN)
                revap = min(revap, self.GW - self.GWQMN)
                self.GW -= revap
                self.SW += revap * 0.5  # Partial return
            
            # Ensure groundwater is non-negative
            self.GW = max(0, self.GW)
            
            # Total discharge
            discharge[t] = Q_surf + Q_base
        
        return discharge
    
    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray,
                  n_iterations: int = 100) -> Dict[str, float]:
        """
        Simple calibration using random search
        
        Args:
            X_train: Training features [precipitation, temperature, pet]
            y_train: Observed discharge
            n_iterations: Number of calibration iterations
            
        Returns:
            Best parameters
        """
        from ..utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        np.random.seed(42)
        for _ in range(n_iterations):
            # Random parameter sampling
            CN = np.random.uniform(60, 90)
            ESCO = np.random.uniform(0.7, 1.0)
            EPCO = np.random.uniform(0.8, 1.0)
            GWQMN = np.random.uniform(500, 2000)
            GW_REVAP = np.random.uniform(0.01, 0.1)
            ALPHA_BF = np.random.uniform(0.01, 0.1)
            REVAPMN = np.random.uniform(300, 1500)
            
            self.CN = CN
            self.S = 25.4 * (1000.0 / CN - 10.0)
            self.ESCO = ESCO
            self.EPCO = EPCO
            self.GWQMN = GWQMN
            self.GW_REVAP = GW_REVAP
            self.ALPHA_BF = ALPHA_BF
            self.REVAPMN = REVAPMN
            
            # Simulate
            y_pred = self.run(X_train[:, 0], X_train[:, 1], X_train[:, 2])
            
            # Evaluate
            nse = nash_sutcliffe_efficiency(y_train, y_pred)
            
            if nse > best_nse:
                best_nse = nse
                best_params = {
                    'CN': CN, 'ESCO': ESCO, 'EPCO': EPCO,
                    'GWQMN': GWQMN, 'GW_REVAP': GW_REVAP,
                    'ALPHA_BF': ALPHA_BF, 'REVAPMN': REVAPMN, 'NSE': nse
                }
        
        # Set best parameters
        if best_params:
            self.CN = best_params['CN']
            self.S = 25.4 * (1000.0 / self.CN - 10.0)
            self.ESCO = best_params['ESCO']
            self.EPCO = best_params['EPCO']
            self.GWQMN = best_params['GWQMN']
            self.GW_REVAP = best_params['GW_REVAP']
            self.ALPHA_BF = best_params['ALPHA_BF']
            self.REVAPMN = best_params['REVAPMN']
        
        return best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discharge
        
        Args:
            X: Features [precipitation, temperature, pet]
            
        Returns:
            Predicted discharge
        """
        return self.run(X[:, 0], X[:, 1], X[:, 2])
