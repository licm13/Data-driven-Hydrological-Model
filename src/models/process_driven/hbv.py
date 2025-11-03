"""
HBV (Hydrologiska Byråns Vattenbalansavdelning) - A conceptual hydrological model
Simplified implementation for demonstration
"""

import numpy as np
from typing import Dict


class HBV:
    """
    HBV hydrological model
    
    Parameters:
        FC: Maximum soil moisture storage (mm)
        BETA: Shape parameter for soil moisture routine
        LP: Soil moisture threshold for evapotranspiration
        K0: Recession coefficient for upper zone
        K1: Recession coefficient for lower zone
        PERC: Percolation from upper to lower zone (mm/day)
    """
    
    def __init__(self, FC: float = 200.0, BETA: float = 2.0, 
                 LP: float = 0.7, K0: float = 0.1, K1: float = 0.05, 
                 PERC: float = 1.0):
        """
        Initialize HBV model
        
        Args:
            FC: Maximum soil moisture storage (mm)
            BETA: Shape parameter
            LP: Soil moisture threshold
            K0: Upper zone recession coefficient
            K1: Lower zone recession coefficient
            PERC: Percolation rate (mm/day)
        """
        self.FC = FC
        self.BETA = BETA
        self.LP = LP
        self.K0 = K0
        self.K1 = K1
        self.PERC = PERC
        
        # State variables
        self.SM = None  # Soil moisture
        self.SUZ = None  # Upper zone storage
        self.SLZ = None  # Lower zone storage
        
    def initialize_states(self):
        """Initialize model states"""
        self.SM = self.FC * 0.5
        self.SUZ = 10.0
        self.SLZ = 50.0
        
    def run(self, precipitation: np.ndarray, temperature: np.ndarray, 
            evapotranspiration: np.ndarray) -> np.ndarray:
        """
        Run HBV model
        
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
        
        for t in range(n):
            P = precipitation[t]
            T = temperature[t]
            EP = evapotranspiration[t]
            
            # Snow routine (simplified - assuming temperature threshold)
            if T < 0:
                # Snow accumulation
                Prain = 0
            else:
                # Rain or snowmelt
                Prain = P
            
            # Soil moisture routine
            # Actual evapotranspiration
            if self.SM / self.FC < self.LP:
                EA = EP * (self.SM / (self.LP * self.FC))
            else:
                EA = EP
            
            EA = min(EA, self.SM)
            self.SM -= EA
            
            # Recharge to groundwater
            if Prain > 0:
                recharge = Prain * (self.SM / self.FC) ** self.BETA
                self.SM += Prain - recharge
                
                # Ensure soil moisture doesn't exceed capacity
                if self.SM > self.FC:
                    recharge += (self.SM - self.FC)
                    self.SM = self.FC
            else:
                recharge = 0
            
            # Response routine
            # Upper zone
            self.SUZ += recharge
            
            # Percolation to lower zone
            perc = min(self.PERC, self.SUZ)
            self.SUZ -= perc
            self.SLZ += perc
            
            # Discharge from upper zone
            Q0 = self.K0 * self.SUZ
            self.SUZ -= Q0
            
            # Discharge from lower zone
            Q1 = self.K1 * self.SLZ
            self.SLZ -= Q1
            
            # Total discharge
            discharge[t] = Q0 + Q1
            
            # Ensure states are non-negative
            self.SM = max(0, self.SM)
            self.SUZ = max(0, self.SUZ)
            self.SLZ = max(0, self.SLZ)
        
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
        try:
            from utils.metrics import nash_sutcliffe_efficiency
        except ImportError:
            from ...utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        np.random.seed(42)
        for _ in range(n_iterations):
            # Random parameter sampling
            FC = np.random.uniform(100, 500)
            BETA = np.random.uniform(1, 5)
            LP = np.random.uniform(0.3, 0.9)
            K0 = np.random.uniform(0.05, 0.3)
            K1 = np.random.uniform(0.01, 0.1)
            PERC = np.random.uniform(0.5, 3)
            
            self.FC, self.BETA, self.LP = FC, BETA, LP
            self.K0, self.K1, self.PERC = K0, K1, PERC
            
            # Simulate
            y_pred = self.run(X_train[:, 0], X_train[:, 1], X_train[:, 2])
            
            # Evaluate
            nse = nash_sutcliffe_efficiency(y_train, y_pred)
            
            if nse > best_nse:
                best_nse = nse
                best_params = {
                    'FC': FC, 'BETA': BETA, 'LP': LP,
                    'K0': K0, 'K1': K1, 'PERC': PERC, 'NSE': nse
                }
        
        # Set best parameters
        if best_params:
            self.FC = best_params['FC']
            self.BETA = best_params['BETA']
            self.LP = best_params['LP']
            self.K0 = best_params['K0']
            self.K1 = best_params['K1']
            self.PERC = best_params['PERC']
        
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
