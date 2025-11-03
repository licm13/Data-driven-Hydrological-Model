"""
GR4J (Génie Rural à 4 paramètres Journalier) - A 4-parameter daily hydrological model
Simplified implementation for demonstration
"""

import numpy as np
from typing import Tuple, Dict


class GR4J:
    """
    GR4J hydrological model
    
    Parameters:
        X1: Maximum capacity of production store (mm)
        X2: Groundwater exchange coefficient (mm)
        X3: One day ahead maximum capacity of routing store (mm)
        X4: Time base of unit hydrograph (days)
    """
    
    def __init__(self, X1: float = 350.0, X2: float = 0.0, 
                 X3: float = 90.0, X4: float = 1.7):
        """
        Initialize GR4J model
        
        Args:
            X1: Maximum capacity of production store (mm)
            X2: Groundwater exchange coefficient (mm)
            X3: Maximum capacity of routing store (mm)
            X4: Time base of unit hydrograph (days)
        """
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        
        # State variables
        self.S = None  # Production store level
        self.R = None  # Routing store level
        
    def initialize_states(self):
        """Initialize model states"""
        self.S = self.X1 * 0.5  # Start at 50% capacity
        self.R = self.X3 * 0.5
        
    def run(self, precipitation: np.ndarray, evapotranspiration: np.ndarray) -> np.ndarray:
        """
        Run GR4J model
        
        Args:
            precipitation: Daily precipitation (mm)
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
            E = evapotranspiration[t]
            
            # Net precipitation and evaporation
            if P >= E:
                Pn = P - E
                En = 0
            else:
                Pn = 0
                En = E - P
            
            # Production store
            if Pn > 0:
                Ps = self.X1 * (1 - (self.S / self.X1) ** 2) * np.tanh(Pn / self.X1)
                Ps = Ps / (1 + (self.S / self.X1) * np.tanh(Pn / self.X1))
                self.S += Ps
            else:
                Ps = 0
            
            # Evaporation from production store
            if En > 0:
                Es = self.S * (2 - self.S / self.X1) * np.tanh(En / self.X1)
                Es = Es / (1 + (1 - self.S / self.X1) * np.tanh(En / self.X1))
                self.S -= Es
            
            # Percolation
            Perc = self.S * (1 - (1 + (4.0 * self.S / (9.0 * self.X1)) ** 4) ** -0.25)
            self.S -= Perc
            
            # Ensure S is within bounds
            self.S = np.clip(self.S, 0, self.X1)
            
            # Total water for routing
            Pr = Perc + (Pn - Ps)
            
            # Split into routing components (90% to routing store, 10% direct)
            Pr9 = 0.9 * Pr
            Pr1 = 0.1 * Pr
            
            # Routing store
            F = self.X2 * (self.R / self.X3) ** 3.5  # Exchange
            self.R = max(0, self.R + Pr9 + F)
            
            # Discharge from routing store
            Qr = self.R * (1 - (1 + (self.R / self.X3) ** 4) ** -0.25)
            self.R -= Qr
            
            # Ensure R is within bounds
            self.R = np.clip(self.R, 0, self.X3)
            
            # Total discharge (simplified unit hydrograph)
            Qd = max(0, Pr1 + F)
            discharge[t] = Qr + Qd
        
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
            X1 = np.random.uniform(100, 1000)
            X2 = np.random.uniform(-5, 5)
            X3 = np.random.uniform(10, 300)
            X4 = np.random.uniform(0.5, 4)
            
            self.X1, self.X2, self.X3, self.X4 = X1, X2, X3, X4
            
            # Simulate
            y_pred = self.run(X_train[:, 0], X_train[:, 2])
            
            # Evaluate
            nse = nash_sutcliffe_efficiency(y_train, y_pred)
            
            if nse > best_nse:
                best_nse = nse
                best_params = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'NSE': nse}
        
        # Set best parameters
        if best_params:
            self.X1 = best_params['X1']
            self.X2 = best_params['X2']
            self.X3 = best_params['X3']
            self.X4 = best_params['X4']
        
        return best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discharge
        
        Args:
            X: Features [precipitation, temperature, pet]
            
        Returns:
            Predicted discharge
        """
        return self.run(X[:, 0], X[:, 2])
