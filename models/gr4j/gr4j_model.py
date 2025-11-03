"""
GR4J Model Implementation
Génie Rural à 4 paramètres Journalier

A 4-parameter daily rainfall-runoff model developed by IRSTEA (formerly Cemagref).

Parameters:
    X1: Maximum capacity of production store (mm)
    X2: Groundwater exchange coefficient (mm/day)
    X3: Maximum capacity of routing store (mm)
    X4: Time base of unit hydrograph (days)

Reference:
    Perrin, C., Michel, C., & Andréassian, V. (2003).
    Improvement of a parsimonious model for streamflow simulation.
    Journal of Hydrology, 279(1-4), 275-289.
"""

import numpy as np


class GR4J:
    """
    GR4J rainfall-runoff model
    
    A 4-parameter daily conceptual model for catchment hydrology.
    """
    
    def __init__(self, X1, X2, X3, X4):
        """
        Initialize GR4J model with parameters.
        
        Args:
            X1 (float): Maximum capacity of production store (mm), typically 100-1200
            X2 (float): Groundwater exchange coefficient (mm/day), typically -5 to 3
            X3 (float): Maximum capacity of routing store (mm), typically 20-300
            X4 (float): Time base of unit hydrograph (days), typically 1.1-2.9
        """
        self.X1 = X1  # Production store capacity
        self.X2 = X2  # Groundwater exchange coefficient
        self.X3 = X3  # Routing store capacity
        self.X4 = X4  # Unit hydrograph time base
        
        # Initialize state variables
        self.S = 0.0  # Production store level (mm)
        self.R = 0.0  # Routing store level (mm)
        
        # Unit hydrograph ordinates
        self.UH1 = None
        self.UH2 = None
        self._compute_unit_hydrographs()
        
    def _compute_unit_hydrographs(self):
        """Compute unit hydrograph ordinates"""
        # UH1 has time base of X4
        # UH2 has time base of 2*X4
        
        nUH1 = int(np.ceil(self.X4))
        nUH2 = int(np.ceil(2 * self.X4))
        
        self.UH1 = np.zeros(nUH1)
        self.UH2 = np.zeros(nUH2)
        
        # Compute SH1 ordinates
        for t in range(1, nUH1 + 1):
            if t <= self.X4:
                self.UH1[t-1] = self._s_curve1(t / self.X4)
            if t > 1:
                self.UH1[t-1] -= self._s_curve1((t-1) / self.X4)
                
        # Compute SH2 ordinates
        for t in range(1, nUH2 + 1):
            if t <= 2 * self.X4:
                self.UH2[t-1] = self._s_curve2(t / (2 * self.X4))
            if t > 1:
                self.UH2[t-1] -= self._s_curve2((t-1) / (2 * self.X4))
                
    def _s_curve1(self, t):
        """S-curve for unit hydrograph 1"""
        if t <= 0:
            return 0.0
        elif t < 1:
            return (t / 1) ** 2.5
        else:
            return 1.0
            
    def _s_curve2(self, t):
        """S-curve for unit hydrograph 2"""
        if t <= 0:
            return 0.0
        elif t < 0.5:
            return 0.5 * (t / 0.5) ** 2.5
        elif t < 1:
            return 1.0 - 0.5 * (2 - t / 0.5) ** 2.5
        else:
            return 1.0
            
    def run(self, P, E, S_init=None, R_init=None):
        """
        Run the GR4J model for a time series of precipitation and evapotranspiration.
        
        Args:
            P (array-like): Daily precipitation (mm)
            E (array-like): Daily potential evapotranspiration (mm)
            S_init (float, optional): Initial production store level
            R_init (float, optional): Initial routing store level
            
        Returns:
            dict: Dictionary containing:
                - Q: Simulated discharge (mm/day)
                - S: Production store levels
                - R: Routing store levels
        """
        P = np.array(P, dtype=float)
        E = np.array(E, dtype=float)
        n = len(P)
        
        # Initialize state variables
        if S_init is not None:
            self.S = S_init
        if R_init is not None:
            self.R = R_init
            
        # Output arrays
        Q = np.zeros(n)
        S_store = np.zeros(n)
        R_store = np.zeros(n)
        
        # Store for unit hydrograph routing
        nUH1 = len(self.UH1)
        nUH2 = len(self.UH2)
        UH1_store = np.zeros(nUH1)
        UH2_store = np.zeros(nUH2)
        
        for t in range(n):
            # Step 1: Net precipitation and evaporation
            if P[t] >= E[t]:
                Pn = P[t] - E[t]
                En = 0.0
                # Capacity ratio
                ws = self.S / self.X1
                Ps = self.X1 * (1 - ws**2) * np.tanh(Pn / self.X1) / (1 + ws * np.tanh(Pn / self.X1))
                # Update production store
                self.S = self.S + Ps
            else:
                Pn = 0.0
                En = E[t] - P[t]
                ws = self.S / self.X1
                Es = self.S * (2 - ws) * np.tanh(En / self.X1) / (1 + (1 - ws) * np.tanh(En / self.X1))
                Ps = 0.0
                # Update production store
                self.S = self.S - Es
                
            # Ensure S stays within bounds
            self.S = max(0, min(self.S, self.X1))
            
            # Percolation from production store
            Perc = self.S * (1 - (1 + (self.S / (2.25 * self.X1))**4)**(-0.25))
            self.S = self.S - Perc
            
            # Total water reaching routing
            Pr = Perc + Pn - Ps
            
            # Split into two flow components (90% and 10%)
            Pr9 = 0.9 * Pr
            Pr1 = 0.1 * Pr
            
            # Unit hydrograph routing for 90% component
            UH1_store = np.roll(UH1_store, 1)
            UH1_store[0] = Pr9
            Q9 = np.sum(UH1_store * self.UH1[:nUH1])
            
            # Unit hydrograph routing for 10% component
            UH2_store = np.roll(UH2_store, 1)
            UH2_store[0] = Pr1
            Q1 = np.sum(UH2_store * self.UH2[:nUH2])
            
            # Groundwater exchange
            F = self.X2 * (self.R / self.X3) ** 3.5
            
            # Update routing store
            self.R = max(0, self.R + Q9 + F)
            
            # Outflow from routing store
            Qr = self.R * (1 - (1 + (self.R / self.X3)**4)**(-0.25))
            self.R = self.R - Qr
            
            # Total discharge
            Qd = max(0, Qr + Q1)
            
            # Store outputs
            Q[t] = Qd
            S_store[t] = self.S
            R_store[t] = self.R
            
        return {
            'Q': Q,
            'S': S_store,
            'R': R_store
        }
        
    def reset(self):
        """Reset model state variables"""
        self.S = 0.0
        self.R = 0.0
