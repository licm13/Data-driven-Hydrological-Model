"""
HBV Model Implementation
Hydrologiska Byråns Vattenbalansavdelning

A conceptual hydrological model originally developed by the Swedish Meteorological 
and Hydrological Institute (SMHI).

The model includes:
- Snow accumulation and melt routine
- Soil moisture accounting
- Response function for runoff generation
- Routing routine

Reference:
    Bergström, S. (1995). The HBV model. In Computer Models of Watershed Hydrology 
    (pp. 443-476). Water Resources Publications.
"""

import numpy as np


class HBV:
    """
    HBV conceptual rainfall-runoff model
    
    A semi-distributed conceptual model with snow, soil, and routing routines.
    """
    
    def __init__(self, params):
        """
        Initialize HBV model with parameters.
        
        Args:
            params (dict): Model parameters including:
                Snow routine:
                - TT: Threshold temperature for snow/rain (°C)
                - CFMAX: Degree-day factor (mm/°C/day)
                - CFSLOPE: Spatial variation in CFMAX (optional)
                - CFR: Refreezing coefficient
                - CWH: Water holding capacity of snow
                
                Soil routine:
                - FC: Maximum soil moisture storage (mm)
                - LP: Soil moisture threshold for reduction of evap (-)
                - BETA: Shape coefficient (-)
                
                Response routine:
                - PERC: Maximum percolation to lower zone (mm/day)
                - UZL: Threshold for quick runoff (mm)
                - K0: Recession coefficient for quick flow (1/day)
                - K1: Recession coefficient for upper zone (1/day)
                - K2: Recession coefficient for lower zone (1/day)
                
                Routing:
                - MAXBAS: Routing parameter (days)
        """
        # Snow routine parameters
        self.TT = params.get('TT', 0.0)
        self.CFMAX = params.get('CFMAX', 3.5)
        self.CFSLOPE = params.get('CFSLOPE', 0.0)
        self.CFR = params.get('CFR', 0.05)
        self.CWH = params.get('CWH', 0.1)
        
        # Soil routine parameters
        self.FC = params.get('FC', 200.0)
        self.LP = params.get('LP', 0.7)
        self.BETA = params.get('BETA', 2.0)
        
        # Response routine parameters
        self.PERC = params.get('PERC', 2.0)
        self.UZL = params.get('UZL', 20.0)
        self.K0 = params.get('K0', 0.5)
        self.K1 = params.get('K1', 0.1)
        self.K2 = params.get('K2', 0.05)
        
        # Routing parameter
        self.MAXBAS = params.get('MAXBAS', 3.0)
        
        # State variables
        self.SP = 0.0   # Snow pack (mm)
        self.WC = 0.0   # Water content in snow (mm)
        self.SM = 0.0   # Soil moisture (mm)
        self.SUZ = 0.0  # Upper zone storage (mm)
        self.SLZ = 0.0  # Lower zone storage (mm)
        
        # Triangular weighting function for routing
        self._compute_routing_weights()
        
    def _compute_routing_weights(self):
        """Compute triangular weighting function for routing"""
        n = int(np.ceil(self.MAXBAS))
        self.routing_weights = np.zeros(n)
        
        for i in range(n):
            if i < self.MAXBAS / 2:
                self.routing_weights[i] = (i + 1) / (self.MAXBAS / 2)
            else:
                self.routing_weights[i] = (self.MAXBAS - i) / (self.MAXBAS / 2)
                
        # Normalize
        self.routing_weights = self.routing_weights / np.sum(self.routing_weights)
        
    def _snow_routine(self, P, T):
        """
        Snow accumulation and melt routine.
        
        Args:
            P: Precipitation (mm)
            T: Temperature (°C)
            
        Returns:
            Liquid water input to soil (mm)
        """
        # Snow or rain
        if T < self.TT:
            # Snow
            self.SP += P
            rain = 0.0
        else:
            # Rain
            rain = P
            
        # Snow melt
        if T > self.TT:
            melt = self.CFMAX * (T - self.TT)
            melt = min(melt, self.SP)
            self.SP -= melt
            self.WC += melt + rain
        else:
            # Refreezing
            refreeze = self.CFR * self.CFMAX * (self.TT - T)
            refreeze = min(refreeze, self.WC)
            self.SP += refreeze
            self.WC -= refreeze
            
        # Water holding capacity of snow
        if self.WC > self.CWH * self.SP:
            liquid = self.WC - self.CWH * self.SP
            self.WC = self.CWH * self.SP
        else:
            liquid = 0.0
            
        return liquid
        
    def _soil_routine(self, liquid, EP):
        """
        Soil moisture accounting routine.
        
        Args:
            liquid: Liquid water input (mm)
            EP: Potential evapotranspiration (mm)
            
        Returns:
            Recharge to response routine (mm)
        """
        # Actual evapotranspiration
        lp_fc = self.LP * self.FC
        if lp_fc > 0 and self.SM / self.FC < self.LP:
            EA = EP * (self.SM / lp_fc)
        else:
            EA = EP
            
        EA = min(EA, self.SM)
        self.SM -= EA
        
        # Recharge to groundwater
        if self.FC > 0:
            recharge = liquid * (self.SM / self.FC) ** self.BETA
        else:
            recharge = 0.0
            
        self.SM += liquid - recharge
        
        # Soil moisture cannot exceed field capacity
        if self.SM > self.FC:
            recharge += self.SM - self.FC
            self.SM = self.FC
            
        return recharge
        
    def _response_routine(self, recharge):
        """
        Response and routing routine.
        
        Args:
            recharge: Recharge from soil routine (mm)
            
        Returns:
            Total runoff (mm)
        """
        # Add recharge to upper zone
        self.SUZ += recharge
        
        # Percolation to lower zone
        perc = min(self.PERC, self.SUZ)
        self.SUZ -= perc
        self.SLZ += perc
        
        # Quick runoff (only if SUZ > UZL)
        if self.SUZ > self.UZL:
            Q0 = self.K0 * (self.SUZ - self.UZL)
            self.SUZ -= Q0
        else:
            Q0 = 0.0
            
        # Upper zone runoff
        Q1 = self.K1 * self.SUZ
        self.SUZ -= Q1
        
        # Lower zone runoff
        Q2 = self.K2 * self.SLZ
        self.SLZ -= Q2
        
        # Total runoff
        Q_total = Q0 + Q1 + Q2
        
        return Q_total
        
    def run(self, P, T, EP, SP_init=None, SM_init=None, SUZ_init=None, SLZ_init=None):
        """
        Run the HBV model for a time series.
        
        Args:
            P (array-like): Daily precipitation (mm)
            T (array-like): Daily temperature (°C)
            EP (array-like): Daily potential evapotranspiration (mm)
            SP_init (float, optional): Initial snow pack
            SM_init (float, optional): Initial soil moisture
            SUZ_init (float, optional): Initial upper zone storage
            SLZ_init (float, optional): Initial lower zone storage
            
        Returns:
            dict: Dictionary containing:
                - Q: Simulated discharge (mm/day)
                - SP: Snow pack storage
                - SM: Soil moisture
                - SUZ: Upper zone storage
                - SLZ: Lower zone storage
        """
        P = np.array(P, dtype=float)
        T = np.array(T, dtype=float)
        EP = np.array(EP, dtype=float)
        n = len(P)
        
        # Initialize state variables
        if SP_init is not None:
            self.SP = SP_init
        if SM_init is not None:
            self.SM = SM_init
        if SUZ_init is not None:
            self.SUZ = SUZ_init
        if SLZ_init is not None:
            self.SLZ = SLZ_init
            
        # Output arrays
        Q = np.zeros(n)
        SP_store = np.zeros(n)
        SM_store = np.zeros(n)
        SUZ_store = np.zeros(n)
        SLZ_store = np.zeros(n)
        
        # Routing store
        n_routing = len(self.routing_weights)
        routing_store = np.zeros(n_routing)
        
        for t in range(n):
            # Snow routine
            liquid = self._snow_routine(P[t], T[t])
            
            # Soil routine
            recharge = self._soil_routine(liquid, EP[t])
            
            # Response routine
            Q_direct = self._response_routine(recharge)
            
            # Routing
            routing_store = np.roll(routing_store, 1)
            routing_store[0] = Q_direct
            Q[t] = np.sum(routing_store * self.routing_weights[:n_routing])
            
            # Store state variables
            SP_store[t] = self.SP
            SM_store[t] = self.SM
            SUZ_store[t] = self.SUZ
            SLZ_store[t] = self.SLZ
            
        return {
            'Q': Q,
            'SP': SP_store,
            'SM': SM_store,
            'SUZ': SUZ_store,
            'SLZ': SLZ_store
        }
        
    def reset(self):
        """Reset model state variables"""
        self.SP = 0.0
        self.WC = 0.0
        self.SM = 0.0
        self.SUZ = 0.0
        self.SLZ = 0.0
