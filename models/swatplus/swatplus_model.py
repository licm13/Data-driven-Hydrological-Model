"""
SWAT+ Model Implementation
Soil and Water Assessment Tool Plus

A simplified implementation of SWAT+ for watershed-scale hydrological modeling.
SWAT+ is a comprehensive watershed model that simulates water, sediment, nutrient,
and pesticide transport at the watershed scale.

This implementation provides a simplified version focusing on core hydrological processes:
- Surface runoff using SCS Curve Number method
- Percolation and lateral flow
- Groundwater flow
- Channel routing

Reference:
    Bieger, K., Arnold, J. G., Rathjens, H., White, M. J., Bosch, D. D., Allen, P. M., 
    Volk, M., & Srinivasan, R. (2017). Introduction to SWAT+, a completely restructured 
    version of the soil and water assessment tool. JAWRA Journal of the American Water 
    Resources Association, 53(1), 115-130.
"""

import numpy as np


class SWATPlus:
    """
    Simplified SWAT+ watershed model
    
    Implements core hydrological processes for watershed simulation.
    """
    
    def __init__(self, params):
        """
        Initialize SWAT+ model with parameters.
        
        Args:
            params (dict): Model parameters including:
                - CN2: SCS Curve Number for moisture condition II
                - ESCO: Soil evaporation compensation factor (0-1)
                - EPCO: Plant uptake compensation factor (0-1)
                - SURLAG: Surface runoff lag coefficient (days)
                - ALPHA_BF: Baseflow recession constant (days)
                - GW_DELAY: Groundwater delay time (days)
                - GW_REVAP: Groundwater revap coefficient
                - REVAPMN: Threshold depth of water in shallow aquifer for revap (mm)
                - RCHRG_DP: Deep aquifer percolation fraction
                - GWQMN: Threshold depth of water in shallow aquifer for return flow (mm)
                - SOL_AWC: Available water capacity of soil layer (mm/mm)
                - SOL_Z: Soil depth (mm)
        """
        # Surface runoff parameters
        self.CN2 = params.get('CN2', 75.0)
        self.SURLAG = params.get('SURLAG', 4.0)
        
        # Evapotranspiration parameters
        self.ESCO = params.get('ESCO', 0.95)
        self.EPCO = params.get('EPCO', 1.0)
        
        # Groundwater parameters
        self.ALPHA_BF = params.get('ALPHA_BF', 0.048)
        self.GW_DELAY = params.get('GW_DELAY', 31.0)
        self.GW_REVAP = params.get('GW_REVAP', 0.02)
        self.REVAPMN = params.get('REVAPMN', 750.0)
        self.RCHRG_DP = params.get('RCHRG_DP', 0.05)
        self.GWQMN = params.get('GWQMN', 1000.0)
        
        # Soil parameters
        self.SOL_AWC = params.get('SOL_AWC', 0.15)  # mm/mm
        self.SOL_Z = params.get('SOL_Z', 1000.0)    # mm
        
        # Calculate maximum soil water storage
        self.SOL_SW_MAX = self.SOL_AWC * self.SOL_Z
        
        # State variables
        self.SW = 0.0           # Soil water content (mm)
        self.GW = 0.0           # Shallow aquifer storage (mm)
        self.GW_DEEP = 0.0      # Deep aquifer storage (mm)
        self.SURQ_lag = []      # Surface runoff lag storage
        self.RCHRG_lag = []     # Recharge lag storage
        
    def _calculate_curve_number(self, SW):
        """
        Adjust curve number based on soil moisture.
        
        Args:
            SW: Current soil water content (mm)
            
        Returns:
            Adjusted curve number
        """
        # Retention parameter
        S = 25.4 * (1000.0 / self.CN2 - 10.0)
        
        # Adjust based on soil moisture
        w1 = SW / self.SOL_SW_MAX
        w2 = 1.0 - w1
        
        # Modified CN
        if w1 > 0.5:
            CN = self.CN2 + (100 - self.CN2) * (w1 - 0.5) * 2
        else:
            CN = self.CN2 - (self.CN2 - 40) * (0.5 - w1) * 2
            
        return np.clip(CN, 40, 100)
        
    def _surface_runoff(self, P, SW):
        """
        Calculate surface runoff using SCS Curve Number method.
        
        Args:
            P: Precipitation (mm)
            SW: Current soil water (mm)
            
        Returns:
            Surface runoff (mm)
        """
        if P <= 0:
            return 0.0
            
        CN = self._calculate_curve_number(SW)
        S = 25.4 * (1000.0 / CN - 10.0)
        Ia = 0.2 * S
        
        if P > Ia:
            Q_surf = ((P - Ia) ** 2) / (P - Ia + S)
        else:
            Q_surf = 0.0
            
        return Q_surf
        
    def _percolation(self, SW, excess):
        """
        Calculate percolation to groundwater.
        
        Args:
            SW: Current soil water (mm)
            excess: Excess water in soil (mm)
            
        Returns:
            Percolation amount (mm)
        """
        if excess > 0:
            # Percolation rate depends on soil water content
            perc_rate = excess * (SW / self.SOL_SW_MAX)
            return perc_rate
        return 0.0
        
    def _evapotranspiration(self, PET, SW):
        """
        Calculate actual evapotranspiration.
        
        Args:
            PET: Potential evapotranspiration (mm)
            SW: Current soil water (mm)
            
        Returns:
            Actual evapotranspiration (mm)
        """
        # Soil evaporation demand
        E_soil = PET * self.ESCO
        
        # Reduce based on soil water availability
        if SW < 0.5 * self.SOL_SW_MAX:
            E_soil *= SW / (0.5 * self.SOL_SW_MAX)
            
        return min(E_soil, SW)
        
    def _groundwater_flow(self):
        """
        Calculate groundwater contribution to streamflow.
        
        Returns:
            Baseflow (mm)
        """
        if self.GW > self.GWQMN:
            Q_gw = self.ALPHA_BF * (self.GW - self.GWQMN)
            return Q_gw
        return 0.0
        
    def run(self, P, PET, SW_init=None, GW_init=None):
        """
        Run the SWAT+ model for a time series.
        
        Args:
            P (array-like): Daily precipitation (mm)
            PET (array-like): Daily potential evapotranspiration (mm)
            SW_init (float, optional): Initial soil water content
            GW_init (float, optional): Initial shallow aquifer storage
            
        Returns:
            dict: Dictionary containing:
                - Q_total: Total streamflow (mm/day)
                - Q_surf: Surface runoff (mm/day)
                - Q_gw: Groundwater flow (mm/day)
                - SW: Soil water content (mm)
                - GW: Shallow aquifer storage (mm)
        """
        P = np.array(P, dtype=float)
        PET = np.array(PET, dtype=float)
        n = len(P)
        
        # Initialize state variables
        if SW_init is not None:
            self.SW = SW_init
        else:
            self.SW = 0.5 * self.SOL_SW_MAX
            
        if GW_init is not None:
            self.GW = GW_init
            
        # Initialize lag storages
        lag_steps = int(np.ceil(max(self.SURLAG, self.GW_DELAY)))
        self.SURQ_lag = [0.0] * lag_steps
        self.RCHRG_lag = [0.0] * lag_steps
        
        # Output arrays
        Q_total = np.zeros(n)
        Q_surf = np.zeros(n)
        Q_gw = np.zeros(n)
        SW_store = np.zeros(n)
        GW_store = np.zeros(n)
        
        for t in range(n):
            # Surface runoff
            Q_s = self._surface_runoff(P[t], self.SW)
            
            # Infiltration
            infil = P[t] - Q_s
            self.SW += infil
            
            # Actual evapotranspiration
            ET_act = self._evapotranspiration(PET[t], self.SW)
            self.SW -= ET_act
            
            # Percolation to groundwater
            if self.SW > self.SOL_SW_MAX:
                excess = self.SW - self.SOL_SW_MAX
                self.SW = self.SOL_SW_MAX
                perc = self._percolation(self.SW, excess)
            else:
                perc = 0.0
                
            # Add percolation to recharge lag
            self.RCHRG_lag.append(perc)
            rchrg = self.RCHRG_lag.pop(0)
            
            # Split recharge between shallow and deep aquifer
            rchrg_shallow = rchrg * (1.0 - self.RCHRG_DP)
            rchrg_deep = rchrg * self.RCHRG_DP
            
            self.GW += rchrg_shallow
            self.GW_DEEP += rchrg_deep
            
            # Groundwater flow to stream
            Q_g = self._groundwater_flow()
            self.GW -= Q_g
            
            # Surface runoff lag
            self.SURQ_lag.append(Q_s)
            Q_s_delayed = self.SURQ_lag.pop(0)
            
            # Total streamflow
            Q_t = Q_s_delayed + Q_g
            
            # Store outputs
            Q_total[t] = Q_t
            Q_surf[t] = Q_s_delayed
            Q_gw[t] = Q_g
            SW_store[t] = self.SW
            GW_store[t] = self.GW
            
        return {
            'Q_total': Q_total,
            'Q_surf': Q_surf,
            'Q_gw': Q_gw,
            'SW': SW_store,
            'GW': GW_store
        }
        
    def reset(self):
        """Reset model state variables"""
        self.SW = 0.0
        self.GW = 0.0
        self.GW_DEEP = 0.0
        self.SURQ_lag = []
        self.RCHRG_lag = []
