# Data-driven-Hydrological-Model

How well do process-based and data-driven hydrological models learn from limited discharge data?

## Overview

This repository provides implementations of three widely-used process-based hydrological models:

1. **GR4J** (Génie Rural à 4 paramètres Journalier) - A 4-parameter daily rainfall-runoff model
2. **HBV** (Hydrologiska Byråns Vattenbalansavdelning) - A conceptual hydrological model with snow routine
3. **SWAT+** (Soil and Water Assessment Tool Plus) - A watershed-scale hydrological model

These models can be used to study and compare process-based approaches with data-driven methods in hydrological modeling.

## Installation

### Requirements

- Python 3.7 or higher
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/licm13/Data-driven-Hydrological-Model.git
cd Data-driven-Hydrological-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Models

### GR4J Model

A 4-parameter daily conceptual rainfall-runoff model developed by IRSTEA (formerly Cemagref).

**Parameters:**
- X1: Maximum capacity of production store (mm)
- X2: Groundwater exchange coefficient (mm/day)
- X3: Maximum capacity of routing store (mm)
- X4: Time base of unit hydrograph (days)

**Usage:**
```python
from models import GR4J

model = GR4J(X1=350.0, X2=0.0, X3=90.0, X4=1.7)
results = model.run(precipitation, evapotranspiration)
discharge = results['Q']
```

### HBV Model

A conceptual hydrological model with snow accumulation and melt routines, making it suitable for cold climates.

**Key Features:**
- Snow accumulation and melt
- Soil moisture accounting
- Response function for runoff generation
- Routing routine

**Usage:**
```python
from models import HBV

params = {
    'TT': 0.0,      # Threshold temperature
    'CFMAX': 3.5,   # Degree-day factor
    'FC': 200.0,    # Field capacity
    'BETA': 2.0,    # Shape coefficient
    'K1': 0.1,      # Upper zone recession
    'K2': 0.05,     # Lower zone recession
    'MAXBAS': 3.0   # Routing parameter
}

model = HBV(params)
results = model.run(precipitation, temperature, evapotranspiration)
discharge = results['Q']
```

### SWAT+ Model

A watershed-scale model that simulates hydrological processes with explicit representation of surface runoff and baseflow.

**Key Features:**
- SCS Curve Number method for surface runoff
- Soil moisture accounting
- Groundwater flow
- Separate surface and subsurface flow components

**Usage:**
```python
from models import SWATPlus

params = {
    'CN2': 75.0,        # SCS Curve Number
    'ALPHA_BF': 0.048,  # Baseflow recession
    'GW_DELAY': 31.0,   # Groundwater delay
    'SOL_AWC': 0.15,    # Available water capacity
    'SOL_Z': 1000.0     # Soil depth
}

model = SWATPlus(params)
results = model.run(precipitation, evapotranspiration)
discharge = results['Q_total']
```

## Examples

The `examples/` directory contains demonstration scripts:

- `example_gr4j.py` - Basic GR4J model usage
- `example_hbv.py` - HBV model with snow simulation
- `example_swatplus.py` - SWAT+ model with flow components
- `compare_models.py` - Compare all three models

Run an example:
```bash
python examples/example_gr4j.py
python examples/compare_models.py
```

## Utilities

The `utils/` module provides performance metrics for model evaluation:

- Nash-Sutcliffe Efficiency (NSE)
- Root Mean Square Error (RMSE)
- Percent Bias (PBIAS)
- Kling-Gupta Efficiency (KGE)

**Usage:**
```python
from utils import calculate_all_metrics

metrics = calculate_all_metrics(observed, simulated)
print(f"NSE: {metrics['NSE']:.3f}")
print(f"RMSE: {metrics['RMSE']:.3f}")
```

## Project Structure

```
Data-driven-Hydrological-Model/
├── models/
│   ├── gr4j/
│   │   ├── __init__.py
│   │   └── gr4j_model.py
│   ├── hbv/
│   │   ├── __init__.py
│   │   └── hbv_model.py
│   ├── swatplus/
│   │   ├── __init__.py
│   │   └── swatplus_model.py
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
├── examples/
│   ├── example_gr4j.py
│   ├── example_hbv.py
│   ├── example_swatplus.py
│   └── compare_models.py
├── requirements.txt
└── README.md
```

## References

### GR4J
Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.

### HBV
Bergström, S. (1995). The HBV model. In Computer Models of Watershed Hydrology (pp. 443-476). Water Resources Publications.

### SWAT+
Bieger, K., Arnold, J. G., Rathjens, H., White, M. J., Bosch, D. D., Allen, P. M., Volk, M., & Srinivasan, R. (2017). Introduction to SWAT+, a completely restructured version of the soil and water assessment tool. JAWRA Journal of the American Water Resources Association, 53(1), 115-130.

## License

This project is open source and available for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
