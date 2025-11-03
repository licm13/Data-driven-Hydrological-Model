"""
Data-driven Hydrological Model Package

∞ Staudinger et al. (2025) - How well do process-based and data-driven
hydrological models learn from limited discharge data
"""

__version__ = '0.1.0'
__author__ = 'Replication Team'

# ¸e;Å!W
from . import models
from . import metrics
from . import calibration
from . import sampling
from . import utils

__all__ = [
    'models',
    'metrics',
    'calibration',
    'sampling',
    'utils',
]
