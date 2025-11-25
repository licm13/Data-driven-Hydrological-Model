"""
Experiment framework for hydrological model studies.

This package provides a unified interface for running complex hydrological
modeling experiments, including:
    - Learning curve analysis
    - Sampling strategy comparison
    - Information content assessment
    - Spatial distribution studies

All experiments follow a consistent workflow pattern defined by BaseExperiment.
"""

from dmg.experiments.base_experiment import BaseExperiment
from dmg.experiments.task_registry import ExperimentRegistry

__all__ = [
    'BaseExperiment',
    'ExperimentRegistry',
]
