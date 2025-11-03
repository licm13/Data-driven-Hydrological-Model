"""
Model registry and lightweight imports for hydrological models.

This module exposes a lazy get_model() to avoid importing heavy dependencies
like PyTorch unless the corresponding model is actually requested.
It also keeps HBV directly importable for convenience in scripts.
"""
from importlib import import_module
from typing import Any

# Keep HBV directly importable (used by some experiment scripts)
from .hbv import HBV  # noqa: F401

__all__ = [
    'HBV',
    'get_model',
    'PROCESS_BASED_MODELS',
    'DATA_DRIVEN_MODELS',
    'ALL_MODELS',
]


def get_model(model_name: str, **kwargs: Any):
    """Factory that returns a model instance by name.

    Supported names (case-insensitive): GR4J, HBV, SWAT+, EDDIS, RTREE, ANN, LSTM.
    Heavy backends (e.g., PyTorch for LSTM) are imported lazily only when needed.
    """
    mapping = {
        'gr4j': ('src.models.gr4j', 'GR4J'),
        'hbv': ('src.models.hbv', 'HBV'),
        'swat+': ('src.models.swat_plus', 'SWATPlus'),
        'swat': ('src.models.swat_plus', 'SWATPlus'),
        'eddis': ('src.models.eddis', 'EDDIS'),
        'rtree': ('src.models.rtree', 'RTREE'),
        'ann': ('src.models.ann', 'ANN'),
        'lstm': ('src.models.lstm', 'LSTM'),
    }

    key = model_name.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(mapping.keys())}"
        )

    module_name, class_name = mapping[key]
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls(**kwargs)


PROCESS_BASED_MODELS = ['GR4J', 'HBV', 'SWAT+']
DATA_DRIVEN_MODELS = ['EDDIS', 'RTREE', 'ANN', 'LSTM']
ALL_MODELS = PROCESS_BASED_MODELS + DATA_DRIVEN_MODELS
