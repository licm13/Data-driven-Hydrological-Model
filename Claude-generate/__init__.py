"""
模型工厂和便捷导入
"""
from .base_model import BaseHydrologicalModel
from .gr4j import GR4J
from .hbv import HBV
from .swat_plus import SWATPlus
from .eddis import EDDIS
from .rtree import RTREE
from .ann import ANN
from .lstm import LSTM

__all__ = [
    'BaseHydrologicalModel',
    'GR4J',
    'HBV',
    'SWATPlus',
    'EDDIS',
    'RTREE',
    'ANN',
    'LSTM',
    'get_model',
]


def get_model(model_name: str, **kwargs):
    """
    模型工厂函数
    
    Parameters:
    -----------
    model_name : str, 模型名称
    **kwargs : 模型初始化参数
    
    Returns:
    --------
    model : 模型实例
    
    Example:
    --------
    >>> model = get_model('HBV', n_elevation_zones=3)
    >>> model = get_model('LSTM', hidden_size=128)
    """
    models = {
        'gr4j': GR4J,
        'hbv': HBV,
        'swat+': SWATPlus,
        'swat': SWATPlus,
        'eddis': EDDIS,
        'rtree': RTREE,
        'ann': ANN,
        'lstm': LSTM,
    }
    
    model_name_lower = model_name.lower()
    
    if model_name_lower not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )
    
    return models[model_name_lower](**kwargs)


# 模型分类
PROCESS_BASED_MODELS = ['GR4J', 'HBV', 'SWAT+']
DATA_DRIVEN_MODELS = ['EDDIS', 'RTREE', 'ANN', 'LSTM']
ALL_MODELS = PROCESS_BASED_MODELS + DATA_DRIVEN_MODELS