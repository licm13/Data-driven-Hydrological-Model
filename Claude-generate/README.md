# 水文模型学习曲线研究

复刻论文：*Staudinger et al. (2025) - How well do process-based and data-driven hydrological models learn from limited discharge data?*

## 项目概述

本项目实现了论文中的三个过程驱动水文模型（GR4J、HBV、SWAT+）以及完整的实验框架，用于比较不同模型在有限径流数据下的学习能力。

## 主要功能

- **三个过程驱动模型**：
  - GR4J：4参数日尺度集总式模型 + CemaNeige雪模块
  - HBV：半分布式模型，支持高程带划分
  - SWAT+：基于HRU的半分布式模型
  
- **评估指标**：
  - 信息熵（Joint Entropy, Conditional Entropy）
  - KGE（Kling-Gupta Efficiency）
  - NSE（Nash-Sutcliffe Efficiency）

- **采样策略**：
  - 完全随机采样
  - 随机连续采样
  - Douglas-Peucker最优采样

## 安装
```bash
pip install -r requirements.txt
```

## 快速开始
```python
from src.models.hbv import HBV
from src.metrics.entropy import evaluate_model_entropy
import numpy as np

# 创建HBV模型
model = HBV(n_elevation_zones=3)

# 初始化参数
params = {
    'TT': 0.0,
    'CFMAX': 3.5,
    'FC': 250.0,
    'BETA': 2.0,
    'K0': 0.2,
    'K1': 0.1,
    'K2': 0.05,
}
model.initialize(params)

# 模拟
precip = np.random.rand(365) * 10
temp = np.random.randn(365) * 10
pet = np.random.rand(365) * 3
discharge = model.simulate(precip, temp, pet)

# 评估
metrics = evaluate_model_entropy(obs, discharge)
print(f"Conditional Entropy: {metrics['H_conditional']:.3f} bits")
```

## 实验复现

### 实验1：学习曲线
```bash
python experiments/experiment_1_learning_curves.py --catchment Iller
```

### 实验2：采样策略
```bash
python experiments/experiment_2_sampling_strategies.py --model HBV
```

## 项目结构

详见文档顶部的项目结构树。

## 引用

如果使用本代码，请引用原论文：
```bibtex
@article{staudinger2025learning,
  title={How well do process-based and data-driven hydrological models learn from limited discharge data?},
  author={Staudinger, Maria and Herzog, Anna and Loritz, Ralf and others},
  journal={Hydrology and Earth System Sciences},
  volume={29},
  pages={5005--5029},
  year={2025},
  publisher={Copernicus GmbH}
}
```

## 许可证

MIT License