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
## How well do process-based and data-driven hydrological models learn from limited discharge data?

### 概述 (Overview)

本项目对比三种过程驱动模型（GR4J、HBV、SWAT+）与四种数据驱动模型（EDDIS、RTREE、ANN、LSTM）在不同训练数据量下的学习曲线，揭示了在有限径流资料条件下二者学习性能的分界与优势互补规律。

This project compares the learning curves of three process-driven models (GR4J, HBV, SWAT+) and four data-driven models (EDDIS, RTREE, ANN, LSTM) under different training data amounts, revealing the boundary and complementary advantages of their learning performance under limited runoff data conditions.

### 核心研究问题 (Core Research Question)

系统回答了水文学核心问题之一：**"数据稀缺条件下模型的学习能力与信息效率"**

This work addresses one of the core hydrological questions: **"Model learning capability and information efficiency under data-scarce conditions"**

---

## 项目结构 (Project Structure)

```
Data-driven-Hydrological-Model/
├── src/
│   ├── models/
│   │   ├── process_driven/      # 过程驱动模型
│   │   │   ├── gr4j.py          # GR4J 模型
│   │   │   ├── hbv.py           # HBV 模型
│   │   │   └── swat_plus.py     # SWAT+ 模型
│   │   └── data_driven/         # 数据驱动模型
│   │       ├── eddis.py         # EDDIS 模型
│   │       ├── rtree.py         # 随机森林回归树
│   │       ├── ann.py           # 人工神经网络
│   │       └── lstm.py          # 长短期记忆网络
│   ├── utils/
│   │   ├── data_loader.py       # 数据加载与预处理
│   │   ├── metrics.py           # 评估指标 (NSE, RMSE, KGE等)
│   │   └── visualization.py     # 可视化工具
│   └── evaluation/
│       └── learning_curves.py   # 学习曲线评估框架
├── examples/
│   └── compare_models.py        # 主示例脚本
├── requirements.txt             # 依赖包列表
└── README.md
```

---

## 模型说明 (Model Descriptions)

### 过程驱动模型 (Process-Driven Models)

1. **GR4J** (Génie Rural à 4 paramètres Journalier)
   - 4参数日尺度水文模型
   - 基于生产库和汇流库的概念模型
   - 参数: X1 (生产库容量), X2 (地下水交换), X3 (汇流库容量), X4 (单位线时间)

2. **HBV** (Hydrologiska Byråns Vattenbalansavdelning)
   - 概念性水文模型
   - 包含积雪、土壤水分和响应模块
   - 参数: FC (土壤水容量), BETA (形状参数), LP (蒸散发阈值), K0/K1 (退水系数)

3. **SWAT+** (Soil and Water Assessment Tool Plus)
   - 流域尺度水文模型
   - 基于SCS径流曲线数方法
   - 参数: CN (曲线数), ESCO (土壤蒸发补偿), ALPHA_BF (基流系数)

### 数据驱动模型 (Data-Driven Models)

1. **EDDIS** (Event-Driven Data-Informed System)
   - 基于事件检测的数据驱动系统
   - 使用多项式特征和岭回归

2. **RTREE** (Regression Tree)
   - 随机森林回归
   - 使用时间滞后特征和滚动统计量

3. **ANN** (Artificial Neural Network)
   - 多层感知器神经网络
   - 包含数据标准化和早停机制

4. **LSTM** (Long Short-Term Memory)
   - 循环神经网络
   - 适用于序列预测问题

---

## 安装与环境配置 (Installation)

### 1. 克隆仓库 (Clone Repository)

```bash
git clone https://github.com/licm13/Data-driven-Hydrological-Model.git
cd Data-driven-Hydrological-Model
```

2. Install dependencies:
### 2. 安装依赖 (Install Dependencies)

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
### 主要依赖包 (Main Dependencies):
- numpy, pandas: 数据处理
- scikit-learn: 机器学习模型
- torch: 深度学习 (LSTM)
- matplotlib, seaborn: 可视化
- tqdm: 进度条

---

## 使用方法 (Usage)

### 运行主示例 (Run Main Example)

```bash
python examples/compare_models.py
```

### 主要输出 (Main Outputs):

1. **学习曲线图** (`learning_curves.png`)
   - 展示所有模型在不同训练数据量下的性能

2. **性能对比图** (`performance_comparison.png`)
   - 在特定训练规模下的模型对比

3. **模型类型对比图** (`model_type_comparison.png`)
   - 过程驱动与数据驱动模型的平均性能对比

4. **效率分析图** (`efficiency_analysis.png`)
   - 学习效率和性能改进分析

5. **结果CSV文件** (`learning_curves_results.csv`)
   - 详细的数值结果

---

## 评估指标 (Evaluation Metrics)

- **NSE** (Nash-Sutcliffe Efficiency): 纳什效率系数
- **RMSE** (Root Mean Squared Error): 均方根误差
- **MAE** (Mean Absolute Error): 平均绝对误差
- **KGE** (Kling-Gupta Efficiency): KGE效率系数
- **PBIAS** (Percent Bias): 百分比偏差

---

## 主要发现 (Key Findings)

### 1. 学习性能分界 (Learning Performance Boundary)

- **数据稀缺时** (< 1年训练数据):
  - 过程驱动模型表现更稳定
  - 融入物理知识和约束
  - 具有更好的泛化能力

- **数据充足时** (> 2年训练数据):
  - 数据驱动模型性能优越
  - 能够学习复杂的非线性模式
  - 更灵活适应局地条件

### 2. 优势互补规律 (Complementary Advantages)

- **过程驱动模型优势**:
  - 物理可解释性强
  - 参数具有物理意义
  - 在无资料流域具有更好的迁移性

- **数据驱动模型优势**:
  - 学习能力强
  - 能够捕捉复杂的时空模式
  - 在数据丰富区域性能更优

### 3. 实践建议 (Practical Recommendations)

- **数据稀缺地区**: 优先使用过程驱动模型或混合方法
- **数据丰富地区**: 数据驱动模型可提供卓越性能
- **综合策略**: 考虑集成方法，结合两种模型的优势

---

## 自定义使用 (Custom Usage)

### 使用自己的数据 (Using Your Own Data)

```python
from src.utils.data_loader import HydrologicalDataLoader
from src.models.process_driven.gr4j import GR4J

# 加载数据
loader = HydrologicalDataLoader(data_path='your_data.csv')
# 数据应包含: precipitation, temperature, pet, discharge

# 初始化模型
model = GR4J()

# 训练模型
X_train = data[['precipitation', 'temperature', 'pet']].values
y_train = data['discharge'].values
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
```

### 添加新模型 (Adding New Models)

```python
class YourModel:
    def __init__(self, **params):
        # 初始化参数
        pass
    
    def fit(self, X_train, y_train):
        # 训练逻辑
        pass
    
    def predict(self, X):
        # 预测逻辑
        return predictions
```

---

## 引用 (Citation)

如果您使用了本项目，请引用：

If you use this project, please cite:

```
@software{hydrological_model_comparison,
  title={Data-driven Hydrological Model: Learning Curve Comparison},
  author={Hydrological Model Team},
  year={2024},
  url={https://github.com/licm13/Data-driven-Hydrological-Model}
}
```

---

## 贡献 (Contributing)

欢迎贡献代码、报告问题或提出改进建议！

Contributions, bug reports, and feature requests are welcome!

---

## 许可证 (License)

MIT License

---

## 联系方式 (Contact)

如有问题，请通过GitHub Issues联系。

For questions, please contact via GitHub Issues.
