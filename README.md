# 数据驱动水文模型学习曲线研究

> **基于论文复现**：*Staudinger et al. (2025) - How well do process-based and data-driven hydrological models learn from limited discharge data?*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📖 项目简介

本项目实现了一个完整的水文模型对比框架，用于评估**过程驱动模型**和**数据驱动模型**在有限径流数据下的学习能力。项目包含7个水文模型、多种评估指标和完整的实验框架，可用于：

- 🔬 **科学研究**：水文模型对比、信息论分析、学习曲线研究
- 📚 **教学培训**：水文建模教学、模型原理演示
- 🏞️ **实际应用**：流域径流模拟、模型选择、参数校准

### 核心研究问题

1. **过程驱动模型 vs 数据驱动模型**：哪类模型在数据有限时表现更好？
2. **样本效率**：达到满意性能需要多少训练数据？
3. **信息内容**：训练数据的质量（信息量）如何影响模型性能？
4. **采样策略**：如何选择最具信息量的训练样本？

---

## ✨ 主要特性

### 🎯 7个水文模型

#### 过程驱动模型 (Process-Based Models)
| 模型 | 类型 | 参数数 | 特点 | 适用场景 |
|------|------|--------|------|----------|
| **GR4J** | 集总式 | 4-6 | 简单高效，全球广泛应用 | 日尺度径流模拟 |
| **HBV** | 半分布式 | 10-15 | 支持高程带，物理过程清晰 | 山区流域、雪融径流 |
| **SWAT+** | 半分布式 | 100+ | HRU结构，适合复杂流域 | 流域尺度水文水质模拟 |

#### 数据驱动模型 (Data-Driven Models)
| 模型 | 类型 | 特点 | 训练数据需求 |
|------|------|------|--------------|
| **EDDIS** | 经验分布 | 无参数，完全基于数据分布 | 极少 |
| **RTREE** | 决策树 | 非线性，可解释性强 | 中等 |
| **ANN** | 前馈神经网络 | 强拟合能力，需手动特征工程 | 较多 |
| **LSTM** | 循环神经网络 | 自动学习时序依赖，适合长期记忆 | 较多 |

### 📊 评估指标体系

#### 传统水文指标
- **KGE** (Kling-Gupta Efficiency)：综合相关性、变异性和偏差
- **NSE** (Nash-Sutcliffe Efficiency)：经典水文效率系数

#### 信息论指标
- **联合熵** H(X,Y)：观测和模拟的总不确定性
- **条件熵** H(Y|X)：给定模拟值时观测值的剩余不确定性（越低越好）
- **互信息** I(X;Y)：模拟与观测的共享信息
- **归一化条件熵**：标准化后的条件熵，便于跨流域对比

### 🧪 4个完整实验

1. **实验1：学习曲线对比** - 评估模型性能随训练数据量的变化
2. **实验2：采样策略影响** - 对比随机采样、连续采样和Douglas-Peucker采样
3. **实验3：信息内容分析** - 分析径流数据的信息熵特性
4. **实验4：空间分布效应** - 研究高程带对HBV模型的影响

### 🚀 实用工具

- ✅ **自动数据检测**：智能识别多种数据格式（CSV, ASCII等）
- ✅ **自动参数校准**：集成SPOTPY优化框架
- ✅ **丰富可视化**：学习曲线、散点图、雷达图、时间序列等
- ✅ **合成数据生成**：无需真实数据即可测试
- ✅ **模块化设计**：易于扩展新模型和指标

---

## 🔧 安装

### 环境要求

- Python 3.8+
- 操作系统：Linux, macOS, Windows

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/licm13/Data-driven-Hydrological-Model.git
cd Data-driven-Hydrological-Model

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 核心依赖包

- **数值计算**: numpy, pandas, scipy
- **机器学习**: scikit-learn, torch
- **水文工具**: spotpy (参数校准)
- **可视化**: matplotlib, seaborn
- **其他**: pyyaml, tqdm, jupyter

---

## 🚀 快速开始

### 方式1：运行示例代码（推荐初学者）

```bash
# 示例1：基础模型使用
python examples/01_basic_model_usage.py

# 示例2：多模型对比
python examples/02_compare_multiple_models.py

# 示例3：真实数据工作流
python examples/03_real_data_workflow.py

# 示例4：学习曲线分析
python examples/04_learning_curves_analysis.py
```

详细说明见 [`examples/README.md`](examples/README.md)

### 方式2：快速测试（使用合成数据）

```bash
# 运行快速测试（约5-10分钟）
python run_all_experiments.py --synthetic --quick_test
```

### 方式3：使用真实数据

#### 步骤1：准备数据

支持两种数据格式：

**格式A：标准CSV**
```
data/raw/
├── Iller/
│   ├── meteorology.csv  # 列名：date, precip, temp, pet
│   └── discharge.csv    # 列名：date, discharge
├── Saale/
└── Selke/
```

**格式B：IMPRO ASCII**
```
IMPRO_catchment_data_infotheo/
├── iller/
│   ├── precipitation_iller.asc
│   ├── temperature_iller.asc
│   ├── pet_iller.asc
│   └── discharge_iller.csv
```

#### 步骤2：运行实验

```bash
# 单个实验
python experiments/experiment_1_learning_curves.py \
    --catchment Iller \
    --data_dir ./data/raw \
    --output_dir ./results

# 所有实验
python run_all_experiments.py \
    --catchments Iller Saale Selke \
    --data_dir ./data/raw \
    --n_replicates 30
```

### 方式4：使用单个模型

```python
from src.models import get_model
from src.utils.data_loader import generate_synthetic_data
from src.metrics.kge import kge

# 生成数据
data = generate_synthetic_data(n_days=1000)

# 创建HBV模型
model = get_model('HBV', n_elevation_zones=3)

# 初始化参数（典型值）
params = {
    'TT': 0.0,      # 阈值温度 [°C]
    'CFMAX': 3.5,   # 度日因子 [mm/°C/day]
    'FC': 250.0,    # 土壤最大含水量 [mm]
    'BETA': 2.0,    # 形状系数
    'K0': 0.2,      # 快速径流系数 [1/day]
    'K1': 0.1,      # 中速径流系数 [1/day]
    'K2': 0.05,     # 慢速径流系数 [1/day]
    'MAXBAS': 3.0,  # 汇流时间 [day]
}
model.initialize(params)

# 模拟
discharge = model.simulate(data.precip, data.temp, data.pet)

# 评估（去除预热期）
warmup = 365
kge_value = kge(data.discharge[warmup:], discharge[warmup:])
print(f"KGE: {kge_value:.3f}")
```

---

## 📁 项目结构

```
Data-driven-Hydrological-Model/
│
├── README.md                    # 本文件
├── QUICKSTART.md               # 快速开始指南
├── requirements.txt            # Python依赖
├── run_all_experiments.py      # 实验总控脚本
│
├── src/                        # 源代码
│   ├── models/                 # 7个水文模型
│   │   ├── base_model.py       # 模型基类
│   │   ├── gr4j.py            # GR4J模型
│   │   ├── hbv.py             # HBV模型
│   │   ├── swat_plus.py       # SWAT+模型
│   │   ├── eddis.py           # EDDIS模型
│   │   ├── rtree.py           # 回归树模型
│   │   ├── ann.py             # 神经网络模型
│   │   └── lstm.py            # LSTM模型
│   │
│   ├── metrics/                # 评估指标
│   │   ├── kge.py             # KGE指标
│   │   └── entropy.py         # 信息熵指标
│   │
│   ├── utils/                  # 工具函数
│   │   ├── data_loader.py     # 数据加载
│   │   ├── impro_loader.py    # IMPRO数据加载
│   │   └── visualization.py   # 可视化工具
│   │
│   ├── calibration/            # 参数校准
│   │   └── spotpy_wrapper.py  # SPOTPY包装器
│   │
│   └── sampling/               # 采样策略
│       └── strategies.py       # 各种采样方法
│
├── examples/                   # 🆕 完整示例代码
│   ├── README.md              # 示例说明文档
│   ├── 01_basic_model_usage.py         # 基础模型使用
│   ├── 02_compare_multiple_models.py   # 多模型对比
│   ├── 03_real_data_workflow.py        # 真实数据工作流
│   └── 04_learning_curves_analysis.py  # 学习曲线分析
│
├── experiments/                # 论文实验复现
│   ├── experiment_1_learning_curves.py
│   ├── experiment_2_sampling_strategies.py
│   ├── experiment_3_information_content.py
│   └── experiment_4_spatial_distribution.py
│
├── notebooks/                  # Jupyter分析
│   └── analysis.ipynb
│
├── scripts/                    # 辅助脚本
│   └── convert_impro_ascii_to_csv.py
│
├── tests/                      # 测试文件
└── results/                    # 输出结果
```

---

## 📊 使用场景

### 场景1：模型选择
**问题**：我有1000天的径流数据，应该选择哪个模型？

**解决方案**：
```bash
python examples/02_compare_multiple_models.py
# 对比不同模型在1000样本下的性能
```

### 场景2：数据收集规划
**问题**：我需要收集多少年的数据才能达到KGE > 0.7？

**解决方案**：
```bash
python examples/04_learning_curves_analysis.py
# 分析学习曲线，找到性价比最高的数据量
```

### 场景3：流域径流模拟
**问题**：如何对我的流域进行径流模拟？

**解决方案**：
```bash
python examples/03_real_data_workflow.py
# 完整的数据处理→校准→验证→评估流程
```

### 场景4：模型教学
**问题**：如何教学生理解不同水文模型的原理？

**解决方案**：
```bash
python examples/01_basic_model_usage.py
# 详细的代码注释，展示模型计算过程
```

---

## 📖 文档

- [快速开始指南](QUICKSTART.md) - 详细的安装和使用说明
- [示例代码文档](examples/README.md) - 完整示例说明
- [数据检测指南](DATA_DETECTION_GUIDE.md) - 数据格式和加载
- [实现总结](IMPLEMENTATION_SUMMARY.md) - 技术细节

### API文档

核心API说明：

**模型使用**
```python
from src.models import get_model

# 获取模型
model = get_model('模型名称', **模型参数)

# 初始化
model.initialize(params_dict)

# 模拟
discharge = model.simulate(precip, temp, pet)

# 获取参数范围（过程模型）
bounds = model.get_parameter_bounds()
```

**数据加载**
```python
from src.utils.data_loader import load_catchment_from_csv

# 加载数据
data = load_catchment_from_csv('流域名称', data_dir='数据目录')

# 数据属性
data.precip       # 降水 [mm/day]
data.temp         # 温度 [°C]
data.pet          # 潜在蒸散发 [mm/day]
data.discharge    # 径流 [mm/day]
data.dates        # 日期索引
```

**评估指标**
```python
from src.metrics.kge import kge
from src.metrics.entropy import evaluate_model_entropy

# KGE评估
kge_value = kge(observed, simulated)

# 信息熵评估
metrics = evaluate_model_entropy(observed, simulated, n_bins=20)
```

---

## 🔬 论文复现

### 主要实验结果

根据论文，主要发现包括：

1. **学习曲线**：
   - 数据驱动模型（LSTM、ANN）在大样本（>1000天）时表现优异
   - 过程驱动模型（GR4J、HBV）在小样本（<500天）时更稳定
   - EDDIS在极小样本（<50天）时表现相对较好

2. **采样策略**：
   - Douglas-Peucker采样在信息量方面优于随机采样
   - 但性能提升有限（~5%）

3. **信息内容**：
   - 条件熵与KGE呈负相关
   - 高信息量的训练样本能显著提升模型性能

### 复现步骤

```bash
# 1. 准备数据（使用IMPRO数据集）
# 2. 运行所有实验
python run_all_experiments.py --catchments Iller Saale Selke --n_replicates 30

# 3. 分析结果
jupyter notebook notebooks/analysis.ipynb
```

---

## 🛠️ 扩展开发

### 添加新模型

```python
# 1. 创建模型文件 src/models/my_model.py
from .base_model import BaseHydrologicalModel

class MyModel(BaseHydrologicalModel):
    def __init__(self, **kwargs):
        super().__init__("MyModel")
        # 初始化代码

    def initialize(self, params):
        # 参数初始化
        pass

    def run_timestep(self, precip, temp, pet):
        # 单步模拟
        return discharge

    def simulate(self, precip, temp, pet):
        # 完整模拟
        pass

# 2. 注册模型 src/models/__init__.py
from .my_model import MyModel

def get_model(name, **kwargs):
    if name == 'MyModel':
        return MyModel(**kwargs)
    # ...
```

### 添加新指标

```python
# 创建文件 src/metrics/my_metric.py
def my_metric(observed, simulated):
    """
    自定义评估指标

    Parameters:
    -----------
    observed : array
        观测值
    simulated : array
        模拟值

    Returns:
    --------
    score : float
        指标值
    """
    # 计算代码
    return score
```

---

## ❓ 常见问题

### Q1: 如何选择合适的模型？

**A**: 根据以下因素选择：
- **数据量**: <500天选GR4J/HBV，>1000天可考虑LSTM/ANN
- **物理解释性**: 需要物理意义选过程模型
- **外推能力**: 需要外推选过程模型
- **计算资源**: 受限选GR4J/RTREE

### Q2: 模型性能不好怎么办？

**A**: 检查以下方面：
1. **数据质量**: 是否有缺失值、异常值？
2. **预热期**: 是否设置了足够的预热期（推荐365天）？
3. **参数校准**: 是否进行了参数优化？
4. **评估指标**: 是否选择了合适的指标？

### Q3: 如何提高运行速度？

**A**: 优化建议：
- 减少校准迭代次数（用于测试时）
- 减少实验重复次数
- 使用更少的训练样本大小
- 神经网络减少训练轮次

### Q4: 可以用于我的流域吗？

**A**: 可以！步骤：
1. 准备流域数据（降水、温度、潜在蒸散发、径流）
2. 转换为标准CSV格式
3. 运行示例3进行校准和验证
4. 根据结果选择最佳模型

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

### 贡献指南

- ✅ 代码需有详细的中文注释
- ✅ 新功能需添加单元测试
- ✅ 更新相关文档
- ✅ 遵循PEP 8代码风格

---

## 📝 引用

如使用本代码进行研究，请引用原论文：

```bibtex
@article{staudinger2025learning,
  title={How well do process-based and data-driven hydrological models learn from limited discharge data?},
  author={Staudinger, Maria and Herzog, Anna and Loritz, Ralf and Struck, Julian and Blume, Theresa and Bárdossy, András and Sivapalan, Murugesu and Zehe, Erwin},
  journal={Hydrology and Earth System Sciences},
  volume={29},
  pages={5005--5029},
  year={2025},
  publisher={Copernicus GmbH},
  doi={10.5194/hess-29-5005-2025}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

- **问题反馈**: 请提交 [GitHub Issue](https://github.com/licm13/Data-driven-Hydrological-Model/issues)
- **功能建议**: 欢迎提交 Pull Request
- **学术讨论**: 请参考原论文联系作者

---

## 🙏 致谢

- 感谢 Staudinger et al. (2025) 提供的科学方法和思路
- 感谢所有开源软件包的开发者
- 感谢IMPRO项目提供的流域数据

---

## 📊 项目统计

- **代码行数**: 4389 行 Python代码
- **模型数量**: 7 个水文模型
- **实验数量**: 4 个完整实验
- **示例数量**: 4 个详细示例
- **文档页数**: 5 份详细文档

---

<p align="center">
  <i>✨ 让水文建模更简单、更科学 ✨</i>
</p>

<p align="center">
  Made with ❤️ by the Data-driven Hydrological Modeling Community
</p>
