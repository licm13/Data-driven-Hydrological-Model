# 架构指南 (Architecture Guide)

本文档详细解释项目的架构设计，包括设计模式的应用和扩展指南。

---

## 目录

1. [架构概览](#1-架构概览)
2. [设计模式详解](#2-设计模式详解)
3. [核心组件](#3-核心组件)
4. [扩展指南](#4-扩展指南)
5. [最佳实践](#5-最佳实践)

---

## 1. 架构概览

### 1.1 系统层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │   Experiments   │  │   CLI Entry     │  │   Notebooks    │  │
│  │   (实验脚本)     │  │   (__main__)    │  │   (教学示例)    │  │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼───────────────────┼────────────┘
            │                    │                   │
            ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Service Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │    Trainers     │  │   Calibrators   │  │  Experiments   │  │
│  │   (训练器)       │  │   (校准器)       │  │   (实验框架)    │  │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼───────────────────┼────────────┘
            │                    │                   │
            ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │     Models      │  │   Data Loaders  │  │    Metrics     │  │
│  │   (水文模型)     │  │   (数据加载)     │  │   (评估指标)    │  │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼───────────────────┼────────────┘
            │                    │                   │
            ▼                    ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │    PyTorch      │  │     NumPy       │  │    Hydra       │  │
│  │   (深度学习)     │  │   (数值计算)     │  │   (配置管理)    │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 代码组织

```
Data-driven-Hydrological-Model/
├── src/                          # 核心源代码 (Legacy Layer)
│   ├── models/                   # 水文模型
│   │   ├── base_model.py        # 模型基类
│   │   ├── hbv.py               # HBV模型 (NumPy)
│   │   └── lstm.py              # LSTM模型
│   ├── metrics/                  # 评估指标
│   └── utils/                    # 工具函数
│
├── HBA-Model/src/dmg/           # 新架构 (dMG Framework)
│   ├── core/
│   │   └── data/loaders/        # 数据加载器
│   ├── models/
│   │   └── phy_models/          # PyTorch物理模型
│   ├── trainers/                # 训练器
│   └── experiments/             # 实验框架
│
├── experiments/                  # 实验脚本
├── notebooks/                    # Jupyter notebooks
└── tests/                        # 测试代码
```

---

## 2. 设计模式详解

### 2.1 适配器模式 (Adapter Pattern)

#### 应用场景
将Legacy NumPy模型适配为PyTorch接口，实现渐进式迁移。

#### 实现

```python
# 问题：Legacy模型和新框架接口不兼容
class LegacyHBV:
    def simulate(self, precip, temp, pet):
        # NumPy实现
        pass

# 解决方案：使用适配器包装
class LegacyHBVAdapter(nn.Module):
    """
    适配器：将Legacy HBV模型包装为dMG框架兼容的接口
    """
    def __init__(self, legacy_model, config):
        super().__init__()
        self.legacy_model = legacy_model
        self.config = config
        
    def forward(self, data_dict, parameters):
        """
        dMG框架期望的接口
        """
        # 从字典中提取数据
        precip = data_dict['precip'].numpy()
        temp = data_dict['temp'].numpy()
        pet = data_dict['pet'].numpy()
        
        # 调用Legacy模型
        discharge = self.legacy_model.simulate(precip, temp, pet)
        
        # 转换回Tensor
        return {'flow': torch.from_numpy(discharge)}
```

#### UML图

```
┌─────────────────┐         ┌─────────────────┐
│   dMG Framework │         │   Legacy Code   │
│   (Client)      │         │   (Adaptee)     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ forward(data, params)     │
         ▼                           │
┌─────────────────┐                  │
│ LegacyHBVAdapter│ ──── wraps ─────►│
│   (Adapter)     │                  │
└─────────────────┘                  │
         │                           │
         │ simulate(precip, temp)    │
         └───────────────────────────┘
```

### 2.2 策略模式 (Strategy Pattern)

#### 应用场景
在运行时切换不同的采样策略、训练策略或损失函数。

#### 采样策略实现

```python
from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    """采样策略抽象基类"""
    
    @abstractmethod
    def generate_samples(self, n_total, sample_size, n_replicates):
        """生成采样索引"""
        pass

class ConsecutiveRandomSampling(SamplingStrategy):
    """连续随机采样：保持时序连续性"""
    
    def generate_samples(self, n_total, sample_size, n_replicates):
        samples = []
        for _ in range(n_replicates):
            start = np.random.randint(0, n_total - sample_size)
            samples.append(np.arange(start, start + sample_size))
        return samples

class DouglasPeuckerSampling(SamplingStrategy):
    """Douglas-Peucker采样：信息驱动采样"""
    
    def generate_samples(self, n_total, sample_size, n_replicates):
        # 使用DP算法选择最具代表性的点
        pass

class StratifiedSampling(SamplingStrategy):
    """分层采样：基于流量分位数"""
    
    def generate_samples(self, n_total, sample_size, n_replicates):
        # 按流量分层采样
        pass

# 使用策略
class DataLoader:
    def __init__(self, strategy: SamplingStrategy):
        self.strategy = strategy
    
    def sample(self, data, sample_size, n_replicates):
        indices = self.strategy.generate_samples(
            len(data), sample_size, n_replicates
        )
        return [data[idx] for idx in indices]
```

#### 训练策略实现

```python
class TrainingStrategy(ABC):
    """训练策略抽象基类"""
    
    @abstractmethod
    def train(self, model, data, config):
        pass

class SpotpyCalibrationStrategy(TrainingStrategy):
    """Spotpy传统校准策略"""
    
    def train(self, model, data, config):
        # 使用Spotpy进行参数优化
        pass

class GradientDescentStrategy(TrainingStrategy):
    """梯度下降策略"""
    
    def train(self, model, data, config):
        # 使用PyTorch优化器
        pass

class HybridStrategy(TrainingStrategy):
    """混合策略：先Spotpy后梯度"""
    
    def train(self, model, data, config):
        # Phase 1: Spotpy预训练
        # Phase 2: 梯度微调
        pass
```

### 2.3 注册表模式 (Registry Pattern)

#### 应用场景
管理多种实验类型、模型类型，支持动态注册和查找。

#### 实现

```python
class ExperimentRegistry:
    """实验注册表"""
    
    _experiments = {}
    
    @classmethod
    def register(cls, name):
        """注册实验的装饰器"""
        def decorator(experiment_class):
            cls._experiments[name] = experiment_class
            return experiment_class
        return decorator
    
    @classmethod
    def create(cls, name, config):
        """创建实验实例"""
        if name not in cls._experiments:
            raise ValueError(f"Unknown experiment: {name}")
        return cls._experiments[name](config)
    
    @classmethod
    def list_experiments(cls):
        """列出所有已注册的实验"""
        return list(cls._experiments.keys())

# 使用方式
@ExperimentRegistry.register('learning_curves')
class LearningCurveExperiment(BaseExperiment):
    """学习曲线实验"""
    pass

@ExperimentRegistry.register('sampling_strategies')
class SamplingStrategyExperiment(BaseExperiment):
    """采样策略实验"""
    pass

# 创建实验
exp = ExperimentRegistry.create('learning_curves', config)
```

### 2.4 模板方法模式 (Template Method Pattern)

#### 应用场景
定义实验的标准执行流程，子类只需实现特定步骤。

#### 实现

```python
class BaseExperiment(ABC):
    """实验基类：定义模板方法"""
    
    def run(self):
        """
        模板方法：定义完整实验流程
        子类不应覆盖此方法
        """
        self._validate_config()
        self._setup_logging()
        
        try:
            # 4个可覆盖的钩子方法
            self.setup()
            results = self.execute()
            metrics = self.evaluate(results)
            self.report(metrics)
            
            return {
                'results': results,
                'metrics': metrics,
                'status': 'success'
            }
        except Exception as e:
            self._handle_error(e)
            return {'status': 'failed', 'error': str(e)}
        finally:
            self._cleanup()
    
    @abstractmethod
    def setup(self):
        """准备阶段 - 子类必须实现"""
        pass
    
    @abstractmethod
    def execute(self):
        """执行阶段 - 子类必须实现"""
        pass
    
    @abstractmethod
    def evaluate(self, results):
        """评估阶段 - 子类必须实现"""
        pass
    
    @abstractmethod
    def report(self, metrics):
        """报告阶段 - 子类必须实现"""
        pass
```

---

## 3. 核心组件

### 3.1 模型组件

```
Models
├── Physics-Based (物理模型)
│   ├── HBV (NumPy) ─────────────────► HBVTorch (PyTorch)
│   ├── GR4J (NumPy) ────────────────► GR4JTorch (PyTorch)
│   └── SWAT+ (简化版)
│
├── Data-Driven (数据驱动)
│   ├── LSTM
│   ├── ANN
│   ├── RTREE
│   └── EDDIS
│
└── Hybrid (混合模型)
    └── dPL (LSTM学习HBV参数)
```

### 3.2 数据加载组件

```python
# 组件关系图
UniversalHydroLoader
├── FormatDetector          # 自动检测数据格式
├── LoaderStrategy          # 加载策略（可插拔）
│   ├── CAMELSLoaderStrategy
│   ├── IMPROLoaderStrategy
│   └── CSVLoaderStrategy
├── SamplingStrategy        # 采样策略（可插拔）
│   ├── ConsecutiveRandomSampling
│   ├── DouglasPeuckerSampling
│   └── StratifiedSampling
└── DataTransformer         # 数据转换
    ├── Normalizer
    └── TensorConverter
```

### 3.3 训练组件

```python
# 训练器层次结构
BaseTrainer
├── TraditionalTrainer      # 传统校准
│   └── SpotpyCalibrator
│
├── NeuralNetTrainer        # 神经网络训练
│   └── PyTorchTrainer
│
└── HybridTrainer           # 混合训练
    ├── Phase1: SpotpyCalibrator
    └── Phase2: PyTorchTrainer
```

---

## 4. 扩展指南

### 4.1 添加新模型

**步骤1**: 创建模型文件

```python
# src/models/my_model.py
from .base_model import BaseHydrologicalModel

class MyModel(BaseHydrologicalModel):
    """我的自定义水文模型"""
    
    def __init__(self, **kwargs):
        super().__init__("MyModel")
        # 初始化参数
    
    def initialize(self, params):
        """初始化模型参数"""
        self.parameters = params
    
    def run_timestep(self, precip, temp, pet, timestep):
        """运行单个时间步"""
        # 实现计算逻辑
        return discharge
    
    def get_parameter_bounds(self):
        """返回参数范围（用于校准）"""
        return {
            'param1': (min_val, max_val),
            'param2': (min_val, max_val),
        }
```

**步骤2**: 注册模型

```python
# src/models/__init__.py
from .my_model import MyModel

def get_model(name, **kwargs):
    models = {
        'HBV': HBV,
        'GR4J': GR4J,
        'MyModel': MyModel,  # 添加
    }
    return models[name](**kwargs)
```

### 4.2 添加新采样策略

```python
# src/sampling/my_strategy.py
from .strategies import SamplingStrategy

class MyCustomSampling(SamplingStrategy):
    """我的自定义采样策略"""
    
    def __init__(self, seed=42):
        self.seed = seed
    
    def generate_samples(self, n_total, sample_size, n_replicates):
        """
        生成采样索引
        
        Parameters:
        -----------
        n_total : int
            总样本数
        sample_size : int
            每个样本的大小
        n_replicates : int
            重复次数
            
        Returns:
        --------
        samples : list of np.ndarray
            每个元素是一个采样索引数组
        """
        np.random.seed(self.seed)
        samples = []
        
        # 实现自定义采样逻辑
        for _ in range(n_replicates):
            indices = self._custom_sampling_logic(n_total, sample_size)
            samples.append(indices)
        
        return samples
    
    def _custom_sampling_logic(self, n_total, sample_size):
        # 具体实现
        pass
```

### 4.3 添加新实验

```python
# experiments/my_experiment.py
from dmg.experiments.base_experiment import BaseExperiment
from dmg.experiments.task_registry import ExperimentRegistry

@ExperimentRegistry.register('my_experiment')
class MyExperiment(BaseExperiment):
    """我的自定义实验"""
    
    def setup(self):
        """准备实验"""
        self.data = self._load_data()
        self.models = self._init_models()
    
    def execute(self):
        """执行实验逻辑"""
        results = {}
        for model_name, model in self.models.items():
            # 实验逻辑
            results[model_name] = self._run_model(model)
        return results
    
    def evaluate(self, results):
        """评估结果"""
        metrics = {}
        for model_name, result in results.items():
            metrics[model_name] = self._calculate_metrics(result)
        return metrics
    
    def report(self, metrics):
        """生成报告"""
        self._save_results(metrics)
        self._generate_plots(metrics)
```

---

## 5. 最佳实践

### 5.1 配置管理

```yaml
# 使用Hydra管理配置
defaults:
  - model: hbv
  - data: camels
  - training: hybrid

experiment:
  name: my_experiment
  seed: 42

# 支持命令行覆盖
# python -m dmg experiment=my_experiment model=lstm
```

### 5.2 类型提示

```python
from typing import Dict, List, Optional, Tuple
from torch import Tensor

def simulate_hbv(
    precip: Tensor,             # [T, N]
    temp: Tensor,               # [T, N]
    params: Tensor,             # [N, 13]
    initial_states: Optional[Dict[str, Tensor]] = None
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    运行HBV模型模拟
    
    Args:
        precip: 降水，形状 (时间步, 批次)
        temp: 温度，形状 (时间步, 批次)
        params: 参数，形状 (批次, 参数数)
        initial_states: 初始状态字典
        
    Returns:
        output: 径流输出
        states: 最终状态字典
    """
    ...
```

### 5.3 错误处理

```python
class HydrologicalModelError(Exception):
    """水文模型相关错误的基类"""
    pass

class ParameterOutOfBoundsError(HydrologicalModelError):
    """参数超出范围错误"""
    def __init__(self, param_name, value, bounds):
        self.param_name = param_name
        self.value = value
        self.bounds = bounds
        message = f"Parameter {param_name}={value} out of bounds {bounds}"
        super().__init__(message)

# 使用
def validate_parameters(params, bounds):
    for name, value in params.items():
        if name in bounds:
            min_val, max_val = bounds[name]
            if not (min_val <= value <= max_val):
                raise ParameterOutOfBoundsError(name, value, bounds[name])
```

### 5.4 日志记录

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseExperiment:
    def run(self):
        logger.info(f"Starting experiment: {self.config['name']}")
        try:
            result = self.execute()
            logger.info(f"Experiment completed successfully")
            return result
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
```

---

## 附录：设计决策记录

### ADR-001: 选择PyTorch而非TensorFlow

**背景**: 需要支持可微分物理模型

**决策**: 使用PyTorch

**原因**:
- 更好的动态计算图支持
- Pythonic API
- 水文学社区（如Neuralhydrology）广泛使用

### ADR-002: 使用Hydra进行配置管理

**背景**: 需要灵活的配置系统

**决策**: 采用Hydra + OmegaConf

**原因**:
- 支持配置组合和继承
- 命令行覆盖
- 自动保存配置到输出目录

### ADR-003: 保留Legacy代码

**背景**: 现有NumPy模型已经过验证

**决策**: 使用适配器模式渐进迁移

**原因**:
- 降低迁移风险
- 保持向后兼容
- 可以逐步验证新实现的正确性
