# 水文模型代码库重构架构方案

## 📋 Executive Summary

本重构方案旨在将 **Legacy Experimental Layer** 的业务逻辑（4个核心实验）迁移并集成到 **dMG Framework Layer** 的现代架构中，创建一个统一的、支持微分参数学习（dPL）的高复杂度水文建模系统。

## 🎯 重构目标

1. **架构统一**：基于 Hydra 的配置驱动实验系统
2. **模型适配**：Legacy NumPy 模型 → PyTorch Differentiable 模型
3. **数据层升级**：支持 CAMELS + CSV/ASCII + 采样策略
4. **训练融合**：Spotpy 传统校准 + PyTorch 梯度下降
5. **配置驱动**：通过 YAML 定义完整实验流程

---

## 📐 系统架构

### 高层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Unified Hydra Entry Point                       │
│                   (HBA-Model/src/dmg/__main__.py)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ├── Config Manager (Hydra + OmegaConf)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│ Experiment    │   │ Model Pipeline │   │ Data Pipeline│
│ Task System   │   │                │   │              │
└───────────────┘   └────────────────┘   └──────────────┘
        │                    │                    │
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────────────────────────────────────────────┐
│              Learning Curve Experiment                │
│              Sampling Strategy Experiment             │
│              Information Entropy Experiment           │
│              Spatial Distribution Experiment          │
└───────────────────────────────────────────────────────┘
```

### 核心组件层次结构

```
dMG Framework
├── Core Infrastructure
│   ├── dmg/core/data/loaders/
│   │   ├── hydro_loader.py (扩展)
│   │   └── legacy_csv_loader.py (新)
│   ├── dmg/core/data/samplers/
│   │   ├── base_sampler.py
│   │   └── learning_curve_sampler.py (新)
│   ├── dmg/core/calc/
│   │   ├── metrics.py (扩展)
│   │   └── entropy.py (新)
│   └── dmg/core/calibration/
│       ├── base_calibrator.py (新)
│       └── spotpy_calibrator.py (新)
│
├── Models
│   ├── dmg/models/phy_models/
│   │   ├── hbv_torch.py (新 - PyTorch 实现)
│   │   ├── gr4j_torch.py (新)
│   │   └── legacy_model_adapter.py (新 - Wrapper)
│   ├── dmg/models/delta_models/
│   │   └── dpl_model.py (保持)
│   └── dmg/models/neural_networks/
│       └── lstm.py (保持)
│
├── Training
│   ├── dmg/trainers/
│   │   ├── base.py
│   │   ├── trainer.py (扩展)
│   │   └── hybrid_trainer.py (新 - Spotpy + Gradient)
│   └── dmg/trainers/strategies/
│       ├── gradient_descent_strategy.py (新)
│       └── spotpy_strategy.py (新)
│
└── Experiments
    ├── dmg/experiments/ (新模块)
    │   ├── __init__.py
    │   ├── base_experiment.py (抽象基类)
    │   ├── learning_curves.py (实验1)
    │   ├── sampling_strategies.py (实验2)
    │   ├── information_content.py (实验3)
    │   └── spatial_distribution.py (实验4)
    └── dmg/experiments/tasks/
        ├── task_registry.py
        └── task_executor.py
```

---

## 🔧 关键组件设计

### 1. PyTorch Model Adapter

**设计模式**: Adapter Pattern + Template Method

```python
# HBA-Model/src/dmg/models/phy_models/legacy_model_adapter.py

class LegacyModelAdapter(torch.nn.Module):
    """
    将 Legacy NumPy 模型适配为 PyTorch Differentiable 模型

    策略：
    1. Eager Mode: 直接转换 NumPy → Torch (性能)
    2. Trace Mode: 使用 torch.jit.trace (兼容性)
    3. Rewrite Mode: 完全重写为 PyTorch 操作 (最优)
    """

    def __init__(self, legacy_model, adaptation_strategy='eager'):
        super().__init__()
        self.legacy_model = legacy_model
        self.strategy = adaptation_strategy

    def forward(self, forcings, parameters):
        # 自动选择策略
        if self.strategy == 'rewrite':
            return self._forward_torch(forcings, parameters)
        else:
            return self._forward_numpy_wrapped(forcings, parameters)
```

### 2. Unified Data Loader

**设计模式**: Strategy Pattern + Factory Pattern

```python
# HBA-Model/src/dmg/core/data/loaders/universal_hydro_loader.py

class UniversalHydroLoader(BaseLoader):
    """
    统一数据加载器，支持：
    - CAMELS dataset (NetCDF)
    - CSV/ASCII files (Legacy)
    - IMPRO format (特殊格式)
    - 采样策略（Douglas-Peucker, Random, Stratified）
    """

    def __init__(self, config):
        super().__init__(config)
        self.format = config['observations']['format']
        self.loader_strategy = self._create_loader_strategy()
        self.sampler = self._create_sampler()

    def _create_loader_strategy(self):
        strategies = {
            'camels': CAMELSLoaderStrategy(),
            'csv': CSVLoaderStrategy(),
            'impro': IMPROLoaderStrategy(),
        }
        return strategies[self.format]
```

### 3. Hybrid Trainer

**设计模式**: Strategy Pattern + Command Pattern

```python
# HBA-Model/src/dmg/trainers/hybrid_trainer.py

class HybridTrainer(BaseTrainer):
    """
    混合训练器，支持：
    1. Traditional Calibration (Spotpy): 用于纯物理模型
    2. Gradient Descent (PyTorch): 用于 dPL 和神经网络
    3. Hybrid Mode: 先 Spotpy 预训练，再 Gradient Fine-tune
    """

    def __init__(self, config, model, train_dataset, eval_dataset):
        super().__init__(config, model, train_dataset, eval_dataset)
        self.training_strategy = self._select_training_strategy()

    def _select_training_strategy(self):
        model_type = self.config['model']['type']

        if model_type == 'physics':
            return SpotpyCalibrationStrategy(self.config)
        elif model_type == 'dpl':
            return GradientDescentStrategy(self.config)
        elif model_type == 'hybrid':
            return HybridStrategy(self.config)  # Spotpy → Gradient
        else:
            return NeuralNetworkStrategy(self.config)
```

### 4. Experiment Task System

**设计模式**: Command Pattern + Registry Pattern

```python
# HBA-Model/src/dmg/experiments/base_experiment.py

class BaseExperiment(ABC):
    """
    实验抽象基类

    所有实验遵循统一流程：
    1. Setup: 数据加载、模型初始化
    2. Execute: 运行实验逻辑
    3. Evaluate: 计算指标
    4. Report: 保存结果和可视化
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.results = {}

    @abstractmethod
    def setup(self) -> None:
        """准备实验环境"""

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """执行实验逻辑"""

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """评估实验结果"""

    @abstractmethod
    def report(self, output_dir: Path) -> None:
        """生成报告"""

    def run(self) -> Dict[str, Any]:
        """完整实验流程"""
        self.setup()
        results = self.execute()
        metrics = self.evaluate()
        self.report(Path(self.config['output_dir']))
        return {'results': results, 'metrics': metrics}
```

```python
# HBA-Model/src/dmg/experiments/learning_curves.py

class LearningCurveExperiment(BaseExperiment):
    """
    实验1：学习曲线分析

    目标：评估不同模型在不同训练样本量下的学习能力

    配置示例：
    experiment:
      name: learning_curves
      sample_sizes: [50, 100, 500, 1000, 2000, 3654]
      n_replicates: 30
      models: ['HBV', 'GR4J', 'LSTM', 'dPL-HBV']
      metrics: ['KGE', 'H_conditional', 'H_normalized']
    """

    def execute(self) -> Dict[str, Any]:
        results = {}

        for model_name in self.config['experiment']['models']:
            model_results = {}

            for sample_size in self.config['experiment']['sample_sizes']:
                # 生成采样索引
                sampling_indices = self.sampler.generate_samples(
                    n_total=len(self.train_data),
                    sample_size=sample_size,
                    n_replicates=self.config['experiment']['n_replicates'],
                    strategy='consecutive_random'
                )

                replicate_results = []

                for rep_idx, indices in enumerate(sampling_indices):
                    # 训练模型
                    trained_model = self.train_model(
                        model_name,
                        self.train_data[indices]
                    )

                    # 评估模型
                    predictions = trained_model.predict(self.test_data)
                    metrics = self.compute_metrics(predictions)
                    replicate_results.append(metrics)

                model_results[sample_size] = replicate_results

            results[model_name] = model_results

        return results
```

---

## 📝 配置文件设计

### 实验配置结构

```yaml
# HBA-Model/conf/experiments/learning_curves.yaml

# 实验元数据
experiment:
  name: learning_curves
  description: "Analyze model learning ability vs training data size"
  type: replicated_sampling  # single, replicated_sampling, spatial_cv

# 数据配置
data:
  source: csv  # camels, csv, impro
  catchments: ['Iller', 'Saale', 'Selke']
  data_dir: ${oc.env:DATA_DIR,'./Dataset/IMPRO_catchment_data_infotheo'}

  # 时间划分
  periods:
    train:
      start: '2001-01-01'
      end: '2010-12-31'
    test:
      start: '2012-01-01'
      end: '2015-12-31'
    warmup_days: 365

  # 采样策略
  sampling:
    strategy: consecutive_random  # consecutive_random, douglas_peucker, stratified
    sample_sizes: [2, 10, 50, 100, 250, 500, 1000, 2000, 3000, 3654]
    n_replicates: 30
    seed: 42

# 模型配置
models:
  # 纯物理模型（使用 Spotpy 校准）
  - name: HBV
    type: physics
    training:
      method: spotpy
      algorithm: lhs
      n_iterations: 500
      objective: kge
    parameters:
      TT: [0.0, -2.5, 2.5]  # [default, min, max]
      CFMAX: [3.5, 1.0, 10.0]
      FC: [250.0, 50.0, 500.0]
      BETA: [2.0, 1.0, 6.0]
      # ... 其他参数

  - name: GR4J
    type: physics
    training:
      method: spotpy
      algorithm: lhs
      n_iterations: 500
      objective: kge

  # 数据驱动模型（梯度下降训练）
  - name: LSTM
    type: neural_network
    training:
      method: gradient_descent
      optimizer: Adam
      learning_rate: 0.001
      epochs: 50
      batch_size: 256
    architecture:
      sequence_length: 365
      hidden_size: 64
      num_layers: 2
      dropout: 0.1

  # 微分参数学习模型（混合训练）
  - name: dPL-HBV
    type: dpl
    training:
      method: hybrid  # spotpy_pretrain + gradient_finetune
      pretrain:
        algorithm: lhs
        n_iterations: 200
      finetune:
        optimizer: Adadelta
        learning_rate: 1.0
        epochs: 30
    components:
      nn_model:
        type: LSTM
        hidden_size: 32
        num_layers: 1
      phy_model:
        type: HBV
        learnable_params: ['FC', 'BETA', 'K0', 'K1', 'K2']  # LSTM 学习这些参数

# 评估指标
metrics:
  performance:
    - name: KGE
      description: Kling-Gupta Efficiency
    - name: NSE
      description: Nash-Sutcliffe Efficiency
    - name: RMSE
      description: Root Mean Square Error

  information_theory:
    - name: H_conditional
      description: Conditional Entropy
      params:
        n_bins: 12
    - name: H_normalized
      description: Normalized Entropy
      params:
        n_bins: 12
    - name: mutual_information
      description: Mutual Information between obs and sim

# 输出配置
output:
  base_dir: ./results/experiments/learning_curves
  save_format: ['pickle', 'csv', 'netcdf']

  visualization:
    enabled: true
    plots:
      - type: learning_curve
        x_axis: sample_size
        y_axis: H_conditional
        groupby: model
        style: median_with_iqr  # median + 25th-75th percentile
      - type: metric_comparison
        metrics: ['KGE', 'H_conditional']
        models: all

  reports:
    generate_latex: false
    generate_html: true

# 计算资源
compute:
  device: cuda  # cuda, cpu
  num_workers: 4
  parallel_replicates: true  # 并行运行 replicates

# 随机种子（可重复性）
random_seed: 42
```

### 采样策略配置

```yaml
# HBA-Model/conf/sampling/douglas_peucker.yaml

name: douglas_peucker
description: "Douglas-Peucker algorithm for information-driven sampling"

algorithm:
  type: iterative_reduction
  distance_metric: perpendicular_distance

  # 距离计算方式
  feature_space:
    - discharge  # 使用径流作为特征
    - precip     # 可选：多维特征空间

  normalization: minmax  # minmax, zscore, none

  # 迭代参数
  initial_sample: full_timeseries
  reduction_strategy: greedy  # greedy, balanced

# 用于实验2
experiment:
  target_sample_sizes: [50, 100, 250, 500, 1000]
  comparison_baseline: random_sampling
```

---

## 🔄 迁移路径

### Phase 1: 基础设施 (Week 1-2)

1. ✅ **Model Adapter**
   - 实现 `LegacyModelAdapter` 基类
   - 移植 HBV 模型到 PyTorch (`hbv_torch.py`)
   - 移植 GR4J 模型到 PyTorch (`gr4j_torch.py`)
   - 单元测试：验证数值一致性

2. ✅ **Data Loader Extension**
   - 扩展 `HydroLoader` 支持 CSV/ASCII
   - 实现 `LegacyCSVLoader`
   - 集成采样策略 (Douglas-Peucker, Random, Stratified)
   - 单元测试：数据加载和采样

3. ✅ **Hybrid Trainer**
   - 实现 Spotpy 校准策略
   - 集成到 `HybridTrainer`
   - 单元测试：校准收敛性

### Phase 2: 实验系统 (Week 3-4)

4. ✅ **Experiment Framework**
   - 实现 `BaseExperiment` 抽象类
   - 创建实验注册表 (`TaskRegistry`)
   - 实现实验执行器 (`TaskExecutor`)

5. ✅ **Migrate Experiments**
   - 实验1: `LearningCurveExperiment`
   - 实验2: `SamplingStrategyExperiment`
   - 实验3: `InformationContentExperiment`
   - 实验4: `SpatialDistributionExperiment`

6. ✅ **Metrics Integration**
   - 扩展 `dmg/core/calc/metrics.py` 添加熵指标
   - 创建 `EntropyMetrics` Pydantic 模型
   - 集成到评估流程

### Phase 3: 配置与入口 (Week 5)

7. ✅ **Configuration System**
   - 设计完整的 YAML 配置结构
   - 创建配置验证 Schema
   - 实现配置继承和组合

8. ✅ **Unified Entry Point**
   - 重构 `dmg/__main__.py`
   - 添加实验模式支持
   - CLI 参数解析

9. ✅ **Documentation**
   - API 文档
   - 用户手册
   - 示例实验配置

### Phase 4: 验证与优化 (Week 6)

10. ✅ **Integration Testing**
    - 端到端测试所有实验
    - 性能基准测试
    - 对比 Legacy 结果验证正确性

11. ✅ **Optimization**
    - GPU 加速优化
    - 并行化实验复制
    - 内存使用优化

---

## 🚀 使用示例

### 命令行接口

```bash
# 运行单个实验（使用默认配置）
python -m dmg --config-name=learning_curves

# 运行实验并覆盖参数
python -m dmg --config-name=learning_curves \
  experiment.sample_sizes=[50,500,1000] \
  data.catchments=['Iller'] \
  compute.device=cuda

# 运行所有实验（批处理）
python -m dmg --config-name=run_all_experiments

# 运行特定实验子集
python -m dmg --config-name=run_all_experiments \
  experiments=[learning_curves,sampling_strategies]

# Quick test mode (开发调试)
python -m dmg --config-name=learning_curves \
  experiment.n_replicates=3 \
  experiment.sample_sizes=[50,500] \
  data.catchments=['Iller']
```

### Python API

```python
from omegaconf import OmegaConf
from dmg.experiments import ExperimentRegistry

# 加载配置
config = OmegaConf.load('conf/experiments/learning_curves.yaml')

# 创建实验
experiment = ExperimentRegistry.create('learning_curves', config)

# 运行实验
results = experiment.run()

# 访问结果
print(results['metrics']['HBV']['KGE']['median'])
```

---

## 📊 性能优化策略

### 1. 并行化

```python
# 并行运行 replicates
from concurrent.futures import ProcessPoolExecutor

def train_single_replicate(args):
    model, data, indices = args
    trained_model = train_model(model, data[indices])
    return evaluate(trained_model, test_data)

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(train_single_replicate, replicate_args))
```

### 2. GPU 批处理

```python
# 批量前向传播（降低 GPU kernel launch overhead）
batch_predictions = []
for batch_indices in batched(all_indices, batch_size=32):
    batch_data = stack_data([data[i] for i in batch_indices])
    predictions = model(batch_data)  # 单次 GPU 调用
    batch_predictions.extend(predictions)
```

### 3. 缓存机制

```python
# 缓存数据加载和预处理结果
@functools.lru_cache(maxsize=128)
def load_and_preprocess_catchment(catchment_name, data_dir):
    data = load_catchment(catchment_name, data_dir)
    return preprocess(data)
```

---

## 🧪 测试策略

### 单元测试

```python
# tests/test_model_adapter.py
def test_hbv_torch_numerical_consistency():
    """验证 PyTorch HBV 与 Legacy NumPy HBV 数值一致性"""
    legacy_model = LegacyHBV()
    torch_model = HBVTorch()

    # 使用相同参数和输入
    params = {...}
    forcings = {...}

    legacy_output = legacy_model.simulate(forcings, params)
    torch_output = torch_model(forcings, params).detach().numpy()

    np.testing.assert_allclose(legacy_output, torch_output, rtol=1e-5)
```

### 集成测试

```python
# tests/test_experiments.py
def test_learning_curve_experiment_end_to_end():
    """端到端测试学习曲线实验"""
    config = load_test_config('learning_curves_quick.yaml')
    experiment = LearningCurveExperiment(config)
    results = experiment.run()

    # 验证结果结构
    assert 'HBV' in results['results']
    assert 50 in results['results']['HBV']
    assert len(results['results']['HBV'][50]) == config.experiment.n_replicates
```

---

## 📚 参考文献

1. Staudinger, M., et al. (2025). "Learning curves and sampling strategies for hydrological models"
2. Kratzert, F., et al. (2019). "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
3. Seibert, J., & Vis, M. J. P. (2012). "Teaching hydrological modeling with a user-friendly catchment-runoff-model software package"

---

## 🔗 相关文件索引

### 核心实现文件

- 模型适配: `HBA-Model/src/dmg/models/phy_models/hbv_torch.py`
- 数据加载: `HBA-Model/src/dmg/core/data/loaders/legacy_csv_loader.py`
- 混合训练: `HBA-Model/src/dmg/trainers/hybrid_trainer.py`
- 实验基类: `HBA-Model/src/dmg/experiments/base_experiment.py`
- 学习曲线: `HBA-Model/src/dmg/experiments/learning_curves.py`

### 配置文件

- 实验配置: `HBA-Model/conf/experiments/`
- 采样策略: `HBA-Model/conf/sampling/`
- 模型配置: `HBA-Model/conf/models/`

### 测试文件

- 单元测试: `HBA-Model/tests/unit/`
- 集成测试: `HBA-Model/tests/integration/`
- 端到端测试: `HBA-Model/tests/e2e/`

---

**Last Updated**: 2025-11-25
**Version**: 1.0
**Author**: Senior Python Architect & Computational Hydrology Expert
