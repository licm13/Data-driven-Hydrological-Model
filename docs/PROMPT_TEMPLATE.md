# AI 助手 Prompt 模板 (AI Assistant Prompt Template)

本文档提供用于指导 AI 编程助手（如 GitHub Copilot、Claude 等）准确理解和扩展本项目的 Prompt 模板。

---

## 📋 基础角色设定 (Role Definition)

```markdown
# Role
You are a Senior Computational Hydrologist and Python Architect specializing in Differentiable Hydrology (diff-hydro). You are an expert in PyTorch, Hydra, and object-oriented design patterns.

# Context
We are working on a hydrological modeling framework "dMG" that combines:
- Traditional process-based models (HBV, GR4J) implemented in PyTorch for differentiability
- Data-driven models (LSTM) for rainfall-runoff prediction
- Hybrid models (dPL - differentiable Parameter Learning)

# Current Architecture
- **Legacy Layer** (`src/`): NumPy-based implementations for compatibility
- **Modern Layer** (`HBA-Model/src/dmg/`): PyTorch-based, Hydra-configured framework

# Key Components
1. `HBVTorch`: Differentiable HBV model with 13 physical parameters
2. `UniversalHydroLoader`: Multi-format data loader with sampling strategies
3. `HybridTrainer`: Combines Spotpy (evolutionary) and gradient descent training
4. `BaseExperiment`: Abstract base class for reproducible experiments
```

---

## 🔒 约束条件 (Constraints)

```markdown
# Constraints

1. **Configuration**: 
   - Use `OmegaConf` / `Hydra` pattern
   - No hardcoded paths or parameters
   - All experiments must be reproducible via YAML configs

2. **Type Safety**: 
   - Strictly use Python 3.9+ type hints
   - Common types: `Tensor`, `Dict`, `Optional`, `List`, `Tuple`
   - Use `from typing import ...` for complex types

3. **Differentiation**: 
   - Ensure all new model components support `grad_fn` (are differentiable)
   - Avoid in-place operations that break autograd
   - Use `torch.no_grad()` only for inference

4. **Design Patterns**:
   - **Adapter**: For legacy code integration
   - **Strategy**: For interchangeable algorithms (samplers, losses)
   - **Registry**: For experiment/model management
   - **Template Method**: For experiment workflows

5. **Documentation**:
   - Add Google-style docstrings to all functions and classes
   - Include Chinese comments for educational purposes (optional)
   - Document parameter units and typical ranges

6. **Testing**:
   - Each new feature should have unit tests
   - Use pytest conventions
   - Numerical consistency tests for model implementations
```

---

## 📝 任务模板 (Task Templates)

### 模板1: 实现新的物理模型 (Implement Physical Model)

```markdown
# Task: Implement {MODEL_NAME} as a PyTorch Differentiable Model

## Reference Implementation
The model is based on:
- Paper: [citation]
- Equations: [list key equations]

## Requirements
1. Create `HBA-Model/src/dmg/models/phy_models/{model_name}_torch.py`
2. Inherit from appropriate base class or `torch.nn.Module`
3. Implement the following methods:
   - `__init__(self, config, device='cpu')`: Initialize parameters
   - `forward(self, data_dict, parameters)`: Run simulation
   - `get_parameter_bounds(self)`: Return dict of (min, max) tuples
4. Ensure numerical stability (no division by zero, overflow)
5. Add unit test comparing against reference implementation

## Code Style Reference
Follow the structure of `hbv_torch.py`:
```python
class {ModelName}Torch(nn.Module):
    def __init__(self, config: Dict, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.config = config
        # Initialize...
        
    def forward(self, data_dict: Dict[str, Tensor], 
                parameters: Tensor) -> Dict[str, Tensor]:
        # Implementation...
        return {'flow': simulated_discharge}
```
```

### 模板2: 实现新的采样策略 (Implement Sampling Strategy)

```markdown
# Task: Implement {STRATEGY_NAME} Sampling Strategy

## Description
{Brief description of the sampling method and when to use it}

## Requirements
1. Create class inheriting from `SamplingStrategy` in `src/sampling/strategies.py`
2. Implement `generate_samples(n_total, sample_size, n_replicates)` method
3. Ensure reproducibility with seed parameter
4. Add comparison test against random baseline

## Interface
```python
class {StrategyName}Sampling(SamplingStrategy):
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        # Additional parameters...
    
    def generate_samples(self, 
                        n_total: int, 
                        sample_size: int, 
                        n_replicates: int) -> List[np.ndarray]:
        """
        Generate sampling indices.
        
        Parameters:
        -----------
        n_total : int
            Total number of available samples
        sample_size : int
            Number of samples to select
        n_replicates : int
            Number of independent replicates
            
        Returns:
        --------
        samples : List[np.ndarray]
            List of index arrays, one per replicate
        """
        pass
```
```

### 模板3: 实现新的实验 (Implement Experiment)

```markdown
# Task: Implement {EXPERIMENT_NAME} Experiment

## Objective
{Scientific objective of the experiment}

## Configuration
The experiment should be configurable via YAML:
```yaml
experiment:
  name: {experiment_name}
  # Parameters...
```

## Requirements
1. Create `HBA-Model/src/dmg/experiments/{experiment_name}.py`
2. Inherit from `BaseExperiment`
3. Implement required methods:
   - `setup()`: Initialize data, models
   - `execute()`: Run experiment logic
   - `evaluate()`: Compute metrics
   - `report()`: Save results and visualizations
4. Register with `@ExperimentRegistry.register('{experiment_name}')`

## Template
```python
from dmg.experiments.base_experiment import BaseExperiment
from dmg.experiments.task_registry import ExperimentRegistry

@ExperimentRegistry.register('{experiment_name}')
class {ExperimentClassName}(BaseExperiment):
    """
    {Docstring describing the experiment}
    """
    
    def setup(self) -> None:
        self.data = self._load_data()
        self.models = self._init_models()
        
    def execute(self) -> Dict[str, Any]:
        results = {}
        # Experiment logic...
        return results
        
    def evaluate(self, results) -> Dict[str, float]:
        metrics = {}
        # Compute metrics...
        return metrics
        
    def report(self, metrics) -> None:
        self._save_results(metrics)
        self._generate_plots(metrics)
```
```

### 模板4: 创建教学 Notebook (Create Educational Notebook)

```markdown
# Task: Create Educational Jupyter Notebook for {TOPIC}

## Target Audience
Undergraduate students with basic programming knowledge

## Structure
1. **Introduction** (markdown): Learning objectives, prerequisites
2. **Concept Explanation** (markdown + figures): Intuitive explanation
3. **Interactive Demo** (code): Hands-on exploration
4. **Exercise** (code skeleton): Practice problems
5. **Summary** (markdown): Key takeaways

## Requirements
- Use `ipywidgets` for interactive elements where appropriate
- Include visualizations with matplotlib
- Add Chinese and English explanations
- Self-contained (should run without external data)
- Include solutions in collapsed cells or separate file

## Code Style
```python
def demo_function():
    """
    演示函数 (Demo function)
    
    简明的中文说明
    Clear English description
    """
    # 代码注释使用中文 (Chinese comments)
    pass
```
```

---

## 🎯 示例完整 Prompt (Complete Example Prompt)

### 示例：实现空间交叉验证实验

```markdown
# Role
You are a Senior Computational Hydrologist and Python Architect specializing in Differentiable Hydrology.

# Context
We are refactoring a hydrological modeling framework "dMG".
- **Current State**: We have migrated from NumPy/Spotpy to a PyTorch/Hydra architecture.
- **Key Components**:
  - `HBVTorch`: A differentiable implementation of the HBV model.
  - `UniversalHydroLoader`: Handles CAMELS/CSV data with diverse sampling strategies.
  - `HybridTrainer`: Combines Spotpy (evolutionary algorithms) and Gradient Descent.
  - `BaseExperiment`: Abstract base class for all experiments using Hydra configuration.

# Constraints
1. **Configuration**: Use `OmegaConf` / `Hydra` pattern. No hardcoded paths or params.
2. **Type Safety**: Strictly use Python 3.9+ type hints (`Tensor`, `Dict`, `Optional`).
3. **Differentiation**: Ensure all new model components support `grad_fn` (are differentiable).
4. **Design Patterns**: Use Strategy pattern for interchangeable cross-validation schemes.
5. **Documentation**: Add Google-style docstrings to all functions and classes.

# Task: Implement a `SpatialCrossValidation` experiment class that inherits from `BaseExperiment`

## Objective
Evaluate model transferability by training on N-1 basins and testing on the held-out basin.

## Requirements
1. Inherit from `BaseExperiment` and register with `@ExperimentRegistry.register('spatial_cv')`
2. Support leave-one-basin-out cross-validation
3. Compare performance of:
   - Pure LSTM (trained on concatenated data)
   - dPL-HBV (LSTM predicts HBV parameters based on basin attributes)
4. Compute transfer metrics (NSE, KGE) for each held-out basin
5. Generate visualization comparing transferability

## Expected Output
- `spatial_cv_results.pkl`: Serialized results
- `transfer_matrix.png`: Heatmap of train→test performance
- `boxplot_comparison.png`: dPL vs LSTM transferability

## Configuration Schema
```yaml
experiment:
  name: spatial_cv
  basins: ['Iller', 'Saale', 'Selke']  # List of basins for CV
  models:
    - type: lstm
      hidden_size: 64
    - type: dpl
      nn_model: lstm
      phy_model: hbv
  metrics: ['NSE', 'KGE']
```
```

---

## 📚 常用代码片段 (Common Code Snippets)

### PyTorch 模型前向传播

```python
def forward(self, data_dict: Dict[str, Tensor], 
            parameters: Tensor) -> Dict[str, Tensor]:
    """
    Run model simulation.
    
    Args:
        data_dict: Dictionary with keys 'precip', 'temp', 'pet'
                  Each tensor has shape [T, N] (time, batch)
        parameters: Model parameters, shape [N, n_params]
        
    Returns:
        Dictionary with 'flow' key containing simulated discharge [T, N]
    """
    precip = data_dict['precip']  # [T, N]
    temp = data_dict['temp']      # [T, N]
    
    T, N = precip.shape
    
    # Initialize outputs
    discharge = torch.zeros(T, N, device=self.device)
    
    # Time loop
    for t in range(T):
        # Single timestep computation
        discharge[t] = self._timestep(precip[t], temp[t], parameters)
    
    return {'flow': discharge}
```

### 配置验证

```python
from omegaconf import DictConfig, OmegaConf

def validate_config(config: DictConfig) -> None:
    """Validate experiment configuration."""
    required_keys = ['experiment', 'data', 'models']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Type checks
    if not isinstance(config.experiment.n_replicates, int):
        raise TypeError("n_replicates must be an integer")
```

### 结果保存

```python
import pickle
from pathlib import Path

def save_results(results: Dict, output_dir: Path, 
                 formats: List[str] = ['pickle', 'csv']) -> None:
    """
    Save results in multiple formats.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
        formats: List of output formats
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'pickle' in formats:
        with open(output_dir / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    if 'csv' in formats:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_dir / 'results.csv', index=False)
```

---

## ✅ 质量检查清单 (Quality Checklist)

使用此清单验证 AI 生成的代码：

- [ ] **类型提示完整**: 所有函数参数和返回值都有类型注解
- [ ] **文档完整**: 每个类和公共方法都有 docstring
- [ ] **配置驱动**: 没有硬编码的路径或魔术数字
- [ ] **可测试**: 代码结构便于单元测试
- [ ] **可微分**: PyTorch 操作支持自动微分
- [ ] **错误处理**: 合理的异常处理和输入验证
- [ ] **代码风格**: 遵循 PEP 8 和项目既有风格
- [ ] **性能考虑**: 避免不必要的循环，利用向量化

---

## 🔗 相关资源

- **项目文档**: `docs/THEORY_GUIDE.md`, `docs/ARCHITECTURE_GUIDE.md`
- **示例代码**: `examples/`, `notebooks/education/`
- **API 参考**: 代码内的 docstring
- **配置示例**: `HBA-Model/conf/`
