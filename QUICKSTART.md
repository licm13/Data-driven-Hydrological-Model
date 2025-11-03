# 快速开始指南

## 安装
```bash
# 克隆仓库
git clone https://github.com/yourusername/hydrological-learning-curves.git
cd hydrological-learning-curves

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速测试（使用合成数据）
```bash
# 运行快速测试（约5-10分钟）
python run_all_experiments.py --synthetic --quick_test
```

## 使用真实数据

### 1. 准备数据

将数据组织为以下结构：
```
data/raw/
├── Iller/
│   ├── meteorology.csv  # date,precip,temp,pet
│   ├── discharge.csv    # date,discharge
│   └── config.yaml
├── Saale/
│   └── ...
└── Selke/
    └── ...
```

### 2. 运行单个实验
```bash
# 实验1：学习曲线
python experiments/experiment_1_learning_curves.py \
    --catchment Iller \
    --data_dir ./data/raw \
    --output_dir ./results

# 实验2：采样策略
python experiments/experiment_2_sampling_strategies.py \
    --catchment Iller \
    --data_dir ./data/raw

# 实验3：信息内容
python experiments/experiment_3_information_content.py \
    --catchments Iller Saale Selke \
    --data_dir ./data/raw

# 实验4：空间分布
python experiments/experiment_4_spatial_distribution.py \
    --catchment Iller \
    --data_dir ./data/raw
```

### 3. 运行所有实验
```bash
python run_all_experiments.py \
    --catchments Iller Saale Selke \
    --data_dir ./data/raw \
    --output_dir ./results \
    --n_replicates 30
```

## 分析结果
```bash
# 启动Jupyter Notebook
jupyter notebook notebooks/analysis.ipynb
```

## 使用单个模型
```python
from src.models import get_model
from src.utils.data_loader import generate_synthetic_data

# 生成数据
data = generate_synthetic_data(n_days=1000)

# 创建HBV模型
model = get_model('HBV', n_elevation_zones=3)

# 初始化参数
params = {
    'TT': 0.0,
    'CFMAX': 3.5,
    'FC': 250.0,
    'BETA': 2.0,
    'K0': 0.2,
    'K1': 0.1,
    'K2': 0.05,
    'MAXBAS': 3.0,
}
model.initialize(params)

# 模拟
discharge = model.simulate(data.precip, data.temp, data.pet)

# 评估
from src.metrics.entropy import evaluate_model_entropy
metrics = evaluate_model_entropy(data.discharge[365:], discharge)
print(f"Conditional Entropy: {metrics['H_conditional']:.3f} bits")
```

## 常见问题

### Q: 如何添加新模型？

1. 在 `src/models/` 创建新文件
2. 继承 `BaseHydrologicalModel`
3. 实现必要方法
4. 在 `src/models/__init__.py` 注册

### Q: 如何修改采样策略？

编辑 `src/sampling/strategies.py` 添加新的采样函数。

### Q: 结果保存在哪里？

默认保存在 `./results/` 目录下，按实验和流域组织。

## 性能提示

- 使用 `--quick_test` 进行快速原型测试
- 减少 `--n_replicates` 可加快运行速度
- 过程模型校准可使用更少的迭代次数
- LSTM训练可减少 `n_epochs` 或 `n_init`

## 引用

如使用本代码，请引用：
```bibtex
@article{staudinger2025learning,
  title={How well do process-based and data-driven hydrological models learn from limited discharge data?},
  author={Staudinger, Maria and others},
  journal={Hydrology and Earth System Sciences},
  year={2025}
}
```
```

---

## 完整项目总结

现在我们已经完成了整个项目的实现，包括：

### ✅ 已实现的功能：

1. **7个水文模型**：
   - 过程驱动：GR4J, HBV, SWAT+
   - 数据驱动：EDDIS, RTREE, ANN, LSTM

2. **评估指标**：
   - 信息熵（联合熵、条件熵）
   - KGE、NSE

3. **4个完整实验**：
   - 实验1：学习曲线对比
   - 实验2：采样策略影响
   - 实验3：信息内容分析
   - 实验4：空间分布效应

4. **工具和可视化**：
   - 数据加载器
   - SPOTPY校准包装
   - 丰富的可视化函数
   - Jupyter分析notebook

5. **文档**：
   - README
   - QUICKSTART
   - 完整代码注释

### 📁 最终项目结构：
```
hydrological-learning-curves/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── setup.py
├── run_all_experiments.py
├── src/
│   ├── models/ (7个模型 + 基类)
│   ├── metrics/ (熵和KGE)
│   ├── calibration/ (SPOTPY)
│   ├── sampling/ (3种策略)
│   └── utils/ (数据+可视化)
├── experiments/ (4个实验脚本)
├── notebooks/ (分析notebook)
└── tests/