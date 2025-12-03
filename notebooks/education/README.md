# 教育模块 (Educational Notebooks)

本目录包含面向大一新生的水文模型教学 Jupyter Notebooks。

---

## 📚 课程结构

| 模块 | 文件 | 主题 | 学习目标 |
|------|------|------|----------|
| 1 | `01_water_cycle_digital_twin.ipynb` | 水文循环的数字孪生 | 理解降雨-径流关系，实现线性水箱模型 |
| 2 | `02_inside_the_black_box.ipynb` | 打开黑盒子 - 物理模型 | 理解HBV模型结构和参数物理意义 |
| 3 | `03_how_machines_learn.ipynb` | 机器如何学习 | 理解LSTM时序建模，观察训练过程 |
| 4 | `04_model_evaluation.ipynb` | 模型评估裁判 | 掌握NSE、KGE等评估指标 |

---

## 🎯 教学设计理念

遵循 **"现象观察 → 原理直觉 → 代码实现 → 探究实验"** 的逻辑：

1. **从直觉开始**：用图形和类比解释概念
2. **交互式学习**：使用可视化和参数调整加深理解
3. **动手实践**：每个模块都有练习和实验
4. **循序渐进**：从简单模型到复杂模型

---

## 🛠️ 环境要求

```bash
# 核心依赖
pip install numpy matplotlib pandas scipy

# 深度学习（Module 3）
pip install torch

# 交互式组件（可选）
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

---

## 📖 使用方式

### 方式1：在 Jupyter Lab/Notebook 中打开

```bash
cd notebooks/education/
jupyter lab
```

### 方式2：在 VS Code 中打开

安装 Python 和 Jupyter 扩展，直接打开 `.ipynb` 文件。

### 方式3：在 Google Colab 中运行

上传 notebook 到 Google Drive，用 Colab 打开。

---

## 🧑‍🏫 教师指南

### 建议课时安排

- **Module 1**: 2 学时（含讲解和练习）
- **Module 2**: 3 学时（HBV结构较复杂）
- **Module 3**: 3 学时（需要解释深度学习基础）
- **Module 4**: 2 学时（评估指标对比）

### 每个模块包含

- 📝 **Markdown 讲解**：概念说明和公式推导
- 💻 **代码演示**：可直接运行的示例
- 🔬 **交互实验**：参数调整和效果观察
- ✏️ **练习题**：课后作业和思考题
- 📚 **总结**：关键概念回顾

---

## 🔗 相关资源

- [理论手册](../../docs/THEORY_GUIDE.md)：详细的模型理论推导
- [数据字典](../../docs/DATA_DICTIONARY.md)：数据格式规范
- [架构指南](../../docs/ARCHITECTURE_GUIDE.md)：代码架构说明

---

## 📧 反馈

如有问题或建议，请提交 Issue 或 Pull Request。
