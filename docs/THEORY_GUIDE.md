# 理论手册 (Theory Guide)

本文档提供水文模型的理论基础，帮助用户深入理解各类模型的原理和应用。

---

## 目录

1. [物理模型](#1-物理模型)
   - [HBV模型](#11-hbv模型)
   - [GR4J模型](#12-gr4j模型)
2. [数据驱动模型](#2-数据驱动模型)
   - [LSTM网络](#21-lstm网络)
   - [神经网络水文应用](#22-神经网络水文应用)
3. [混合建模方法](#3-混合建模方法)
   - [微分参数学习(dPL)](#31-微分参数学习dpl)
4. [评估指标理论](#4-评估指标理论)

---

## 1. 物理模型

物理模型（也称为过程驱动模型）基于水文过程的物理方程来模拟降雨-径流关系。

### 1.1 HBV模型

#### 概述

HBV (Hydrologiska Byråns Vattenbalansavdelning) 模型是瑞典水文研究所开发的概念性水文模型，自1972年以来在全球得到广泛应用。

#### 模型结构

HBV模型包含四个主要模块：

```
[降水 P, 温度 T]
       ↓
┌──────────────────┐
│   1. 雪模块       │ ← 度日法融雪
│   (Snow Routine)  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   2. 土壤模块     │ ← 非线性产流
│   (Soil Routine)  │ → 蒸发 E
└────────┬─────────┘
         ↓
┌──────────────────┐
│   3. 响应模块     │ ← 双层水箱
│   (Response)      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   4. 汇流模块     │ ← 三角形单位线
│   (Routing)       │
└────────┬─────────┘
         ↓
    [径流 Q]
```

#### 核心方程

**1. 雪模块 - 度日法**

融雪量计算：
$$M = CFMAX \times (T - TT), \quad \text{when } T > TT$$

其中：
- $M$: 融雪量 [mm/day]
- $CFMAX$: 度日因子 [mm/°C/day]
- $T$: 气温 [°C]
- $TT$: 阈值温度 [°C]

降水类型判断：
$$P_{snow} = \begin{cases} P \times SFCF & T < TT \\ 0 & T \geq TT \end{cases}$$

**2. 土壤模块 - 非线性产流**

地下水补给（产流）：
$$R = P_{eff} \times \left(\frac{SM}{FC}\right)^{BETA}$$

其中：
- $R$: 地下水补给 [mm/day]
- $P_{eff}$: 有效降水 [mm/day]
- $SM$: 土壤含水量 [mm]
- $FC$: 田间持水量 [mm]
- $BETA$: 形状系数 [-]

实际蒸散发：
$$E_a = E_p \times \min\left(1, \frac{SM}{LP \times FC}\right)$$

**3. 响应模块 - 双层水箱**

上层水箱出流：
- 快速流: $Q_0 = K_0 \times \max(0, SUZ - UZL)$
- 中速流: $Q_1 = K_1 \times SUZ$

下层水箱出流（基流）：
- 基流: $Q_2 = K_2 \times SLZ$

渗漏：
$$PERC = \min(PERC_{max}, SUZ)$$

**4. 汇流模块 - 三角形单位线**

单位线权重：
$$w_i = \begin{cases} \frac{2i}{MAXBAS \times (MAXBAS + 1)} & i \leq \frac{MAXBAS}{2} \\ \frac{2(MAXBAS - i)}{MAXBAS \times (MAXBAS + 1)} & i > \frac{MAXBAS}{2} \end{cases}$$

#### 参数说明

| 参数 | 物理含义 | 典型范围 | 单位 |
|------|----------|----------|------|
| TT | 降雪/融雪阈值温度 | -2.5 ~ 2.5 | °C |
| CFMAX | 度日因子 | 1 ~ 10 | mm/°C/d |
| FC | 田间持水量 | 50 ~ 500 | mm |
| BETA | 产流形状系数 | 1 ~ 6 | - |
| LP | 蒸发限制参数 | 0.3 ~ 1.0 | - |
| K0 | 快速流系数 | 0.05 ~ 0.5 | 1/d |
| K1 | 中速流系数 | 0.01 ~ 0.4 | 1/d |
| K2 | 基流系数 | 0.001 ~ 0.15 | 1/d |
| MAXBAS | 汇流时间 | 1 ~ 7 | d |

---

### 1.2 GR4J模型

#### 概述

GR4J (Génie Rural à 4 paramètres Journalier) 是法国国家农业研究院开发的日尺度集总式水文模型，以仅需4个参数而著称。

#### 模型结构

```
    [P, E]
       ↓
┌──────────────────┐
│ 产流存储库        │ ← 容量 X1
│ (Production Store)│
└────────┬─────────┘
         ↓
    [净降水 Pn]
         ↓
    ┌────┴────┐
    ↓         ↓
  90%       10%
    ↓         ↓
┌────────┐  直接
│汇流存储 │  径流
│库 (X3) │
└────┬───┘
     ↓
    [Q]
```

#### 核心方程

**产流存储库**

存储量变化：
$$\frac{dS}{dt} = P \times \left(1 - \left(\frac{S}{X_1}\right)^2\right) - E \times \left(\frac{S}{X_1}\right) \times \left(2 - \frac{S}{X_1}\right)$$

**汇流过程**

使用两个单位线 UH1 和 UH2 进行汇流计算。

#### 参数说明

| 参数 | 物理含义 | 典型范围 | 单位 |
|------|----------|----------|------|
| X1 | 产流存储库容量 | 100 ~ 1200 | mm |
| X2 | 地下水交换系数 | -5 ~ 3 | mm |
| X3 | 汇流存储库容量 | 20 ~ 300 | mm |
| X4 | 单位线时基 | 1.1 ~ 2.9 | d |

---

## 2. 数据驱动模型

### 2.1 LSTM网络

#### 概述

长短期记忆网络（Long Short-Term Memory, LSTM）是一种特殊的循环神经网络（RNN），能够学习时间序列中的长期依赖关系。

#### 网络结构

LSTM单元包含三个门控机制：

```
                    ┌─────────────┐
         c_{t-1} ──→│   Cell      │──→ c_t
                    │   State     │
                    └──────┬──────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    │          │           │           │          │
    │    ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐    │
    │    │  Forget   │ │ Input │ │  Output   │    │
    │    │   Gate    │ │ Gate  │ │   Gate    │    │
    │    └─────┬─────┘ └───┬───┘ └─────┬─────┘    │
    │          │           │           │          │
    └──────────┴───────────┴───────────┴──────────┘
                           │
              x_t, h_{t-1} │ 输入
```

#### 门控方程

**遗忘门 (Forget Gate)**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门 (Input Gate)**：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门 (Output Gate)**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 2.2 神经网络水文应用

#### 水文意义类比

| LSTM组件 | 水文类比 |
|----------|----------|
| Cell State | 流域存储（土壤水、地下水） |
| Forget Gate | 蒸发、下渗损失 |
| Input Gate | 降雨补给 |
| Output Gate | 径流产出 |

#### 输入特征设计

典型输入特征包括：
- 降水 (P)
- 温度 (T)
- 潜在蒸散发 (PET)
- 日照时长
- 流域属性（面积、高程、坡度等）

#### 回溯窗口选择

回溯窗口（lookback period）的选择影响模型性能：
- 太短：无法捕捉长期记忆
- 太长：计算量大，可能引入噪声
- 典型值：30-365天

---

## 3. 混合建模方法

### 3.1 微分参数学习 (dPL)

#### 概念

微分参数学习（differentiable Parameter Learning, dPL）结合了物理模型的可解释性和深度学习的灵活性。

#### 架构

```
┌─────────────────────────────────────────────────────┐
│                   dPL Framework                      │
│                                                      │
│   ┌─────────────┐         ┌─────────────────────┐  │
│   │    LSTM     │ ──→     │   Physical Model    │  │
│   │  (学习参数)  │ θ(x,t)  │   (HBV/GR4J)        │  │
│   └─────────────┘         └──────────┬──────────┘  │
│         ↑                            │              │
│         │                            ↓              │
│   [气象数据, 流域属性]            [模拟径流]          │
│                                      │              │
│                            Loss = f(Q_sim, Q_obs)   │
│                                      │              │
│                              反向传播 ↓              │
│                         ┌────────────┴────────────┐ │
│                         │ 更新LSTM权重             │ │
│                         └─────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

#### 优势

1. **可解释性**：输出的参数有明确物理意义
2. **数据效率**：物理约束减少所需训练数据
3. **泛化能力**：物理结构提供外推能力
4. **不确定性量化**：可以分析参数分布

#### 实现要点

- 物理模型必须是可微分的（使用PyTorch实现）
- 参数需要有合理的约束范围
- 可以使用预训练（先用Spotpy找好初值）

---

## 4. 评估指标理论

### 4.1 Nash-Sutcliffe效率系数 (NSE)

$$NSE = 1 - \frac{\sum_{t=1}^{n}(Q_{obs}^t - Q_{sim}^t)^2}{\sum_{t=1}^{n}(Q_{obs}^t - \overline{Q_{obs}})^2}$$

**特点**：
- 范围：(-∞, 1]
- NSE=1：完美预测
- NSE=0：与观测均值一样好
- 对洪峰敏感（平方项）

### 4.2 Kling-Gupta效率系数 (KGE)

$$KGE = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$

其中：
- $r$：皮尔逊相关系数
- $\alpha = \sigma_{sim}/\sigma_{obs}$：变异性比
- $\beta = \mu_{sim}/\mu_{obs}$：偏差比

### 4.3 信息论指标

**条件熵**：
$$H(Y|X) = -\sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)}$$

**归一化条件熵**：
$$H_{norm} = \frac{H(Y_{obs}|Y_{sim})}{H(Y_{obs})}$$

**互信息**：
$$I(X;Y) = H(Y) - H(Y|X)$$

---

## 参考文献

1. Bergström, S. (1976). Development and application of a conceptual runoff model for Scandinavian catchments.

2. Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.

3. Kratzert, F., et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23(12), 5089-5110.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

5. Gupta, H. V., et al. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.
