# 数据字典 (Data Dictionary)

本文档详细定义了项目支持的数据格式、变量命名规范和数据质量要求。

---

## 目录

1. [数据格式概览](#1-数据格式概览)
2. [CAMELS数据集](#2-camels数据集)
3. [IMPRO ASCII格式](#3-impro-ascii格式)
4. [标准CSV格式](#4-标准csv格式)
5. [变量命名规范](#5-变量命名规范)
6. [数据质量要求](#6-数据质量要求)
7. [UniversalHydroLoader配置](#7-universalhydroloader配置)

---

## 1. 数据格式概览

本项目支持三种主要数据格式：

| 格式 | 文件类型 | 适用场景 | 加载器 |
|------|----------|----------|--------|
| CAMELS | NetCDF/CSV | 大规模研究、基准测试 | CAMELSLoaderStrategy |
| IMPRO | ASCII | 德国流域研究 | IMPROLoaderStrategy |
| 标准CSV | CSV/TSV | 自定义数据、教学 | CSVLoaderStrategy |

---

## 2. CAMELS数据集

### 2.1 概述

CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) 是美国671个流域的基准数据集。

### 2.2 目录结构

```
CAMELS_US/
├── basin_mean_forcing/
│   ├── daymet/
│   │   └── 01013500_lump_cida_forcing_leap.txt
│   └── nldas/
├── usgs_streamflow/
│   └── 01013500_streamflow_qc.txt
└── camels_attributes_v2.0/
    ├── camels_clim.txt
    ├── camels_geol.txt
    ├── camels_hydro.txt
    ├── camels_soil.txt
    ├── camels_topo.txt
    └── camels_vege.txt
```

### 2.3 强迫数据格式

文件：`{basin_id}_lump_cida_forcing_leap.txt`

| 列号 | 变量名 | 描述 | 单位 | 精度 |
|------|--------|------|------|------|
| 1 | Year | 年份 | - | 整数 |
| 2 | Mnth | 月份 | - | 整数 |
| 3 | Day | 日 | - | 整数 |
| 4 | Hr | 小时 | - | 整数 |
| 5 | dayl | 日照时长 | s | 0.1 |
| 6 | prcp | 降水 | mm/day | 0.01 |
| 7 | srad | 短波辐射 | W/m² | 0.1 |
| 8 | swe | 雪水当量 | mm | 0.1 |
| 9 | tmax | 最高温度 | °C | 0.01 |
| 10 | tmin | 最低温度 | °C | 0.01 |
| 11 | vp | 水汽压 | Pa | 0.1 |

### 2.4 径流数据格式

文件：`{basin_id}_streamflow_qc.txt`

| 列号 | 变量名 | 描述 | 单位 | 精度 |
|------|--------|------|------|------|
| 1 | gauge_id | 站点ID | - | 字符串 |
| 2 | Year | 年份 | - | 整数 |
| 3 | Month | 月份 | - | 整数 |
| 4 | Day | 日 | - | 整数 |
| 5 | streamflow | 日均流量 | ft³/s | 0.01 |
| 6 | QC_flag | 质控标记 | - | A/M |

### 2.5 流域属性

**气候属性 (camels_clim.txt)**

| 变量名 | 描述 | 单位 |
|--------|------|------|
| p_mean | 年均降水 | mm/day |
| pet_mean | 年均PET | mm/day |
| aridity | 干旱指数 | - |
| frac_snow | 降雪比例 | - |
| high_prec_freq | 高降水频率 | days/yr |

**地形属性 (camels_topo.txt)**

| 变量名 | 描述 | 单位 |
|--------|------|------|
| gauge_lat | 站点纬度 | ° |
| gauge_lon | 站点经度 | ° |
| elev_mean | 平均高程 | m |
| slope_mean | 平均坡度 | m/km |
| area_gages2 | 流域面积 | km² |

---

## 3. IMPRO ASCII格式

### 3.1 概述

IMPRO项目使用的ASCII格式，主要用于德国流域的水文研究。

### 3.2 目录结构

```
IMPRO_catchment_data_infotheo/
├── iller/
│   ├── precipitation_iller.asc
│   ├── temperature_iller.asc
│   ├── pet_iller.asc
│   └── discharge_iller.csv
├── saale/
└── selke/
```

### 3.3 气象数据格式 (.asc)

**文件头格式**：
```
ncols         1
nrows         3654
xllcorner     0.0
yllcorner     0.0
cellsize      1.0
NODATA_value  -9999.0
```

**数据部分**：
- 每行一个时间步的值
- 无日期列（需要配合配置文件指定起始日期）
- 缺测值：-9999.0

### 3.4 径流数据格式 (.csv)

| 列名 | 描述 | 单位 | 格式 |
|------|------|------|------|
| date | 日期 | - | YYYY-MM-DD |
| discharge | 日均流量 | mm/day | 浮点数 |

**示例**：
```csv
date,discharge
2001-01-01,1.23
2001-01-02,1.45
2001-01-03,1.67
```

### 3.5 单位转换

IMPRO降水和PET可能使用不同单位，需要注意：

| 变量 | 原始单位 | 目标单位 | 转换 |
|------|----------|----------|------|
| precipitation | mm/day | mm/day | 无需转换 |
| temperature | °C | °C | 无需转换 |
| pet | mm/day | mm/day | 无需转换 |
| discharge | mm/day | mm/day | 无需转换 |

---

## 4. 标准CSV格式

### 4.1 推荐目录结构

```
data/raw/
├── {catchment_name}/
│   ├── meteorology.csv    # 气象数据
│   └── discharge.csv      # 径流数据
```

### 4.2 气象数据格式 (meteorology.csv)

**必需列**：

| 列名 | 描述 | 单位 | 数据类型 | 缺测值 |
|------|------|------|----------|--------|
| date | 日期 | - | datetime | 不允许 |
| precip | 降水量 | mm/day | float | NaN |
| temp | 平均温度 | °C | float | NaN |

**可选列**：

| 列名 | 描述 | 单位 | 数据类型 |
|------|------|------|----------|
| pet | 潜在蒸散发 | mm/day | float |
| tmax | 最高温度 | °C | float |
| tmin | 最低温度 | °C | float |
| humidity | 相对湿度 | % | float |
| wind | 风速 | m/s | float |
| radiation | 太阳辐射 | W/m² | float |

**示例**：
```csv
date,precip,temp,pet
2001-01-01,5.2,3.5,1.2
2001-01-02,0.0,2.1,1.0
2001-01-03,12.3,4.2,1.5
```

### 4.3 径流数据格式 (discharge.csv)

| 列名 | 描述 | 单位 | 数据类型 | 缺测值 |
|------|------|------|----------|--------|
| date | 日期 | - | datetime | 不允许 |
| discharge | 日均径流 | mm/day | float | NaN |

**注意**：径流单位应为mm/day（流域平均），如果原始数据为m³/s，需要转换：

$$Q_{mm/day} = \frac{Q_{m^3/s} \times 86400}{A_{km^2} \times 1000}$$

### 4.4 合并格式

也可以将气象和径流数据合并在一个文件中：

```csv
date,precip,temp,pet,discharge
2001-01-01,5.2,3.5,1.2,2.3
2001-01-02,0.0,2.1,1.0,2.1
```

---

## 5. 变量命名规范

### 5.1 标准变量名

| 标准名 | 允许的别名 | 描述 |
|--------|-----------|------|
| precip | P, precipitation, prcp, rainfall | 降水 |
| temp | T, temperature, tavg, tmean | 平均温度 |
| tmax | T_max, temperature_max | 最高温度 |
| tmin | T_min, temperature_min | 最低温度 |
| pet | PET, evapotranspiration, ET0 | 潜在蒸散发 |
| discharge | Q, streamflow, runoff | 径流 |
| date | Date, time, datetime, timestamp | 日期 |

### 5.2 单位规范

| 变量 | 标准单位 | 允许的输入单位 |
|------|----------|----------------|
| precip | mm/day | mm/h (×24), m/day (×1000) |
| temp | °C | K (-273.15), °F ((F-32)×5/9) |
| discharge | mm/day | m³/s (需提供面积转换) |

---

## 6. 数据质量要求

### 6.1 完整性要求

| 变量 | 最大缺测率 | 处理方法 |
|------|-----------|----------|
| precip | 5% | 线性插值（短期），标记为NaN（长期） |
| temp | 10% | 线性插值 |
| discharge | 20% | 标记为NaN，评估时跳过 |

### 6.2 质量检查

```python
# 推荐的质量检查规则
quality_checks = {
    'precip': {
        'min': 0,
        'max': 500,  # mm/day
        'dtype': 'float'
    },
    'temp': {
        'min': -50,
        'max': 50,  # °C
        'dtype': 'float'
    },
    'discharge': {
        'min': 0,
        'max': None,  # 无上限
        'dtype': 'float'
    }
}
```

### 6.3 时间序列要求

- **时间步长**：日尺度（推荐）
- **时间连续性**：不允许时间跳跃
- **预热期**：建议保留至少365天用于模型预热
- **最小长度**：训练+验证+测试 ≥ 3年

---

## 7. UniversalHydroLoader配置

### 7.1 配置结构

```yaml
observations:
  format: csv  # 'csv', 'impro', 'camels'
  data_dir: ./data/raw
  catchments:
    - Iller
    - Saale
  
  # CSV格式特定配置
  csv_options:
    meteorology_file: meteorology.csv
    discharge_file: discharge.csv
    date_column: date
    date_format: '%Y-%m-%d'
    separator: ','
    encoding: utf-8
    
  # IMPRO格式特定配置
  impro_options:
    start_date: '2001-01-01'
    precipitation_file: precipitation_{catchment}.asc
    temperature_file: temperature_{catchment}.asc
    pet_file: pet_{catchment}.asc
    discharge_file: discharge_{catchment}.csv
    
  # CAMELS格式特定配置
  camels_options:
    forcing_source: daymet  # 'daymet' or 'nldas'
    attributes_version: v2.0

# 时间划分
periods:
  train:
    start: '2001-01-01'
    end: '2010-12-31'
  validation:
    start: '2011-01-01'
    end: '2012-12-31'
  test:
    start: '2013-01-01'
    end: '2015-12-31'
  warmup_days: 365

# 采样策略
sampling:
  strategy: consecutive_random
  n_replicates: 30
  seed: 42
```

### 7.2 配置参数详解

| 参数 | 描述 | 默认值 |
|------|------|--------|
| format | 数据格式 | 'csv' |
| data_dir | 数据根目录 | './data' |
| catchments | 流域名称列表 | [] |
| date_format | 日期解析格式 | '%Y-%m-%d' |
| separator | CSV分隔符 | ',' |
| warmup_days | 预热期天数 | 365 |
| n_replicates | 采样重复次数 | 30 |

### 7.3 使用示例

```python
from dmg.core.data.loaders.universal_hydro_loader import UniversalHydroLoader

config = {
    'observations': {
        'format': 'csv',
        'data_dir': './data/raw',
        'catchments': ['Iller']
    },
    'sampling': {
        'strategy': 'douglas_peucker',
        'n_replicates': 30,
        'seed': 42
    }
}

loader = UniversalHydroLoader(config, test_split=True)
train_data, test_data = loader.get_data()

# 生成学习曲线样本
samples = loader.generate_learning_curve_samples(
    sample_sizes=[50, 100, 500, 1000]
)
```

---

## 附录：常见问题

### Q1: 如何将m³/s转换为mm/day？

```python
def convert_discharge(Q_m3s, area_km2):
    """
    将流量从m³/s转换为mm/day
    
    Q_mm = Q_m3s * 86400 / (area_km2 * 1000)
    """
    return Q_m3s * 86400 / (area_km2 * 1000)
```

### Q2: 如何处理PET缺失？

如果没有PET观测数据，可以使用Hargreaves方法估算：

$$PET = 0.0023 \times R_a \times (T_{mean} + 17.8) \times (T_{max} - T_{min})^{0.5}$$

### Q3: 如何处理闰年？

本项目默认处理闰年：
- CAMELS数据集包含闰年处理版本
- CSV格式按实际日期读取，自动处理闰年
