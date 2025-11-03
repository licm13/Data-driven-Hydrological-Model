# 数据目录自动检测使用说明

## 概述

`run_all_experiments.py` 脚本现在支持自动检测数据目录位置，无需手动指定数据路径。系统会自动搜索 Dataset 目录并验证流域数据的完整性。

## 数据目录结构

脚本支持以下数据组织方式：

### 方式1：流域子目录结构（推荐）
```
Dataset/
└── IMPRO_catchment_data_infotheo/
    ├── iller/
    │   ├── discharge_iller.csv
    │   ├── precipitation_iller.asc
    │   ├── temperature_iller.asc
    │   └── pet_iller.asc
    ├── saale/
    │   ├── discharge_saale.csv
    │   ├── precipitation_saale.asc
    │   ├── temperature_saale.asc
    │   └── pet_saale.asc
    └── selke/
        ├── discharge_selke.csv
        ├── precipitation_selke.asc
        ├── temperature_selke.asc
        └── pet_selke.asc
```

### 方式2：平铺文件结构
```
Dataset/
└── data_directory/
    ├── discharge_iller.csv
    ├── precipitation_iller.asc
    ├── temperature_iller.asc
    ├── pet_iller.asc
    ├── discharge_saale.csv
    ├── precipitation_saale.asc
    ├── temperature_saale.asc
    ├── pet_saale.asc
    ├── discharge_selke.csv
    ├── precipitation_selke.asc
    ├── temperature_selke.asc
    └── pet_selke.asc
```

## 自动检测逻辑

### 1. Dataset 目录搜索
脚本会在以下位置搜索 Dataset 目录：
- `项目根目录/Dataset`
- `项目根目录/../Dataset`
- `项目根目录/../../Dataset`

### 2. 数据子目录搜索
在 Dataset 目录中搜索以下可能的数据子目录：
- `IMPRO_catchment_data_infotheo`
- `catchment_data`
- `data`
- `raw_data`
- `input_data`

### 3. 流域数据验证
对每个指定的流域，验证以下必需文件是否存在：
- `discharge_[catchment].csv` - 径流数据
- `precipitation_[catchment].asc` - 降水数据
- `temperature_[catchment].asc` - 温度数据
- `pet_[catchment].asc` - 蒸发数据

## 使用方法

### 1. 自动检测模式（推荐）
```bash
# 使用自动检测的数据目录
python run_all_experiments.py

# 快速测试模式
python run_all_experiments.py --quick_test

# 指定特定流域
python run_all_experiments.py --catchments Iller Saale
```

### 2. 手动指定数据目录
```bash
# 手动指定数据目录路径
python run_all_experiments.py --data_dir "/path/to/your/data"

# 使用相对路径
python run_all_experiments.py --data_dir "../Dataset/IMPRO_catchment_data_infotheo"
```

### 3. 测试数据检测功能
```bash
# 运行数据检测测试
python test_data_detection.py
```

## 错误处理

### 数据目录未找到
```
错误: 未找到 Dataset 目录！
已搜索以下路径:
  - C:\path\to\project\Dataset
  - C:\path\to\project\..\Dataset
  - C:\path\to\project\..\..\Dataset
请确保 Dataset 目录存在于项目目录或其父目录中。
```

**解决方案：**
1. 确认 Dataset 目录存在
2. 手动指定 `--data_dir` 参数

### 数据子目录未找到
```
错误: 在 Dataset 目录中未找到包含数据文件的子目录！
Dataset 路径: C:\path\to\Dataset
可用的子目录: ['folder1', 'folder2']
```

**解决方案：**
1. 检查数据文件是否存在于子目录中
2. 确认文件格式为 .csv 或 .asc

### 流域数据不完整
```
流域 ILLER 的数据文件不完整！
缺失文件:
  - precipitation: C:\path\to\data\precipitation_iller.asc
  - temperature: C:\path\to\data\temperature_iller.asc
可用文件:
  + discharge: C:\path\to\data\discharge_iller.csv
  + pet: C:\path\to\data\pet_iller.asc
```

**解决方案：**
1. 下载缺失的数据文件
2. 检查文件命名是否正确
3. 在快速测试模式下跳过有问题的流域

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 自动检测 | 数据目录路径 |
| `--catchments` | ['Iller', 'Saale', 'Selke'] | 流域名称列表 |
| `--experiments` | ['1', '2', '3', '4'] | 要运行的实验编号 |
| `--output_dir` | './results' | 结果输出目录 |
| `--quick_test` | False | 快速测试模式 |
| `--synthetic` | False | 使用合成数据 |
| `--n_replicates` | 30 | 每个样本大小的重复次数 |

## 示例输出

### 成功检测
```
正在自动检测数据目录...
找到 Dataset 目录: C:\Users\user\Dataset
找到数据子目录: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo
发现流域子目录结构: ['iller', 'saale', 'selke']
数据组织方式: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo/[catchment_name]/[data_files]
✓ 成功找到数据目录: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo

正在验证流域数据...
使用流域子目录: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo\iller
✓ 流域 ILLER 的所有数据文件已找到
使用流域子目录: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo\saale
✓ 流域 SAALE 的所有数据文件已找到
使用流域子目录: C:\Users\user\Dataset\IMPRO_catchment_data_infotheo\selke
✓ 流域 SELKE 的所有数据文件已找到
```

## 故障排除

1. **权限问题**：确保对 Dataset 目录有读取权限
2. **路径问题**：使用绝对路径避免相对路径问题
3. **文件名大小写**：确保文件名大小写匹配（特别是在 Linux 系统上）
4. **文件格式**：确认数据文件为正确的格式（.csv 或 .asc）

通过这个自动检测系统，您无需担心数据路径配置问题，脚本会自动找到并验证数据的完整性。