# Data Cleaning Sacred Migration Guide

## 概述

`data_cleaning.py` 已从使用 `argparse` 迁移到使用 `Sacred` 进行实验管理，并支持从 YAML 配置文件加载参数。

## 主要变更

### 1. 使用 Sacred 替代 argparse

- 所有命令行参数现在通过 Sacred 配置管理
- 实验运行历史和配置自动记录到 `results/sacred/` 目录
- 支持配置的可复现性追踪

### 2. YAML 配置文件支持

系统现在会自动加载以下配置文件：

- **Dataset 配置**: `src/config/dataset/{dataset_key}.yaml`
- **Agent 配置**: `src/config/agent/{agent_config_file}.yaml` 或 `default.yaml`

配置加载优先级：
1. YAML 配置文件（基础配置）
2. Sacred 命令行参数（覆盖配置）

### 3. DatasetLoader 集成

数据加载现在使用 `DatasetLoader` 类：
- 统一的数据加载接口
- 自动从 YAML 配置读取数据集设置
- 支持自定义 prompt 构建

### 4. LLMAgent 集成（可选）

可以通过配置启用 LLMAgent 工具调用：
- 设置 `use_agent_tools=true` 启用
- 通过 `agent_config_file` 指定 agent 配置文件
- 支持代码执行、搜索、思维导图等工具

## 使用方法

### 基本用法

```bash
python src/data_cleaning.py \
    with dataset=gsm8k \
    model="/path/to/model" \
    output="output.json"
```

### 使用 YAML 配置

```bash
python src/data_cleaning.py \
    with dataset=gsm8k \
    agent_config_file="default.yaml" \
    model="/path/to/model" \
    output="output.json"
```

### 命令行参数覆盖 YAML 配置

```bash
python src/data_cleaning.py \
    with dataset=gsm8k \
    agent_config_file="default.yaml" \
    model="/path/to/model" \
    temperature=0.5 \
    batch_size=16 \
    output="output.json"
```

### 启用 Agent 工具

```bash
python src/data_cleaning.py \
    with dataset=gsm8k \
    agent_config_file="vllm_with_tools.yaml" \
    use_agent_tools=true \
    model="/path/to/model" \
    output="output.json"
```

## 配置文件示例

### Dataset 配置 (`src/config/dataset/gsm8k.yaml`)

```yaml
name: "GSM8K"
key: "gsm8k"
category: "math"
task_type: "math"
default_source: "json"
default_input: "datasets/GSM8k/main/test-00000-of-00001.parquet"
adapter: "dataformat.adapters.gsm8k_adapter"
```

### Agent 配置 (`src/config/agent/default.yaml`)

```yaml
backend: "vllm"
model_path: ""
tokenizer_path: null
vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
generation:
  temperature: 0.7
  top_p: 1.0
  max_tokens: 2048
  max_new_tokens: 2048
```

## 主要参数说明

### 必需参数

- `dataset`: 数据集名称 (squadv2, hotpot, nq, aqua, gsm8k, math, omini)
- `model`: 模型路径（必需）
- `output`: 输出文件路径（必需）

### 可选参数

所有原有 argparse 参数都支持，可以通过 Sacred 命令行或 YAML 配置设置。

## 迁移指南

### 从旧版本迁移

旧版本使用 argparse：

```bash
python data_cleaning.py \
    --dataset gsm8k \
    --model /path/to/model \
    --output output.json
```

新版本使用 Sacred：

```bash
python src/data_cleaning.py \
    with dataset=gsm8k \
    model="/path/to/model" \
    output="output.json"
```

### 配置参数映射

- `--dataset` → `dataset=`
- `--model` → `model=`
- `--output` → `output=`
- `--backend` → `backend=`
- `--temperature` → `temperature=`
- 等等...

## 实验记录

所有实验运行记录在 `results/sacred/` 目录下，包括：
- 配置快照
- 运行日志
- 结果文件
- 可复现性信息

## 注意事项

1. **向后兼容性**: 旧的 `build_args()` 函数仍然存在，但不再使用
2. **配置优先级**: 命令行参数 > YAML 配置文件 > 默认值
3. **DatasetLoader**: 优先使用 DatasetLoader，如果失败会回退到旧的 adapter 方式
4. **LLMAgent**: 需要显式启用 `use_agent_tools=true` 才会使用工具功能

## 依赖项

新增依赖：
- `sacred`: 实验管理框架
- `pyyaml`: YAML 配置文件解析

安装：
```bash
pip install sacred pyyaml
```


