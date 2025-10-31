# Benchmark 使用说明

## 📋 概述

本项目现在专注于**性能测试**，提供了完整的自动化benchmark功能，用于比较Speculative Decoding和标准自回归生成的性能。

## 🚀 快速开始

### 1. 配置GPU分配

编辑 `run_benchmark.sh` 配置GPU分配：

```bash
# 设置可用GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 选择GPU分配策略
GPU_STRATEGY="multi_gpu_ratio"  # 选项: multi_gpu_ratio, separate, same, auto

# GPU比例配置（multi_gpu_ratio策略）
TARGET_GPU_RATIO=6    # Target模型使用6张GPU
DRAFTER_GPU_RATIO=2    # Drafter模型使用2张GPU
```

### 2. 配置Benchmark参数

编辑 `run_benchmark.sh` 设置测试参数：

```bash
# Benchmark模式（二选一）:
# 选项1: 固定prompt数量
export NUM_PROMPTS=100

# 选项2: 基于时间（时长 + 速率）
export AUTO_RATE=1.0              # 每秒请求数
export AUTO_DURATION=300          # 运行时长（秒）

# Batch处理
export ENABLE_BATCH="true"        # 启用batch处理
export BATCH_SIZE=4               # Batch大小
export MAX_BATCH_LENGTH=512      # 最大序列长度

# 生成参数
export GENERATION_LENGTH=100     # 生成长度（tokens）
export GAMMA_VALUE=4             # Gamma参数

# 功能开关
export ENABLE_SPECULATIVE="true" # 启用speculative decoding
export ENABLE_TARGET="true"      # 启用target AR（用于对比）
export ENABLE_GPU_MONITOR="true" # 启用GPU能耗监控
```

### 3. 运行Benchmark

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

## 📊 输出结果

Benchmark完成后会生成以下文件：

- `benchmark_results.json` - 完整结果（speculative + target + GPU指标）
- `benchmark_results_speculative.json` - Speculative decoding结果
- `benchmark_results_target.json` - Target AR结果
- `benchmark_results_gpu.json` - GPU监控结果

控制台会显示格式化的性能摘要。

## 🎯 GPU分配策略

### multi_gpu_ratio（推荐）

按比例分配多张GPU：

```bash
GPU_STRATEGY="multi_gpu_ratio"
TARGET_GPU_RATIO=6
DRAFTER_GPU_RATIO=2
```

**效果**:
- Target模型（8B）: GPU 0-5（层自动分配到6张GPU）
- Drafter模型（1.7B）: GPU 6-7（层自动分配到2张GPU）

### separate

分离部署：

```bash
GPU_STRATEGY="separate"
```

**效果**:
- Target模型 → GPU 0
- Drafter模型 → GPU 1

### same

共享GPU：

```bash
GPU_STRATEGY="same"
```

**效果**:
- Target模型和Drafter模型共享GPU 0

### auto

自动分配：

```bash
GPU_STRATEGY="auto"
```

**效果**: 让transformers自动决定GPU分配

详细说明请参考：[GPU部署指南](docs/GPU_DEPLOYMENT_CN.md)

## 📈 性能指标

Benchmark会收集以下指标：

### 推理指标
- **TTFT (Time To First Token)** - 首token延迟
- **端到端延迟** - 每个请求的总生成时间
- **吞吐量** - 每秒token数
- **Token数量** - Prompt tokens、生成tokens、总tokens
- **接受率** - 草稿接受率（speculative decoding）

### GPU指标
- **能耗** - 平均、峰值、总能耗
- **GPU利用率** - 平均GPU利用率百分比
- **内存使用** - 平均内存使用百分比
- **温度** - 峰值GPU温度

## 🔧 高级配置

### 自定义模型路径

编辑 `benchmark.py` 修改模型路径：

```python
target_model = "/path/to/target/model"
drafter_model = "/path/to/drafter/model"
```

### 数据集配置

配置ShareGPT数据集目录：

```bash
export SHAREGPT_DIR="/path/to/sharegpt/directory"
```

Benchmark会从以下文件加载prompts：
- `sharegpt_gpt4.jsonl`
- `sharegpt_V3_format.jsonl`
- `sharegpt_zh_38K_format.jsonl`

### 采样间隔配置

```bash
export GPU_MONITOR_INTERVAL=1.0  # GPU监控采样间隔（秒）
```

## 🐛 故障排除

### GPU内存不足
- 减少 `BATCH_SIZE` 或 `MAX_BATCH_LENGTH`
- 减少 `GAMMA_VALUE`
- 为target模型分配更多GPU

### 模型加载失败
- 检查 `benchmark.py` 中的模型路径
- 确保模型兼容（相同tokenizer，相同logit形状）
- 使用 `nvidia-smi` 验证GPU可用性

### Benchmark问题
- 检查ShareGPT数据集目录路径
- 验证环境变量设置正确
- 检查GPU分配是否匹配可用GPU

## 📚 相关文档

- [GPU部署详细指南](docs/GPU_DEPLOYMENT_CN.md)
- [主README](README.md)
