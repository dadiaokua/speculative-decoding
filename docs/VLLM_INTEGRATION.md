# vLLM Integration Guide

## 概述

本项目现在支持两种推理引擎：

1. **Transformers** (默认): 使用 Hugging Face Transformers 库
2. **vLLM**: 使用 vLLM 高性能推理引擎

## vLLM 简介

vLLM 是一个高性能的 LLM 推理引擎，具有以下特性：

- **连续批处理 (Continuous Batching)**: 动态管理批处理，提高吞吐量
- **PagedAttention**: 高效的注意力机制内存管理
- **张量并行**: 支持多GPU并行推理
- **流水线并行**: 支持大模型的层间并行
- **高吞吐量**: 比标准 Transformers 快 2-10倍

## 安装 vLLM

```bash
# 激活虚拟环境
source spec_venv/bin/activate

# 安装 vLLM
pip install vllm
```

或者取消注释 `requirements.txt` 中的 vLLM 行：

```bash
# 编辑 requirements.txt
# 将 # vllm>=0.5.0 改为 vllm>=0.5.0

pip install -r requirements.txt
```

## 使用方法

### 1. 在 `run_benchmark.sh` 中配置

编辑 `run_benchmark.sh` 文件，设置推理引擎：

```bash
# 推理引擎选择
export INFERENCE_ENGINE="vllm"   # 选项: "transformers", "vllm"
```

### 2. vLLM 参数配置

`run_benchmark.sh` 中提供了以下 vLLM 参数：

```bash
# vLLM引擎参数
export VLLM_TENSOR_PARALLEL_SIZE=8       # 张量并行大小（通常等于GPU数量）
export VLLM_GPU_MEMORY_UTILIZATION=0.9   # GPU显存利用率（0-1之间）
export VLLM_MAX_MODEL_LEN=4096           # 最大模型长度
export VLLM_MAX_NUM_SEQS=128             # 最大并发序列数
export VLLM_MAX_NUM_BATCHED_TOKENS=8192  # 批处理最大token数（可选，默认自动计算）
export VLLM_DISABLE_LOG_STATS=true       # 是否禁用日志统计
export VLLM_DTYPE="half"                 # 数据类型: "half", "float16", "bfloat16"

# vLLM推测解码参数（可选，启用后使用vLLM原生推测解码）
export VLLM_ENABLE_SPECULATIVE="true"    # 是否启用vLLM推测解码
export VLLM_NUM_SPECULATIVE_TOKENS=5     # 推测token数量（默认使用GAMMA_VALUE）
export VLLM_USE_V2_BLOCK_MANAGER="true"  # 是否使用v2块管理器（推荐）
```

### 3. 启用 vLLM 原生推测解码（推荐）

vLLM 原生支持推测解码，性能更优！

```bash
# 1. 设置推理引擎为 vLLM
export INFERENCE_ENGINE="vllm"

# 2. 设置推理方法为推测解码
export INFERENCE_METHOD="speculative"

# 3. 启用 vLLM 推测解码
export VLLM_ENABLE_SPECULATIVE="true"

# 4. 设置推测 token 数量（可选，默认使用 GAMMA_VALUE）
export VLLM_NUM_SPECULATIVE_TOKENS=5
```

**关键优势**：
- ✅ vLLM 使用单引擎模式加载 Target 和 Drafter
- ✅ 自动优化的推测解码实现
- ✅ 更高的推测接受率
- ✅ 更低的延迟和显存占用

### 4. 运行基准测试

```bash
bash run_benchmark.sh
```

## 参数说明

### `VLLM_TENSOR_PARALLEL_SIZE`
- **含义**: 张量并行的 GPU 数量
- **建议**: 
  - 单机8卡: 设置为 8
  - 单机4卡: 设置为 4
  - 单卡: 设置为 1
- **注意**: 必须能被 GPU 总数整除

### `VLLM_GPU_MEMORY_UTILIZATION`
- **含义**: GPU 显存利用率
- **建议**: 
  - 0.9 (默认): 平衡性能与稳定性
  - 0.95: 追求最大性能（可能OOM）
  - 0.8: 更保守，留更多显存余量

### `VLLM_MAX_MODEL_LEN`
- **含义**: 模型支持的最大序列长度
- **建议**:
  - 4096 (默认): 适合大多数对话场景
  - 8192: 长文本场景
  - 2048: 短文本场景，可减少显存占用

### `VLLM_MAX_NUM_SEQS`
- **含义**: 最大并发序列数（批大小上限）
- **建议**:
  - 128 (默认): 高吞吐量场景
  - 256: 极高并发场景
  - 64: 低并发、低延迟场景

### `VLLM_MAX_NUM_BATCHED_TOKENS`
- **含义**: 批处理中最大 token 数量
- **默认**: `None` (由 vLLM 自动计算，通常为 `max_num_seqs * max_model_len`)
- **建议**:
  - 不设置: 让 vLLM 自动优化（推荐）
  - 手动设置: 用于精细控制内存和性能
  - 高吞吐量: 设置更大值（如 16384）
  - 低延迟: 设置更小值（如 4096）
- **注意**: 
  - 设置过大可能导致 OOM
  - 设置过小会限制批处理效率
  - 与 `max_num_seqs` 和 `max_model_len` 密切相关

### `VLLM_DTYPE`
- **含义**: 数据类型
- **选项**:
  - `"half"` (默认): FP16，平衡精度与速度
  - `"float16"`: 同 half
  - `"bfloat16"`: BF16，某些硬件支持更好
  - `"float32"`: FP32，最高精度但速度慢

## 性能对比

| 引擎 | 吞吐量 | 延迟 | 显存占用 | 易用性 |
|------|--------|------|----------|--------|
| Transformers | 低 | 高 | 中 | ⭐⭐⭐⭐⭐ |
| vLLM | 高 (2-10x) | 低 | 低 | ⭐⭐⭐⭐ |

## 参数详细说明

### `VLLM_ENABLE_SPECULATIVE`
- **含义**: 是否启用 vLLM 原生推测解码
- **选项**: 
  - `"true"`: 启用（需要同时设置 `INFERENCE_METHOD="speculative"`）
  - `"false"`: 禁用，使用标准 AR 生成
- **推荐**: `"true"` （性能更优）

### `VLLM_NUM_SPECULATIVE_TOKENS`
- **含义**: 推测生成的 token 数量（对应 Transformers 模式的 `GAMMA_VALUE`）
- **建议**: 
  - 5 (默认): 平衡性能与接受率
  - 3-4: 更保守，接受率更高
  - 6-8: 更激进，可能接受率较低
- **自动回退**: 如果不设置，会自动使用 `GAMMA_VALUE`

### `VLLM_USE_V2_BLOCK_MANAGER`
- **含义**: 是否使用 v2 块管理器（vLLM 新版本特性）
- **建议**: `"true"` （推荐，性能更好）
- **注意**: v2 块管理器是推测解码的推荐配置

## 限制与注意事项

### 当前限制

1. **TTFT 指标**: vLLM 的 TTFT (Time To First Token) 指标当前设为 0，因为难以精确测量
   - 未来版本将实现更准确的 TTFT 测量

2. **批处理**: vLLM 自带连续批处理，无需额外配置
   - `ENABLE_BATCH` 参数对 vLLM 无效

3. **接受率统计**: vLLM 原生推测解码的接受率统计可能与 Transformers 实现略有不同

### 硬件要求

- **GPU**: NVIDIA GPU with Compute Capability ≥ 7.0
  - V100, A100, A10, RTX 20/30/40 系列等
- **CUDA**: 11.8 或更高
- **显存**: 
  - Qwen3-8B: 至少 16GB (FP16)
  - Qwen3-1.7B: 至少 4GB (FP16)
  - 多卡分布式: 可突破单卡显存限制

## 故障排查

### 错误: "vLLM is not installed"

```bash
pip install vllm
```

### 错误: "CUDA error: out of memory"

降低参数：
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_MAX_MODEL_LEN=2048
export VLLM_MAX_NUM_SEQS=64
```

### 错误: "Tensor parallel size must divide the number of GPUs"

确保 `VLLM_TENSOR_PARALLEL_SIZE` 与可用 GPU 数量匹配：

```bash
# 8卡
export VLLM_TENSOR_PARALLEL_SIZE=8

# 4卡
export VLLM_TENSOR_PARALLEL_SIZE=4

# 单卡
export VLLM_TENSOR_PARALLEL_SIZE=1
```

## 示例配置

### 配置 1: vLLM 推测解码（推荐，8x V100 32GB）

```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=128
export VLLM_DTYPE="half"
export VLLM_USE_V2_BLOCK_MANAGER="true"
```

### 配置 2: 高吞吐量（标准 AR，8x V100 32GB）

```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="target_ar"
export VLLM_ENABLE_SPECULATIVE="false"
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=256
export VLLM_DTYPE="half"
```

### 配置 3: 低延迟（推测解码，8x V100 32GB）

```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=4
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_MAX_MODEL_LEN=2048
export VLLM_MAX_NUM_SEQS=32
export VLLM_DTYPE="half"
```

### 配置 4: 单卡测试（推测解码，1x V100 32GB）

```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=64
export VLLM_DTYPE="half"
```

## 未来计划

- [x] ~~实现 vLLM + 推测解码集成~~ ✅ 已完成（使用 vLLM 原生推测解码）
- [ ] 改进 TTFT 测量精度
- [ ] 支持更多 vLLM 高级特性（如 prefix caching）
- [ ] 添加 vLLM 专用性能指标
- [ ] 优化 vLLM 推测解码的接受率统计

## 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

