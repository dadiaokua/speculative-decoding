# vLLM Integration Guide

vLLM 是一个高性能的 LLM 推理引擎，可以提供 2-10x 的推理加速。

## 安装 vLLM

```bash
# 基础安装
pip install vllm

# 或指定版本
pip install vllm==0.6.3

# 依赖更新
pip install "transformers>=4.51.1" "tokenizers>=0.21.1"
```

## 快速开始

### 1. 配置 vLLM 引擎

编辑 `run_benchmark.sh`:

```bash
# 选择 vLLM 引擎
export INFERENCE_ENGINE="vllm"

# vLLM 基础配置
export VLLM_TENSOR_PARALLEL_SIZE=8       # 张量并行大小（= GPU 数量）
export VLLM_GPU_MEMORY_UTILIZATION=0.85  # GPU 显存利用率
export VLLM_MAX_MODEL_LEN=4096           # 最大序列长度
export VLLM_MAX_NUM_SEQS=128             # 最大并发序列数
export VLLM_DTYPE="half"                 # 数据类型

# 可选：批处理最大 token 数（默认自动计算）
export VLLM_MAX_NUM_BATCHED_TOKENS=8192
```

### 2. 启用 vLLM 推测解码（推荐）

```bash
# 推测解码配置
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5     # 推测 token 数
export VLLM_USE_V2_BLOCK_MANAGER="false" # V2 块管理器
```

### 3. 运行基准测试

```bash
bash run_benchmark.sh
```

## 配置说明

### 必需参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行大小 | GPU 数量 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 显存利用率 | 0.85 |
| `VLLM_MAX_MODEL_LEN` | 最大序列长度 | 4096 |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_MAX_NUM_SEQS` | 最大并发序列数 | 128 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 批处理最大 token 数 | 自动 |
| `VLLM_DTYPE` | 数据类型 | half |

### 推测解码参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `VLLM_ENABLE_SPECULATIVE` | 是否启用 | true |
| `VLLM_NUM_SPECULATIVE_TOKENS` | 推测 token 数 | 5 |
| `VLLM_USE_V2_BLOCK_MANAGER` | 使用 V2 管理器 | false |

## 性能优化

### 1. 显存优化

**OOM 问题**：
```bash
# 降低显存利用率
export VLLM_GPU_MEMORY_UTILIZATION=0.8

# 减少并发序列数
export VLLM_MAX_NUM_SEQS=64

# 减少最大序列长度
export VLLM_MAX_MODEL_LEN=2048
```

### 2. 吞吐量优化

**提高并发**：
```bash
# 增加并发序列数
export VLLM_MAX_NUM_SEQS=256

# 增加批处理 token 数
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
```

### 3. 推测解码优化

```bash
# 调整推测 token 数（通常 4-6 最优）
export VLLM_NUM_SPECULATIVE_TOKENS=5

# 确保启用推测解码
export VLLM_ENABLE_SPECULATIVE="true"
export INFERENCE_METHOD="speculative"
```

## 常见问题

### Q1: CUDA OOM

**问题**: `CUDA error: out of memory`

**解决方案**:
1. 降低 `VLLM_GPU_MEMORY_UTILIZATION` 到 0.7-0.8
2. 减少 `VLLM_MAX_NUM_SEQS`
3. 减少 `VLLM_MAX_MODEL_LEN`
4. 使用更少的 GPU（降低 `VLLM_TENSOR_PARALLEL_SIZE`）

### Q2: 推测解码不工作

**问题**: 推测解码未启用

**检查清单**:
```bash
# 1. 确认环境变量
export VLLM_ENABLE_SPECULATIVE="true"
export INFERENCE_METHOD="speculative"

# 2. 检查日志输出
# 应该看到: "使用 vLLM 原生推测解码"
```

### Q3: transformers 版本冲突

**问题**: `ValueError: 'xxx' is already used by a Transformers config`

**解决方案**:
```bash
pip uninstall -y vllm transformers tokenizers
pip install "transformers>=4.51.1" "tokenizers>=0.21.1"
pip install vllm
```

### Q4: FlashAttention 不可用

**问题**: V100 不支持 FlashAttention-2

**说明**: 这是正常的，vLLM 会自动回退到 XFormers，性能仍然很好。

## vLLM vs Transformers

### 性能对比

| 引擎 | 吞吐量 | 延迟 | 显存 | 并发 |
|------|--------|------|------|------|
| Transformers | 基准 | 基准 | 基准 | 低 |
| vLLM | 2-10x | 0.3-0.5x | 1.2-1.5x | 高 |

### 特性对比

| 特性 | Transformers | vLLM |
|------|--------------|------|
| 推测解码 | ✅ 手动实现 | ✅ 原生支持 |
| 批处理 | ✅ 基础 | ✅ 高级（连续批处理） |
| 异步推理 | ❌ | ✅ |
| PagedAttention | ❌ | ✅ |
| CUDA Graphs | ❌ | ✅ |

### 使用建议

**使用 Transformers 当**:
- 研究和实验
- 需要精细控制
- 单请求推理
- CPU 推理

**使用 vLLM 当**:
- 生产部署
- 高吞吐量需求
- 并发请求
- 性能测试

## 示例配置

### 配置 1: 单卡测试
```bash
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=64
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5
```

### 配置 2: 8 卡高性能
```bash
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_SEQS=128
export VLLM_MAX_NUM_BATCHED_TOKENS=8192
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5
```

### 配置 3: 最大吞吐量
```bash
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_MAX_MODEL_LEN=2048
export VLLM_MAX_NUM_SEQS=256
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENABLE_SPECULATIVE="false"  # 关闭推测解码以提高吞吐
```

## 更多信息

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [推测解码论文](https://arxiv.org/abs/2302.01318)

---

**更新日期**: 2025-11-07

