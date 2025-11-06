# vLLM 快速开始指南

## 🚀 一分钟快速上手

### 方案 A: vLLM 标准推理（最简单）

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 编辑 run_benchmark.sh，修改一行：
export INFERENCE_ENGINE="vllm"

# 3. 运行
bash run_benchmark.sh
```

### 方案 B: vLLM 推测解码（推荐，性能最优）

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 编辑 run_benchmark.sh，修改三行：
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"

# 3. 运行
bash run_benchmark.sh
```

就这么简单！🎉

## ⚙️ 默认配置

vLLM 的默认配置已经针对 8x V100 32GB 优化：

```bash
# 基础配置
export VLLM_TENSOR_PARALLEL_SIZE=8       # 8卡张量并行
export VLLM_GPU_MEMORY_UTILIZATION=0.9   # 90% 显存利用率
export VLLM_MAX_MODEL_LEN=4096           # 4K 上下文长度
export VLLM_MAX_NUM_SEQS=128             # 128 并发序列
export VLLM_DTYPE="half"                 # FP16 精度

# 推测解码配置（可选）
export VLLM_ENABLE_SPECULATIVE="false"   # 默认关闭，设为true启用
export VLLM_NUM_SPECULATIVE_TOKENS=5     # 推测token数（对应GAMMA_VALUE）
export VLLM_USE_V2_BLOCK_MANAGER="true"  # V2块管理器（推荐）
```

这些参数通常不需要修改，除非遇到 OOM 或其他问题。

## 📊 性能对比

| 指标 | Transformers | vLLM | 提升 |
|------|-------------|------|------|
| 吞吐量 | 基准 | 2-10x | 🔥 |
| 延迟 | 基准 | -30-50% | ✅ |
| 显存占用 | 基准 | -10-20% | 💚 |

## ✨ vLLM 推测解码优势

vLLM 原生推测解码相比 Transformers 实现有以下优势：

1. **单引擎模式**: Target 和 Drafter 在同一引擎中加载，共享内存管理
2. **自动优化**: vLLM 内部优化的推测解码实现
3. **更高效率**: PagedAttention + 连续批处理 + 推测解码的完美结合
4. **更少显存**: 统一的内存管理，避免重复分配

**推荐使用场景**：
- ✅ 生产环境部署（最佳性能）
- ✅ 高吞吐量场景
- ✅ 多用户并发请求

## 🔧 故障排查

### 问题：OOM (Out of Memory)

**解决方案**：降低显存利用率

```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.8   # 从 0.9 降到 0.8
export VLLM_MAX_MODEL_LEN=2048           # 从 4096 降到 2048
```

### 问题：Tensor parallel size 错误

**解决方案**：确保张量并行大小与 GPU 数量匹配

```bash
# 8卡
export VLLM_TENSOR_PARALLEL_SIZE=8

# 4卡
export VLLM_TENSOR_PARALLEL_SIZE=4

# 单卡
export VLLM_TENSOR_PARALLEL_SIZE=1
```

### 问题：vLLM 未安装

**解决方案**：

```bash
pip install vllm
```

## 📖 更多信息

- 详细配置说明：[vLLM Integration Guide](VLLM_INTEGRATION.md)
- GPU 部署策略：[GPU Deployment Guide](GPU_DEPLOYMENT.md)
- 主要文档：[README.md](../README.md)

## 💡 提示

1. **首次使用**: 建议先用 Transformers 引擎跑一次，确保环境正常，然后再切换到 vLLM

2. **性能优化**: 如果追求极致吞吐量，可以提高 `VLLM_MAX_NUM_SEQS` 到 256 或更高

3. **延迟优化**: 如果追求低延迟，可以降低 `VLLM_MAX_NUM_SEQS` 到 32 或更低

4. **显存不足**: 优先降低 `VLLM_MAX_MODEL_LEN`，其次降低 `VLLM_GPU_MEMORY_UTILIZATION`

5. **监控**: GPU 监控功能在 vLLM 模式下同样有效，可以实时查看功耗和性能

## 🎯 推荐配置

### 推测解码 + 高吞吐量（最推荐）
```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=5
export VLLM_MAX_NUM_SEQS=256
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export AUTO_RATE=10.0
```

### 推测解码 + 低延迟
```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="speculative"
export VLLM_ENABLE_SPECULATIVE="true"
export VLLM_NUM_SPECULATIVE_TOKENS=4
export VLLM_MAX_NUM_SEQS=32
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export AUTO_RATE=1.0
```

### 标准推理 + 长文本
```bash
export INFERENCE_ENGINE="vllm"
export INFERENCE_METHOD="target_ar"
export VLLM_ENABLE_SPECULATIVE="false"
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_NUM_SEQS=64
```

