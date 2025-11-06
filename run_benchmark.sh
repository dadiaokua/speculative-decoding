#!/bin/bash

# =============================================================================
# 推测解码性能基准测试脚本 (Speculative Decoding Performance Benchmark)
# =============================================================================
#
# 功能说明：
# 本脚本用于运行推测解码（Speculative Decoding）的性能基准测试，
# 对比推测解码与标准自回归生成的性能差异。
#
# 主要特性：
# 1. GPU灵活分配：支持单GPU、多GPU分离、多GPU共享等多种策略
# 2. 实时监控：GPU功率、温度、利用率、能耗等硬件指标
# 3. 性能指标：TTFT、延迟、吞吐量、token生成数、接受率等
# 4. 能效分析：每焦耳/千瓦时生成的token数
# 5. 多种测试模式：固定请求数或基于时间+速率的持续测试
#
# 使用方法：
#   bash run_benchmark.sh                    # 使用默认配置
#   bash run_benchmark.sh --target-model ... # 指定模型路径
#
# =============================================================================

set -e  # 遇到错误立即退出

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored messages
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Project Configuration
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_info "🚀 Starting Speculative Decoding Benchmark"
print_info "Project directory: $PROJECT_DIR"

# =============================================================================
# GPU配置 (GPU Configuration)
# =============================================================================

# 可用的GPU设备列表（逗号分隔，从0开始编号）
# 例如：0,1,2,3,4,5,6,7 表示使用全部8张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU分配策略：multi_gpu_ratio, separate, same, shared_all, auto
#
# 策略说明：
# - multi_gpu_ratio: 按比例分配GPU（如7:1，Target用7张，Drafter用1张）
# - shared_all: 两个模型共享所有GPU（推荐用于32GB+显存的GPU，如V100 32GB）
# - separate: Target用GPU 0，Drafter用GPU 1（双GPU场景）
# - same: 两个模型都用GPU 0（单GPU场景，显存需足够）
# - auto: 自动分配（让transformers库决定）
#
# 性能建议：
# - V100 32GB × 8: 推荐 shared_all（最佳性能）
# - V100 16GB × 8: 推荐 multi_gpu_ratio（避免OOM）
# - 单卡测试: same
GPU_STRATEGY="shared_all"

# GPU比例配置（仅在GPU_STRATEGY="multi_gpu_ratio"时生效）
TARGET_GPU_RATIO=7    # Target模型使用的GPU数量（GPU 0-6）
DRAFTER_GPU_RATIO=1   # Drafter模型使用的GPU数量（GPU 7）

# Validate GPU ratio
TOTAL_GPUS=$((TARGET_GPU_RATIO + DRAFTER_GPU_RATIO))
if [ $TOTAL_GPUS -ne 8 ]; then
    print_error "GPU ratio sum ($TOTAL_GPUS) does not equal 8 GPUs"
    exit 1
fi

# Set GPU allocation based on strategy
case $GPU_STRATEGY in
    "multi_gpu_ratio")
        TARGET_GPUS=""
        DRAFTER_GPUS=""
        
        for ((i=0; i<TARGET_GPU_RATIO; i++)); do
            if [ $i -eq 0 ]; then
                TARGET_GPUS="cuda:$i"
            else
                TARGET_GPUS="$TARGET_GPUS,cuda:$i"
            fi
        done
        
        for ((i=TARGET_GPU_RATIO; i<8; i++)); do
            if [ $i -eq $TARGET_GPU_RATIO ]; then
                DRAFTER_GPUS="cuda:$i"
            else
                DRAFTER_GPUS="$DRAFTER_GPUS,cuda:$i"
            fi
        done
        
        export TARGET_GPU="$TARGET_GPUS"
        export DRAFTER_GPU="$DRAFTER_GPUS"
        
        print_info "GPU Strategy: Multi-GPU Ratio ($TARGET_GPU_RATIO:$DRAFTER_GPU_RATIO)"
        print_info "  Target (8B): GPUs $TARGET_GPUS"
        print_info "  Drafter (1.7B): GPUs $DRAFTER_GPUS"
        ;;
    "separate")
        export TARGET_GPU="cuda:0"
        export DRAFTER_GPU="cuda:1"
        print_info "GPU Strategy: Separate GPUs"
        ;;
    "same")
        export TARGET_GPU="cuda:0"
        export DRAFTER_GPU="cuda:0"
        print_info "GPU Strategy: Shared GPU (single GPU)"
        ;;
    "shared_all")
        # Both models use all 8 GPUs - best for high-memory GPUs (32GB+)
        export TARGET_GPU="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        export DRAFTER_GPU="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        print_info "GPU Strategy: Shared All GPUs (8:8)"
        print_info "  Both Target and Drafter use all 8 GPUs"
        print_info "  Optimal for V100 32GB or A100"
        ;;
    "auto")
        export TARGET_GPU="auto"
        export DRAFTER_GPU="auto"
        print_info "GPU Strategy: Auto allocation"
        ;;
    *)
        print_error "Unknown GPU strategy: $GPU_STRATEGY"
        exit 1
        ;;
esac

# =============================================================================
# Model Configuration
# =============================================================================

# Model paths (local paths or Hugging Face model IDs)
export TARGET_MODEL="/home/llm/model_hub/Qwen3-8B"      # Target model path
export DRAFTER_MODEL="/home/llm/model_hub/Qwen3-0.6B"  # Drafter model path

# =============================================================================
# Dataset Configuration
# =============================================================================

export SHAREGPT_DIR="$PROJECT_DIR/sharegpt_gpt4"
export PROMPT_MIN_LENGTH=10
export PROMPT_MAX_LENGTH=500
export MAX_LOAD_LINES=10000

# =============================================================================
# 基准测试参数 (Benchmark Parameters)
# =============================================================================

# 测试模式：基于时间 或 基于数量
#
# 模式1：基于数量（NUM_PROMPTS > 0）
#   - 运行固定数量的请求后停止
#   - 适合快速测试和对比
#
# 模式2：基于时间（NUM_PROMPTS = 0）
#   - 按指定速率运行指定时长
#   - 更接近生产环境的持续负载测试
#
export NUM_PROMPTS=0                    # 0 = 使用时间模式, >0 = 运行指定数量的请求
export AUTO_RATE=1.0                     # 请求速率（prompts/秒，仅时间模式）
export AUTO_DURATION=300                 # 测试时长（秒，仅时间模式）

# 批处理配置（当前实现为单请求模式，批处理功能待启用）
export ENABLE_BATCH="false"               # 是否启用批处理
export BATCH_SIZE=4                      # 批大小
export MAX_BATCH_LENGTH=512               # 批内最大序列长度

# 生成参数
export GENERATION_LENGTH=100             # 每个请求生成的token数量
export GAMMA_VALUE=4                     # Gamma参数（推测解码的草稿token数）

# 推理引擎选择
# - "transformers": 使用Hugging Face Transformers（默认）
# - "vllm": 使用vLLM高性能推理引擎
export INFERENCE_ENGINE="vllm"   # 选项: "transformers", "vllm"

# 推理方法选择
# - "speculative": 推测解码（Drafter生成草稿 + Target验证）
# - "target_ar": 标准自回归生成（仅使用Target模型）
export INFERENCE_METHOD="speculative"    # 选项: "speculative", "target_ar"
export ENABLE_DEBUG="false"              # 是否启用调试输出

# vLLM引擎参数（仅在INFERENCE_ENGINE="vllm"时生效）
export VLLM_TENSOR_PARALLEL_SIZE=8       # 张量并行大小（通常等于GPU数量）
export VLLM_GPU_MEMORY_UTILIZATION=0.9   # GPU显存利用率（0-1之间）
export VLLM_MAX_MODEL_LEN=4096           # 最大模型长度
export VLLM_MAX_NUM_SEQS=128             # 最大并发序列数
export VLLM_MAX_NUM_BATCHED_TOKENS=8192  # 批处理最大token数（可选，默认自动计算）
export VLLM_DISABLE_LOG_STATS=true       # 是否禁用日志统计
export VLLM_DTYPE="half"                 # 数据类型: "half", "float16", "bfloat16"

# vLLM推测解码参数（可选，启用后使用vLLM原生推测解码）
export VLLM_ENABLE_SPECULATIVE="false"   # 是否启用vLLM推测解码
export VLLM_NUM_SPECULATIVE_TOKENS=5     # 推测token数量（对应GAMMA_VALUE）
export VLLM_USE_V2_BLOCK_MANAGER="true"  # 是否使用v2块管理器（推荐）

# GPU监控配置
#
# 监控内容：
# - 功率消耗（瓦特）
# - GPU利用率（%）
# - 显存使用（MB）
# - 温度（℃）
# - 能耗（焦耳/千瓦时）
#
# 采样间隔建议：
# - 0.1s: 极高精度，可能跟不上（不推荐）
# - 0.5s: 高精度，稳定可靠（推荐）✅
# - 1.0s: 良好精度，低开销
# - 10.0s: 中等精度，极低开销
#
export ENABLE_GPU_MONITOR="true"         # 是否启用GPU监控
export GPU_MONITOR_INTERVAL=0.5          # 采样间隔（秒，0.5s推荐，平衡精度与稳定性）

# Output configuration
# Output filename will automatically include inference method suffix
# e.g., "benchmark_results.json" -> "benchmark_results_speculative.json" or "benchmark_results_target_ar.json"
export OUTPUT_FILE="benchmark_results.json"

# =============================================================================
# Environment Setup
# =============================================================================

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# Display Configuration
# =============================================================================

echo ""
print_info "📋 Configuration Summary:"
echo "  Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Target GPU: $TARGET_GPU"
echo "  Drafter GPU: $DRAFTER_GPU"
echo ""
echo "  Target Model: $TARGET_MODEL"
echo "  Drafter Model: $DRAFTER_MODEL"
echo ""
echo "  Dataset: $SHAREGPT_DIR"
if [ "$NUM_PROMPTS" -gt 0 ]; then
    echo "  Benchmark Mode: Fixed count"
    echo "  Total Prompts: $NUM_PROMPTS"
else
    echo "  Benchmark Mode: Time-based"
    echo "  Rate: $AUTO_RATE prompts/s"
    echo "  Duration: $AUTO_DURATION s"
fi
echo ""
echo "  Batch Processing: $ENABLE_BATCH"
if [ "$ENABLE_BATCH" = "true" ]; then
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Max Batch Length: $MAX_BATCH_LENGTH"
fi
echo ""
echo "  Generation Length: $GENERATION_LENGTH tokens"
echo "  Gamma: $GAMMA_VALUE"
echo "  Inference Engine: $INFERENCE_ENGINE"
echo "  Inference Method: $INFERENCE_METHOD"
if [ "$INFERENCE_ENGINE" = "vllm" ]; then
    echo ""
    echo "  vLLM Configuration:"
    echo "    Tensor Parallel: $VLLM_TENSOR_PARALLEL_SIZE"
    echo "    GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
    echo "    Max Model Length: $VLLM_MAX_MODEL_LEN"
    echo "    Max Num Seqs: $VLLM_MAX_NUM_SEQS"
    if [ ! -z "$VLLM_MAX_NUM_BATCHED_TOKENS" ]; then
        echo "    Max Num Batched Tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
    fi
    echo "    Data Type: $VLLM_DTYPE"
    if [ "$VLLM_ENABLE_SPECULATIVE" = "true" ]; then
        echo ""
        echo "  vLLM Speculative Decoding:"
        echo "    Enabled: Yes"
        echo "    Num Speculative Tokens: $VLLM_NUM_SPECULATIVE_TOKENS"
        echo "    Use V2 Block Manager: $VLLM_USE_V2_BLOCK_MANAGER"
    else
        echo ""
        echo "  vLLM Speculative Decoding: Disabled"
    fi
fi
echo ""
echo "  GPU Monitoring: $ENABLE_GPU_MONITOR"
if [ "$ENABLE_GPU_MONITOR" = "true" ]; then
    echo "  Monitor Interval: $GPU_MONITOR_INTERVAL s"
fi
echo ""
echo "  Output File: $OUTPUT_FILE"
echo ""

# =============================================================================
# Run Benchmark
# =============================================================================

print_success "Starting benchmark..."
# Note: GPU allocation is controlled by TARGET_GPU and DRAFTER_GPU environment variables above
# Model paths can be passed as command line arguments (--target-model, --drafter-model)
# If not provided, will use TARGET_MODEL and DRAFTER_MODEL environment variables
python benchmark.py "$@"

print_success "Benchmark completed! Results saved to $OUTPUT_FILE"

