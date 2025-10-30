#!/bin/bash

# =============================================================================
# Speculative Decoding Performance Benchmark
# =============================================================================

set -e  # Exit on error

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
# GPU Configuration
# =============================================================================

# Available GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU allocation strategy: multi_gpu_ratio, separate, same, auto
GPU_STRATEGY="multi_gpu_ratio"

# GPU ratio configuration
TARGET_GPU_RATIO=6    # Target model uses this many GPUs (0-5)
DRAFTER_GPU_RATIO=2   # Drafter model uses this many GPUs (6-7)

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
        print_info "GPU Strategy: Shared GPU"
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
# Dataset Configuration
# =============================================================================

export SHAREGPT_DIR="$PROJECT_DIR/sharegpt_gpt4"
export PROMPT_MIN_LENGTH=10
export PROMPT_MAX_LENGTH=500
export MAX_LOAD_LINES=10000

# =============================================================================
# Benchmark Parameters
# =============================================================================

# Benchmark mode: time-based or count-based
# If NUM_PROMPTS > 0, run exactly that many prompts
# Otherwise, use AUTO_DURATION and AUTO_RATE
export NUM_PROMPTS=0                    # 0 = use duration, >0 = exact count
export AUTO_RATE=1.0                     # Requests per second (when using duration)
export AUTO_DURATION=300                 # Duration in seconds (when NUM_PROMPTS=0)

# Batch processing
export ENABLE_BATCH="true"               # Enable batch processing
export BATCH_SIZE=4                      # Batch size
export MAX_BATCH_LENGTH=512               # Max sequence length in batch

# Generation parameters
export GENERATION_LENGTH=100             # Generation length in tokens
export GAMMA_VALUE=4                     # Gamma parameter for speculative decoding

# Feature flags
export ENABLE_SPECULATIVE="true"         # Enable speculative decoding
export ENABLE_TARGET="true"              # Enable target AR generation (for comparison)
export ENABLE_DEBUG="false"              # Enable debug output

# GPU monitoring
export ENABLE_GPU_MONITOR="true"         # Enable GPU power/performance monitoring
export GPU_MONITOR_INTERVAL=1.0          # GPU monitoring sampling interval (seconds)

# Output configuration
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
echo "  Speculative Decoding: $ENABLE_SPECULATIVE"
echo "  Target AR: $ENABLE_TARGET"
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
python benchmark.py --device cuda:0

print_success "Benchmark completed! Results saved to $OUTPUT_FILE"

