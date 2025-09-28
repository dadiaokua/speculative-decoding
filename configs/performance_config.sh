#!/bin/bash

# =============================================================================
# 性能优化配置启动脚本
# =============================================================================

echo "⚡ 性能优化配置启动..."

PROJECT_DIR="/Users/myrick/GithubProjects/Speculative-Decoding"
cd "$PROJECT_DIR"

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 性能优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "🎯 性能优化设置:"
echo "  - CUDA内存分配优化"
echo "  - OpenMP线程数: 8"
echo "  - 禁用Tokenizer并行警告"

echo ""
echo "📈 推荐性能参数:"
echo "  Gamma: 4-6 (根据acceptance rate调整)"
echo "  Temperature: 0.7-0.9"
echo "  Top-p: 0.8-0.95"
echo "  缓存: 启用 (如果模型支持)"

echo ""
echo "💡 性能监控命令:"
echo "  nvidia-smi -l 1  # 实时GPU监控"
echo "  htop             # CPU和内存监控"

echo ""
echo "🚀 启动高性能模式..."

# 创建自动配置命令
cat > /tmp/auto_perf_config.txt << EOF
/gamma 5
/length 80
/processor nucleus 0.8 0.9
/cache
/speculative
/target
EOF

echo "📋 将自动应用性能配置，启动后手动输入以下命令:"
cat /tmp/auto_perf_config.txt

python infer.py --device cuda:0
