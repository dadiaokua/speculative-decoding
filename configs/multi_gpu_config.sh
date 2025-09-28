#!/bin/bash

# =============================================================================
# 多GPU配置启动脚本 - 大模型4卡 + 小模型1卡
# =============================================================================

echo "🔧 多GPU配置启动..."

# 项目路径
PROJECT_DIR="/Users/myrick/GithubProjects/Speculative-Decoding"
cd "$PROJECT_DIR"

# 5卡配置：前4张给大模型，第5张给小模型
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 启动前检查GPU状态
echo "📊 GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits

echo ""
echo "🎯 配置说明:"
echo "  - Target Model (大模型): GPU 0-3 (流水线并行)"
echo "  - Drafter Model (小模型): GPU 4 (单卡)"
echo "  - 主控设备: cuda:0"

echo ""
echo "⚙️ 推荐运行时配置:"
echo "  /gamma 4"
echo "  /length 50"
echo "  /processor nucleus 0.8 0.9"
echo "  /speculative  # 确保启用speculative decoding"

echo ""
echo "🚀 启动程序..."
python infer.py --device cuda:0
