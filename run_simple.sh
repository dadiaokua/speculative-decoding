#!/bin/bash

# =============================================================================
# Speculative Decoding 简化启动脚本
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 启动 Speculative Decoding...${NC}"

# 基础配置
PROJECT_DIR="/Users/myrick/GithubProjects/Speculative-Decoding"
cd "$PROJECT_DIR"

# =============================================================================
# GPU分配配置
# =============================================================================

# 可用GPU设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 可用的GPU卡

# 8卡并行配置 - 充分利用所有GPU
GPU_STRATEGY="multi_gpu_ratio"  # multi_gpu_ratio, separate, same, auto

# GPU分配比例配置 - 可以自由调整比例
# 推荐配置:
#   8B模型较大 -> 分配更多GPU (如6张)
#   1.7B模型较小 -> 分配较少GPU (如2张)
# 其他比例选项: 5:3, 4:4, 7:1 等

TARGET_GPU_RATIO=6    # Target模型使用6张GPU (0-5) - 可调整
DRAFTER_GPU_RATIO=2   # Drafter模型使用2张GPU (6-7) - 可调整

# 验证比例总和
TOTAL_GPUS=$((TARGET_GPU_RATIO + DRAFTER_GPU_RATIO))
if [ $TOTAL_GPUS -ne 8 ]; then
    echo -e "${RED}❌ 错误: GPU比例总和 ($TOTAL_GPUS) 不等于8张卡${NC}"
    exit 1
fi

case $GPU_STRATEGY in
    "multi_gpu_ratio")
        # 策略1: 按比例分配所有8张GPU
        TARGET_GPUS=""
        DRAFTER_GPUS=""
        
        # 生成Target GPU列表 (0到TARGET_GPU_RATIO-1)
        for ((i=0; i<TARGET_GPU_RATIO; i++)); do
            if [ $i -eq 0 ]; then
                TARGET_GPUS="cuda:$i"
            else
                TARGET_GPUS="$TARGET_GPUS,cuda:$i"
            fi
        done
        
        # 生成Drafter GPU列表 (TARGET_GPU_RATIO到7)
        for ((i=TARGET_GPU_RATIO; i<8; i++)); do
            if [ $i -eq $TARGET_GPU_RATIO ]; then
                DRAFTER_GPUS="cuda:$i"
            else
                DRAFTER_GPUS="$DRAFTER_GPUS,cuda:$i"
            fi
        done
        
        export TARGET_GPU="$TARGET_GPUS"
        export DRAFTER_GPU="$DRAFTER_GPUS"
        
        echo -e "${YELLOW}🎯 GPU策略: 8卡并行 (比例 $TARGET_GPU_RATIO:$DRAFTER_GPU_RATIO)${NC}"
        echo -e "  Target (8B): GPUs $TARGET_GPUS"
        echo -e "  Drafter (1.7B): GPUs $DRAFTER_GPUS"
        ;;
    "separate")
        # 策略2: 简单分离 (兼容旧配置)
        export TARGET_GPU="cuda:0"
        export DRAFTER_GPU="cuda:1"
        echo -e "${YELLOW}🎯 GPU策略: 分离模型${NC}"
        echo -e "  Target (8B): GPU 0"
        echo -e "  Drafter (1.7B): GPU 1"
        ;;
    "same")
        # 策略3: 共享GPU
        export TARGET_GPU="cuda:0"
        export DRAFTER_GPU="cuda:0"
        echo -e "${YELLOW}🎯 GPU策略: 共享GPU${NC}"
        echo -e "  Target (8B): GPU 0"
        echo -e "  Drafter (1.7B): GPU 0"
        ;;
    "auto")
        # 策略4: 自动分配
        export TARGET_GPU="auto"
        export DRAFTER_GPU="auto"
        echo -e "${YELLOW}🎯 GPU策略: 自动分配${NC}"
        echo -e "  让transformers自动决定GPU分配"
        ;;
esac

# 环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

echo -e "${YELLOW}📋 配置信息:${NC}"
echo -e "  项目目录: $PROJECT_DIR"
echo -e "  可用GPU: $CUDA_VISIBLE_DEVICES"
echo -e "  Target GPU: $TARGET_GPU"
echo -e "  Drafter GPU: $DRAFTER_GPU"

# =============================================================================
# 高级GPU配置 (可选)
# =============================================================================

# 如果需要手动指定层分配，取消注释以下配置
# export MANUAL_LAYER_MAPPING="true"
# export TARGET_LAYER_CONFIG="cuda:0,cuda:1"  # 将target模型层分布到GPU 0和1
# export DRAFTER_LAYER_CONFIG="cuda:2"        # drafter模型全部在GPU 2

# =============================================================================
# ShareGPT 自动Prompt配置
# =============================================================================

# ShareGPT数据文件路径
export SHAREGPT_PATH="/Users/myrick/GithubProjects/Speculative-Decoding/sharegpt_gpt4/sharegpt_gpt4.jsonl"

# 自动运行配置
export AUTO_MODE="true"                    # 启用自动模式，不需要手动输入
export NUM_PROMPTS=5                       # 自动运行的prompt数量
export PROMPT_MIN_LENGTH=10                # Prompt最小长度
export PROMPT_MAX_LENGTH=500               # Prompt最大长度
export MAX_LOAD_LINES=10000                # 最大加载数据行数

# Batch处理配置
export ENABLE_BATCH="true"                 # 启用batch处理
export BATCH_SIZE=4                        # 每个batch的大小
export MAX_BATCH_LENGTH=512                # batch中最大序列长度

# 推理参数配置
export GENERATION_LENGTH=100              # 生成长度
export GAMMA_VALUE=4                      # Gamma参数
export ENABLE_SPECULATIVE="true"          # 是否启用speculative decoding
export ENABLE_TARGET="true"               # 是否启用target生成
export ENABLE_NGRAM="true"                # 是否启用ngram辅助生成
export ENABLE_DEBUG="false"               # 是否启用debug模式

echo -e "\n${GREEN}🤖 自动模式配置:${NC}"
echo -e "  自动模式: $AUTO_MODE"
echo -e "  处理数量: $NUM_PROMPTS 个prompts"
echo -e "  生成长度: $GENERATION_LENGTH tokens"
echo -e "  Gamma值: $GAMMA_VALUE"
echo -e "  Speculative: $ENABLE_SPECULATIVE"
echo -e "  Target生成: $ENABLE_TARGET" 
echo -e "  Ngram辅助: $ENABLE_NGRAM"

echo -e "\n${YELLOW}📦 Batch处理配置:${NC}"
echo -e "  Batch模式: $ENABLE_BATCH"
echo -e "  Batch大小: $BATCH_SIZE"
echo -e "  最大序列长度: $MAX_BATCH_LENGTH"

if [ "$AUTO_MODE" = "true" ]; then
    echo -e "\n${YELLOW}📝 程序将自动从ShareGPT选择prompts运行，无需手动输入${NC}"
else
    echo -e "\n${GREEN}💡 手动模式 - 启动后可用命令:${NC}"
    echo -e "  /gamma 4          # 设置gamma值"
    echo -e "  /length 100       # 设置生成长度" 
    echo -e "  /random           # 随机prompt"
    echo -e "  /help            # 查看所有命令"
fi

echo -e "\n${BLUE}正在启动程序...${NC}"
python infer.py --device cuda:0
