#!/bin/bash

# 配置检查脚本

echo "========================================="
echo "环境变量检查"
echo "========================================="

echo "INFERENCE_ENGINE = '$INFERENCE_ENGINE'"
echo "INFERENCE_METHOD = '$INFERENCE_METHOD'"
echo "VLLM_ENABLE_SPECULATIVE = '$VLLM_ENABLE_SPECULATIVE'"
echo ""

echo "如果上面的值为空，说明环境变量没有设置"
echo ""

echo "========================================="
echo "检查 run_benchmark.sh 文件"
echo "========================================="

if [ -f "run_benchmark.sh" ]; then
    echo "✅ run_benchmark.sh 存在"
    echo ""
    echo "INFERENCE_ENGINE 设置："
    grep "^export INFERENCE_ENGINE=" run_benchmark.sh
    echo ""
    echo "VLLM_ENABLE_SPECULATIVE 设置："
    grep "^export VLLM_ENABLE_SPECULATIVE=" run_benchmark.sh
else
    echo "❌ run_benchmark.sh 不存在"
fi

