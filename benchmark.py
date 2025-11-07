"""Speculative Decoding Performance Benchmark - 主入口

这个脚本是性能基准测试的主入口，支持：
- Transformers 引擎（CPU/GPU推理）
- vLLM 引擎（高性能 GPU 推理）
- 推测解码 vs 标准自回归生成
- 详细的性能指标收集（TTFT、延迟、吞吐量、接受率、能耗等）

使用方法:
    # 使用环境变量配置（推荐）
    bash run_benchmark.sh
    
    # 或直接运行，指定模型路径
    python benchmark.py --target-model /path/to/target --drafter-model /path/to/drafter
"""

import argparse
import os
from termcolor import colored

from engine.vllm_engine import is_vllm_available


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用环境变量配置（推荐）
  bash run_benchmark.sh
  
  # 指定模型路径
  python benchmark.py --target-model /path/to/target --drafter-model /path/to/drafter
  
  # GPU 分配由环境变量 TARGET_GPU 和 DRAFTER_GPU 控制（在 run_benchmark.sh 中设置）
        """
    )
    
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="目标模型路径（覆盖 TARGET_MODEL 环境变量）"
    )
    parser.add_argument(
        "--drafter-model",
        type=str,
        default=None,
        help="草稿模型路径（覆盖 DRAFTER_MODEL 环境变量）"
    )
    
    args = parser.parse_args()
    
    # 转换空字符串为 None
    target_model = args.target_model if args.target_model and args.target_model.strip() else None
    drafter_model = args.drafter_model if args.drafter_model and args.drafter_model.strip() else None
    
    # 根据推理引擎选择运行器
    inference_engine = os.getenv("INFERENCE_ENGINE", "transformers").lower()
    
    if inference_engine == "vllm":
        # 使用 vLLM 引擎
        if not is_vllm_available():
            print(colored("❌ Error: vLLM engine selected but vLLM is not installed!", "red"))
            print(colored("   Please install vLLM: pip install vllm", "yellow"))
            return
        
        from engine.vllm_benchmark import VLLMBenchmarkRunner
        VLLMBenchmarkRunner(target_model=target_model, drafter_model=drafter_model)
    else:
        # 使用 Transformers 引擎
        from engine.benchmark_runner import BenchmarkRunner
        BenchmarkRunner(target_model=target_model, drafter_model=drafter_model)


if __name__ == "__main__":
    main()

