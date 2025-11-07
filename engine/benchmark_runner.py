"""Benchmark Runner - 性能测试执行器

负责协调整个性能测试流程，包括配置加载、模型加载、测试执行等。
"""

import os
import time
import random
import numpy as np
import torch
from termcolor import colored
from typing import Optional

from engine.dataset import load_sharegpt_multi
from engine.infer_engine import infer_batch
from engine.metrics import BenchmarkResults, print_benchmark_summary
from engine.gpu_monitor import GPUMonitor, print_gpu_summary
from utils.logits_processor import GreedyProcessor


class BenchmarkRunner:
    """性能基准测试运行器（Transformers引擎）"""
    
    def __init__(self, target_model: Optional[str] = None, drafter_model: Optional[str] = None):
        """初始化基准测试运行器
        
        Args:
            target_model: 目标模型路径（覆盖环境变量 TARGET_MODEL）
            drafter_model: 草稿模型路径（覆盖环境变量 DRAFTER_MODEL）
        """
        print(colored("Speculative Decoding Performance Benchmark", "red", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        # 存储命令行参数
        self.target_model_arg = target_model
        self.drafter_model_arg = drafter_model
        
        # 加载配置
        self._load_config()
        
        # 加载模型
        from engine.model_loader import load_transformers_models
        models_data = load_transformers_models(
            target_model_path=self.target_model_arg or os.getenv("TARGET_MODEL", "/home/llm/model_hub/Qwen3-8B"),
            drafter_model_path=self.drafter_model_arg or os.getenv("DRAFTER_MODEL", "/home/llm/model_hub/Qwen3-1.7B"),
            target_gpu_env=os.getenv("TARGET_GPU", "cuda:0"),
            drafter_gpu_env=os.getenv("DRAFTER_GPU", "cuda:0")
        )
        
        # 解包模型数据
        self.target = models_data['target']
        self.drafter = models_data['drafter']
        self.tokenizer = models_data['tokenizer']
        self.end_tokens = models_data['end_tokens']
        self.target_device = models_data['target_device']
        self.drafter_device = models_data['drafter_device']
        self.target_gpu_ids = models_data['target_gpu_ids']
        self.drafter_gpu_ids = models_data['drafter_gpu_ids']
        
        # 加载数据集
        self._load_sharegpt_data()
        
        # 初始化处理器
        self.processor = GreedyProcessor()
        
        # 运行基准测试
        self._run_benchmark()
    
    def _load_config(self):
        """从环境变量加载配置"""
        self.gamma = int(os.getenv("GAMMA_VALUE", "4"))
        self.gen_len = int(os.getenv("GENERATION_LENGTH", "100"))
        
        # 推理方法: "speculative" 或 "target_ar"
        inference_method = os.getenv("INFERENCE_METHOD", "speculative").lower()
        if inference_method == "speculative":
            self.spec = True
            self.target_gen = False
            self.inference_method_name = "speculative"
        elif inference_method == "target_ar":
            self.spec = False
            self.target_gen = True
            self.inference_method_name = "target_ar"
        else:
            print(colored(f"⚠️  Warning: Unknown INFERENCE_METHOD '{inference_method}', defaulting to 'speculative'", "yellow"))
            self.spec = True
            self.target_gen = False
            self.inference_method_name = "speculative"
        
        self.debug = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
        
        # 批处理配置
        self.enable_batch = os.getenv("ENABLE_BATCH", "false").lower() == "true"
        self.batch_size = int(os.getenv("BATCH_SIZE", "4"))
        self.max_batch_length = int(os.getenv("MAX_BATCH_LENGTH", "512"))
        
        # 基准测试参数
        self.num_prompts = int(os.getenv("NUM_PROMPTS", "0"))
        self.auto_rate = float(os.getenv("AUTO_RATE", "1.0"))
        self.auto_duration = float(os.getenv("AUTO_DURATION", "300"))
        self.prompt_min_length = int(os.getenv("PROMPT_MIN_LENGTH", "10"))
        self.prompt_max_length = int(os.getenv("PROMPT_MAX_LENGTH", "500"))
        self.max_load_lines = int(os.getenv("MAX_LOAD_LINES", "10000"))
        
        # 数据集路径
        self.sharegpt_dir = os.getenv(
            "SHAREGPT_DIR",
            "/Users/myrick/GithubProjects/Speculative-Decoding/sharegpt_gpt4",
        )
        self.sharegpt_paths = [
            os.path.join(self.sharegpt_dir, "sharegpt_gpt4.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_V3_format.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_zh_38K_format.jsonl"),
        ]
        
        # 输出文件
        base_output_file = os.getenv("OUTPUT_FILE", "benchmark_results.json")
        if base_output_file.endswith(".json"):
            self.output_file = base_output_file.replace(".json", f"_{self.inference_method_name}.json")
        else:
            self.output_file = f"{base_output_file}_{self.inference_method_name}.json"
        
        # GPU 监控配置
        self.enable_gpu_monitor = os.getenv("ENABLE_GPU_MONITOR", "true").lower() == "true"
        self.gpu_monitor_interval = float(os.getenv("GPU_MONITOR_INTERVAL", "1.0"))
        
        # 推理上下文属性（infer_batch 需要）
        self.chat = True
        self.cache = False
        self.reset_in_between = False
        self.ngram = None
    
    def _load_sharegpt_data(self):
        """加载 ShareGPT 提示数据"""
        try:
            parts = load_sharegpt_multi(
                self.sharegpt_paths,
                max_lines=self.max_load_lines,
                min_len=self.prompt_min_length,
                max_len=self.prompt_max_length,
            )
            self.sharegpt_parts = parts
            flat = []
            for p in parts:
                flat.extend(p)
            self.sharegpt_data = flat if flat else None
            print(colored(f"✅ Loaded {len(flat)} prompts from ShareGPT", "green"))
        except Exception as e:
            print(colored(f"❌ Error loading ShareGPT data: {e}", "red"))
            self.sharegpt_data = None
    
    def _get_random_prompt(self):
        """从 ShareGPT 数据中获取随机提示"""
        if not self.sharegpt_data:
            return "Tell me a story about artificial intelligence."
        
        if hasattr(self, "sharegpt_parts") and self.sharegpt_parts:
            non_empty = [p for p in self.sharegpt_parts if p]
            if non_empty:
                chosen_part = random.choice(non_empty)
                return random.choice(chosen_part)
        return random.choice(self.sharegpt_data)
    
    def _set_seed(self, seed: int):
        """设置随机种子以保证可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _run_benchmark(self):
        """运行性能基准测试"""
        if not self.sharegpt_data:
            print(colored("❌ No ShareGPT data available, cannot run benchmark", "red"))
            return
        
        # 验证基准测试参数
        if self.num_prompts <= 0:
            if self.auto_duration <= 0 or self.auto_rate <= 0:
                print(colored("❌ Invalid benchmark parameters: AUTO_DURATION and AUTO_RATE must be > 0 when NUM_PROMPTS=0", "red"))
                return
        
        print(colored("\n🚀 Starting Benchmark", "magenta", attrs=["bold"]))
        if self.num_prompts > 0:
            print(colored(f"  Total Prompts: {self.num_prompts}", "yellow"))
        else:
            print(colored(f"  Rate: {self.auto_rate:.2f} prompts/s", "yellow"))
            print(colored(f"  Duration: {self.auto_duration:.1f} s", "yellow"))
        print(colored(f"  Batch mode: {self.enable_batch}", "yellow"))
        if self.enable_batch:
            print(colored(f"  Batch size: {self.batch_size}", "yellow"))
        print(colored(f"  Inference Method: {'Speculative Decoding' if self.spec else 'Target AR'}", "yellow"))
        print(colored("=" * 70, "cyan"))
        
        # 初始化结果收集器
        if self.spec:
            spec_results = BenchmarkResults(method="speculative")
            target_results = None
        else:
            spec_results = None
            target_results = BenchmarkResults(method="target_ar")
        
        # 启动 GPU 监控
        gpu_monitor = None
        if self.enable_gpu_monitor:
            try:
                cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
                else:
                    gpu_ids = None
                
                def get_performance_metrics():
                    """性能指标回调函数"""
                    tokens_gen = 0
                    tokens_acc = 0
                    requests_done = 0
                    throughput_val = 0.0
                    avg_ttft_val = 0.0
                    avg_latency_val = 0.0
                    
                    if spec_results:
                        tokens_gen += spec_results.total_tokens
                        requests_done += spec_results.total_requests
                        if spec_results.avg_ttft > 0:
                            avg_ttft_val = spec_results.avg_ttft
                        if spec_results.avg_latency > 0:
                            avg_latency_val = spec_results.avg_latency
                        if spec_results.overall_throughput > 0:
                            throughput_val = spec_results.overall_throughput
                        if spec_results.method == "speculative":
                            if spec_results.avg_acceptance_rate > 0:
                                tokens_acc = int(tokens_gen * spec_results.avg_acceptance_rate)
                    
                    if target_results:
                        tokens_gen += target_results.total_tokens
                        requests_done += target_results.total_requests
                        if throughput_val == 0 and target_results.overall_throughput > 0:
                            throughput_val = target_results.overall_throughput
                    
                    return {
                        'total_tokens_generated': tokens_gen,
                        'total_tokens_accepted': tokens_acc,
                        'requests_completed': requests_done,
                        'throughput': throughput_val,
                        'avg_ttft': avg_ttft_val,
                        'avg_latency': avg_latency_val
                    }
                
                gpu_monitor = GPUMonitor(
                    gpu_ids=gpu_ids,
                    sampling_interval=self.gpu_monitor_interval,
                    performance_callback=get_performance_metrics
                )
                gpu_monitor.start()
            except Exception as e:
                print(colored(f"⚠️  Warning: Could not start GPU monitor: {e}", "yellow"))
        
        # 设置基准测试开始时间
        benchmark_start_time = time.time()
        if spec_results:
            spec_results.start_time = benchmark_start_time
        if target_results:
            target_results.start_time = benchmark_start_time
        
        # 运行基准测试主循环
        from engine.benchmark_executor import execute_benchmark_loop
        total_requests = execute_benchmark_loop(
            runner=self,
            spec_results=spec_results,
            target_results=target_results
        )
        
        # 停止 GPU 监控
        gpu_monitor_results = None
        if gpu_monitor:
            try:
                gpu_monitor.stop()
                gpu_monitor_results = gpu_monitor.get_results()
                
                total_tokens_gen = 0
                total_tokens_acc = 0
                total_reqs = 0
                
                if spec_results:
                    total_tokens_gen += spec_results.total_tokens
                    total_reqs += spec_results.total_requests
                    if spec_results.method == "speculative" and spec_results.avg_acceptance_rate > 0:
                        total_tokens_acc = int(spec_results.total_tokens * spec_results.avg_acceptance_rate)
                
                if target_results:
                    total_tokens_gen += target_results.total_tokens
                    total_reqs += target_results.total_requests
                
                gpu_monitor_results.total_tokens_generated = total_tokens_gen
                gpu_monitor_results.total_tokens_accepted = total_tokens_acc
                gpu_monitor_results.total_requests = total_reqs
                
            except Exception as e:
                print(colored(f"⚠️  Warning: Error stopping GPU monitor: {e}", "yellow"))
        
        # 完成结果统计
        if spec_results:
            spec_results.end_time = time.time()
            spec_results.total_batches = len(spec_results.batches)
        
        if target_results:
            target_results.end_time = time.time()
            target_results.total_batches = len(target_results.batches)
        
        # 打印结果
        print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))
        print(colored("📊 Benchmark Complete", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan", attrs=["bold"]))
        
        if spec_results:
            print_benchmark_summary(spec_results)
        if target_results:
            print_benchmark_summary(target_results)
        
        if gpu_monitor_results:
            print_gpu_summary(gpu_monitor_results)
            gpu_output_file = self.output_file.replace(".json", "_gpu.json")
            if gpu_monitor:
                gpu_monitor.save_results(gpu_output_file, results=gpu_monitor_results)
        
        # 保存结果
        import json
        combined = {}
        if spec_results:
            combined["speculative"] = spec_results.to_dict()
            spec_results.save_json(self.output_file)
        if target_results:
            combined["target_ar"] = target_results.to_dict()
            target_results.save_json(self.output_file)
        if gpu_monitor_results:
            combined["gpu_monitoring"] = gpu_monitor_results.to_dict()
        
        if combined:
            with open(self.output_file, 'w') as f:
                json.dump(combined, f, indent=2)
            print(colored(f"✅ Results saved to {self.output_file}", "green"))

