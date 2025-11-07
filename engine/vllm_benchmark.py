"""vLLM Benchmark Runner - vLLM性能测试执行器

负责使用 vLLM 引擎进行性能测试。
"""

import os
import time
import asyncio
import random
import logging
from termcolor import colored
from typing import Optional

from engine.dataset import load_sharegpt_multi
from engine.metrics import BenchmarkResults, RequestMetrics, BatchMetrics, print_benchmark_summary
from engine.gpu_monitor import GPUMonitor, print_gpu_summary
from engine.vllm_engine import VLLMEngineManager, create_vllm_config_from_env


class VLLMBenchmarkRunner:
    """vLLM 性能基准测试运行器"""
    
    def __init__(self, target_model: Optional[str] = None, drafter_model: Optional[str] = None):
        """初始化 vLLM 基准测试运行器
        
        Args:
            target_model: 目标模型路径（覆盖环境变量 TARGET_MODEL）
            drafter_model: 草稿模型路径（覆盖环境变量 DRAFTER_MODEL）
        """
        print(colored("Speculative Decoding Performance Benchmark (vLLM)", "red", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        # 存储命令行参数
        self.target_model_arg = target_model
        self.drafter_model_arg = drafter_model
        
        # 加载配置
        self._load_config()
        
        # 加载数据集
        self._load_sharegpt_data()
        
        # 运行基准测试（异步）
        asyncio.run(self._run_benchmark_vllm())
    
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
    
    async def _vllm_process_request(self, prompt_idx: int, prompt: str, submit_time: float,
                                     start_time: float, target_results):
        """处理单个 vLLM 请求（异步）
        
        Args:
            prompt_idx: 请求索引
            prompt: 输入提示文本
            submit_time: 请求提交时间
            start_time: 基准测试开始时间
            target_results: 结果收集对象
        
        Returns:
            bool: 是否成功
        """
        print(colored(
            f"\n🎲 Request #{prompt_idx} submitted (elapsed {submit_time - start_time:.1f}s)",
            "magenta", attrs=["bold"]
        ))
        
        request_start = time.time()
        try:
            output = await self.vllm_target.generate(
                prompt,
                max_tokens=self.gen_len,
                temperature=1.0,
                top_p=1.0
            )
            request_end = time.time()
            
            if output:
                # 创建指标
                req_metric = RequestMetrics()
                req_metric.start_time = request_start
                req_metric.end_time = request_end
                req_metric.first_token_time = request_start
                req_metric.ttft = 0.0
                req_metric.total_latency = request_end - request_start
                req_metric.generated_tokens = len(output.split())
                req_metric.prompt_tokens = len(prompt.split())
                req_metric.total_tokens = req_metric.prompt_tokens + req_metric.generated_tokens
                
                # 创建批次指标（单请求批次）
                batch_metric = BatchMetrics()
                batch_metric.batch_size = 1
                batch_metric.batch_start_time = request_start
                batch_metric.batch_end_time = request_end
                batch_metric.requests.append(req_metric)
                
                target_results.batches.append(batch_metric)
                target_results.total_requests += 1
                
                queue_time = request_start - submit_time
                
                # 打印请求完成信息
                print(colored(
                    f"✅ Request #{prompt_idx} completed: {req_metric.generated_tokens} tokens in {req_metric.total_latency:.3f}s "
                    f"(queue_time: {queue_time:.3f}s)",
                    "green"
                ))
                
                # 打印 Prompt 和 LLM 返回结果
                print(colored("─" * 70, "cyan"))
                print(colored("📝 Prompt:", "yellow", attrs=["bold"]))
                prompt_display = prompt if len(prompt) <= 200 else prompt[:200] + "..."
                print(colored(f"   {prompt_display}", "white"))
                
                print(colored("\n💬 LLM Response:", "yellow", attrs=["bold"]))
                output_display = output if len(output) <= 300 else output[:300] + "..."
                print(colored(f"   {output_display}", "white"))
                print(colored("─" * 70, "cyan"))
                
                return True
            else:
                print(colored(f"❌ Request #{prompt_idx} failed", "red"))
                return False
        except Exception as e:
            print(colored(f"❌ Request #{prompt_idx} error: {e}", "red"))
            return False
    
    async def _run_benchmark_vllm(self):
        """运行 vLLM 基准测试（异步版本）"""
        logger = logging.getLogger(__name__)
        
        print(colored("\n🚀 Starting Benchmark", "cyan", attrs=["bold"]))
        
        # 获取模型路径
        target_model = (
            self.target_model_arg
            if self.target_model_arg is not None
            else os.getenv("TARGET_MODEL", "/home/llm/model_hub/Qwen3-8B")
        )
        drafter_model = (
            self.drafter_model_arg
            if self.drafter_model_arg is not None
            else os.getenv("DRAFTER_MODEL", "/home/llm/model_hub/Qwen3-0.6B")
        )
        
        # 初始化 vLLM 引擎
        print(colored("Initializing vLLM engine...", "yellow"))
        
        vllm_config = create_vllm_config_from_env()
        vllm_config.model_path = target_model
        
        # 检查是否使用推测解码
        use_spec = self.spec and vllm_config.enable_speculative
        
        if self.spec and not vllm_config.enable_speculative:
            print(colored("⚠️  Warning: INFERENCE_METHOD=speculative but VLLM_ENABLE_SPECULATIVE=false", "yellow"))
            print(colored("   Set VLLM_ENABLE_SPECULATIVE=true to enable vLLM native speculative decoding", "yellow"))
            print(colored("   Falling back to target-only generation", "yellow"))
        
        if use_spec:
            vllm_config.speculative_model = drafter_model
            print(colored(f"✅ 使用 vLLM 原生推测解码", "green"))
            print(colored(f"   Target: {target_model}", "cyan"))
            print(colored(f"   Drafter: {drafter_model}", "cyan"))
            print(colored(f"   推测token数: {vllm_config.num_speculative_tokens}", "cyan"))
        
        # 初始化引擎
        self.vllm_target = VLLMEngineManager(vllm_config, logger)
        
        if not await self.vllm_target.initialize():
            print(colored("❌ Failed to initialize vLLM engine", "red"))
            return
        
        # 显示配置
        if self.num_prompts > 0:
            print(f"Rate: {self.auto_rate:.2f} prompts/s")
            print(f"Total Prompts: {self.num_prompts}")
        else:
            print(f"Rate: {self.auto_rate:.2f} prompts/s")
            print(f"Duration: {self.auto_duration:.1f} s")
        
        print(f"Batch mode: {self.enable_batch}")
        if use_spec:
            print(f"Inference Method: Speculative Decoding (vLLM Native)")
        else:
            print(f"Inference Method: Target AR (vLLM)")
        print("=" * 70)
        
        # 初始化结果
        method_name = "speculative_vllm" if use_spec else "target_ar_vllm"
        target_results = BenchmarkResults(method=method_name)
        
        # 启动 GPU 监控
        gpu_monitor = None
        gpu_monitor_results = None
        if self.enable_gpu_monitor:
            gpu_ids = list(range(8))
            gpu_monitor = GPUMonitor(
                gpu_ids=gpu_ids,
                sampling_interval=self.gpu_monitor_interval
            )
            gpu_monitor.start()
            print(colored(f"✅ GPU Monitor started (GPUs: {gpu_ids}, interval: {self.gpu_monitor_interval}s)", "green"))
        
        # 运行基准测试（并发请求）
        start_time = time.time()
        target_results.start_time = start_time
        
        use_num_prompts = self.num_prompts > 0
        if use_num_prompts:
            end_time = None
            target_requests = self.num_prompts
        else:
            end_time = start_time + self.auto_duration
            target_requests = None
        
        # 请求发送循环
        tasks = []
        total_requests = 0
        interval = 1.0 / self.auto_rate if not use_num_prompts else 0
        prompt_idx = 0
        
        while True:
            now = time.time()
            
            # 检查是否达到停止条件
            if use_num_prompts:
                if total_requests >= target_requests:
                    break
            else:
                if now >= end_time:
                    break
            
            # 发送新请求
            prompt = self._get_random_prompt()
            prompt_idx += 1
            total_requests += 1
            
            # 创建异步任务（不等待完成）
            task = asyncio.create_task(
                self._vllm_process_request(prompt_idx, prompt, now, start_time, target_results)
            )
            tasks.append(task)
            
            # 控制发送速率
            if interval > 0:
                await asyncio.sleep(interval)
        
        # 等待所有请求完成
        print(colored(f"\n⏳ Waiting for all {len(tasks)} requests to complete...", "cyan"))
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 完成结果统计
        target_results.end_time = time.time()
        target_results.total_batches = len(target_results.batches)
        
        # 停止 GPU 监控
        if gpu_monitor:
            gpu_monitor_results = gpu_monitor.stop()
            if gpu_monitor_results:
                print(colored(f"🛑 GPU Monitor stopped (collected {len(gpu_monitor_results.snapshots)} snapshots)", "cyan"))
                gpu_monitor_results.total_tokens_generated = target_results.total_tokens
                gpu_monitor_results.total_tokens_accepted = 0
                gpu_monitor_results.total_requests = target_results.total_requests
            else:
                print(colored("🛑 GPU Monitor stopped", "cyan"))
        
        # 打印结果
        print(colored("\n" + "=" * 70, "cyan"))
        print(colored("📊 Benchmark Results", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        print_benchmark_summary(target_results)
        
        if gpu_monitor_results:
            print_gpu_summary(gpu_monitor_results)
            gpu_output_file = self.output_file.replace(".json", "_gpu.json")
            if gpu_monitor:
                gpu_monitor.save_results(gpu_output_file, results=gpu_monitor_results)
        
        # 保存结果
        import json
        combined = {
            "target_ar_vllm": target_results.to_dict()
        }
        if gpu_monitor_results:
            combined["gpu_monitoring"] = gpu_monitor_results.to_dict()
        
        with open(self.output_file, 'w') as f:
            json.dump(combined, f, indent=2)
        print(colored(f"✅ Results saved to {self.output_file}", "green"))
        
        # 清理
        await self.vllm_target.shutdown()

