"""Speculative Decoding Performance Benchmark

This script runs automated performance tests comparing Speculative Decoding
with standard autoregressive generation, collecting detailed metrics including:
- TTFT (Time To First Token)
- Latency
- Throughput
- Token counts
- Acceptance rates (for speculative decoding)
"""

import argparse
import random
import numpy as np
import torch
from engine.dataset import load_sharegpt_multi
from engine.infer_engine import infer_batch
from engine.metrics import BenchmarkResults, print_benchmark_summary, print_comparison
from engine.gpu_monitor import GPUMonitor, print_gpu_summary
from engine.vllm_engine import VLLMEngineManager, VLLMConfig, is_vllm_available, create_vllm_config_from_env
from utils.logits_processor import GreedyProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import asyncio
from termcolor import colored


class BenchmarkRunner:
    """Performance benchmark runner for Speculative Decoding."""
    
    def __init__(self, target_model=None, drafter_model=None):
        """
        Initialize benchmark runner.
        
        Args:
            target_model: Optional path to target model (overrides TARGET_MODEL env var)
            drafter_model: Optional path to drafter model (overrides DRAFTER_MODEL env var)
        
        Note: GPU allocation is controlled by environment variables (TARGET_GPU, DRAFTER_GPU)
        set in run_benchmark.sh, not by a device parameter. The device parameter is kept
        for compatibility but has no effect.
        """
        print(colored("Speculative Decoding Performance Benchmark", "red", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        # Store command line arguments (if provided)
        self.target_model_arg = target_model
        self.drafter_model_arg = drafter_model
        
        # Configuration from environment variables
        self._load_config()
        
        # Load models
        if self.inference_engine == "transformers":
            self._load_models()
        else:  # vllm
            # vLLM engine will be initialized asynchronously later
            self.vllm_target = None
            self.vllm_drafter = None
            self.target = None
            self.drafter = None
            self.target_tokenizer = None
            self.drafter_tokenizer = None
        
        # Load dataset
        self._load_sharegpt_data()
        
        # Initialize processor
        self.processor = GreedyProcessor()
        
        # Run benchmark
        if self.inference_engine == "vllm":
            # vLLM requires async execution
            asyncio.run(self._run_benchmark_vllm())
        else:
            self._run_benchmark()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        self.gamma = int(os.getenv("GAMMA_VALUE", "4"))
        self.gen_len = int(os.getenv("GENERATION_LENGTH", "100"))
        
        # Inference engine: "transformers" or "vllm"
        self.inference_engine = os.getenv("INFERENCE_ENGINE", "transformers").lower()
        if self.inference_engine not in ["transformers", "vllm"]:
            print(colored(f"⚠️  Warning: Unknown INFERENCE_ENGINE '{self.inference_engine}', defaulting to 'transformers'", "yellow"))
            self.inference_engine = "transformers"
        
        print(colored(f"🔧 Inference Engine: {self.inference_engine}", "cyan"))
        
        # Check vLLM availability
        if self.inference_engine == "vllm" and not is_vllm_available():
            print(colored("❌ Error: vLLM engine selected but vLLM is not installed!", "red"))
            print(colored("   Please install vLLM: pip install vllm", "yellow"))
            raise RuntimeError("vLLM not available")
        
        # Inference method: "speculative" or "target_ar" (only one at a time)
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
        
        # Batch processing
        self.enable_batch = os.getenv("ENABLE_BATCH", "false").lower() == "true"
        self.batch_size = int(os.getenv("BATCH_SIZE", "4"))
        self.max_batch_length = int(os.getenv("MAX_BATCH_LENGTH", "512"))
        
        # Benchmark parameters
        self.num_prompts = int(os.getenv("NUM_PROMPTS", "0"))  # 0 means use duration instead
        self.auto_rate = float(os.getenv("AUTO_RATE", "1.0"))
        self.auto_duration = float(os.getenv("AUTO_DURATION", "300"))
        self.prompt_min_length = int(os.getenv("PROMPT_MIN_LENGTH", "10"))
        self.prompt_max_length = int(os.getenv("PROMPT_MAX_LENGTH", "500"))
        self.max_load_lines = int(os.getenv("MAX_LOAD_LINES", "10000"))
        
        # Model paths
        self.sharegpt_dir = os.getenv(
            "SHAREGPT_DIR",
            "/Users/myrick/GithubProjects/Speculative-Decoding/sharegpt_gpt4",
        )
        self.sharegpt_paths = [
            os.path.join(self.sharegpt_dir, "sharegpt_gpt4.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_V3_format.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_zh_38K_format.jsonl"),
        ]
        
        # Output file - automatically includes inference method in filename
        base_output_file = os.getenv("OUTPUT_FILE", "benchmark_results.json")
        # Generate filename with inference method
        # e.g., "benchmark_results.json" -> "benchmark_results_speculative.json"
        if base_output_file.endswith(".json"):
            self.output_file = base_output_file.replace(".json", f"_{self.inference_method_name}.json")
        else:
            self.output_file = f"{base_output_file}_{self.inference_method_name}.json"
        
        # GPU monitoring
        self.enable_gpu_monitor = os.getenv("ENABLE_GPU_MONITOR", "true").lower() == "true"
        self.gpu_monitor_interval = float(os.getenv("GPU_MONITOR_INTERVAL", "1.0"))
        
        # Inference context attributes (required by infer_batch)
        self.chat = True
        self.cache = False
        self.reset_in_between = False  # Whether to reset ngram between batches (not used in benchmark)
        self.ngram = None  # N-gram storage (not used in benchmark, set to None)
    
    def _load_models(self):
        """Load target and drafter models."""
        # Model paths: command line args override environment variables
        target_model = (
            self.target_model_arg 
            if self.target_model_arg is not None 
            else os.getenv("TARGET_MODEL", "/home/llm/model_hub/Qwen3-8B")
        )
        drafter_model = (
            self.drafter_model_arg 
            if self.drafter_model_arg is not None 
            else os.getenv("DRAFTER_MODEL", "/home/llm/model_hub/Qwen3-1.7B")
        )
        
        model_dtype = torch.float16
        
        # GPU allocation
        # Reads GPU configuration from environment variables set by run_benchmark.sh
        # Default "cuda:0" means: if TARGET_GPU/DRAFTER_GPU are not set, both models will use GPU 0
        target_gpu_env = os.getenv("TARGET_GPU", "cuda:0")
        drafter_gpu_env = os.getenv("DRAFTER_GPU", "cuda:0")
        
        def parse_device_map(gpu_string):
            """
            Parse GPU string into device_map format for transformers.
            
            Input formats:
            - "auto" -> "auto" (let transformers decide)
            - "cuda:0" -> "cuda:0" (single GPU)
            - "cuda:0,cuda:1,..." -> "auto" (multi-GPU, let transformers distribute)
            
            Note: For multi-GPU, we use "auto" mode with CUDA_VISIBLE_DEVICES
            to ensure proper device allocation without conflicts.
            """
            if gpu_string == "auto":
                return "auto"
            elif "," in gpu_string:
                # Multi-GPU case: use "auto" to let transformers distribute layers
                # The actual GPUs are controlled by setting CUDA_VISIBLE_DEVICES temporarily
                return "auto"
            else:
                # Single GPU case: "cuda:0"
                return gpu_string
        
        target_device_map = parse_device_map(target_gpu_env)
        drafter_device_map = parse_device_map(drafter_gpu_env)
        
        # For multi-GPU, we need to set CUDA_VISIBLE_DEVICES before loading models
        # Extract GPU IDs from the environment variable strings
        def get_gpu_ids(gpu_string):
            """Extract GPU IDs from string like 'cuda:0,cuda:1,cuda:2' -> [0, 1, 2]"""
            if "," in gpu_string:
                gpu_ids = []
                for gpu in gpu_string.split(","):
                    if gpu.startswith("cuda:"):
                        gpu_ids.append(int(gpu.split(":")[1]))
                return gpu_ids
            elif gpu_string.startswith("cuda:"):
                return [int(gpu_string.split(":")[1])]
            return []
        
        target_gpu_ids = get_gpu_ids(target_gpu_env)
        drafter_gpu_ids = get_gpu_ids(drafter_gpu_env)

        def resolve_primary_device(gpu_ids, fallback: str = "cuda:0"):
            """
            Resolve primary device for a model.
            
            Important: When using CUDA_VISIBLE_DEVICES with device_map="auto",
            the GPUs are remapped. For example:
            - CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" makes physical GPU 0 → logical cuda:0
            - CUDA_VISIBLE_DEVICES="6,7" makes physical GPU 6 → logical cuda:0
            
            Since we set CUDA_VISIBLE_DEVICES during model loading, we always use
            cuda:0 as the primary device (which maps to the first physical GPU in the list).
            """
            if gpu_ids:
                # Always use cuda:0 because CUDA_VISIBLE_DEVICES remaps the first visible GPU to index 0
                return torch.device("cuda:0")
            if torch.cuda.is_available():
                return torch.device(fallback)
            return torch.device("cpu")

        # Store physical GPU IDs for reference (used in logging and monitoring)
        self.target_gpu_ids = target_gpu_ids
        self.drafter_gpu_ids = drafter_gpu_ids
        
        # Primary devices for tensor operations (always cuda:0 after CUDA_VISIBLE_DEVICES remapping)
        self.target_device = resolve_primary_device(target_gpu_ids)
        self.drafter_device = resolve_primary_device(drafter_gpu_ids)
        
        print(colored("Loading models...", "light_grey"))
        print(colored(f"Target: {target_model} on GPUs {target_gpu_ids} (device_map={target_device_map})", "yellow"))
        print(colored(f"Drafter: {drafter_model} on GPUs {drafter_gpu_ids} (device_map={drafter_device_map})", "yellow"))
        
        # Save original CUDA_VISIBLE_DEVICES
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        # Load target model with temporary CUDA_VISIBLE_DEVICES
        if target_gpu_ids and target_device_map == "auto":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target_gpu_ids))
        
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=None,
            dtype=model_dtype,  # Use dtype instead of torch_dtype (deprecated)
            device_map=target_device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.target.eval()
        
        # Restore CUDA_VISIBLE_DEVICES after loading target
        if original_cuda_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        
        self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load drafter model with temporary CUDA_VISIBLE_DEVICES
        if drafter_gpu_ids and drafter_device_map == "auto":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, drafter_gpu_ids))
        
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=None,
            dtype=model_dtype,  # Use dtype instead of torch_dtype (deprecated)
            device_map=drafter_device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        # Restore CUDA_VISIBLE_DEVICES after loading drafter
        if original_cuda_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        
        # End tokens - tokens that signal the end of generation
        # When any of these tokens are generated, the model will stop generating
        self.end_tokens = [self.tokenizer.eos_token_id]  # Standard EOS (End Of Sequence) token
        
        # Qwen models use a special token "<|im_end|>" to mark the end of assistant responses
        # We add it to end_tokens so generation stops correctly when using Qwen models
        try:
            qwen_end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if qwen_end_token is not None and qwen_end_token != self.tokenizer.unk_token_id:
                self.end_tokens.append(qwen_end_token)
        except:
            # If the tokenizer doesn't have this token (non-Qwen model), silently ignore
            pass
        
        print(colored("✅ Models loaded successfully", "green"))
    
    def _load_sharegpt_data(self):
        """Load ShareGPT prompts."""
        try:
            # load_sharegpt_multi expects a list of full file paths
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
        """Get a random prompt from ShareGPT data."""
        if not self.sharegpt_data:
            return "Tell me a story about artificial intelligence."
        
        if hasattr(self, "sharegpt_parts") and self.sharegpt_parts:
            non_empty = [p for p in self.sharegpt_parts if p]
            if non_empty:
                chosen_part = random.choice(non_empty)
                return random.choice(chosen_part)
        return random.choice(self.sharegpt_data)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _run_benchmark(self):
        """Run the performance benchmark."""
        if not self.sharegpt_data:
            print(colored("❌ No ShareGPT data available, cannot run benchmark", "red"))
            return
        
        # Validate benchmark parameters based on mode
        if self.num_prompts <= 0:
            # Time-based mode: need valid duration and rate
            if self.auto_duration <= 0 or self.auto_rate <= 0:
                print(colored("❌ Invalid benchmark parameters: AUTO_DURATION and AUTO_RATE must be > 0 when NUM_PROMPTS=0", "red"))
                return
        # else: num_prompts > 0, count-based mode - no additional validation needed
        
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
        
        # Initialize results based on selected inference method
        if self.spec:
            spec_results = BenchmarkResults(method="speculative")
            target_results = None
        else:
            spec_results = None
            target_results = BenchmarkResults(method="target_ar")
        
        # Start GPU monitoring (before setting start_time to capture accurate timing)
        gpu_monitor = None
        if self.enable_gpu_monitor:
            try:
                # Parse GPU IDs from CUDA_VISIBLE_DEVICES
                cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
                else:
                    gpu_ids = None  # Monitor all GPUs
                
                # Performance callback function for GPU monitor
                def get_performance_metrics():
                    """Callback to get current performance metrics."""
                    tokens_gen = 0
                    tokens_acc = 0
                    requests_done = 0
                    throughput_val = 0.0
                    avg_ttft_val = 0.0
                    avg_latency_val = 0.0
                    
                    # Aggregate from both spec and target results
                    if spec_results:
                        tokens_gen += spec_results.total_tokens
                        requests_done += spec_results.total_requests
                        if spec_results.avg_ttft > 0:
                            avg_ttft_val = spec_results.avg_ttft
                        if spec_results.avg_latency > 0:
                            avg_latency_val = spec_results.avg_latency
                        if spec_results.overall_throughput > 0:
                            throughput_val = spec_results.overall_throughput
                        # For speculative decoding, count accepted tokens
                        if spec_results.method == "speculative":
                            # Estimate accepted tokens from acceptance rate
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
        
        # Set benchmark start time after GPU monitor is initialized
        # This ensures accurate timing capture
        benchmark_start_time = time.time()
        if spec_results:
            spec_results.start_time = benchmark_start_time
        if target_results:
            target_results.start_time = benchmark_start_time
        
        # Run benchmark
        start_time = time.time()
        use_num_prompts = self.num_prompts > 0
        if use_num_prompts:
            end_time = None
            target_requests = self.num_prompts
        else:
            end_time = start_time + self.auto_duration
            target_requests = None
        total_requests = 0
        
        if self.enable_batch:
            prompts_per_iter = max(1, self.batch_size)
            interval = prompts_per_iter / self.auto_rate if not use_num_prompts else 0
            batch_idx = 0
            
            while True:
                now = time.time()
                if use_num_prompts:
                    if total_requests >= target_requests:
                        break
                else:
                    if now >= end_time:
                        break
                
                # Check if we'll exceed target
                if use_num_prompts and total_requests + prompts_per_iter > target_requests:
                    prompts_per_iter = target_requests - total_requests
                
                batch_idx += 1
                iteration_start = time.time()
                
                prompts = [self._get_random_prompt() for _ in range(prompts_per_iter)]
                
                print(colored(
                    f"\n📦 Batch {batch_idx}: {len(prompts)} prompts (elapsed {iteration_start - start_time:.1f}s)",
                    "magenta", attrs=["bold"]
                ))
                
                self._set_seed(42)
                spec_metrics, target_metrics = infer_batch(self, prompts)
                
                if spec_results and spec_metrics:
                    spec_results.batches.append(spec_metrics)
                    spec_results.total_requests += len(prompts)
                
                if target_results and target_metrics:
                    target_results.batches.append(target_metrics)
                    target_results.total_requests += len(prompts)
                
                total_requests += len(prompts)
                
                elapsed = time.time() - iteration_start
                if not use_num_prompts:
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, max(0.0, end_time - time.time())))
        else:
            interval = 1.0 / self.auto_rate if not use_num_prompts else 0
            prompt_idx = 0
            
            while True:
                now = time.time()
                if use_num_prompts:
                    if total_requests >= target_requests:
                        break
                else:
                    if now >= end_time:
                        break
                
                prompt = self._get_random_prompt()
                prompt_idx += 1
                
                print(colored(
                    f"\n🎲 Request #{prompt_idx} (elapsed {now - start_time:.1f}s)",
                    "magenta", attrs=["bold"]
                ))
                
                self._set_seed(42)
                spec_metrics, target_metrics = infer_batch(self, [prompt])
                
                if spec_results and spec_metrics:
                    spec_results.batches.append(spec_metrics)
                    spec_results.total_requests += 1
                
                if target_results and target_metrics:
                    target_results.batches.append(target_metrics)
                    target_results.total_requests += 1
                
                total_requests += 1
                
                if not use_num_prompts:
                    elapsed = time.time() - now
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, max(0.0, end_time - time.time())))
        
        # Stop GPU monitoring
        gpu_monitor_results = None
        if gpu_monitor:
            try:
                gpu_monitor.stop()
                gpu_monitor_results = gpu_monitor.get_results()
                
                # Set final performance metrics
                total_tokens_gen = 0
                total_tokens_acc = 0
                total_reqs = 0
                
                if spec_results:
                    total_tokens_gen += spec_results.total_tokens
                    total_reqs += spec_results.total_requests
                    if spec_results.method == "speculative" and spec_results.avg_acceptance_rate > 0:
                        # Estimate accepted tokens from acceptance rate
                        total_tokens_acc = int(spec_results.total_tokens * spec_results.avg_acceptance_rate)
                
                if target_results:
                    total_tokens_gen += target_results.total_tokens
                    total_reqs += target_results.total_requests
                
                gpu_monitor_results.total_tokens_generated = total_tokens_gen
                gpu_monitor_results.total_tokens_accepted = total_tokens_acc
                gpu_monitor_results.total_requests = total_reqs
                
            except Exception as e:
                print(colored(f"⚠️  Warning: Error stopping GPU monitor: {e}", "yellow"))
        
        # Finalize results
        if spec_results:
            spec_results.end_time = time.time()
            spec_results.total_batches = len(spec_results.batches)
        
        if target_results:
            target_results.end_time = time.time()
            target_results.total_batches = len(target_results.batches)
        
        # Print results
        print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))
        print(colored("📊 Benchmark Complete", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan", attrs=["bold"]))
        
        # Print results summary
        if spec_results:
            print_benchmark_summary(spec_results)
        if target_results:
            print_benchmark_summary(target_results)
        
        # Print GPU monitoring results
        if gpu_monitor_results:
            print_gpu_summary(gpu_monitor_results)
            gpu_output_file = self.output_file.replace(".json", "_gpu.json")
            if gpu_monitor:
                gpu_monitor.save_results(gpu_output_file, results=gpu_monitor_results)
        
        # Save results (file name already includes inference method)
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
        
        # Save combined results file
        if combined:
            with open(self.output_file, 'w') as f:
                json.dump(combined, f, indent=2)
            print(colored(f"✅ Results saved to {self.output_file}", "green"))
    
    async def _vllm_process_request(self, prompt_idx: int, prompt: str, submit_time: float, 
                                     start_time: float, target_results):
        """处理单个vLLM请求（异步）
        
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
                # Create metrics
                from engine.metrics import InferenceMetrics
                metrics = InferenceMetrics()
                metrics.latency = request_end - request_start
                metrics.ttft = 0.0  # vLLM doesn't provide TTFT easily
                metrics.num_generated_tokens = len(output.split())  # Approximate
                metrics.throughput = metrics.num_generated_tokens / metrics.latency if metrics.latency > 0 else 0
                
                target_results.batches.append(metrics)
                target_results.total_requests += 1
                
                queue_time = request_start - submit_time
                print(colored(
                    f"✅ Request #{prompt_idx} completed: {metrics.num_generated_tokens} tokens in {metrics.latency:.3f}s "
                    f"(queue_time: {queue_time:.3f}s)",
                    "green"
                ))
                return True
            else:
                print(colored(f"❌ Request #{prompt_idx} failed", "red"))
                return False
        except Exception as e:
            print(colored(f"❌ Request #{prompt_idx} error: {e}", "red"))
            return False
    
    async def _run_benchmark_vllm(self):
        """Run benchmark using vLLM engine (async version)."""
        import logging
        logger = logging.getLogger(__name__)
        
        print(colored("\n🚀 Starting Benchmark", "cyan", attrs=["bold"]))
        
        # Get model paths
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
        
        # Initialize vLLM engines
        print(colored("Initializing vLLM engine...", "yellow"))
        
        # 从环境变量创建配置
        vllm_config = create_vllm_config_from_env()
        vllm_config.model_path = target_model
        
        # 检查是否使用推测解码
        use_spec = self.spec and vllm_config.enable_speculative
        
        if self.spec and not vllm_config.enable_speculative:
            print(colored("⚠️  Warning: INFERENCE_METHOD=speculative but VLLM_ENABLE_SPECULATIVE=false", "yellow"))
            print(colored("   Set VLLM_ENABLE_SPECULATIVE=true to enable vLLM native speculative decoding", "yellow"))
            print(colored("   Falling back to target-only generation", "yellow"))
        
        if use_spec:
            # vLLM 原生推测解码模式
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
        
        # Display configuration
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
        
        # Initialize results
        method_name = "speculative_vllm" if use_spec else "target_ar_vllm"
        target_results = BenchmarkResults(method_name=method_name)
        
        # Start GPU monitoring
        gpu_monitor = None
        gpu_monitor_results = None
        if self.gpu_monitor_enabled:
            gpu_ids = list(range(8))  # Monitor all 8 GPUs for vLLM
            gpu_monitor = GPUMonitor(
                gpu_ids=gpu_ids,
                sampling_interval=self.gpu_monitor_interval
            )
            gpu_monitor.start()
            print(colored(f"✅ GPU Monitor started (GPUs: {gpu_ids}, interval: {self.gpu_monitor_interval}s)", "green"))
        
        # Run benchmark with concurrent requests
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
        
        # Stop GPU monitoring
        if gpu_monitor:
            gpu_monitor_results = gpu_monitor.stop()
            print(colored(f"🛑 GPU Monitor stopped (collected {len(gpu_monitor_results.snapshots)} snapshots)", "cyan"))
        
        # Print results
        print(colored("\n" + "=" * 70, "cyan"))
        print(colored("📊 Benchmark Results", "cyan", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        print_benchmark_summary(target_results)
        
        if gpu_monitor_results:
            print_gpu_summary(gpu_monitor_results)
            gpu_output_file = self.output_file.replace(".json", "_gpu.json")
            if gpu_monitor:
                gpu_monitor.save_results(gpu_output_file, results=gpu_monitor_results)
        
        # Save results
        import json
        combined = {
            "target_ar_vllm": target_results.to_dict()
        }
        if gpu_monitor_results:
            combined["gpu_monitoring"] = gpu_monitor_results.to_dict()
        
        with open(self.output_file, 'w') as f:
            json.dump(combined, f, indent=2)
        print(colored(f"✅ Results saved to {self.output_file}", "green"))
        
        # Cleanup
        await self.vllm_target.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding Performance Benchmark")
    
    # Model path arguments (can override environment variables)
    # Use None as default so environment variables are used when not provided
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Path to target model (overrides TARGET_MODEL env var, default: use env var or /home/llm/model_hub/Qwen3-8B)"
    )
    parser.add_argument(
        "--drafter-model",
        type=str,
        default=None,
        help="Path to drafter model (overrides DRAFTER_MODEL env var, default: use env var or /home/llm/model_hub/Qwen3-1.7B)"
    )
    
    # Note: GPU allocation is controlled by environment variables (TARGET_GPU, DRAFTER_GPU)
    # set in run_benchmark.sh
    
    args = parser.parse_args()
    
    # Convert empty strings to None (treat as not provided)
    target_model = args.target_model if args.target_model and args.target_model.strip() else None
    drafter_model = args.drafter_model if args.drafter_model and args.drafter_model.strip() else None
    
    BenchmarkRunner(target_model=target_model, drafter_model=drafter_model)

