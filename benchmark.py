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
from utils.logits_processor import GreedyProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
from termcolor import colored


class BenchmarkRunner:
    """Performance benchmark runner for Speculative Decoding."""
    
    def __init__(self, device: str = "cuda"):
        print(colored("Speculative Decoding Performance Benchmark", "red", attrs=["bold"]))
        print(colored("=" * 70, "cyan"))
        
        self.device = device
        
        # Configuration from environment variables
        self._load_config()
        
        # Load models
        self._load_models()
        
        # Load dataset
        self._load_sharegpt_data()
        
        # Initialize processor
        self.processor = GreedyProcessor()
        
        # Run benchmark
        self._run_benchmark()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        self.gamma = int(os.getenv("GAMMA_VALUE", "4"))
        self.gen_len = int(os.getenv("GENERATION_LENGTH", "100"))
        self.spec = os.getenv("ENABLE_SPECULATIVE", "true").lower() == "true"
        self.target_gen = os.getenv("ENABLE_TARGET", "true").lower() == "true"
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
        
        # Output file
        self.output_file = os.getenv("OUTPUT_FILE", "benchmark_results.json")
        
        # GPU monitoring
        self.enable_gpu_monitor = os.getenv("ENABLE_GPU_MONITOR", "true").lower() == "true"
        self.gpu_monitor_interval = float(os.getenv("GPU_MONITOR_INTERVAL", "1.0"))
        
        self.chat = True
        self.cache = False
    
    def _load_models(self):
        """Load target and drafter models."""
        target_model = "/home/llm/model_hub/Qwen3-8B"
        drafter_model = "/home/llm/model_hub/Qwen3-1.7B"
        
        model_dtype = torch.float16
        
        # GPU allocation
        # Reads GPU configuration from environment variables set by run_benchmark.sh
        target_gpu_env = os.getenv("TARGET_GPU", "cuda:0")
        drafter_gpu_env = os.getenv("DRAFTER_GPU", "cuda:0")
        
        def parse_device_map(gpu_string):
            """
            Parse GPU string into device_map format for transformers.
            
            Input formats:
            - "auto" -> "auto" (let transformers decide)
            - "cuda:0" -> "cuda:0" (single GPU)
            - "cuda:0,cuda:1,cuda:2" -> {"": [0,1,2]} (multi-GPU auto-distribute)
            
            Output formats:
            - Single GPU: "cuda:0"
            - Multi-GPU: {"": [0,1,2]} where "" means auto-distribute layers
            - Auto: "auto"
            """
            if gpu_string == "auto":
                return "auto"
            elif "," in gpu_string:
                # Multi-GPU case: "cuda:0,cuda:1,cuda:2"
                # Extract GPU IDs and return auto-distribution format
                gpu_ids = []
                for gpu in gpu_string.split(","):
                    if gpu.startswith("cuda:"):
                        gpu_ids.append(int(gpu.split(":")[1]))
                # {"": [0,1,2]} tells transformers to auto-distribute model layers
                # across these GPUs using accelerate library
                return {"": gpu_ids}
            else:
                # Single GPU case: "cuda:0"
                return gpu_string
        
        target_device_map = parse_device_map(target_gpu_env)
        drafter_device_map = parse_device_map(drafter_gpu_env)
        
        print(colored("Loading models...", "light_grey"))
        print(colored(f"Target: {target_model} on {target_device_map}", "yellow"))
        print(colored(f"Drafter: {drafter_model} on {drafter_device_map}", "yellow"))
        
        # Print detailed GPU allocation info
        if isinstance(target_device_map, dict) and "" in target_device_map:
            print(colored(f"  → Target model layers will be auto-distributed across GPUs: {target_device_map['']}", "light_grey"))
        if isinstance(drafter_device_map, dict) and "" in drafter_device_map:
            print(colored(f"  → Drafter model layers will be auto-distributed across GPUs: {drafter_device_map['']}", "light_grey"))
        
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=None,
            torch_dtype=model_dtype,
            device_map=target_device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.target.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=None,
            torch_dtype=model_dtype,
            device_map=drafter_device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        # End tokens
        self.end_tokens = [self.tokenizer.eos_token_id]
        try:
            qwen_end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if qwen_end_token is not None and qwen_end_token != self.tokenizer.unk_token_id:
                self.end_tokens.append(qwen_end_token)
        except:
            pass
        
        print(colored("✅ Models loaded successfully", "green"))
    
    def _load_sharegpt_data(self):
        """Load ShareGPT prompts."""
        try:
            parts = load_sharegpt_multi(
                self.sharegpt_dir,
                self.sharegpt_file_names,
                prompt_min_length=self.prompt_min_length,
                prompt_max_length=self.prompt_max_length,
                max_load_lines=self.max_load_lines,
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
        
        if self.num_prompts <= 0:
            if self.auto_duration <= 0 or self.auto_rate <= 0:
                print(colored("❌ Invalid benchmark parameters", "red"))
                return
        elif self.num_prompts <= 0:
            print(colored("❌ Invalid NUM_PROMPTS parameter", "red"))
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
        print(colored("=" * 70, "cyan"))
        
        # Initialize results
        spec_results = BenchmarkResults(method="speculative") if self.spec else None
        target_results = BenchmarkResults(method="target_ar") if self.target_gen else None
        
        if spec_results:
            spec_results.start_time = time.time()
        if target_results:
            target_results.start_time = time.time()
        
        # Start GPU monitoring
        gpu_monitor = None
        if self.enable_gpu_monitor:
            try:
                # Parse GPU IDs from CUDA_VISIBLE_DEVICES
                cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
                else:
                    gpu_ids = None  # Monitor all GPUs
                
                gpu_monitor = GPUMonitor(gpu_ids=gpu_ids, sampling_interval=self.gpu_monitor_interval)
                gpu_monitor.start()
            except Exception as e:
                print(colored(f"⚠️  Warning: Could not start GPU monitor: {e}", "yellow"))
        
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
        
        if spec_results:
            print_benchmark_summary(spec_results)
            spec_results.save_json(self.output_file.replace(".json", "_speculative.json"))
        
        if target_results:
            print_benchmark_summary(target_results)
            target_results.save_json(self.output_file.replace(".json", "_target.json"))
        
        if spec_results and target_results:
            print_comparison(spec_results, target_results)
        
        # Print GPU monitoring results
        if gpu_monitor_results:
            print_gpu_summary(gpu_monitor_results)
            gpu_output_file = self.output_file.replace(".json", "_gpu.json")
            if gpu_monitor:
                gpu_monitor.save_results(gpu_output_file, results=gpu_monitor_results)
        
        # Save combined results
        if spec_results or target_results:
            combined = {}
            if spec_results:
                combined["speculative"] = spec_results.to_dict()
            if target_results:
                combined["target_ar"] = target_results.to_dict()
            if gpu_monitor_results:
                combined["gpu_monitoring"] = gpu_monitor_results.to_dict()
            
            import json
            with open(self.output_file, 'w') as f:
                json.dump(combined, f, indent=2)
            print(colored(f"✅ Combined results saved to {self.output_file}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding Performance Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()
    
    BenchmarkRunner(device=args.device)

