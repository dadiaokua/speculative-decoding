"""Benchmark Executor - 基准测试执行逻辑

负责执行基准测试的主循环，包括批处理和单请求模式。
"""

import time
from termcolor import colored
from engine.infer_engine import infer_batch


def execute_benchmark_loop(runner, spec_results, target_results):
    """执行基准测试主循环
    
    Args:
        runner: BenchmarkRunner 实例
        spec_results: 推测解码结果收集器（如果启用）
        target_results: 目标模型结果收集器（如果启用）
    
    Returns:
        int: 总请求数
    """
    start_time = time.time()
    use_num_prompts = runner.num_prompts > 0
    
    if use_num_prompts:
        end_time = None
        target_requests = runner.num_prompts
    else:
        end_time = start_time + runner.auto_duration
        target_requests = None
    
    total_requests = 0
    
    if runner.enable_batch:
        total_requests = _execute_batch_mode(
            runner, spec_results, target_results,
            start_time, end_time, target_requests, use_num_prompts
        )
    else:
        total_requests = _execute_single_mode(
            runner, spec_results, target_results,
            start_time, end_time, target_requests, use_num_prompts
        )
    
    return total_requests


def _execute_batch_mode(runner, spec_results, target_results, 
                       start_time, end_time, target_requests, use_num_prompts):
    """执行批处理模式的基准测试"""
    prompts_per_iter = max(1, runner.batch_size)
    interval = prompts_per_iter / runner.auto_rate if not use_num_prompts else 0
    batch_idx = 0
    total_requests = 0
    
    while True:
        now = time.time()
        if use_num_prompts:
            if total_requests >= target_requests:
                break
        else:
            if now >= end_time:
                break
        
        # 检查是否会超过目标数量
        if use_num_prompts and total_requests + prompts_per_iter > target_requests:
            prompts_per_iter = target_requests - total_requests
        
        batch_idx += 1
        iteration_start = time.time()
        
        prompts = [runner._get_random_prompt() for _ in range(prompts_per_iter)]
        
        print(colored(
            f"\n📦 Batch {batch_idx}: {len(prompts)} prompts (elapsed {iteration_start - start_time:.1f}s)",
            "magenta", attrs=["bold"]
        ))
        
        runner._set_seed(42)
        spec_metrics, target_metrics = infer_batch(runner, prompts)
        
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
    
    return total_requests


def _execute_single_mode(runner, spec_results, target_results,
                        start_time, end_time, target_requests, use_num_prompts):
    """执行单请求模式的基准测试"""
    interval = 1.0 / runner.auto_rate if not use_num_prompts else 0
    prompt_idx = 0
    total_requests = 0
    
    while True:
        now = time.time()
        if use_num_prompts:
            if total_requests >= target_requests:
                break
        else:
            if now >= end_time:
                break
        
        prompt = runner._get_random_prompt()
        prompt_idx += 1
        
        print(colored(
            f"\n🎲 Request #{prompt_idx} (elapsed {now - start_time:.1f}s)",
            "magenta", attrs=["bold"]
        ))
        
        runner._set_seed(42)
        spec_metrics, target_metrics = infer_batch(runner, [prompt])
        
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
    
    return total_requests

