from typing import List, Tuple, Optional
from termcolor import colored
import time
import torch

from .batch_decode import decode_batch_with_chat_template
from .metrics import BatchMetrics, RequestMetrics


def infer_batch(ctx, prompts: List[str]) -> Tuple[Optional[BatchMetrics], Optional[BatchMetrics]]:
    """
    Run batch inference on multiple prompts.
    
    This is the main entry point for batch inference. It handles:
    1. Prompt formatting (chat template if needed)
    2. Tokenization
    3. Running speculative decoding and/or target AR generation
    4. Collecting performance metrics
    
    Args:
        ctx: Context object (BenchmarkRunner instance) containing:
            - tokenizer: Tokenizer for the model
            - chat: Whether to use chat template
            - reset_in_between: Whether to reset ngram between batches
            - ngram: Optional ngram storage object
            - spec: Whether to run speculative decoding
            - target_gen: Whether to run target AR generation
            - max_batch_length: Maximum sequence length for batch
        prompts: List of raw prompt strings
    
    Returns:
        Tuple of (speculative_metrics, target_metrics):
            - spec_metrics: BatchMetrics for speculative decoding (None if disabled)
            - target_metrics: BatchMetrics for target AR (None if disabled)
    """
    # Step 1: Format prompts with chat template if needed
    # Chat models (like Qwen, Llama) require special formatting
    if ctx.chat:
        formatted_prompts = [
            ctx.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], 
                add_generation_prompt=True,  # Add assistant's turn marker
                tokenize=False  # We'll tokenize in batch_decode
            )
            for p in prompts
        ]
    else:
        # Non-chat models: use prompts as-is
        formatted_prompts = prompts

    # Step 2: Tokenize the batch with padding and truncation
    # This converts text prompts to token IDs that the model can process
    input_ids, attention_mask = decode_batch_with_chat_template(
        ctx.tokenizer, 
        formatted_prompts, 
        max_length=ctx.max_batch_length, 
        chat=False  # Already formatted above if needed
    )

    drafter_device = getattr(ctx, "drafter_device", None)
    target_device = getattr(ctx, "target_device", None)

    if ctx.spec and drafter_device is not None:
        input_ids = input_ids.to(drafter_device)
        attention_mask = attention_mask.to(drafter_device)
    elif ctx.target_gen and target_device is not None:
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device)

    # Step 3: Reset ngram storage if needed (for ngram-assisted decoding)
    if ctx.reset_in_between and ctx.ngram is not None:
        ctx.ngram.reset()

    batch_size = len(prompts)
    spec_metrics = None
    target_metrics = None

    # Step 4: Run inference based on configured method (only one at a time)
    # Either speculative decoding (drafter + target) or target AR (target only)
    if ctx.spec:
        # Speculative decoding: uses both drafter and target models for faster generation
        print(colored("🚀 Running Speculative Decoding on batch...", "green"))
        spec_metrics = run_batch_speculative(ctx, input_ids, attention_mask, batch_size)
        target_metrics = None
    elif ctx.target_gen:
        # Target AR: uses only the target model (standard autoregressive generation)
        print(colored("🎯 Running Target AR on batch...", "blue"))
        spec_metrics = None
        target_metrics = run_batch_target(ctx, input_ids, attention_mask, batch_size)
    else:
        # No inference method enabled (should not happen if configured correctly)
        print(colored("⚠️  Warning: No inference method enabled", "yellow"))
        spec_metrics = None
        target_metrics = None

    return spec_metrics, target_metrics


def run_batch_speculative(ctx, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int) -> Optional[BatchMetrics]:
    """Run speculative decoding on batch - engine function with metrics collection."""
    batch_metrics = BatchMetrics(batch_size=batch_size)
    batch_metrics.batch_start_time = time.time()
    
    # Track per-request start times for TTFT calculation
    request_start_times = [time.time()] * batch_size
    first_token_times = [None] * batch_size
    
    def set_first_token_time(idx):
        if idx < batch_size and first_token_times[idx] is None:
            first_token_times[idx] = time.time()
    
    try:
        batch_outputs, batch_accept_rates = batch_speculative_generate(
            ctx, input_ids, attention_mask, batch_size, 
            first_token_callback=set_first_token_time
        )
        
        batch_metrics.batch_end_time = time.time()
        
        # Collect metrics for each request
        for i in range(batch_size):
            req_metrics = RequestMetrics()
            req_metrics.start_time = request_start_times[i]
            req_metrics.prompt_tokens = torch.sum(attention_mask[i]).item()
            req_metrics.generated_tokens = len(batch_outputs[i]) - req_metrics.prompt_tokens
            req_metrics.total_tokens = len(batch_outputs[i])
            req_metrics.acceptance_rate = batch_accept_rates[i] if i < len(batch_accept_rates) else 0.0
            req_metrics.end_time = batch_metrics.batch_end_time
            
            # Calculate TTFT and latency
            if first_token_times[i] is not None:
                req_metrics.first_token_time = first_token_times[i]
                req_metrics.ttft = first_token_times[i] - request_start_times[i]
            else:
                # Fallback estimate if first token time not captured
                req_metrics.ttft = (batch_metrics.batch_end_time - request_start_times[i]) / max(req_metrics.generated_tokens, 1)
            
            req_metrics.total_latency = batch_metrics.batch_end_time - request_start_times[i]
            
            batch_metrics.requests.append(req_metrics)
        
        return batch_metrics
        
    except Exception as e:
        print(colored(f"❌ Batch speculative decoding failed: {e}", "red"))
        return None


def batch_speculative_generate(ctx, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int, first_token_callback=None) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Core batch speculative decoding implementation.
    
    This is the heart of speculative decoding - it generates tokens faster by:
    1. Using a small drafter model to propose multiple candidate tokens (gamma tokens)
    2. Using the large target model to verify all proposals in parallel
    3. Accepting/rejecting proposals based on probability ratios (rejection sampling)
    4. Re-sampling from corrected distribution when proposals are rejected
    
    Algorithm Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Drafter generates γ draft tokens sequentially             │
    │    (using KV cache for efficiency)                           │
    │                                                               │
    │ 2. Target verifies all γ drafts in parallel                  │
    │    (computes probability for each draft token)               │
    │                                                               │
    │ 3. For each draft, accept/reject based on:                   │
    │    accept_prob = min(1, p_target / p_drafter)                │
    │                                                               │
    │ 4. If rejected, sample from residual distribution:          │
    │    residual = max(0, p_target - p_drafter)                   │
    │                                                               │
    │ 5. Repeat until gen_len tokens generated                      │
    └─────────────────────────────────────────────────────────────┘
    
    Args:
        ctx: Context object containing models and configuration
        input_ids: Token IDs for prompts [batch_size, prompt_len]
        attention_mask: Attention mask [batch_size, prompt_len]
        batch_size: Number of prompts in batch
        first_token_callback: Optional callback to record first token time
    
    Returns:
        Tuple of:
        - batch_outputs: List of full token sequences (prompt + generated)
        - batch_accept_rates: List of acceptance rates per sequence
    """
    batch_outputs: List[torch.Tensor] = []
    batch_accept_rates: List[float] = []

    device = input_ids.device
    target_device = getattr(ctx, "target_device", device)
    # 存储所有生成的token，形状: [batch_size, gen_len]
    generated_tokens = torch.zeros(batch_size, ctx.gen_len, device=device, dtype=torch.long)
    # 标记每个序列是否已完成生成（遇到结束token）
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 统计每个序列的草稿生成数和接受数（用于计算接受率）
    drafts_generated_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)
    drafts_accepted_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)

    # ========== 阶段1: 初始化Drafter的KV Cache ==========
    # 预先处理整个prompt，建立KV cache，避免后续重复计算
    drafter_past = None
    with torch.no_grad():
        init_out = ctx.drafter(input_ids, attention_mask=attention_mask, use_cache=True)
        drafter_past = init_out.past_key_values  # 保存KV cache供后续使用

    # ========== 主循环: 每次生成gamma个token ==========
    step = 0  # 当前已生成的token位置
    while step < ctx.gen_len:
        if finished.all():  # 所有序列都完成了，提前退出
            break

        remaining = ctx.gen_len - step
        current_gamma = min(ctx.gamma, remaining)  # 本次要生成的草稿token数

        # 存储本次迭代中drafter生成的所有候选token和概率
        draft_tokens = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.long)
        drafter_sampled_probs = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.float32)
        q_probs_full = torch.zeros(batch_size, current_gamma, ctx.target.config.vocab_size, device=device, dtype=torch.float32)

        # ========== 阶段2: Drafter生成gamma个候选token（顺序生成）==========
        for draft_step in range(current_gamma):
            if finished.all():
                break

            # 确定当前输入的token（根据位置选择上一个token）
            if draft_step == 0:
                # 第一个草稿：从上一步的最后一个token开始（或prompt的最后一个token）
                token_prev = generated_tokens[:, step-1] if step > 0 else input_ids[:, -1]
            else:
                # 后续草稿：从上一个草稿token开始
                token_prev = generated_tokens[:, step + draft_step - 1]
            current_input = token_prev.unsqueeze(1)  # [batch_size, 1]

            # Drafter前向传播（使用KV cache加速）
            with torch.no_grad():
                out = ctx.drafter(current_input, past_key_values=drafter_past, use_cache=True)
                logits = out.logits[:, -1, :]  # 取最后一个位置的logits
                q_probs = torch.softmax(logits, dim=-1)  # drafter的概率分布 q(x)

            drafter_past = out.past_key_values  # 更新KV cache

            # 从drafter的概率分布中采样一个token
            samples_all = torch.multinomial(q_probs, 1).squeeze(-1).to(device)  # [batch_size] - 确保在正确设备上
            q_probs_full[:, draft_step, :] = q_probs  # 保存完整概率分布（用于后续计算）

            # 只更新未完成的序列
            active_mask = ~finished
            if active_mask.any():
                draft_tokens[active_mask, draft_step] = samples_all[active_mask]
                # 保存采样token的概率（用于acceptance计算）
                drafter_sampled_probs[active_mask, draft_step] = q_probs[active_mask, :].gather(
                    1, samples_all[active_mask].unsqueeze(1)
                ).squeeze(1)
                generated_tokens[active_mask, step + draft_step] = samples_all[active_mask]
                drafts_generated_per_seq[active_mask] += 1
                
                # 记录第一个token的生成时间（用于TTFT计算）
                if first_token_callback is not None and draft_step == 0 and step == 0:
                    for idx in torch.where(active_mask)[0]:
                        first_token_callback(idx.item())

        # ========== 阶段3: Target验证所有gamma个候选token（并行验证）==========
        active_mask = ~finished
        if active_mask.any():
            # 构建完整的序列：prompt + 已生成的token + 本次的gamma个草稿token
            draft_segment = generated_tokens[:, :step + current_gamma]
            verify_ids = torch.cat([input_ids, draft_segment], dim=1).to(target_device)
            with torch.no_grad():
                # Target一次性前向传播，验证所有gamma个位置
                t_out = ctx.target(verify_ids)
                # 提取gamma个位置的logits（对应gamma个草稿token的位置）
                t_logits_full = t_out.logits[:, -(current_gamma+1):-1, :]
                p_probs_full = torch.softmax(t_logits_full, dim=-1)  # target的概率分布 p(x)

            # ========== 阶段4: 对每个序列逐个进行accept/reject决策 ==========
            active_indices = torch.where(active_mask)[0]
            for global_idx in active_indices:
                local_row = global_idx.item()
                if finished[global_idx]:
                    continue
                accepted_count = 0  # 本次迭代中接受的token数

                # 顺序验证每个草稿token
                for draft_idx in range(current_gamma):
                    if finished[global_idx]:
                        break

                    sampled_token = draft_tokens[global_idx, draft_idx].item()
                    q_vec = q_probs_full[global_idx, draft_idx]  # drafter在draft_idx位置的概率分布
                    p_vec = p_probs_full[local_row, draft_idx].to(device)   # target在draft_idx位置的概率分布
                    q_vec = q_vec.to(device)

                    # 提取采样token的概率
                    p_sample = p_vec[sampled_token].item()  # target对该token的概率
                    q_sample = q_vec[sampled_token].item()   # drafter对该token的概率

                    # ========== 拒绝采样（Rejection Sampling）==========
                    # 接受概率 = min(1, p_target / p_drafter)
                    # 这确保了输出的分布与target模型一致
                    accept_prob = 1.0 if q_sample <= 0.0 else min(1.0, p_sample / q_sample)
                    
                    if torch.rand(1, device=device).item() < accept_prob:
                        # ✅ 接受该token
                        accepted_count += 1
                        drafts_accepted_per_seq[global_idx] += 1
                        # 检查是否是结束token
                        if sampled_token in ctx.end_tokens:
                            finished[global_idx] = True
                            break
                    else:
                        # ❌ 拒绝该token，需要从修正后的分布中重新采样
                        # residual分布 = max(0, p_target - p_drafter)
                        # 这确保了即使拒绝后，采样仍遵循target的分布
                        residual = torch.clamp(p_vec - torch.minimum(p_vec, q_vec), min=0.0).to(device)
                        denom = residual.sum().item()
                        if denom <= 1e-12:
                            # 如果residual分布为空，直接按target分布采样
                            corrected = torch.multinomial(p_vec.to(device), 1).item()
                        else:
                            # 从residual分布中采样（归一化）
                            # 确保采样结果在正确的设备上
                            corrected = torch.multinomial((residual / denom).to(device), 1).item()
                        generated_tokens[global_idx, step + draft_idx] = corrected
                        # 检查是否是结束token
                        if corrected in ctx.end_tokens:
                            finished[global_idx] = True
                        break  # 拒绝后，后续的draft token全部无效

                # 如果接受了部分token，需要清理后续无效的token位置
                if accepted_count < current_gamma:
                    tail_start = step + accepted_count + 1
                    if tail_start < step + current_gamma:
                        generated_tokens[global_idx, tail_start: step + current_gamma] = 0

        step += current_gamma  # 移动到下一个gamma窗口

    # ========== 阶段5: 后处理 - 构建最终输出并计算接受率 ==========
    for i in range(batch_size):
        gen_seq = generated_tokens[i]
        # 找到最后一个非零token的位置（移除填充的0）
        nonzero = torch.nonzero(gen_seq, as_tuple=True)[0]
        if nonzero.numel() > 0:
            last = nonzero[-1].item() + 1
            final_gen = gen_seq[:last]
        else:
            final_gen = torch.tensor([], dtype=torch.long, device=device)
        # 拼接prompt和生成的token
        full_output = torch.cat([input_ids[i], final_gen])
        batch_outputs.append(full_output)

        # 计算该序列的接受率
        tot = drafts_generated_per_seq[i].item()
        acc = drafts_accepted_per_seq[i].item()
        batch_accept_rates.append((acc / tot) if tot > 0 else 0.0)

    return batch_outputs, batch_accept_rates


def run_batch_target(ctx, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int) -> Optional[BatchMetrics]:
    """Run target AR on batch - engine function with metrics collection."""
    batch_metrics = BatchMetrics(batch_size=batch_size)
    batch_metrics.batch_start_time = time.time()
    
    # Track per-request start times for TTFT calculation
    request_start_times = [time.time()] * batch_size
    first_token_times = [None] * batch_size
    
    def set_first_token_time(idx):
        if idx < batch_size and first_token_times[idx] is None:
            first_token_times[idx] = time.time()
    
    try:
        batch_outputs = batch_autoregressive_generate(ctx, input_ids, attention_mask, batch_size, first_token_callback=set_first_token_time)
        
        batch_metrics.batch_end_time = time.time()
        
        # Collect metrics for each request
        for i in range(batch_size):
            req_metrics = RequestMetrics()
            req_metrics.start_time = request_start_times[i]
            req_metrics.prompt_tokens = torch.sum(attention_mask[i]).item()
            req_metrics.generated_tokens = len(batch_outputs[i]) - req_metrics.prompt_tokens
            req_metrics.total_tokens = len(batch_outputs[i])
            req_metrics.end_time = batch_metrics.batch_end_time
            
            # Calculate TTFT and latency
            if first_token_times[i] is not None:
                req_metrics.first_token_time = first_token_times[i]
                req_metrics.ttft = first_token_times[i] - request_start_times[i]
            else:
                # Fallback estimate
                req_metrics.ttft = (batch_metrics.batch_end_time - request_start_times[i]) / max(req_metrics.generated_tokens, 1)
            
            req_metrics.total_latency = batch_metrics.batch_end_time - request_start_times[i]
            
            batch_metrics.requests.append(req_metrics)
        
        return batch_metrics
        
    except Exception as e:
        print(colored(f"❌ Batch target generation failed: {e}", "red"))
        return None


def batch_autoregressive_generate(ctx, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int, first_token_callback=None) -> List[torch.Tensor]:
    device = input_ids.device
    generated_tokens = torch.zeros(batch_size, ctx.gen_len, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    past_key_values = None

    for step in range(ctx.gen_len):
        if finished.all():
            break

        if step == 0:
            current_input = input_ids
            current_attention_mask = attention_mask
        else:
            current_input = generated_tokens[:, step-1:step]
            current_attention_mask = torch.ones_like(current_input, device=device)

        active_mask = ~finished
        if not active_mask.any():
            break

        active_input = current_input[active_mask]
        active_attention_mask = current_attention_mask[active_mask]

        with torch.no_grad():
            if past_key_values is not None:
                active_past = []
                for layer_past in past_key_values:
                    active_past.append((
                        layer_past[0][active_mask] if layer_past[0] is not None else None,
                        layer_past[1][active_mask] if layer_past[1] is not None else None,
                    ))
            else:
                active_past = None

            outputs = ctx.target(
                active_input,
                attention_mask=active_attention_mask,
                past_key_values=active_past,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]

            if outputs.past_key_values is not None:
                if past_key_values is None:
                    past_key_values = []
                    for layer_kv in outputs.past_key_values:
                        key_shape = list(layer_kv[0].shape); value_shape = list(layer_kv[1].shape)
                        key_shape[0] = batch_size; value_shape[0] = batch_size
                        full_key = torch.zeros(key_shape, device=device, dtype=layer_kv[0].dtype)
                        full_value = torch.zeros(value_shape, device=device, dtype=layer_kv[1].dtype)
                        full_key[active_mask] = layer_kv[0]
                        full_value[active_mask] = layer_kv[1]
                        past_key_values.append((full_key, full_value))
                else:
                    for i, layer_kv in enumerate(outputs.past_key_values):
                        past_key_values[i][0][active_mask] = layer_kv[0]
                        past_key_values[i][1][active_mask] = layer_kv[1]

        # Greedy by default (ctx.processor may have temperature)
        if hasattr(ctx.processor, 'temperature') and ctx.processor.temperature > 0:
            probs = torch.softmax(logits / ctx.processor.temperature, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1).to(device)  # 确保在正确设备上
        else:
            next_tokens = torch.argmax(logits, dim=-1).to(device)  # 确保在正确设备上

        active_indices = torch.where(active_mask)[0]
        for i, active_idx in enumerate(active_indices):
            token = next_tokens[i].item()
            generated_tokens[active_idx, step] = token
            
            # Record first token time for TTFT calculation
            if first_token_callback is not None and step == 0:
                first_token_callback(active_idx.item())
            
            if token in ctx.end_tokens:
                finished[active_idx] = True

    batch_outputs: List[torch.Tensor] = []
    for i in range(batch_size):
        gen_seq = generated_tokens[i]
        nonzero_indices = torch.nonzero(gen_seq, as_tuple=True)[0]
        if len(nonzero_indices) > 0:
            last_token_idx = nonzero_indices[-1].item() + 1
            final_generated = gen_seq[:last_token_idx]
        else:
            final_generated = torch.tensor([], dtype=torch.long, device=device)
        original_length = torch.sum(attention_mask[i]).item()
        original_input = input_ids[i][:original_length]
        full_output = torch.cat([original_input, final_generated])
        batch_outputs.append(full_output)

    return batch_outputs


