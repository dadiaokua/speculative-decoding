from typing import List, Tuple, Optional
from termcolor import colored
import time
import torch

from .batch_decode import decode_batch_with_chat_template
from .metrics import BatchMetrics, RequestMetrics


def infer_batch(ctx, prompts: List[str]) -> Tuple[Optional[BatchMetrics], Optional[BatchMetrics]]:
    """Run batch inference given raw prompts using the ctx object.

    Returns:
        Tuple of (speculative_metrics, target_metrics) - one may be None if disabled
    """
    if ctx.chat:
        formatted_prompts = [
            ctx.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            for p in prompts
        ]
    else:
        formatted_prompts = prompts

    input_ids, attention_mask = decode_batch_with_chat_template(
        ctx.tokenizer, formatted_prompts, max_length=ctx.max_batch_length, chat=False
    )

    if ctx.reset_in_between and ctx.ngram is not None:
        ctx.ngram.reset()

    batch_size = len(prompts)
    spec_metrics = None
    target_metrics = None

    if ctx.spec:
        print(colored("🚀 Running Speculative Decoding on batch...", "green"))
        spec_metrics = run_batch_speculative(ctx, input_ids, attention_mask, batch_size)

    if ctx.target_gen:
        print(colored("🎯 Running Target AR on batch...", "blue"))
        target_metrics = run_batch_target(ctx, input_ids, attention_mask, batch_size)

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
    """True batch speculative generation implementation (engine version).

    Uses:
    - ctx.drafter, ctx.target, ctx.end_tokens, ctx.gamma, ctx.gen_len, ctx.debug
    """
    batch_outputs: List[torch.Tensor] = []
    batch_accept_rates: List[float] = []

    device = input_ids.device
    generated_tokens = torch.zeros(batch_size, ctx.gen_len, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    drafts_generated_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)
    drafts_accepted_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Initialize drafter cache with full prompt
    drafter_past = None
    with torch.no_grad():
        init_out = ctx.drafter(input_ids, attention_mask=attention_mask, use_cache=True)
        drafter_past = init_out.past_key_values

    step = 0
    while step < ctx.gen_len:
        if finished.all():
            break

        remaining = ctx.gen_len - step
        current_gamma = min(ctx.gamma, remaining)

        draft_tokens = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.long)
        drafter_sampled_probs = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.float32)
        q_probs_full = torch.zeros(batch_size, current_gamma, ctx.target.config.vocab_size, device=device, dtype=torch.float32)

        for draft_step in range(current_gamma):
            if finished.all():
                break

            # Build full-batch single-token continuation
            if draft_step == 0:
                token_prev = generated_tokens[:, step-1] if step > 0 else input_ids[:, -1]
            else:
                token_prev = generated_tokens[:, step + draft_step - 1]
            current_input = token_prev.unsqueeze(1)

            with torch.no_grad():
                out = ctx.drafter(current_input, past_key_values=drafter_past, use_cache=True)
                logits = out.logits[:, -1, :]
                q_probs = torch.softmax(logits, dim=-1)

            drafter_past = out.past_key_values

            samples_all = torch.multinomial(q_probs, 1).squeeze(-1)
            q_probs_full[:, draft_step, :] = q_probs

            active_mask = ~finished
            if active_mask.any():
                draft_tokens[active_mask, draft_step] = samples_all[active_mask]
                drafter_sampled_probs[active_mask, draft_step] = q_probs[active_mask, :].gather(
                    1, samples_all[active_mask].unsqueeze(1)
                ).squeeze(1)
                generated_tokens[active_mask, step + draft_step] = samples_all[active_mask]
                drafts_generated_per_seq[active_mask] += 1
                
                # Record first token time for TTFT calculation
                if first_token_callback is not None and draft_step == 0 and step == 0:
                    for idx in torch.where(active_mask)[0]:
                        first_token_callback(idx.item())

        active_mask = ~finished
        if active_mask.any():
            verify_ids = torch.cat([input_ids, generated_tokens[:, :step + current_gamma]], dim=1)
            with torch.no_grad():
                t_out = ctx.target(verify_ids)
                t_logits_full = t_out.logits[:, -(current_gamma+1):-1, :]
                p_probs_full = torch.softmax(t_logits_full, dim=-1)

            for global_idx in torch.where(active_mask)[0]:
                local_row = global_idx.item()
                if finished[global_idx]:
                    continue
                accepted_count = 0

                for draft_idx in range(current_gamma):
                    if finished[global_idx]:
                        break

                    sampled_token = draft_tokens[global_idx, draft_idx].item()
                    q_vec = q_probs_full[global_idx, draft_idx]
                    p_vec = p_probs_full[local_row, draft_idx]

                    p_sample = p_vec[sampled_token].item()
                    q_sample = q_vec[sampled_token].item()

                    accept_prob = 1.0 if q_sample <= 0.0 else min(1.0, p_sample / q_sample)
                    if torch.rand(1, device=device).item() < accept_prob:
                        accepted_count += 1
                        drafts_accepted_per_seq[global_idx] += 1
                        if sampled_token in ctx.end_tokens:
                            finished[global_idx] = True
                            break
                    else:
                        residual = torch.clamp(p_vec - torch.minimum(p_vec, q_vec), min=0.0)
                        denom = residual.sum().item()
                        if denom <= 1e-12:
                            corrected = torch.multinomial(p_vec, 1).item()
                        else:
                            corrected = torch.multinomial(residual / denom, 1).item()
                        generated_tokens[global_idx, step + draft_idx] = corrected
                        if corrected in ctx.end_tokens:
                            finished[global_idx] = True
                        break

                if accepted_count < current_gamma:
                    tail_start = step + accepted_count + 1
                    if tail_start < step + current_gamma:
                        generated_tokens[global_idx, tail_start: step + current_gamma] = 0

        step += current_gamma

    for i in range(batch_size):
        gen_seq = generated_tokens[i]
        nonzero = torch.nonzero(gen_seq, as_tuple=True)[0]
        if nonzero.numel() > 0:
            last = nonzero[-1].item() + 1
            final_gen = gen_seq[:last]
        else:
            final_gen = torch.tensor([], dtype=torch.long, device=device)
        full_output = torch.cat([input_ids[i], final_gen])
        batch_outputs.append(full_output)

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
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

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


