import argparse
import random
import numpy as np
import torch
from sampling import autoregressive_generate, speculative_generate
from ngram_assisted import OneLevelNGramStorage, NGramStorage, ngram_assisted_speculative_generate
from engine.dataset import load_sharegpt_prompts, load_sharegpt_multi
from engine.models import load_causal_lm
from engine.infer_engine import infer_batch
from utils.logits_processor import GreedyProcessor, MultinomialProcessor, TopKProcessor, NucleusProcessor, TopKNucleusProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
import os
import json
from termcolor import colored


class InferenceCLI:

    def __init__(self, device: str = "cuda"):
        print(
            colored("Speculative Decoding", "red"),
            colored("CLI", on_color="on_red", color="white"),
            "\n",
        )
        self.device = device

        self.gamma = 4
        self.gen_len = 35
        self.debug = False
        self.spec = True
        self.dr = False
        self.cache = False
        self.target_gen = True
        # Ngram Assisted Generation
        self.ngram_gen = True
        self.ngram = None
        self.top_k_filler = 3
        self.ngram_n = 3
        self.reset_in_between = True
        
        self.chat = True # If using a chat instructed model, set to True
        
        # ShareGPT configuration from environment variables
        self.sharegpt_dir = os.getenv(
            "SHAREGPT_DIR",
            "/Users/myrick/GithubProjects/Speculative-Decoding/sharegpt_gpt4",
        )
        self.sharegpt_paths = [
            os.path.join(self.sharegpt_dir, "sharegpt_gpt4.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_V3_format.jsonl"),
            os.path.join(self.sharegpt_dir, "sharegpt_zh_38K_format.jsonl"),
        ]
        self.sharegpt_data = None
        
        # Auto mode configuration
        self.auto_mode = os.getenv("AUTO_MODE", "false").lower() == "true"
        self.auto_rate = float(os.getenv("AUTO_RATE", "1.0"))
        self.auto_duration = float(os.getenv("AUTO_DURATION", "300"))
        self.prompt_min_length = int(os.getenv("PROMPT_MIN_LENGTH", "10"))
        self.prompt_max_length = int(os.getenv("PROMPT_MAX_LENGTH", "500"))
        self.max_load_lines = int(os.getenv("MAX_LOAD_LINES", "10000"))
        
        # Override default parameters with environment variables
        self.gamma = int(os.getenv("GAMMA_VALUE", "4"))
        self.gen_len = int(os.getenv("GENERATION_LENGTH", "100"))
        self.spec = os.getenv("ENABLE_SPECULATIVE", "true").lower() == "true"
        self.target_gen = os.getenv("ENABLE_TARGET", "true").lower() == "true"
        self.ngram_gen = os.getenv("ENABLE_NGRAM", "true").lower() == "true"
        self.debug = os.getenv("ENABLE_DEBUG", "false").lower() == "true"
        
        # Batch processing configuration
        self.enable_batch = os.getenv("ENABLE_BATCH", "false").lower() == "true"
        self.batch_size = int(os.getenv("BATCH_SIZE", "4"))
        self.max_batch_length = int(os.getenv("MAX_BATCH_LENGTH", "512"))
        
        self.processors = {
            "greedy": {
                "processor": GreedyProcessor,
                "building_args": {"temperature": float},
            },
            "multinomial": {
                "processor": MultinomialProcessor,
                "building_args": {"temperature": float},
            },
            "topk": {
                "processor": TopKProcessor,
                "building_args": {"temperature": float, "top_k": int},
            },
            "nucleus": {
                "processor": NucleusProcessor,
                "building_args": {"temperature": float, "top_p": float},
            },
            "topknucleus": {
                "processor": TopKNucleusProcessor,
                "building_args": {"temperature": float, "top_k": int, "top_p": float},
            },
        }
        self.selected_processor = {
            "name": "greedy",
            "processor": GreedyProcessor,
            "args": {"temperature": 1.0},
        }
        self.processor = GreedyProcessor()

        self._load_models()
        self._load_sharegpt_data()
        
        if self.auto_mode:
            self._run_auto_mode()
        else:
        self._run()

    def _load_models(self):
        # Target model (larger model for better quality)
        target_model = "/home/llm/model_hub/Qwen3-8B"
        target_quantize = None  # No quantization, use precision specified by torch_dtype
        
        # Drafter model (smaller model for speed)
        drafter_model = "/home/llm/model_hub/Qwen3-1.7B"
        drafter_quantize = None  # No quantization, use precision specified by torch_dtype
        
        # Precision options: torch.float16 (FP16/half), torch.bfloat16 (BF16), torch.float32 (FP32)
        model_dtype = torch.float16  # Using FP16 (half precision)
        
        # GPU allocation configuration - read from environment variables
        target_gpu_env = os.getenv("TARGET_GPU", "cuda:0")
        drafter_gpu_env = os.getenv("DRAFTER_GPU", "cuda:0")
        
        # Parse multi-GPU configuration
        def parse_device_map(gpu_string):
            if gpu_string == "auto":
                return "auto"
            elif "," in gpu_string:
                # Multi-GPU: "cuda:0,cuda:1,cuda:2" -> {"": [0,1,2]} for auto-split
                gpu_ids = []
                for gpu in gpu_string.split(","):
                    if gpu.startswith("cuda:"):
                        gpu_ids.append(int(gpu.split(":")[1]))
                return {"": gpu_ids}  # Let transformers auto-distribute layers
            else:
                # Single GPU: "cuda:0" -> "cuda:0"
                return gpu_string
        
        target_device_map = parse_device_map(target_gpu_env)
        drafter_device_map = parse_device_map(drafter_gpu_env)
        
        # Alternative: Manual configuration (uncomment if you want to override env vars)
        # target_device_map = "cuda:0"   # Force Target model to GPU 0
        # drafter_device_map = "cuda:1"  # Force Drafter model to GPU 1
        # target_device_map = "auto"     # Let transformers auto-assign

        print(colored("Target model:", on_color="on_yellow"), target_model)
        print(colored("Drafter model:", on_color="on_yellow"), drafter_model)
        print(colored("Model precision:", on_color="on_yellow"), str(model_dtype).split('.')[-1])
        print(colored("Target GPU:", on_color="on_yellow"), target_device_map)
        print(colored("Drafter GPU:", on_color="on_yellow"), drafter_device_map)
        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=target_quantize,
            torch_dtype=model_dtype,  # Configurable precision
            device_map=target_device_map,  # Specific GPU allocation
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.target.eval()

        tokenizer_name = target_model
        if tokenizer_name != target_model:
            print(colored("Warning: Tokenizer is different from target model. Use with caution.", "red"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Set pad token for Qwen models if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            quantization_config=drafter_quantize,
            torch_dtype=model_dtype,  # Configurable precision
            device_map=drafter_device_map,  # Specific GPU allocation
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        self.ngram = NGramStorage(n=3, vocab_size=self.target.config.vocab_size)
        
        # End tokens - adapt for Qwen models
        self.end_tokens = [self.tokenizer.eos_token_id]
        # Try to add Qwen-specific end tokens if they exist
        try:
            qwen_end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if qwen_end_token is not None and qwen_end_token != self.tokenizer.unk_token_id:
                self.end_tokens.append(qwen_end_token)
        except:
            pass  # If token doesn't exist, just use eos_token_id

    def _load_sharegpt_data(self):
        """Load ShareGPT data for random prompt selection"""
        try:
            parts = load_sharegpt_multi(
                self.sharegpt_paths,
                max_lines=self.max_load_lines,
                min_len=self.prompt_min_length,
                max_len=self.prompt_max_length,
            )
            # Flatten but also keep parts for proportional sampling later
            self.sharegpt_parts = parts
            flat = []
            for p in parts:
                flat.extend(p)
            self.sharegpt_data = flat if flat else None
        except Exception as e:
            print(colored(f"Error loading ShareGPT data: {e}", "red"))
            self.sharegpt_data = None

    def _get_random_prompt(self):
        """Get a random prompt from ShareGPT data"""
        if not self.sharegpt_data:
            return "Tell me a story about artificial intelligence."  # Fallback prompt
        # Proportional sampling across three files
        if hasattr(self, "sharegpt_parts") and self.sharegpt_parts:
            non_empty = [p for p in self.sharegpt_parts if p]
            if not non_empty:
                return random.choice(self.sharegpt_data)
            # Uniform over files, then uniform within file (equal proportion)
            chosen_part = random.choice(non_empty)
            return random.choice(chosen_part)
        return random.choice(self.sharegpt_data)

    def _run_auto_mode(self):
        """Run in automatic mode with random prompts from ShareGPT"""
        print(colored("🤖 Running in AUTO MODE", "magenta", attrs=["bold"]))
        print(colored(f"Target rate: {self.auto_rate:.2f} prompts/s", "yellow"))
        print(colored(f"Duration: {self.auto_duration:.1f} s", "yellow"))
        if self.enable_batch:
            print(colored(f"📦 Batch mode: {self.batch_size} prompts per batch", "yellow"))
        print(colored("=" * 60, "magenta"))

        if not self.sharegpt_data:
            print(colored("❌ No ShareGPT data available, cannot run auto mode", "red"))
            return
        
        if self.auto_duration <= 0:
            print(colored("⚠️ AUTO_DURATION must be > 0", "yellow"))
            return

        if self.auto_rate <= 0:
            print(colored("⚠️ AUTO_RATE must be > 0", "yellow"))
            return

        start_time = time.time()
        end_time = start_time + max(0.0, self.auto_duration)
        processed = 0
        
        while True:
            now = time.time()
            if now >= end_time:
                break
            remaining = end_time - now
            if remaining <= 0:
                break
            
            if self.enable_batch:
                prompts = [self._get_random_prompt() for _ in range(self.batch_size)]
                print(colored(f"\n📦 Batch starting at t={now - start_time:.1f}s", "magenta", attrs=["bold"]))
                infer_batch(self, prompts)
                processed += len(prompts)
            else:
                prompt = self._get_random_prompt()
                print(colored(f"\n🎲 Prompt {processed + 1}", "magenta", attrs=["bold"]))
                print(colored(f"'{prompt}'", "cyan"))
                print(colored("-" * 60, "magenta"))
                try:
                    self._infer(prompt)
                except Exception as e:
                    print(colored(f"❌ Error during inference: {e}", "red"))
                    continue
                processed += 1
                print(colored("=" * 60, "magenta"))

            if self.auto_rate > 0:
                elapsed = time.time() - start_time
                expected = processed / self.auto_rate
                sleep_time = expected - elapsed
                if sleep_time > 0:
                    time.sleep(min(sleep_time, max(0.0, end_time - time.time())))
        
        total_elapsed = time.time() - start_time
        print(colored(f"✅ Auto mode finished: processed {processed} prompts in {total_elapsed:.1f}s", "green", attrs=["bold"]))

    def _run_batch_speculative(self, input_ids, attention_mask, batch_size):
        """Run speculative decoding on batch - true batch implementation"""
        print(colored("🚀 Running TRUE batch speculative decoding...", "green"))
        
        try:
            self._set_seed(42)
            start_time = time.time()
            
            # Run batch speculative generation
            batch_outputs, batch_accept_rates = self._batch_speculative_generate(
                input_ids, attention_mask, batch_size
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Decode and display results
            total_tokens = 0
            for i, (output_ids, accept_rate) in enumerate(zip(batch_outputs, batch_accept_rates)):
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                total_tokens += len(output_ids) - len(input_ids[i])  # Only count generated tokens
                
                print(colored(f"  Batch item {i+1}: Acceptance rate: {accept_rate:.3f}", "green"))
                if self.debug:
                    print(colored(f"    Output: {output[:100]}{'...' if len(output) > 100 else ''}", "cyan"))
            
            batch_throughput = total_tokens / total_time
            print(colored(f"📊 Batch throughput: {batch_throughput:.1f} tokens/s "
                         f"({total_tokens} tokens in {total_time:.2f}s)", "green", attrs=["bold"]))
            
        except Exception as e:
            print(colored(f"❌ Batch speculative decoding failed: {e}", "red"))

    def _batch_speculative_generate(self, input_ids, attention_mask, batch_size):
        """True batch speculative generation implementation with:
        - Correct acceptance rule using p_target / p_draft (<= 1)
        - Drafter KV cache for fast draft generation
        - Per-sequence acceptance-rate tracking
        - Residual resampling on rejection (r ∝ max(p - q, 0))
        """
        batch_outputs: list[torch.Tensor] = []
        batch_accept_rates: list[float] = []

        device = input_ids.device
        generated_tokens = torch.zeros(batch_size, self.gen_len, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        drafts_generated_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)
        drafts_accepted_per_seq = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Initialize drafter cache with full prompt
        drafter_past = None
        with torch.no_grad():
            init_out = self.drafter(input_ids, attention_mask=attention_mask, use_cache=True)
            drafter_past = init_out.past_key_values

        step = 0
        while step < self.gen_len:
            if finished.all():
                break

            remaining = self.gen_len - step
            current_gamma = min(self.gamma, remaining)

            draft_tokens = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.long)
            drafter_sampled_probs = torch.zeros(batch_size, current_gamma, device=device, dtype=torch.float32)
            # q_probs_full[b, d, v]: drafter prob distribution for batch row b, draft position d
            q_probs_full = torch.zeros(batch_size, current_gamma, self.target.config.vocab_size, device=device, dtype=torch.float32)

            for draft_step in range(current_gamma):
                if finished.all():
                    break

                # Build a full-batch single-token continuation input to keep KV cache aligned
                if draft_step == 0:
                    token_prev = generated_tokens[:, step-1] if step > 0 else input_ids[:, -1]
                else:
                    token_prev = generated_tokens[:, step + draft_step - 1]
                current_input = token_prev.unsqueeze(1)  # [batch, 1]

                with torch.no_grad():
                    out = self.drafter(current_input, past_key_values=drafter_past, use_cache=True)
                    logits = out.logits[:, -1, :]
                    q_probs = torch.softmax(logits, dim=-1)  # [batch, vocab]

                # Update past for full batch directly (no slicing/merging)
                drafter_past = out.past_key_values

                # Sample drafts for all rows, but only record unfinished ones
                samples_all = torch.multinomial(q_probs, 1).squeeze(-1)  # [batch]
                # Save full q into preallocated tensor
                q_probs_full[:, draft_step, :] = q_probs

                # Apply to unfinished rows
                active_mask = ~finished
                if active_mask.any():
                    draft_tokens[active_mask, draft_step] = samples_all[active_mask]
                    drafter_sampled_probs[active_mask, draft_step] = q_probs[active_mask, :].gather(
                        1, samples_all[active_mask].unsqueeze(1)
                    ).squeeze(1)
                    generated_tokens[active_mask, step + draft_step] = samples_all[active_mask]
                    drafts_generated_per_seq[active_mask] += 1

            # Verify with target
            active_mask = ~finished
            if active_mask.any():
                verify_ids = torch.cat([input_ids, generated_tokens[:, :step + current_gamma]], dim=1)
                with torch.no_grad():
                    t_out = self.target(verify_ids)
                    t_logits_full = t_out.logits[:, -(current_gamma+1):-1, :]
                    p_probs_full = torch.softmax(t_logits_full, dim=-1)  # [batch, current_gamma, vocab]

                for global_idx in torch.where(active_mask)[0]:
                    local_row = global_idx.item()
                    if finished[global_idx]:
                        continue
                    accepted_count = 0

                    for draft_idx in range(current_gamma):
                        if finished[global_idx]:
                            break

                        sampled_token = draft_tokens[global_idx, draft_idx].item()
                        # Directly read q/p for this batch row and draft position
                        q_vec = q_probs_full[global_idx, draft_idx]
                        p_vec = p_probs_full[local_row, draft_idx]
                        p_sample = p_vec[sampled_token].item()
                        q_sample = q_vec[sampled_token].item()

                        accept_prob = 1.0 if q_sample <= 0.0 else min(1.0, p_sample / q_sample)
                        if torch.rand(1, device=device).item() < accept_prob:
                            accepted_count += 1
                            drafts_accepted_per_seq[global_idx] += 1
                            if sampled_token in self.end_tokens:
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
                            if corrected in self.end_tokens:
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

    def _run_batch_target(self, input_ids, attention_mask, batch_size):
        """Run target model on batch - true batch implementation"""
        print(colored("🎯 Running TRUE batch target generation...", "blue"))
        
        try:
            self._set_seed(42)
            start_time = time.time()
            
            # Run batch autoregressive generation
            batch_outputs = self._batch_autoregressive_generate(input_ids, attention_mask, batch_size)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Decode and display results
            total_tokens = 0
            for i, output_ids in enumerate(batch_outputs):
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                generated_tokens = len(output_ids) - len(input_ids[i])
                total_tokens += generated_tokens
                
                print(colored(f"  Batch item {i+1}: Generated {generated_tokens} tokens", "blue"))
                if self.debug:
                    print(colored(f"    Output: {output[:100]}{'...' if len(output) > 100 else ''}", "cyan"))
            
            batch_throughput = total_tokens / total_time
            print(colored(f"📊 Batch throughput: {batch_throughput:.1f} tokens/s "
                         f"({total_tokens} tokens in {total_time:.2f}s)", "blue", attrs=["bold"]))
            
        except Exception as e:
            print(colored(f"❌ Batch target generation failed: {e}", "red"))

    def _batch_autoregressive_generate(self, input_ids, attention_mask, batch_size):
        """True batch autoregressive generation implementation"""
        device = input_ids.device
        max_input_len = input_ids.size(1)
        
        # Initialize generation
        generated_tokens = torch.zeros(batch_size, self.gen_len, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        current_lengths = torch.sum(attention_mask, dim=1)
        
        # Use KV cache for efficiency
        past_key_values = None
        
        for step in range(self.gen_len):
            if finished.all():
                break
            
            # Prepare input for current step
            if step == 0:
                # First step: use original input
                current_input = input_ids
                current_attention_mask = attention_mask
            else:
                # Subsequent steps: only use the last generated token
                current_input = generated_tokens[:, step-1:step]
                current_attention_mask = torch.ones_like(current_input, device=device)
            
            # Only process unfinished sequences
            active_mask = ~finished
            if not active_mask.any():
                break
            
            active_input = current_input[active_mask]
            active_attention_mask = current_attention_mask[active_mask]
            
            # Run target model
            with torch.no_grad():
                if past_key_values is not None:
                    # Use only active past_key_values
                    active_past_key_values = []
                    for layer_past in past_key_values:
                        # Each layer_past is (key, value) tuple
                        active_layer_past = (
                            layer_past[0][active_mask] if layer_past[0] is not None else None,
                            layer_past[1][active_mask] if layer_past[1] is not None else None
                        )
                        active_past_key_values.append(active_layer_past)
                else:
                    active_past_key_values = None
                
                outputs = self.target(
                    active_input,
                    attention_mask=active_attention_mask,
                    past_key_values=active_past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]  # Last token logits
                
                # Update past_key_values for next iteration
                if outputs.past_key_values is not None:
                    if past_key_values is None:
                        # Initialize past_key_values for all sequences
                        past_key_values = []
                        for layer_kv in outputs.past_key_values:
                            # Create tensors for all batch items
                            key_shape = list(layer_kv[0].shape)
                            value_shape = list(layer_kv[1].shape)
                            key_shape[0] = batch_size
                            value_shape[0] = batch_size
                            
                            full_key = torch.zeros(key_shape, device=device, dtype=layer_kv[0].dtype)
                            full_value = torch.zeros(value_shape, device=device, dtype=layer_kv[1].dtype)
                            
                            # Fill in the active sequences
                            full_key[active_mask] = layer_kv[0]
                            full_value[active_mask] = layer_kv[1]
                            
                            past_key_values.append((full_key, full_value))
                    else:
                        # Update existing past_key_values
                        for i, layer_kv in enumerate(outputs.past_key_values):
                            past_key_values[i][0][active_mask] = layer_kv[0]
                            past_key_values[i][1][active_mask] = layer_kv[1]
            
            # Sample next tokens
            if hasattr(self.processor, 'temperature') and self.processor.temperature > 0:
                probs = torch.softmax(logits / self.processor.temperature, dim=-1)
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(logits, dim=-1)
            
            # Update generated tokens for active sequences
            active_indices = torch.where(active_mask)[0]
            for i, active_idx in enumerate(active_indices):
                token = next_tokens[i].item()
                generated_tokens[active_idx, step] = token
                
                # Check for EOS tokens
                if token in self.end_tokens:
                    finished[active_idx] = True
        
        # Prepare final outputs
        batch_outputs = []
        for i in range(batch_size):
            # Find the actual end of generated sequence
            generated_seq = generated_tokens[i]
            
            # Find first zero (padding) or use full sequence
            nonzero_indices = torch.nonzero(generated_seq, as_tuple=True)[0]
            if len(nonzero_indices) > 0:
                last_token_idx = nonzero_indices[-1].item() + 1
                final_generated = generated_seq[:last_token_idx]
            else:
                final_generated = torch.tensor([], dtype=torch.long, device=device)
            
            # Combine input and generated tokens
            original_length = torch.sum(attention_mask[i]).item()
            original_input = input_ids[i][:original_length]
            full_output = torch.cat([original_input, final_generated])
            batch_outputs.append(full_output)
        
        return batch_outputs

    def _perform_command(self, command: str):
        args = command.split(" ")
        if args[0] == "/quit":
            print(colored("Goodbye!", on_color="on_red"))
            exit(0)
        if args[0] == "/debug":
            self.debug = not self.debug
            print(colored(f"Debug mode: {self.debug}", on_color="on_blue"))
            return
        if args[0] == "/speculative":
            self.spec = not self.spec
            print(colored(f"Speculative Decoding generation: {self.spec}", on_color="on_blue"))
            return
        if args[0] == "/drafter":
            self.dr = not self.dr
            print(colored(f"Drafter generation: {self.dr}", on_color="on_blue"))
            return
        if args[0] == "/cache":
            self.cache = not self.cache
            print(colored(f"Cache: {self.cache}", on_color="on_blue"))
            if self.cache:
                print(colored("Warning, cache feature is very unstable accross different models. It might generate errors or just perturb the generation. Use with caution.", "red"))
            return
        if args[0] == "/target":
            self.target_gen = not self.target_gen
            print(colored(f"Target generation: {self.target_gen}", on_color="on_blue"))
            return
        if args[0] == "/chat":
            self.chat = not self.chat
            print(colored(f"Chat mode: {self.chat}", on_color="on_blue"))
            return
        if args[0] == "/length":
            if len(args) < 2:
                print(colored("Usage: /length <value>", "red"))
                return
            self.gen_len = int(args[1])
            print(colored(f"Generation length: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/gamma":
            if len(args) < 2:
                print(colored("Usage: /gamma <value>", "red"))
                return
            self.gamma = int(args[1])
            print(colored(f"Gamma: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            return
        if args[0] == "/processor":
            # /processor <processor_name> <args0> <args1> ...
            if len(args) < 2:
                print(colored("Usage: /processor <processor_name> <args0> <args1> ...", "red"))
                return
            processor_name = args[1]
            if processor_name not in self.processors:
                print(colored("Invalid processor name", "red"))
                print(colored("Available processors:", "red"))
                for processor in self.processors.keys():
                    print(colored(f"\t{processor}", "red"))
                return
            processor = self.processors[processor_name]
            print(colored(f"Selected processor: {processor_name}", "blue"))
            building_args = processor["building_args"]
            args = args[2:]
            processor_args = {}
            for arg_name, arg_type in building_args.items():
                if len(args) == 0:
                    print(colored(f"Missing argument {arg_name}", "red"))
                    return
                try:
                    processor_args[arg_name] = arg_type(args[0])
                    print(colored(f"\t{arg_name}: {arg_type(args[0])}", "blue"))
                except ValueError:
                    print(colored(f"Invalid argument {arg_name} of type {arg_type}", "red"))
                    return
                args = args[1:]
            self.selected_processor = {
                "name": processor_name,
                "processor": processor["processor"],
                "args": processor_args,
            }
            self.processor = processor["processor"](**processor_args)
            return
        # Ngram Assisted Generation
        if args[0] == "/ngram":
            self.ngram_gen = not self.ngram_gen
            print(colored(f"Ngram assisted generation: {self.ngram_gen}", on_color="on_blue"))
            return
        if args[0] == "/top_k_filler":
            if len(args) < 2:
                print(colored("Usage: /top_k_filler <value>", "red"))
                return
            self.top_k_filler = int(args[1])
            print(colored(f"Top k filler: {int(args[1])}", on_color="on_blue"))
            return
        if args[0] == "/set_ngramstorage":
            if len(args) < 3:
                print(colored("Usage: /set_ngramstorage <basic/onelevel> <n>", "red"))
                return
            if args[1] == "onelevel":
                ntype = OneLevelNGramStorage
            elif args[1] == "basic":
                ntype = NGramStorage
            else:
                print(colored("Invalid ngram type", "red"))
                return
            self.ngram = ntype(n=int(args[2]), vocab_size=self.target.config.vocab_size)
            self.ngram_n = int(args[2])
            print(colored(f"Ngram type: {args[1]}", "blue"))
            print(colored(f"Ngram n: {int(args[2])}", "blue"))
            return
        if args[0] == "/reset_in_between":
            self.reset_in_between = not self.reset_in_between
            print(colored(f"Reset ngram in between each generation: {self.reset_in_between}", on_color="on_blue"))
            return
        if args[0] == "/random":
            # Get a random prompt from ShareGPT and run inference
            random_prompt = self._get_random_prompt()
            print(colored("🎲 Random prompt from ShareGPT:", "magenta"))
            print(colored(f"'{random_prompt}'", "cyan"))
            print(colored("=" * 50, "magenta"))
            self._infer(random_prompt)
            return
        print(colored("Unknown command", "red"))
        self._help()

    def _help(self):
        print(colored("Commands:", on_color="on_blue"))
        print("/quit: quit the program")
        print("/debug: toggle speculative debug mode")
        print(colored(f"\t{self.debug}", "green" if self.debug else "red"))
        print("/clear: clear the screen")
        print("/speculative: toggle speculative decoding")
        print(colored(f"\t{self.spec}", "green" if self.spec else "red"))
        print("/target: toggle target generation")
        print(colored(f"\t{self.target_gen}", "green" if self.target_gen else "red"))
        print("/drafter: toggle drafter generation")
        print(colored(f"\t{self.dr}", "green" if self.dr else "red"))
        print("/cache: toggle cache")
        print(colored(f"\t{self.cache}", "green" if self.cache else "red"))
        print("/chat: toggle chat mode")
        print(colored(f"\t{self.chat}", "green" if self.chat else "red"))
        print("/length <value>: set generation length")
        print(colored(f"\t{self.gen_len}", "blue"))
        print("/gamma <value>: set gamma")
        print(colored(f"\t{self.gamma}", "blue"))
        print("/processor <processor_name> [args0] [args1] ...: set processor")
        print(colored(f"\t{self.selected_processor['name']}", "blue"))
        for arg_name, arg_value in self.selected_processor["args"].items():
            print(colored(f"\t\t{arg_name}: {arg_value}", "blue"))
        # Ngram Assisted Generation
        print("/ngram: toggle ngram assisted generation")
        print(colored(f"\t{self.ngram_gen}", "green" if self.ngram_gen else "red"))
        print("/top_k_filler <value>: set top k filler for ngram update")
        print(colored(f"\t{self.top_k_filler}", "blue"))
        print("/set_ngramstorage <basic/onelevel> <n>: set ngramstorage drafter")
        print(colored(f"\t{self.ngram.__class__.__name__} {self.ngram_n}", "blue"))
        print("/reset_in_between: toggle reset ngram in between each generation")
        print(colored(f"\t{self.reset_in_between}", "green" if self.reset_in_between else "red"))
        print("/random: get a random prompt from ShareGPT and run inference")
        if self.sharegpt_data:
            print(colored(f"\t{len(self.sharegpt_data)} prompts available", "green"))
        else:
            print(colored(f"\tShareGPT data not available", "red"))
        

    def _infer(self, prefix: str):
        if self.chat:
            prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": prefix}], add_generation_prompt=True, tokenize=False)
            
        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
        
        if self.reset_in_between:
            self.ngram.reset()
        
        spec_throughput = 0.0
        base_throughput = 0.0
        drafter_throughput = 0.0

        if self.spec:
            self._set_seed(42)
            spec_start_time = time.time()
            output_ids, accept_rate = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )
            spec_end_time = time.time()
            spec_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Speculative ==========", "green"))
            print(colored("Out:", "green"), spec_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))
            spec_throughput = len(spec_output) / (spec_end_time - spec_start_time)
            print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
            print(colored("========== Speculative ==========", "green"))
            
        if self.ngram_gen:
            self._set_seed(42)
            ngram_start_time = time.time()
            output_ids, accept_rate = ngram_assisted_speculative_generate(
                tokenized,
                self.ngram,
                self.target,
                tokenizer=self.tokenizer,
                gamma=self.gamma,
                filler_top_k=self.top_k_filler,
                logits_processor=self.processor,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
                first_target=True,
                stop_if_unknown=True,
            )
            ngram_end_time = time.time()
            ngram_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("========== Ngram Assisted ==========", "yellow"))
            print(colored("Out:", "yellow"), ngram_output)
            print(colored(f"Acceptance rate: {accept_rate:.3f}", "yellow"))
            ngram_throughput = len(ngram_output) / (ngram_end_time - ngram_start_time)
            print(colored(f"Throughput: {ngram_throughput:.1f} tokens/s", "yellow"))
            print(colored("========== Ngram Assisted ==========", "yellow"))
            if self.spec and ngram_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput / ngram_throughput)) * 100:.1f}%", "magenta"))

        if self.target_gen:
            self._set_seed(42)
            start_time = time.time()
            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            end_time = time.time()
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(colored("=========== Target AR ===========", "blue"))
            print(colored("Out:", "blue"), output)
            base_throughput = len(output) / (end_time - start_time)
            print(colored(f"Throughput: {base_throughput:.1f} tokens/s", "blue"))
            print(colored("=========== Target AR ===========", "blue"))
            if self.spec and base_throughput > 0.0:
                print(colored(f"Throughput increase: {((spec_throughput / base_throughput)) * 100:.1f}%", "magenta"))

        if self.dr:
            self._set_seed(42)
            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            print(colored("========== Drafter AR ==========", "cyan"))
            drafter_throughput = len(output) / (end_time - start_time)
            print(colored("Out:", "cyan"), output)
            print(colored(f"Throughput: {drafter_throughput:.1f} tokens/s", "cyan"))
            print(colored("========== Drafter AR ==========", "cyan"))

    def _run(self):
        while True:
            command = input("> ").replace('\\n', '\n').replace('\\t', '\t')
            if command.startswith("/"):
                self._perform_command(command)
                continue

            self._infer(command)
            
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding CLI")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    args = parser.parse_args()

    InferenceCLI(device=args.device)
