from typing import List, Tuple
import torch
from termcolor import colored


def decode_batch_with_chat_template(tokenizer, prompts: List[str], max_length: int, chat: bool = True):
    """Apply chat template and tokenize a list of prompts to a padded batch."""
    if chat:
        formatted = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": p}
            ], add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
    else:
        formatted = prompts

    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encoded.input_ids, encoded.attention_mask


