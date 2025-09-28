from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored


def load_causal_lm(model_name: str,
                   device_map,
                   torch_dtype,
                   quantization_config=None,
                   trust_remote_code: bool = True,
                   attn_implementation: str = "sdpa"):
    """Load a Causal LM with consistent options.

    Returns (model, tokenizer)
    """
    print(colored("Loading model:", "light_grey"), model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return model, tokenizer


