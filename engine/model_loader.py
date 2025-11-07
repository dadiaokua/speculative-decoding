"""Model Loader - 模型加载器

负责加载 Transformers 模型和 tokenizer。
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
from typing import Dict, List, Any


def load_transformers_models(
    target_model_path: str,
    drafter_model_path: str,
    target_gpu_env: str = "cuda:0",
    drafter_gpu_env: str = "cuda:0",
    model_dtype=torch.float16
) -> Dict[str, Any]:
    """加载 Transformers 模型
    
    Args:
        target_model_path: 目标模型路径
        drafter_model_path: 草稿模型路径
        target_gpu_env: 目标模型GPU配置（如 "cuda:0" 或 "cuda:0,cuda:1,..."）
        drafter_gpu_env: 草稿模型GPU配置
        model_dtype: 模型数据类型
    
    Returns:
        包含模型、tokenizer、设备等信息的字典
    """
    def parse_device_map(gpu_string):
        """解析GPU字符串为 device_map 格式"""
        if gpu_string == "auto":
            return "auto"
        elif "," in gpu_string:
            return "auto"
        else:
            return gpu_string
    
    def get_gpu_ids(gpu_string):
        """从字符串提取 GPU IDs"""
        if "," in gpu_string:
            gpu_ids = []
            for gpu in gpu_string.split(","):
                if gpu.startswith("cuda:"):
                    gpu_ids.append(int(gpu.split(":")[1]))
            return gpu_ids
        elif gpu_string.startswith("cuda:"):
            return [int(gpu_string.split(":")[1])]
        return []
    
    def resolve_primary_device(gpu_ids, fallback: str = "cuda:0"):
        """解析主设备（考虑 CUDA_VISIBLE_DEVICES 重映射）"""
        if gpu_ids:
            return torch.device("cuda:0")
        if torch.cuda.is_available():
            return torch.device(fallback)
        return torch.device("cpu")
    
    target_device_map = parse_device_map(target_gpu_env)
    drafter_device_map = parse_device_map(drafter_gpu_env)
    
    target_gpu_ids = get_gpu_ids(target_gpu_env)
    drafter_gpu_ids = get_gpu_ids(drafter_gpu_env)
    
    target_device = resolve_primary_device(target_gpu_ids)
    drafter_device = resolve_primary_device(drafter_gpu_ids)
    
    print(colored("Loading models...", "light_grey"))
    print(colored(f"Target: {target_model_path} on GPUs {target_gpu_ids} (device_map={target_device_map})", "yellow"))
    print(colored(f"Drafter: {drafter_model_path} on GPUs {drafter_gpu_ids} (device_map={drafter_device_map})", "yellow"))
    
    # 保存原始 CUDA_VISIBLE_DEVICES
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    # 加载目标模型
    if target_gpu_ids and target_device_map == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, target_gpu_ids))
    
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        quantization_config=None,
        dtype=model_dtype,
        device_map=target_device_map,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    target.eval()
    
    # 恢复 CUDA_VISIBLE_DEVICES
    if original_cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载草稿模型
    if drafter_gpu_ids and drafter_device_map == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, drafter_gpu_ids))
    
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_model_path,
        quantization_config=None,
        dtype=model_dtype,
        device_map=drafter_device_map,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    drafter.eval()
    
    # 恢复 CUDA_VISIBLE_DEVICES
    if original_cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    
    # 结束标记
    end_tokens = [tokenizer.eos_token_id]
    
    # Qwen 特殊结束标记
    try:
        qwen_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if qwen_end_token is not None and qwen_end_token != tokenizer.unk_token_id:
            end_tokens.append(qwen_end_token)
    except:
        pass
    
    print(colored("✅ Models loaded successfully", "green"))
    
    return {
        'target': target,
        'drafter': drafter,
        'tokenizer': tokenizer,
        'end_tokens': end_tokens,
        'target_device': target_device,
        'drafter_device': drafter_device,
        'target_gpu_ids': target_gpu_ids,
        'drafter_gpu_ids': drafter_gpu_ids,
    }

