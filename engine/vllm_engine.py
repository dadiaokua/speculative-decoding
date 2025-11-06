"""vLLM引擎模块

本模块封装vLLM推理引擎的初始化和推理逻辑。
vLLM是一个高性能的LLM推理引擎，支持：
- 连续批处理（Continuous Batching）
- PagedAttention内存优化
- 张量并行和流水线并行
- 高吞吐量推理

使用方法：
    engine_manager = VLLMEngineManager(
        model_path="/path/to/model",
        tensor_parallel_size=8,
        gpu_memory_utilization=0.9
    )
    await engine_manager.initialize()
    output = await engine_manager.generate(prompt, max_tokens=100)
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 尝试导入vLLM
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.outputs import RequestOutput
    vllm_available = True
except ImportError:
    vllm_available = False
    AsyncLLMEngine = None
    AsyncEngineArgs = None
    SamplingParams = None
    RequestOutput = None


@dataclass
class VLLMConfig:
    """vLLM引擎配置"""
    model_path: str                          # 模型路径
    tensor_parallel_size: int = 8            # 张量并行大小
    pipeline_parallel_size: int = 1          # 流水线并行大小
    gpu_memory_utilization: float = 0.9      # GPU显存利用率
    max_model_len: int = 4096                # 最大模型长度
    max_num_seqs: int = 128                  # 最大并发序列数
    max_num_batched_tokens: Optional[int] = None  # 批处理最大token数（None=自动计算）
    disable_log_stats: bool = True           # 禁用日志统计
    dtype: str = "half"                      # 数据类型
    quantization: Optional[str] = None       # 量化方式
    enable_prefix_caching: bool = False      # 启用前缀缓存
    scheduling_policy: str = "priority"      # 调度策略
    
    # 推测解码参数
    enable_speculative: bool = False         # 是否启用推测解码
    speculative_model: Optional[str] = None  # 推测模型路径（drafter模型）
    num_speculative_tokens: int = 5          # 推测token数量
    use_v2_block_manager: bool = True        # 使用v2块管理器


class VLLMEngineManager:
    """vLLM引擎管理器
    
    负责vLLM引擎的初始化、配置和推理。
    """
    
    def __init__(self, config: VLLMConfig, logger: Optional[logging.Logger] = None):
        """
        初始化vLLM引擎管理器
        
        Args:
            config: vLLM配置对象
            logger: 日志记录器
        """
        if not vllm_available:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.engine: Optional[AsyncLLMEngine] = None
        self.request_counter = 0
        
    async def initialize(self):
        """初始化vLLM引擎"""
        try:
            # 设置环境变量
            self._setup_environment()
            
            # 创建引擎参数
            engine_args_dict = {
                "model": self.config.model_path,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "max_num_seqs": self.config.max_num_seqs,
                "disable_log_stats": self.config.disable_log_stats,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "dtype": self.config.dtype,
                "quantization": self.config.quantization,
            }
            
            # 添加 max_num_batched_tokens（如果设置）
            if self.config.max_num_batched_tokens is not None:
                engine_args_dict["max_num_batched_tokens"] = self.config.max_num_batched_tokens
            
            # 如果启用推测解码，添加推测解码参数
            if self.config.enable_speculative and self.config.speculative_model:
                engine_args_dict["speculative_model"] = self.config.speculative_model
                engine_args_dict["num_speculative_tokens"] = self.config.num_speculative_tokens
                engine_args_dict["use_v2_block_manager"] = self.config.use_v2_block_manager
                self.logger.info("🚀 正在启动vLLM引擎（推测解码模式）...")
            else:
                self.logger.info("🚀 正在启动vLLM引擎（标准模式）...")
            
            engine_args = AsyncEngineArgs(**engine_args_dict)
            
            self.logger.info(f"  Target模型: {engine_args.model}")
            if self.config.enable_speculative and self.config.speculative_model:
                self.logger.info(f"  Drafter模型: {self.config.speculative_model}")
                self.logger.info(f"  推测token数: {self.config.num_speculative_tokens}")
                self.logger.info(f"  V2块管理器: {self.config.use_v2_block_manager}")
            self.logger.info(f"  张量并行: {engine_args.tensor_parallel_size}")
            self.logger.info(f"  显存利用率: {engine_args.gpu_memory_utilization}")
            self.logger.info(f"  最大序列长度: {engine_args.max_model_len}")
            self.logger.info(f"  最大并发数: {engine_args.max_num_seqs}")
            if self.config.max_num_batched_tokens is not None:
                self.logger.info(f"  批处理最大tokens: {self.config.max_num_batched_tokens}")
            self.logger.info(f"  数据类型: {engine_args.dtype}")
            
            # 创建引擎实例
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # 给引擎初始化时间
            await asyncio.sleep(3)
            
            self.logger.info("✅ vLLM引擎启动成功！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ vLLM引擎启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_environment(self):
        """设置vLLM所需的环境变量"""
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        
        # 抑制vLLM详细日志
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> Optional[str]:
        """
        使用vLLM生成文本
        
        Args:
            prompt: 输入提示文本
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            
        Returns:
            生成的文本，如果失败返回None
        """
        if self.engine is None:
            self.logger.error("引擎未初始化")
            return None
        
        try:
            # 创建采样参数
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            
            # 生成唯一请求ID
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1
            
            # 提交生成请求
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id
            )
            
            # 等待生成完成
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                return final_output.outputs[0].text
            return None
            
        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            return None
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Optional[str]]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示文本列表
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            
        Returns:
            生成的文本列表
        """
        if self.engine is None:
            self.logger.error("引擎未初始化")
            return [None] * len(prompts)
        
        try:
            # 创建采样参数
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            # 提交所有请求
            request_ids = []
            for i, prompt in enumerate(prompts):
                request_id = f"batch_req_{self.request_counter}_{i}"
                request_ids.append(request_id)
                # vLLM会自动处理批处理
                self.engine.add_request(request_id, prompt, sampling_params)
            
            self.request_counter += 1
            
            # 等待所有请求完成
            # TODO: 实现批量结果收集
            # 当前简化为单个生成
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, max_tokens, temperature, top_p)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量生成失败: {e}")
            return [None] * len(prompts)
    
    async def shutdown(self):
        """关闭vLLM引擎"""
        if self.engine:
            self.logger.info("正在关闭vLLM引擎...")
            # vLLM没有显式的shutdown方法，但可以清理资源
            self.engine = None
            self.logger.info("✅ vLLM引擎已关闭")


def create_vllm_config_from_env() -> VLLMConfig:
    """从环境变量创建vLLM配置"""
    # 检查是否启用推测解码
    enable_speculative = os.getenv("VLLM_ENABLE_SPECULATIVE", "false").lower() == "true"
    
    # 如果启用推测解码，使用GAMMA_VALUE作为num_speculative_tokens的默认值
    num_speculative_tokens = int(os.getenv("VLLM_NUM_SPECULATIVE_TOKENS") or os.getenv("GAMMA_VALUE", "5"))
    
    # 读取 max_num_batched_tokens（可选参数）
    max_num_batched_tokens_str = os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS")
    max_num_batched_tokens = int(max_num_batched_tokens_str) if max_num_batched_tokens_str else None
    
    config = VLLMConfig(
        model_path=os.getenv("TARGET_MODEL", "/home/llm/model_hub/Qwen3-8B"),
        tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "8")),
        pipeline_parallel_size=int(os.getenv("VLLM_PIPELINE_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
        max_num_seqs=int(os.getenv("VLLM_MAX_NUM_SEQS", "128")),
        max_num_batched_tokens=max_num_batched_tokens,
        disable_log_stats=os.getenv("VLLM_DISABLE_LOG_STATS", "true").lower() == "true",
        dtype=os.getenv("VLLM_DTYPE", "half"),
        quantization=os.getenv("VLLM_QUANTIZATION") if os.getenv("VLLM_QUANTIZATION") else None,
        # 推测解码参数
        enable_speculative=enable_speculative,
        speculative_model=os.getenv("DRAFTER_MODEL") if enable_speculative else None,
        num_speculative_tokens=num_speculative_tokens,
        use_v2_block_manager=os.getenv("VLLM_USE_V2_BLOCK_MANAGER", "true").lower() == "true",
    )
    
    return config


# 检查vLLM是否可用
def is_vllm_available() -> bool:
    """检查vLLM是否已安装且可用"""
    return vllm_available

