# Speculative Decoding Performance Benchmark

This repository is a PyTorch implementation of Speculative Decoding / Speculative Sampling ([Leviathan et al., 2023](#1); [Chen et al., 2023](#2)) with comprehensive performance benchmarking capabilities.

The project focuses on **automated performance testing** comparing Speculative Decoding with standard autoregressive generation, collecting detailed metrics including TTFT, latency, throughput, token counts, and GPU power consumption.

<p align="center">
    <img src="figures/example.png" alt="Example of generation." width="600"/>
    <br>
    <em>Figure 1: Example of generation, comparing Speculative Decoding and Vanilla Decoding.</em>
</p>

## What is Speculative Decoding?

Speculative Decoding is a decoding strategy for transformers that allows to generate sequences faster than the classic auto-regressive decoding without changing the output distribution or requiring further fine-tuning. It uses a smaller, more efficient approximation model (called a "drafter") to generate speculative token prefixes. These prefixes are then evaluated in parallel by the larger target model, reducing the number of serial decoding steps required and leading to inference speedups.

<p align="center">
    <img src="figures/specdec_method.png" alt="Overview of Speculative Decoding." width="600"/>
    <br>
    <em>Figure 2: Overview of Speculative Decoding.</em>
</p>

## Key Features

- ✅ **Automated Performance Benchmarking** - Compare Speculative Decoding vs. Target AR generation
- ✅ **Comprehensive Metrics Collection** - TTFT, latency, throughput, token counts, acceptance rates
- ✅ **GPU Power Monitoring** - Real-time GPU power consumption and performance tracking
- ✅ **Batch Inference Support** - Efficient batch processing for both speculative and autoregressive decoding
- ✅ **Multi-GPU Deployment** - Flexible GPU allocation strategies for optimal resource utilization
- ✅ **Detailed Performance Reports** - JSON output with complete benchmark results

## Installation

This project requires Python 3.7 or later and the following dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
```
rich
tqdm
termcolor
tokenizers>=0.19.1
torch>=2.3.0
transformers>=4.41.1
accelerate>=0.30.1
bitsandbytes>=0.43.1
numpy
```

## Quick Start

### 1. Configure GPU Allocation

Edit `run_benchmark.sh` to configure GPU allocation:

```bash
# Set available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Choose GPU allocation strategy
GPU_STRATEGY="multi_gpu_ratio"  # Options: multi_gpu_ratio, separate, same, auto

# GPU ratio configuration (for multi_gpu_ratio strategy)
TARGET_GPU_RATIO=6    # Target model uses 6 GPUs
DRAFTER_GPU_RATIO=2   # Drafter model uses 2 GPUs
```

For detailed GPU deployment information, see [GPU Deployment Guide](docs/GPU_DEPLOYMENT.md).

### 2. Configure Benchmark Parameters

Edit `run_benchmark.sh` to set benchmark parameters:

```bash
# Benchmark mode (choose one):
# Option 1: Fixed number of prompts
export NUM_PROMPTS=100

# Option 2: Time-based (duration + rate)
export AUTO_RATE=1.0              # Requests per second
export AUTO_DURATION=300          # Duration in seconds

# Batch processing
export ENABLE_BATCH="true"        # Enable batch processing
export BATCH_SIZE=4               # Batch size
export MAX_BATCH_LENGTH=512       # Max sequence length

# Generation parameters
export GENERATION_LENGTH=100      # Generation length in tokens
export GAMMA_VALUE=4              # Gamma parameter for speculative decoding

# Feature flags
export ENABLE_SPECULATIVE="true"  # Enable speculative decoding
export ENABLE_TARGET="true"       # Enable target AR (for comparison)
export ENABLE_GPU_MONITOR="true"  # Enable GPU power monitoring
```

### 3. Run Benchmark

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

The benchmark will:
1. Load models (Target and Drafter) according to GPU configuration
2. Load prompts from ShareGPT dataset
3. Run automated performance tests
4. Collect performance metrics and GPU power consumption
5. Generate comprehensive reports

### 4. View Results

Results are saved to:
- `benchmark_results.json` - Complete benchmark results (speculative + target + GPU metrics)
- `benchmark_results_speculative.json` - Speculative decoding results only
- `benchmark_results_target.json` - Target AR results only
- `benchmark_results_gpu.json` - GPU monitoring results only

A formatted summary is also printed to the console.

## Performance Metrics

The benchmark collects the following metrics:

### Inference Metrics
- **TTFT (Time To First Token)** - Latency to first generated token
- **End-to-End Latency** - Total generation time per request
- **Throughput** - Tokens per second
- **Token Counts** - Prompt tokens, generated tokens, total tokens
- **Acceptance Rate** - Draft acceptance rate (for speculative decoding)

### GPU Metrics
- **Power Consumption** - Average, peak, and total energy consumption
- **GPU Utilization** - Average GPU utilization percentage
- **Memory Usage** - Average memory usage percentage
- **Temperature** - Peak GPU temperature

## Project Structure

```
Speculative-Decoding/
├── benchmark.py              # Main benchmark runner
├── run_benchmark.sh          # Benchmark configuration and launch script
├── engine/                   # Core engine modules
│   ├── infer_engine.py      # Batch inference engine
│   ├── metrics.py           # Performance metrics collection
│   ├── gpu_monitor.py       # GPU power/performance monitoring
│   ├── models.py            # Model loading utilities
│   ├── dataset.py           # Dataset loading utilities
│   └── batch_decode.py      # Batch tokenization utilities
├── sampling/                 # Decoding strategies
│   ├── speculative_decoding.py
│   ├── base_decoding.py
│   └── ...
├── ngram_assisted/          # N-gram assisted decoding
├── docs/                    # Documentation
│   ├── GPU_DEPLOYMENT.md    # GPU deployment guide
│   └── GPU_DEPLOYMENT_CN.md # GPU deployment guide (Chinese)
└── sharegpt_gpt4/          # ShareGPT dataset
```

## GPU Deployment

The project supports flexible GPU allocation strategies:

- **multi_gpu_ratio** (default): Proportional allocation across multiple GPUs
- **separate**: Target and Drafter on different GPUs
- **same**: Both models share the same GPU
- **auto**: Automatic allocation by transformers

For detailed information, see [GPU Deployment Guide](docs/GPU_DEPLOYMENT.md).

Example: 8-GPU deployment with 6:2 ratio
```
Target model (8B):  GPUs 0-5 (layers auto-distributed)
Drafter model (1.7B): GPUs 6-7 (layers auto-distributed)
```

## Advanced Usage

### Programmatic API

You can also use the benchmark programmatically:

```python
from benchmark import BenchmarkRunner

# Initialize benchmark runner
runner = BenchmarkRunner(device="cuda:0")

# Benchmark will run automatically based on environment variables
```

### Custom Model Paths

Configure model paths via command line arguments (recommended):

```bash
python benchmark.py --target-model /path/to/target/model --drafter-model /path/to/drafter/model
```

Or set environment variables in `run_benchmark.sh`:

```bash
export TARGET_MODEL="/path/to/target/model"
export DRAFTER_MODEL="/path/to/drafter/model"
```

Command line arguments override environment variables if both are provided.

### Dataset Configuration

The benchmark loads prompts from ShareGPT JSONL files. Configure the dataset directory:

```bash
export SHAREGPT_DIR="/path/to/sharegpt/directory"
```

The benchmark will load prompts from:
- `sharegpt_gpt4.jsonl`
- `sharegpt_V3_format.jsonl`
- `sharegpt_zh_38K_format.jsonl`

## Model Requirements

For Speculative Decoding to work:
- The target model must be a transformer model (decoder only or encoder-decoder)
- The drafter model must share the same tokenizer as the target model
- Both models should output same shape logits
- The target model should be large enough to benefit from acceleration
- The drafter model should be small enough to be faster than the target model

## Batch Inference

The project supports efficient batch inference for both speculative and autoregressive decoding:

- **Batch Speculative Decoding**: Processes multiple prompts simultaneously
- **Batch Target AR**: Processes multiple prompts simultaneously
- **KV Cache Utilization**: Optimized KV cache usage for faster inference
- **Independent Acceptance Rates**: Per-sequence acceptance rate tracking

## N-gram Assisted Speculative Decoding

The project also includes **N-gram Assisted Speculative Decoding** (NASD), which replaces the drafter model with an N-gram storage. This approach is training-free and model-agnostic.

<p align="center">
    <img src="figures/nasd_method.png" alt="Overview of NASD method." width="600"/>
    <br>
    <em>Figure 3: Overview of NASD method.</em>
</p>

## Known Issues

### Cache Feature
The cache feature can be inconsistent depending on the model. For production use, ensure your models support KV cache properly, or disable cache if encountering issues.

## Troubleshooting

### GPU Memory Issues
- Reduce `BATCH_SIZE` or `MAX_BATCH_LENGTH`
- Reduce `GAMMA_VALUE`
- Use more GPUs for the target model

### Model Loading Issues
- Check model paths in `benchmark.py`
- Ensure models are compatible (same tokenizer, same logit shape)
- Verify GPU availability with `nvidia-smi`

### Benchmark Issues
- Check ShareGPT dataset directory path
- Verify environment variables are set correctly
- Check GPU allocation matches available GPUs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

<a id="1">[1]</a> Leviathan, Y., Kalman, M. & Matias, Y.. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of the 40th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 202:19274-19286 Available from https://proceedings.mlr.press/v202/leviathan23a.html.

<a id="2">[2]</a> Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318.

<a id="3">[3]</a> Jie Ou, Yueming Chen, Wenhong Tian. (2024). Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding. *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 10–22*

## License

See [LICENSE](LICENSE) file for details.
