# GPU 部署机制说明

本文档详细说明 Speculative Decoding 项目中两个模型（Target 和 Drafter）的 GPU 部署机制。

## 📋 目录
1. [整体架构](#整体架构)
2. [GPU 选择流程](#gpu-选择流程)
3. [部署策略](#部署策略)
4. [device_map 参数详解](#devicemap-参数详解)
5. [配置示例](#配置示例)

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  run_benchmark.sh (Shell配置层)                        │
│  - 设置 CUDA_VISIBLE_DEVICES                           │
│  - 配置 GPU_STRATEGY                                    │
│  - 导出 TARGET_GPU 和 DRAFTER_GPU 环境变量              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  benchmark.py (Python应用层)                           │
│  - 读取 TARGET_GPU 和 DRAFTER_GPU 环境变量              │
│  - parse_device_map() 解析GPU字符串                     │
│  - 转换为 device_map 格式                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  transformers.AutoModelForCausalLM.from_pretrained()    │
│  - device_map 参数控制GPU分配                           │
│  - accelerate 库自动处理多GPU并行                       │
└─────────────────────────────────────────────────────────┘
```

## 🔄 GPU 选择流程

### 1. Shell 脚本层（run_benchmark.sh）

**步骤 1: 设置可见GPU**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
这限制了程序只能看到这8张GPU卡（0-7）。

**步骤 2: 选择GPU分配策略**
```bash
GPU_STRATEGY="multi_gpu_ratio"  # 可选: multi_gpu_ratio, separate, same, auto
```

**步骤 3: 根据策略生成GPU字符串**

- **multi_gpu_ratio 策略**（默认）:
  ```bash
  TARGET_GPU_RATIO=6    # Target模型使用6张GPU
  DRAFTER_GPU_RATIO=2   # Drafter模型使用2张GPU
  
  # 生成:
  TARGET_GPU="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5"
  DRAFTER_GPU="cuda:6,cuda:7"
  ```

- **separate 策略**:
  ```bash
  TARGET_GPU="cuda:0"
  DRAFTER_GPU="cuda:1"
  ```

- **same 策略**:
  ```bash
  TARGET_GPU="cuda:0"
  DRAFTER_GPU="cuda:0"  # 两个模型共享同一张GPU
  ```

- **auto 策略**:
  ```bash
  TARGET_GPU="auto"
  DRAFTER_GPU="auto"    # 让transformers自动决定
  ```

**步骤 4: 导出环境变量**
```bash
export TARGET_GPU="$TARGET_GPUS"
export DRAFTER_GPU="$DRAFTER_GPUS"
```

### 2. Python 代码层（benchmark.py）

**步骤 1: 读取环境变量**
```python
target_gpu_env = os.getenv("TARGET_GPU", "cuda:0")
drafter_gpu_env = os.getenv("DRAFTER_GPU", "cuda:0")
```

**步骤 2: 解析GPU字符串**
```python
def parse_device_map(gpu_string):
    if gpu_string == "auto":
        return "auto"
    elif "," in gpu_string:
        # 多GPU情况: "cuda:0,cuda:1,cuda:2"
        gpu_ids = []
        for gpu in gpu_string.split(","):
            if gpu.startswith("cuda:"):
                gpu_ids.append(int(gpu.split(":")[1]))
        return {"": gpu_ids}  # 关键：空字符串键表示自动分配层
    else:
        # 单GPU情况: "cuda:0"
        return gpu_string
```

**解析结果示例：**
- `"cuda:0"` → `"cuda:0"` (单GPU)
- `"cuda:0,cuda:1,cuda:2"` → `{"": [0, 1, 2]}` (多GPU自动分配)
- `"auto"` → `"auto"` (完全自动)

**步骤 3: 加载模型**
```python
self.target = AutoModelForCausalLM.from_pretrained(
    target_model,
    device_map=target_device_map,  # 例如: {"": [0,1,2,3,4,5]}
    ...
)

self.drafter = AutoModelForCausalLM.from_pretrained(
    drafter_model,
    device_map=drafter_device_map,  # 例如: {"": [6,7]}
    ...
)
```

### 3. Transformers/Accelerate 层

当 `device_map={"": [0,1,2]}` 时，transformers 会：

1. **自动分割模型层**：
   - 将模型的Transformer层均匀分配到指定的GPU上
   - 例如：24层模型 + 3张GPU = 每张GPU 8层

2. **处理层间数据传输**：
   - 前一层输出自动传输到下一层所在的GPU
   - 使用CUDA流优化通信

3. **管理显存**：
   - 每个GPU只存储分配到的层
   - 优化KV cache的存储位置

## 🎯 部署策略详解

### 策略 1: multi_gpu_ratio（推荐）

**适用场景**: 有多个GPU，希望充分利用硬件资源

**工作原理**:
```
8张GPU分配示例（TARGET_GPU_RATIO=6, DRAFTER_GPU_RATIO=2）:

Target模型 (8B):
├─ GPU 0 ─┤
├─ GPU 1 ─┤
├─ GPU 2 ─┤─ 模型层自动分割到6张GPU
├─ GPU 3 ─┤
├─ GPU 4 ─┤
└─ GPU 5 ─┘

Drafter模型 (1.7B):
├─ GPU 6 ─┤─ 模型层自动分割到2张GPU
└─ GPU 7 ─┘
```

**优点**:
- 充分利用多GPU资源
- 大模型分配更多GPU，小模型分配较少GPU
- 两个模型可以并行推理

**配置示例**:
```bash
# run_benchmark.sh
GPU_STRATEGY="multi_gpu_ratio"
TARGET_GPU_RATIO=6
DRAFTER_GPU_RATIO=2
```

### 策略 2: separate

**适用场景**: 只有2张GPU，简单分离部署

**工作原理**:
```
Target模型 → GPU 0
Drafter模型 → GPU 1
```

**优点**: 简单明了，资源隔离

**缺点**: 资源利用率可能不如多GPU分配

### 策略 3: same

**适用场景**: 只有1张GPU，或想测试单GPU性能

**工作原理**:
```
Target模型 ─┐
            ├─→ GPU 0 (共享)
Drafter模型 ┘
```

**优点**: 适合资源受限环境

**缺点**: 两个模型竞争GPU资源，可能影响性能

### 策略 4: auto

**适用场景**: 让系统自动决定最优分配

**工作原理**: transformers会自动检测GPU并分配

**优点**: 无需手动配置

**缺点**: 可能不是最优分配

## 🔧 device_map 参数详解

### 格式 1: 字符串格式（单GPU）
```python
device_map="cuda:0"
device_map="cuda:1"
```
- 整个模型放在指定GPU上
- 适用于单GPU部署

### 格式 2: 自动分配格式（多GPU）
```python
device_map={"": [0, 1, 2, 3]}
```
- `""` (空字符串) 表示"自动分配所有层"
- `[0, 1, 2, 3]` 是要使用的GPU ID列表
- Transformers会将模型层均匀分配到这些GPU
- 由 `accelerate` 库处理层间通信

### 格式 3: 手动层分配格式（高级）
```python
device_map={
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 1,
    "model.layers.3": 1,
    "model.norm": 1,
    "lm_head": 1,
}
```
- 手动指定每一层所在的GPU
- 适用于需要精确控制的场景
- 本项目暂不使用此格式

### 格式 4: 完全自动
```python
device_map="auto"
```
- 让 transformers 完全自动决定
- 会考虑所有可见GPU和模型大小

## 📝 配置示例

### 示例 1: 8卡部署（6:2分配）

**run_benchmark.sh**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_STRATEGY="multi_gpu_ratio"
TARGET_GPU_RATIO=6
DRAFTER_GPU_RATIO=2
```

**结果**:
- Target模型: `device_map={"": [0,1,2,3,4,5]}`
- Drafter模型: `device_map={"": [6,7]}`

### 示例 2: 4卡部署（3:1分配）

**run_benchmark.sh**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_STRATEGY="multi_gpu_ratio"
TARGET_GPU_RATIO=3
DRAFTER_GPU_RATIO=1
```

**结果**:
- Target模型: `device_map={"": [0,1,2]}`
- Drafter模型: `device_map={"": [3]}`

### 示例 3: 2卡分离部署

**run_benchmark.sh**:
```bash
export CUDA_VISIBLE_DEVICES=0,1
GPU_STRATEGY="separate"
```

**结果**:
- Target模型: `device_map="cuda:0"`
- Drafter模型: `device_map="cuda:1"`

## 🔍 调试和验证

### 查看实际GPU分配

在模型加载后，可以检查：

```python
# 查看target模型的设备分配
for name, param in self.target.named_parameters():
    print(f"{name}: {param.device}")

# 查看drafter模型的设备分配
for name, param in self.drafter.named_parameters():
    print(f"{name}: {param.device}")
```

### 使用 nvidia-smi 监控

```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 查看每个GPU的进程
nvidia-smi pmon
```

## ⚠️ 注意事项

1. **CUDA_VISIBLE_DEVICES 的影响**:
   - 如果设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`
   - 在程序中，这4张卡会被映射为 `cuda:0, cuda:1, cuda:2, cuda:3`
   - 程序无法访问物理GPU 4,5,6,7

2. **多GPU通信开销**:
   - 使用多GPU时，层间数据传输会有通信开销
   - 通常在8层以上才会看到明显的性能提升

3. **显存管理**:
   - 多GPU部署时，每个GPU只需要存储部分层
   - 但KV cache仍然会占用显存，需要合理规划

4. **两个模型的独立性**:
   - Target和Drafter是完全独立的模型
   - 它们可以在不同的GPU上并行运行
   - 不会相互干扰

## 📚 参考资源

- [Hugging Face device_map 文档](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map)
- [Accelerate 库文档](https://huggingface.co/docs/accelerate/)
- [多GPU推理最佳实践](https://huggingface.co/docs/transformers/parallelism)

