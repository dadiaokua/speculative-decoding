# GPU 部署机制说明（中文版）

## 📋 概述

本项目实现了两个模型的GPU部署：
- **Target模型** (Qwen3-8B): 大模型，用于高质量生成
- **Drafter模型** (Qwen3-1.7B): 小模型，用于快速草稿生成

## 🔄 部署流程

### 第一步：Shell脚本配置（run_benchmark.sh）

1. **设置可见GPU**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```
   限制程序只能看到这8张GPU

2. **选择GPU分配策略**
   ```bash
   GPU_STRATEGY="multi_gpu_ratio"  # 多GPU比例分配
   TARGET_GPU_RATIO=6              # Target用6张
   DRAFTER_GPU_RATIO=2             # Drafter用2张
   ```

3. **生成GPU字符串**
   - Target: `"cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5"`
   - Drafter: `"cuda:6,cuda:7"`

4. **导出环境变量**
   ```bash
   export TARGET_GPU="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5"
   export DRAFTER_GPU="cuda:6,cuda:7"
   ```

### 第二步：Python代码解析（benchmark.py）

1. **读取环境变量**
   ```python
   target_gpu_env = os.getenv("TARGET_GPU")  # "cuda:0,cuda:1,..."
   drafter_gpu_env = os.getenv("DRAFTER_GPU")  # "cuda:6,cuda:7"
   ```

2. **解析为device_map格式**
   ```python
   # "cuda:0,cuda:1,cuda:2" → {"": [0, 1, 2]}
   # 空字符串键 "" 表示自动分配模型层
   ```

3. **加载模型**
   ```python
   self.target = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map={"": [0,1,2,3,4,5]}  # 6张GPU自动分配层
   )
   
   self.drafter = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map={"": [6,7]}  # 2张GPU自动分配层
   )
   ```

### 第三步：Transformers自动分配

当使用 `device_map={"": [0,1,2]}` 时：

1. **自动分割模型层**
   - 例如：24层模型 + 3张GPU = 每张GPU 8层
   - Transformer层均匀分配到各个GPU

2. **处理层间通信**
   - 前一层输出自动传输到下一层所在的GPU
   - 使用CUDA流优化通信效率

3. **显存管理**
   - 每个GPU只存储分配到的层
   - KV cache也会相应分配到各GPU

## 🎯 部署示例

### 示例1：8卡部署（6:2分配）

```bash
# run_benchmark.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_STRATEGY="multi_gpu_ratio"
TARGET_GPU_RATIO=6
DRAFTER_GPU_RATIO=2
```

**实际部署**:
```
Target模型 (8B):
├─ GPU 0: 层 0-3
├─ GPU 1: 层 4-7
├─ GPU 2: 层 8-11
├─ GPU 3: 层 12-15
├─ GPU 4: 层 16-19
└─ GPU 5: 层 20-23

Drafter模型 (1.7B):
├─ GPU 6: 层 0-11
└─ GPU 7: 层 12-23
```

### 示例2：2卡分离部署

```bash
GPU_STRATEGY="separate"
```

**实际部署**:
```
Target模型 → GPU 0 (全部层)
Drafter模型 → GPU 1 (全部层)
```

### 示例3：单卡共享部署

```bash
GPU_STRATEGY="same"
```

**实际部署**:
```
Target模型 ─┐
            ├─→ GPU 0 (全部层，共享显存)
Drafter模型 ┘
```

## 🔍 关键点说明

### 1. device_map格式说明

**格式1：单GPU**
```python
device_map="cuda:0"  # 整个模型放在GPU 0
```

**格式2：多GPU自动分配（本项目使用）**
```python
device_map={"": [0,1,2]}  
# "" 空字符串 = 自动分配所有层
# [0,1,2] = 使用的GPU ID列表
```

**格式3：完全自动**
```python
device_map="auto"  # 让transformers完全自动决定
```

### 2. 为什么使用 {"": [0,1,2]} 格式？

- `{"": [0,1,2]}` 告诉transformers：
  - 使用GPU 0, 1, 2
  - 自动将模型层均匀分配到这些GPU
  - 自动处理层间数据传输

- 这样做的好处：
  - 无需手动指定每一层的位置
  - Transformers/Accelerate库会自动优化分配
  - 支持动态batch和KV cache管理

### 3. 两个模型的独立性

- Target和Drafter是**完全独立**的模型
- 它们可以在**不同的GPU上并行运行**
- 不会相互干扰，互不影响
- 这正是speculative decoding的优势：可以同时利用多个GPU

## 📊 实际GPU分配验证

加载模型后，可以通过以下方式验证：

```python
# 查看Target模型的参数分布
print("Target模型GPU分配:")
for name, param in self.target.named_parameters():
    if 'layers.0' in name or 'layers.11' in name or 'layers.23' in name:
        print(f"  {name}: {param.device}")

# 查看Drafter模型的参数分布  
print("Drafter模型GPU分配:")
for name, param in self.drafter.named_parameters():
    if 'layers.0' in name or 'layers.11' in name or 'layers.23' in name:
        print(f"  {name}: {param.device}")
```

## ⚠️ 注意事项

1. **CUDA_VISIBLE_DEVICES的影响**
   - 如果设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`
   - 程序只能看到这4张卡，会被映射为cuda:0,1,2,3
   - 无法访问物理GPU 4,5,6,7

2. **显存限制**
   - 多GPU部署时，每个GPU只需要存储部分层
   - 但KV cache仍会占用显存
   - 需要确保每个GPU有足够显存

3. **通信开销**
   - 多GPU时，层间数据传输有通信开销
   - 通常在模型较大（8层以上）时多GPU才有明显优势

4. **比例选择建议**
   - Target模型（8B）较大，建议分配更多GPU（如6张）
   - Drafter模型（1.7B）较小，分配较少GPU（如2张）即可
   - 总比例应该等于可用GPU数量

## 📚 相关文档

详细的技术说明请参考：`docs/GPU_DEPLOYMENT.md`

