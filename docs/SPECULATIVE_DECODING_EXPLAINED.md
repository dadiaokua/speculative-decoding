# Speculative Decoding 核心算法详解

本文档详细解释 `batch_speculative_generate` 方法的工作原理，这是整个 Speculative Decoding 系统的核心实现。

## 📖 什么是 Speculative Decoding？

Speculative Decoding（推测解码）是一种加速大模型推理的技术，核心思想是：
- **用小模型（Drafter）快速生成多个候选token**
- **用大模型（Target）并行验证这些候选**
- **通过拒绝采样确保输出分布与大模型一致**

**优势**：可以在不改变输出分布的前提下，加速2-3倍。

---

## 🔄 算法流程概览

```
┌─────────────────────────────────────────────────────────┐
│  输入: prompt tokens [batch_size, prompt_len]           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段1: 初始化Drafter KV Cache                          │
│  - 用prompt初始化drafter的KV cache                      │
│  - 避免后续重复计算prompt的attention                    │
└─────────────────────────────────────────────────────────┘
                        ↓
          ┌─────────────────────────┐
          │   主循环（每次γ个token）  │
          └─────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2: Drafter顺序生成γ个候选token                      │
│  - 使用KV cache逐token生成                              │
│  - 保存每个token的概率分布 q(x)                         │
│  - 从q(x)中采样得到候选token                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段3: Target并行验证所有γ个候选                        │
│  - 一次性前向传播，计算所有位置的logits                  │
│  - 得到每个位置的概率分布 p(x)                          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段4: 拒绝采样（Rejection Sampling）                   │
│  对每个候选token:                                         │
│  - 计算接受概率: accept_prob = min(1, p/q)              │
│  - 如果接受: 保留该token                                 │
│  - 如果拒绝: 从residual分布重新采样                      │
│    residual = max(0, p - q)                             │
└─────────────────────────────────────────────────────────┘
                        ↓
          ┌─────────────────────────┐
          │  继续下一个γ窗口?        │
          └─────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段5: 后处理                                           │
│  - 拼接prompt和生成的tokens                              │
│  - 计算每个序列的接受率                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🧩 关键步骤详解

### 阶段1: 初始化Drafter KV Cache

```python
init_out = ctx.drafter(input_ids, attention_mask=attention_mask, use_cache=True)
drafter_past = init_out.past_key_values
```

**作用**：
- 预先处理整个prompt，建立KV cache
- 后续生成时只需处理新token，大幅加速

**为什么重要**：
- 如果没有KV cache，每次生成都要重新计算整个序列的attention
- 使用KV cache后，只需计算新token与历史KV的attention

---

### 阶段2: Drafter顺序生成γ个候选token

```python
for draft_step in range(current_gamma):
    # 1. 获取当前位置的上一个token
    token_prev = generated_tokens[:, step + draft_step - 1]
    
    # 2. Drafter前向传播（使用KV cache）
    out = ctx.drafter(token_prev, past_key_values=drafter_past, use_cache=True)
    q_probs = torch.softmax(out.logits, dim=-1)
    
    # 3. 从概率分布中采样
    sampled_token = torch.multinomial(q_probs, 1)
    
    # 4. 更新KV cache
    drafter_past = out.past_key_values
```

**关键点**：
- **顺序生成**：每个draft token依赖于前一个draft token
- **使用KV cache**：避免重复计算历史token的attention
- **保存概率分布**：后续需要用到 `q(x)` 进行acceptance计算

**为什么顺序生成**：
- Drafter必须按照token序列的顺序生成，因为下一个token依赖于前面的context
- 但我们可以用KV cache加速这个过程

---

### 阶段3: Target并行验证所有γ个候选

```python
# 构建完整序列: prompt + 已生成tokens + γ个draft tokens
verify_ids = torch.cat([input_ids, generated_tokens[:, :step + current_gamma]], dim=1)

# Target一次性前向传播
t_out = ctx.target(verify_ids)
p_probs_full = torch.softmax(t_out.logits[:, -(current_gamma+1):-1, :], dim=-1)
```

**关键点**：
- **并行验证**：一次前向传播计算所有γ个位置的概率
- **不采样**：只计算概率分布，不生成token
- **高效**：相比顺序生成γ次，速度提升明显

**为什么能并行**：
- Target不需要逐token生成，只需计算给定序列的概率分布
- 可以使用矩阵运算一次性计算所有位置

---

### 阶段4: 拒绝采样（Rejection Sampling）

这是整个算法的核心，确保输出分布与Target模型一致。

#### 4.1 接受概率计算

```python
p_sample = p_vec[sampled_token]  # Target对采样token的概率
q_sample = q_vec[sampled_token]  # Drafter对采样token的概率

accept_prob = min(1.0, p_sample / q_sample)
```

**数学原理**：
- 如果 `p_target > p_drafter`：接受概率 = 1（总是接受）
- 如果 `p_target < p_drafter`：接受概率 = `p_target / p_drafter`（按比例接受）

**为什么这样设计**：
- 确保最终输出的token分布完全遵循Target模型的分布
- 即使Drafter采样了概率很低的token，只要Target认为它合理，也会接受

#### 4.2 接受token

```python
if torch.rand(1) < accept_prob:
    # 接受该token
    accepted_count += 1
    if sampled_token in ctx.end_tokens:
        finished[global_idx] = True  # 遇到结束token，停止生成
        break
```

#### 4.3 拒绝后重新采样

```python
else:
    # 拒绝该token，从residual分布重新采样
    residual = torch.clamp(p_vec - torch.minimum(p_vec, q_vec), min=0.0)
    corrected = torch.multinomial(residual / residual.sum(), 1)
```

**Residual分布的含义**：
- `residual = max(0, p_target - p_drafter)`
- 表示"Target认为重要但Drafter低估的部分"
- 从residual采样可以确保即使拒绝后，仍遵循Target的分布

**为什么需要residual采样**：
- 如果简单拒绝后从 `p_target` 采样，会导致过度偏向Target的分布
- Residual采样保持了正确的概率质量分配

**特殊情况处理**：
```python
if residual.sum() <= 1e-12:
    # 如果residual为空，直接按target分布采样
    corrected = torch.multinomial(p_vec, 1)
```

---

### 阶段5: 后处理

```python
# 移除填充的0，只保留实际生成的tokens
nonzero = torch.nonzero(gen_seq, as_tuple=True)[0]
final_gen = gen_seq[:nonzero[-1] + 1]

# 拼接prompt和生成的tokens
full_output = torch.cat([input_ids[i], final_gen])

# 计算接受率
accept_rate = drafts_accepted / drafts_generated
```

---

## 🎯 为什么这个算法能加速？

### 1. **并行化优势**
- **Drafter**：顺序生成γ个token（小模型，速度快）
- **Target**：并行验证γ个token（大模型，但只需一次前向传播）

### 2. **KV Cache优化**
- Drafter使用KV cache，避免重复计算
- 每个draft token只需一次前向传播

### 3. **接受率通常较高**
- 如果Drafter和Target分布相似，大部分token会被接受
- 平均每次可以接受2-3个token（取决于γ设置）

### 4. **数学保证**
- 拒绝采样确保输出分布完全符合Target模型
- 不会因为加速而牺牲质量

---

## 📊 性能指标

### 接受率（Acceptance Rate）
```
accept_rate = drafts_accepted / drafts_generated
```
- **理想情况**：接受率 > 70%，意味着大部分时间都在接受Drafter的提案
- **低接受率**：说明Drafter和Target差异较大，可能需要调整Drafter模型

### 加速比（Speedup）
```
speedup ≈ 1 / (1/γ + 1/accept_rate)
```
- 接受率越高，加速比越大
- γ越大，单次迭代处理的token越多，但可能降低接受率

---

## ⚠️ 关键实现细节

### 1. **Batch处理**
- 所有计算都是batch-wise的，充分利用GPU并行能力
- 需要处理不同序列可能在不同时刻完成的情况

### 2. **Early Stopping**
- 遇到结束token（EOS）时立即停止
- 每个序列独立判断是否完成

### 3. **Token位置管理**
- 需要准确跟踪每个token在序列中的位置
- 拒绝后需要清理后续无效的draft tokens

### 4. **概率分布保存**
- 必须保存Drafter的完整概率分布 `q_probs_full`
- 不能只保存采样token的概率，因为acceptance计算需要完整的q分布

---

## 🔍 与标准AR生成的区别

| 特性 | 标准AR生成 | Speculative Decoding |
|------|-----------|---------------------|
| **模型数量** | 1个（Target） | 2个（Drafter + Target） |
| **生成方式** | 顺序生成 | Drafter顺序 + Target并行验证 |
| **每次迭代** | 1个token | γ个token（可能） |
| **加速** | 1x | 2-3x（取决于接受率） |
| **输出分布** | Target分布 | Target分布（通过拒绝采样保证） |

---

## 📚 参考资料

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Speculative Decoding Explained](https://lilianweng.github.io/posts/2023-10-25-inference-optimization/)

---

## 💡 优化建议

1. **选择合适的γ值**
   - γ太小：加速不明显
   - γ太大：Drafter质量下降，接受率降低
   - 建议：γ = 3-5

2. **选择合适的Drafter模型**
   - 太小：分布差异大，接受率低
   - 太大：Drafter自己就很慢，失去加速意义
   - 建议：Drafter约为Target的1/3-1/5大小

3. **监控接受率**
   - 如果接受率 < 50%，考虑：
     - 减小γ值
     - 更换更好的Drafter模型
     - 调整Drafter的采样策略（如temperature）

