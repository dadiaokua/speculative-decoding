# 启动脚本使用说明

## 📁 脚本文件说明

### 1. `run_speculative.sh` - 完整功能脚本
包含所有配置选项和环境检查的完整启动脚本。

**特性:**
- 🔍 自动环境检查
- 🎛️ 丰富的配置选项
- 🔧 命令行参数支持
- 📊 配置摘要显示
- 🎯 交互式确认

### 2. `run_simple.sh` - 简化启动脚本
快速启动的简化版本。

**特性:**
- 🚀 一键启动
- 📋 基础配置显示
- 💡 常用命令提示

### 3. `configs/multi_gpu_config.sh` - 多GPU专用配置
针对多GPU部署优化的启动脚本。

**特性:**
- 🔧 5卡配置 (4+1)
- 📊 GPU状态检查
- 🎯 流水线并行说明

### 4. `configs/performance_config.sh` - 性能优化配置
性能调优的启动配置。

**特性:**
- ⚡ 性能优化设置
- 📈 推荐参数
- 💡 监控命令

## 🚀 使用方法

### 基础使用
```bash
# 给脚本添加执行权限
chmod +x run_simple.sh
chmod +x run_speculative.sh
chmod +x configs/*.sh

# 快速启动
./run_simple.sh

# 完整功能启动
./run_speculative.sh
```

### 多GPU部署
```bash
# 多GPU配置启动
./configs/multi_gpu_config.sh

# 或者使用完整脚本的多GPU模式
./run_speculative.sh --device-mode multi_gpu
```

### 性能优化启动
```bash
# 性能优化配置
./configs/performance_config.sh

# 或者使用完整脚本的自定义参数
./run_speculative.sh --gamma 6 --temperature 0.9 --debug
```

### 命令行参数 (完整脚本)
```bash
# 查看帮助
./run_speculative.sh --help

# 自定义配置
./run_speculative.sh --device-mode single --gamma 4 --length 100
./run_speculative.sh --device-mode multi_gpu --temperature 0.8
./run_speculative.sh --processor nucleus --no-speculative --debug
```

## ⚙️ 配置修改

### 修改模型配置
编辑脚本中的以下变量：
```bash
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFTER_MODEL="meta-llama/Llama-3.2-1B-Instruct"
```

### 修改GPU配置
```bash
# 单GPU
SINGLE_DEVICE="cuda:0"

# 多GPU
MULTI_GPU_DEVICES="0,1,2,3,4"
```

### 修改生成参数
```bash
GAMMA=4                    # Speculative decoding的gamma值
GENERATION_LENGTH=50       # 生成长度
TEMPERATURE=0.8           # 温度参数
TOP_P=0.9                 # Nucleus采样的top-p
```

## 🎯 运行时配置

启动后，可以使用以下交互命令：

### 基础控制
```bash
/help                     # 显示所有命令
/quit                     # 退出程序
/clear                    # 清屏
```

### 参数调整
```bash
/gamma 6                  # 设置gamma值
/length 100               # 设置生成长度
/processor nucleus 0.8 0.9 # 设置采样器
```

### 功能开关
```bash
/speculative              # 切换speculative decoding
/target                   # 切换target模型生成
/drafter                  # 切换drafter独立生成
/ngram                    # 切换n-gram辅助
/debug                    # 切换调试模式
/cache                    # 切换缓存功能
```

## 🔧 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 减少gamma值或启用量化
   /gamma 2
   # 检查GPU内存使用
   nvidia-smi
   ```

2. **模型加载失败**
   ```bash
   # 检查网络连接和HuggingFace访问
   # 或使用本地模型路径
   ```

3. **性能较差**
   ```bash
   # 调整gamma值找到最优acceptance rate
   /gamma 4
   /gamma 6
   # 启用缓存
   /cache
   ```

### 环境检查
```bash
# 检查CUDA
nvidia-smi

# 检查Python环境
python --version
pip list | grep torch

# 检查项目文件
ls -la /Users/myrick/GithubProjects/Speculative-Decoding/
```

## 📊 性能监控

### GPU监控
```bash
# 实时GPU使用率
nvidia-smi -l 1

# GPU内存使用详情
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 1
```

### 系统监控
```bash
# CPU和内存使用
htop

# 进程监控
ps aux | grep python
```

## 💡 最佳实践

1. **首次运行**: 使用 `run_simple.sh` 快速测试
2. **生产环境**: 使用 `run_speculative.sh` 完整配置
3. **多GPU部署**: 使用 `configs/multi_gpu_config.sh`
4. **性能调优**: 使用 `configs/performance_config.sh`

5. **参数调优顺序**:
   - 先确保模型正常加载
   - 调整gamma值优化acceptance rate
   - 调整采样参数优化生成质量
   - 启用缓存和其他优化功能
