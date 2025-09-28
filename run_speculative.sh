#!/bin/bash

# =============================================================================
# Speculative Decoding 启动脚本
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 配置参数
# =============================================================================

# 🎯 基础配置
PROJECT_DIR="/Users/myrick/GithubProjects/Speculative-Decoding"
PYTHON_ENV="python"  # 或者指定虚拟环境路径，如 "/path/to/venv/bin/python"

# 🔧 设备配置
DEVICE_MODE="single"  # single, multi_gpu, cpu
SINGLE_DEVICE="cuda:0"
MULTI_GPU_DEVICES="0,1,2,3,4"  # GPU IDs for multi-GPU setup

# 🧠 模型配置 (可选，如果要修改代码中的模型)
TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFTER_MODEL="meta-llama/Llama-3.2-1B-Instruct"
ENABLE_QUANTIZATION=true

# 🎛️ 生成参数
GAMMA=4
GENERATION_LENGTH=50
TEMPERATURE=0.8
TOP_P=0.9
PROCESSOR_TYPE="nucleus"  # greedy, multinomial, topk, nucleus, topknucleus

# 🔀 功能开关
ENABLE_SPECULATIVE=true
ENABLE_TARGET=true
ENABLE_DRAFTER=false
ENABLE_NGRAM=true
ENABLE_DEBUG=false
ENABLE_CACHE=false
ENABLE_CHAT=true

# 🧠 N-gram配置
NGRAM_TYPE="basic"  # basic, onelevel
NGRAM_N=3
TOP_K_FILLER=3
RESET_BETWEEN=true

# =============================================================================
# 函数定义
# =============================================================================

show_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Speculative Decoding                     ║"
    echo "║                     启动配置脚本                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_requirements() {
    print_info "检查环境要求..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "Python未找到，请安装Python 3.7+"
        exit 1
    fi
    
    # 检查CUDA (如果使用GPU)
    if [[ $DEVICE_MODE != "cpu" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            print_warning "nvidia-smi未找到，GPU功能可能不可用"
        else
            print_info "CUDA设备信息:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while read line; do
                echo -e "  ${CYAN}GPU $line${NC}"
            done
        fi
    fi
    
    # 检查项目目录
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "项目目录不存在: $PROJECT_DIR"
        exit 1
    fi
    
    # 检查主要文件
    if [ ! -f "$PROJECT_DIR/infer.py" ]; then
        print_error "入口文件不存在: $PROJECT_DIR/infer.py"
        exit 1
    fi
    
    print_success "环境检查完成"
}

setup_environment() {
    print_info "设置环境变量..."
    
    # 切换到项目目录
    cd "$PROJECT_DIR"
    
    # 设置CUDA设备
    case $DEVICE_MODE in
        "single")
            export CUDA_VISIBLE_DEVICES=${SINGLE_DEVICE#cuda:}
            print_info "单GPU模式: $SINGLE_DEVICE"
            ;;
        "multi_gpu")
            export CUDA_VISIBLE_DEVICES="$MULTI_GPU_DEVICES"
            print_info "多GPU模式: $MULTI_GPU_DEVICES"
            ;;
        "cpu")
            export CUDA_VISIBLE_DEVICES=""
            print_info "CPU模式"
            ;;
    esac
    
    # 设置其他环境变量
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告
    
    print_success "环境设置完成"
}

create_auto_config() {
    print_info "创建自动配置脚本..."
    
    local config_file="$PROJECT_DIR/auto_config.py"
    cat > "$config_file" << EOF
# 自动生成的配置文件
import sys
import time

def auto_configure():
    """自动配置参数"""
    commands = []
    
    # 基础参数配置
    commands.append("/gamma $GAMMA")
    commands.append("/length $GENERATION_LENGTH")
    
    # 处理器配置
    if "$PROCESSOR_TYPE" == "nucleus":
        commands.append("/processor nucleus $TEMPERATURE $TOP_P")
    elif "$PROCESSOR_TYPE" == "topk":
        commands.append("/processor topk $TEMPERATURE 50")  # 默认top_k=50
    elif "$PROCESSOR_TYPE" == "greedy":
        commands.append("/processor greedy $TEMPERATURE")
    elif "$PROCESSOR_TYPE" == "multinomial":
        commands.append("/processor multinomial $TEMPERATURE")
    
    # 功能开关
    if not $ENABLE_SPECULATIVE:
        commands.append("/speculative")
    if not $ENABLE_TARGET:
        commands.append("/target")
    if $ENABLE_DRAFTER:
        commands.append("/drafter")
    if not $ENABLE_NGRAM:
        commands.append("/ngram")
    if $ENABLE_DEBUG:
        commands.append("/debug")
    if $ENABLE_CACHE:
        commands.append("/cache")
    if not $ENABLE_CHAT:
        commands.append("/chat")
    
    # N-gram配置
    commands.append("/set_ngramstorage $NGRAM_TYPE $NGRAM_N")
    commands.append("/top_k_filler $TOP_K_FILLER")
    if not $RESET_BETWEEN:
        commands.append("/reset_in_between")
    
    return commands

if __name__ == "__main__":
    print("🤖 自动配置Speculative Decoding...")
    for cmd in auto_configure():
        print(f"执行命令: {cmd}")
        time.sleep(0.1)
EOF
    
    print_success "配置脚本已创建: $config_file"
}

show_config_summary() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                        配置摘要                               ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "${YELLOW}🎯 基础配置:${NC}"
    echo -e "  项目目录: ${GREEN}$PROJECT_DIR${NC}"
    echo -e "  Python环境: ${GREEN}$PYTHON_ENV${NC}"
    echo -e "  设备模式: ${GREEN}$DEVICE_MODE${NC}"
    
    echo -e "\n${YELLOW}🔧 设备配置:${NC}"
    case $DEVICE_MODE in
        "single")
            echo -e "  单GPU设备: ${GREEN}$SINGLE_DEVICE${NC}"
            ;;
        "multi_gpu")
            echo -e "  多GPU设备: ${GREEN}$MULTI_GPU_DEVICES${NC}"
            ;;
        "cpu")
            echo -e "  使用CPU运行"
            ;;
    esac
    
    echo -e "\n${YELLOW}🎛️ 生成参数:${NC}"
    echo -e "  Gamma: ${GREEN}$GAMMA${NC}"
    echo -e "  生成长度: ${GREEN}$GENERATION_LENGTH${NC}"
    echo -e "  采样器: ${GREEN}$PROCESSOR_TYPE${NC}"
    echo -e "  温度: ${GREEN}$TEMPERATURE${NC}"
    echo -e "  Top-p: ${GREEN}$TOP_P${NC}"
    
    echo -e "\n${YELLOW}🔀 功能开关:${NC}"
    echo -e "  Speculative: ${GREEN}$ENABLE_SPECULATIVE${NC}"
    echo -e "  Target: ${GREEN}$ENABLE_TARGET${NC}"
    echo -e "  Drafter: ${GREEN}$ENABLE_DRAFTER${NC}"
    echo -e "  N-gram: ${GREEN}$ENABLE_NGRAM${NC}"
    echo -e "  调试: ${GREEN}$ENABLE_DEBUG${NC}"
    echo -e "  缓存: ${GREEN}$ENABLE_CACHE${NC}"
    echo -e "  聊天: ${GREEN}$ENABLE_CHAT${NC}"
    
    echo -e "\n${YELLOW}🧠 N-gram配置:${NC}"
    echo -e "  类型: ${GREEN}$NGRAM_TYPE${NC}"
    echo -e "  N值: ${GREEN}$NGRAM_N${NC}"
    echo -e "  Top-k填充: ${GREEN}$TOP_K_FILLER${NC}"
    echo -e "  重置间隔: ${GREEN}$RESET_BETWEEN${NC}"
    
    echo ""
}

start_application() {
    print_info "启动Speculative Decoding..."
    
    # 构建启动命令
    local device_arg=""
    case $DEVICE_MODE in
        "single")
            device_arg="--device $SINGLE_DEVICE"
            ;;
        "multi_gpu")
            device_arg="--device cuda:0"  # 主设备
            ;;
        "cpu")
            device_arg="--device cpu"
            ;;
    esac
    
    # 启动应用
    print_success "正在启动应用程序..."
    echo -e "${PURPLE}提示: 启动后可以输入以下命令进行配置:${NC}"
    echo -e "  ${CYAN}/gamma $GAMMA${NC}"
    echo -e "  ${CYAN}/length $GENERATION_LENGTH${NC}"
    echo -e "  ${CYAN}/processor $PROCESSOR_TYPE $TEMPERATURE $TOP_P${NC}"
    echo -e "  ${CYAN}/help${NC} - 查看所有命令"
    echo ""
    
    # 执行启动命令
    exec $PYTHON_ENV infer.py $device_arg
}

# =============================================================================
# 主程序
# =============================================================================

main() {
    show_banner
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --device-mode)
                DEVICE_MODE="$2"
                shift 2
                ;;
            --gamma)
                GAMMA="$2"
                shift 2
                ;;
            --length)
                GENERATION_LENGTH="$2"
                shift 2
                ;;
            --temperature)
                TEMPERATURE="$2"
                shift 2
                ;;
            --processor)
                PROCESSOR_TYPE="$2"
                shift 2
                ;;
            --no-speculative)
                ENABLE_SPECULATIVE=false
                shift
                ;;
            --debug)
                ENABLE_DEBUG=true
                shift
                ;;
            --help|-h)
                echo "Speculative Decoding 启动脚本"
                echo ""
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --device-mode MODE     设备模式 (single|multi_gpu|cpu)"
                echo "  --gamma VALUE          设置gamma值"
                echo "  --length VALUE         设置生成长度"
                echo "  --temperature VALUE    设置温度"
                echo "  --processor TYPE       设置采样器类型"
                echo "  --no-speculative       禁用speculative decoding"
                echo "  --debug                启用调试模式"
                echo "  --help, -h             显示帮助信息"
                echo ""
                echo "示例:"
                echo "  $0                                    # 使用默认配置"
                echo "  $0 --device-mode multi_gpu --gamma 6 # 多GPU模式，gamma=6"
                echo "  $0 --debug --temperature 0.9         # 调试模式，高温度"
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                print_info "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 执行主流程
    check_requirements
    setup_environment
    show_config_summary
    
    # 询问是否继续
    echo -e "${YELLOW}是否继续启动? (y/N):${NC} \c"
    read -r response
    case $response in
        [yY][eE][sS]|[yY])
            start_application
            ;;
        *)
            print_info "启动已取消"
            exit 0
            ;;
    esac
}

# 运行主程序
main "$@"
