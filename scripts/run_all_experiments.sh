#!/bin/bash
################################################################################
# 论文改进实验一键运行脚本
# 按照优先级顺序运行所有关键实验
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
DATA_DIR="./data"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./experiments/results"
VIS_DIR="./visualization/results"

# 实验配置
NUM_STATISTICAL_RUNS=10
EPOCHS_MAIN=30
EPOCHS_CUB=30
EPOCHS_HYPERPARAM=20
SUBSET_RATIO=0.2

################################################################################
# 辅助函数
################################################################################

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_file_exists() {
    if [ -f "$1" ]; then
        print_success "找到文件: $1"
        return 0
    else
        print_error "文件不存在: $1"
        return 1
    fi
}

check_dir_exists() {
    if [ -d "$1" ]; then
        print_success "找到目录: $1"
        return 0
    else
        print_warning "目录不存在，将创建: $1"
        mkdir -p "$1"
        return 0
    fi
}

################################################################################
# 前置检查
################################################################################

print_header "前置条件检查"

# 检查Python环境
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
else
    print_error "未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 检查必要的Python包
print_warning "检查Python依赖包..."
python -c "import torch; import torchvision; import numpy; import matplotlib; import scipy" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "所有必要的Python包已安装"
else
    print_error "缺少必要的Python包，请运行: pip install -r requirements.txt"
    exit 1
fi

# 检查GPU可用性
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    print_success "GPU可用: $GPU_NAME"
    DEVICE="cuda"
else
    print_warning "GPU不可用，将使用CPU（速度会很慢）"
    DEVICE="cpu"
fi

# 检查目录
check_dir_exists "$CHECKPOINT_DIR"
check_dir_exists "$RESULTS_DIR"
check_dir_exists "$VIS_DIR"

# 检查数据集
if [ -d "$DATA_DIR/food-101" ]; then
    print_success "Food-101数据集已存在"
else
    print_warning "Food-101数据集不存在，将自动下载..."
fi

################################################################################
# 检查模型checkpoints
################################################################################

print_header "检查模型Checkpoints"

BASELINE_CKPT="$CHECKPOINT_DIR/baseline_best.pth"
TEACHER_CKPT="$CHECKPOINT_DIR/teacher_best.pth"
SIMAM_CKPT="$CHECKPOINT_DIR/simam_best.pth"
SIMAM_KD_CKPT="$CHECKPOINT_DIR/simam_kd_best.pth"

NEED_TRAIN=false

if ! check_file_exists "$BASELINE_CKPT"; then
    print_warning "需要训练基线模型"
    NEED_TRAIN=true
fi

if ! check_file_exists "$TEACHER_CKPT"; then
    print_warning "需要训练教师模型"
    NEED_TRAIN=true
fi

if ! check_file_exists "$SIMAM_CKPT"; then
    print_warning "需要训练SimAM模型"
    NEED_TRAIN=true
fi

if ! check_file_exists "$SIMAM_KD_CKPT"; then
    print_warning "需要训练SimAM+KD模型"
    NEED_TRAIN=true
fi

if [ "$NEED_TRAIN" = true ]; then
    echo -e "\n${YELLOW}需要先训练基础模型，这将需要较长时间...${NC}"
    read -p "是否现在训练？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_header "训练基础模型"
        
        # 训练基线
        if [ ! -f "$BASELINE_CKPT" ]; then
            echo "训练基线MobileNetV3..."
            python train1.py \
                --data-dir "$DATA_DIR" \
                --output-dir "$CHECKPOINT_DIR" \
                --epochs "$EPOCHS_MAIN" \
                --device "$DEVICE"
            mv "$CHECKPOINT_DIR/best_model.pth" "$BASELINE_CKPT" 2>/dev/null || true
        fi
        
        # 训练教师
        if [ ! -f "$TEACHER_CKPT" ]; then
            echo "训练ResNet-50教师..."
            python train_teacher.py \
                --data-dir "$DATA_DIR" \
                --output-dir "$CHECKPOINT_DIR" \
                --epochs "$EPOCHS_MAIN" \
                --device "$DEVICE"
        fi
        
        # 训练SimAM
        if [ ! -f "$SIMAM_CKPT" ]; then
            echo "训练SimAM模型..."
            python train_distillation.py \
                --data-dir "$DATA_DIR" \
                --attention-type simam \
                --output-dir "$CHECKPOINT_DIR" \
                --epochs "$EPOCHS_MAIN" \
                --device "$DEVICE" \
                --no-distillation
            mv "$CHECKPOINT_DIR/best_model.pth" "$SIMAM_CKPT" 2>/dev/null || true
        fi
        
        # 训练SimAM+KD
        if [ ! -f "$SIMAM_KD_CKPT" ]; then
            echo "训练SimAM+KD模型..."
            python train_distillation.py \
                --data-dir "$DATA_DIR" \
                --attention-type simam \
                --teacher-checkpoint "$TEACHER_CKPT" \
                --output-dir "$CHECKPOINT_DIR" \
                --epochs "$EPOCHS_MAIN" \
                --device "$DEVICE"
        fi
        
        print_success "所有基础模型训练完成！"
    else
        print_error "需要先训练基础模型才能继续。请稍后运行。"
        exit 1
    fi
fi

################################################################################
# 实验菜单
################################################################################

print_header "实验菜单"

echo "请选择要运行的实验："
echo "  1. [P0] Grad-CAM可视化分析 (~30分钟)"
echo "  2. [P0] CUB-200-2011泛化验证 (~8-10小时)"
echo "  3. [P1] 统计显著性检验 (~2-3天)"
echo "  4. [P2] 超参数交互分析 (~10-15小时)"
echo "  5. 运行所有实验（自动化）"
echo "  0. 退出"
echo ""
read -p "请输入选项 (0-5): " choice

case $choice in
    1)
        ################################################################################
        # 实验1: Grad-CAM可视化
        ################################################################################
        print_header "实验1: Grad-CAM可视化分析"
        
        python visualization/gradcam_analysis.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$VIS_DIR/gradcam" \
            --baseline-checkpoint "$BASELINE_CKPT" \
            --simam-checkpoint "$SIMAM_CKPT" \
            --simam-kd-checkpoint "$SIMAM_KD_CKPT" \
            --num-samples 20
        
        print_success "Grad-CAM可视化完成！"
        print_warning "请查看: $VIS_DIR/gradcam/"
        ;;
        
    2)
        ################################################################################
        # 实验2: CUB-200泛化验证
        ################################################################################
        print_header "实验2: CUB-200-2011泛化验证"
        
        # 检查CUB数据集
        if [ ! -d "$DATA_DIR/CUB_200_2011" ]; then
            print_warning "CUB-200-2011数据集不存在"
            echo "请从以下地址下载:"
            echo "https://data.caltech.edu/records/65de6-vp158"
            echo "然后解压到: $DATA_DIR/CUB_200_2011/"
            exit 1
        fi
        
        python experiments/train_cub200.py \
            --data-dir "$DATA_DIR/CUB_200_2011" \
            --output-dir "$RESULTS_DIR/cub200" \
            --device "$DEVICE"
        
        print_success "CUB-200实验完成！"
        print_warning "请查看: $RESULTS_DIR/cub200/"
        ;;
        
    3)
        ################################################################################
        # 实验3: 统计显著性检验
        ################################################################################
        print_header "实验3: 统计显著性检验"
        
        print_warning "这将进行 $NUM_STATISTICAL_RUNS 次独立运行，需要很长时间..."
        read -p "确认继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "已取消"
            exit 0
        fi
        
        python experiments/statistical_significance.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$RESULTS_DIR/statistical" \
            --num-runs "$NUM_STATISTICAL_RUNS" \
            --epochs "$EPOCHS_MAIN" \
            --teacher-checkpoint "$TEACHER_CKPT" \
            --device "$DEVICE"
        
        print_success "统计显著性检验完成！"
        print_warning "请查看: $RESULTS_DIR/statistical/"
        ;;
        
    4)
        ################################################################################
        # 实验4: 超参数交互分析
        ################################################################################
        print_header "实验4: 超参数交互分析"
        
        python experiments/hyperparameter_interaction.py \
            --teacher-checkpoint "$TEACHER_CKPT" \
            --data-dir "$DATA_DIR" \
            --output-dir "$RESULTS_DIR/hyperparameter" \
            --attention-type simam \
            --epochs "$EPOCHS_HYPERPARAM" \
            --subset-ratio "$SUBSET_RATIO" \
            --device "$DEVICE"
        
        print_success "超参数交互分析完成！"
        print_warning "请查看: $RESULTS_DIR/hyperparameter/"
        ;;
        
    5)
        ################################################################################
        # 运行所有实验
        ################################################################################
        print_header "运行所有实验"
        
        print_warning "这将依次运行所有4个实验，总共需要约3-5天时间"
        print_warning "建议在tmux或screen会话中运行，避免断线中断"
        read -p "确认继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "已取消"
            exit 0
        fi
        
        # 实验1
        print_header "[1/4] Grad-CAM可视化"
        python visualization/gradcam_analysis.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$VIS_DIR/gradcam" \
            --baseline-checkpoint "$BASELINE_CKPT" \
            --simam-checkpoint "$SIMAM_CKPT" \
            --simam-kd-checkpoint "$SIMAM_KD_CKPT" \
            --num-samples 20
        
        # 实验2
        if [ -d "$DATA_DIR/CUB_200_2011" ]; then
            print_header "[2/4] CUB-200泛化验证"
            python experiments/train_cub200.py \
                --data-dir "$DATA_DIR/CUB_200_2011" \
                --output-dir "$RESULTS_DIR/cub200" \
                --device "$DEVICE"
        else
            print_warning "跳过CUB-200实验（数据集未下载）"
        fi
        
        # 实验3
        print_header "[3/4] 统计显著性检验"
        python experiments/statistical_significance.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$RESULTS_DIR/statistical" \
            --num-runs "$NUM_STATISTICAL_RUNS" \
            --epochs "$EPOCHS_MAIN" \
            --teacher-checkpoint "$TEACHER_CKPT" \
            --device "$DEVICE"
        
        # 实验4
        print_header "[4/4] 超参数交互分析"
        python experiments/hyperparameter_interaction.py \
            --teacher-checkpoint "$TEACHER_CKPT" \
            --data-dir "$DATA_DIR" \
            --output-dir "$RESULTS_DIR/hyperparameter" \
            --attention-type simam \
            --epochs "$EPOCHS_HYPERPARAM" \
            --subset-ratio "$SUBSET_RATIO" \
            --device "$DEVICE"
        
        print_success "所有实验完成！"
        ;;
        
    0)
        print_warning "已退出"
        exit 0
        ;;
        
    *)
        print_error "无效选项"
        exit 1
        ;;
esac

################################################################################
# 完成总结
################################################################################

print_header "实验完成总结"

echo "实验结果位置:"
echo "  - Grad-CAM可视化: $VIS_DIR/gradcam/"
echo "  - CUB-200结果: $RESULTS_DIR/cub200/"
echo "  - 统计显著性: $RESULTS_DIR/statistical/"
echo "  - 超参数分析: $RESULTS_DIR/hyperparameter/"
echo ""
echo "下一步:"
echo "  1. 查看生成的图表和数据"
echo "  2. 更新论文中的占位符 (XX.XX)"
echo "  3. 参考 EXPERIMENTS_GUIDE.md 了解如何使用结果"
echo ""

print_success "所有任务完成！祝论文发表顺利！ 🎉"












