#!/bin/bash
# 快速开始脚本 - 一键运行完整实验流程

echo "=========================================="
echo "轻量级食品图像分类 - 快速开始"
echo "=========================================="

# 设置变量
DATA_DIR="./data"
TEACHER_MODEL="teacher_resnet50.pth"
OUTPUT_DIR="./checkpoints"

# 步骤1: 检查数据集
echo ""
echo "[步骤 1/4] 检查数据集..."
if [ ! -d "$DATA_DIR/food-101" ]; then
    echo "  未找到Food-101数据集，开始下载..."
    python -c "from torchvision import datasets; datasets.Food101(root='$DATA_DIR', download=True)"
else
    echo "  ✓ 数据集已存在"
fi

# 步骤2: 训练教师模型(如果不存在)
echo ""
echo "[步骤 2/4] 准备教师模型..."
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "  未找到教师模型，开始训练..."
    python train_teacher.py \
        --data-dir $DATA_DIR \
        --epochs 10 \
        --batch-size 64 \
        --output $TEACHER_MODEL
else
    echo "  ✓ 教师模型已存在: $TEACHER_MODEL"
fi

# 步骤3: 训练学生模型 (ECA)
echo ""
echo "[步骤 3/4] 训练学生模型 (MobileNetV3 + ECA + KD)..."
python train_distillation.py \
    --data-dir $DATA_DIR \
    --attention-type eca \
    --teacher-checkpoint $TEACHER_MODEL \
    --temperature 4.0 \
    --alpha 0.7 \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.05 \
    --pretrained \
    --output-dir $OUTPUT_DIR/eca

# 步骤4: 测试模型
echo ""
echo "[步骤 4/4] 测试模型..."
if [ -f "$OUTPUT_DIR/eca/best_mobilenetv3_eca.pth" ]; then
    echo "  ✓ 训练完成！模型保存在: $OUTPUT_DIR/eca/best_mobilenetv3_eca.pth"
    echo ""
    echo "  可以使用以下命令测试模型:"
    echo "  python test_model.py --image <图片路径> --model-path $OUTPUT_DIR/eca/best_mobilenetv3_eca.pth --attention-type eca"
else
    echo "  ✗ 训练失败，请检查日志"
fi

echo ""
echo "=========================================="
echo "快速开始完成！"
echo "=========================================="
echo ""
echo "接下来可以："
echo "  1. 运行消融实验: ./run_ablation_study.sh"
echo "  2. 分析模型复杂度: python utils/model_complexity.py"
echo "  3. 测试其他注意力机制 (simam, cbam, se, coord)"
echo ""


